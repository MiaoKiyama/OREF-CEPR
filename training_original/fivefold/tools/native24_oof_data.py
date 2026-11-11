"""Fold-isolated adapters over immutable optical/SAR caches.

Native targets are semantic0..23; both unannotated raw0 and padding are255.
"""
from __future__ import annotations

import json
import copy
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tcmmsline3.native24_fast_data import FastFull120Dataset24
from tcmmsline3.ms75_sar_joint_data import MS75SARJointDataset, digest
from tcmmsline3.ms75_rebuild_data import load_manifest, read_cached_scene
from tcmmsline3.target_dino_data import normalize_bgrn
from tcmmsline3.ms75_data import BalancedPointIndex
from tcmmsline3.dinov3_sat_nir_data import (DINOv3SATNIRDataset,
    four_scene_block_order, sampling_order_audit)
from tcmmsline3.dinov3_sat_nir_protocol import (DINOV3_SAT_NIR_RARE_CLASS_IDS,
    DINOV3_SAT_NIR_HARD_CLASS_IDS)

SCHEMA = "native24_fivefold_data_v1_20261012"


def check_fold_config(config):
    data = config["data"]
    train, held = data["train_ids"], data["held_ids"]
    if (data.get("scope") != "oof_train96" or data.get("updates_per_epoch") != 480
            or data.get("seed") != 1337 or len(train) != 96 or len(held) != 24
            or len(set(train)) != 96 or len(set(held)) != 24 or set(train) & set(held)
            or data.get("fold_id") not in range(5)):
        raise ValueError("OOF data needs one exact, disjoint train96/held24 fold")
    for key in ("fold_id", "train_ids", "held_ids", "plans_manifest"):
        if key in config and config[key] != data[key]:
            raise ValueError("top-level/data fold contract differs: " + key)
    return data


class OOFFastDataset24(FastFull120Dataset24):
    def __init__(self, config, epoch=0, start_sample=0, *, load_sar=True):
        data = check_fold_config(config)
        # This initializer samples selection_train_ids and never broadens to120.
        MS75SARJointDataset.__init__(self, config, epoch, start_sample)
        self.load_sar = bool(load_sar)
        self.fold_id = data["fold_id"]
        bindings = data.get("sar_scene_bindings")
        if bindings is not None:
            if set(bindings) != set(self.scene_ids):
                raise ValueError("SAR bindings must cover exact train96")
            for scene, binding in bindings.items():
                if binding["available"] != self.availability[scene]:
                    raise ValueError("SAR availability changed")
                for asset in binding["assets"]:
                    stat = Path(asset["path"]).stat()
                    if stat.st_size != asset["size"] or stat.st_mtime_ns != asset["mtime_ns"]:
                        raise ValueError("SAR asset identity changed: " + asset["path"])
        if (list(self.scene_ids) != data["train_ids"]
                or self.metadata["held_ids"] != data["held_ids"]
                or self.samples_per_epoch != 30720 or self.updates != 480
                or self.metadata.get("oof_fold_id") != self.fold_id):
            raise ValueError("dataset manifest differs from the sealed OOF fold")
        plans = json.loads(Path(data["plans_manifest"]).read_text())
        expected = plans["epochs"][str(self.epoch)]
        if (plans["fold_id"] != self.fold_id
                or plans["train_ids"] != list(self.scene_ids)
                or plans["held_ids"] != self.metadata["held_ids"]
                or expected["plan_sha256"] != self.plan_sha256):
            raise ValueError("OOF sampling plan changed")

    def audit(self):
        # Deliberately independent of warmup/load_sar for exact epoch resumes.
        value = MS75SARJointDataset.audit(self)
        return dict(value, schema=SCHEMA, fold_id=self.fold_id,
                    train_ids=list(self.scene_ids), held_ids=self.metadata["held_ids"],
                    num_classes=24, samples=30720, updates_per_epoch=480,
                    official30_used=False, held_pixels_used_for_sampling=False,
                    supervision="semantic0..23; raw0_and_padding_ignore255",
                    class0_supervised=False, original_semantic_sampling_preserved=True)


def load_dino_index(data, metadata):
    return BalancedPointIndex.from_class_npz(
        data["dino_class_index"], expected_sha256=data["dino_class_index_sha256"],
        active_scene_ids=data["train_ids"], official_train_scene_ids=metadata["train_ids"],
        official_test_scene_ids=metadata["test_ids"],
        expected_source_split="official_train_oof_fold", require_exact_active_scene_ids=True)


def dino_planner(metadata, class_index, epoch):
    """Use the original balanced-class scheduler, including its precise RNG."""
    planner = object.__new__(DINOv3SATNIRDataset)
    planner.scene_ids = list(metadata["selection_train_ids"])
    planner.seed, planner.samples_per_epoch, planner.max_scene_repeat = 1337, 30720, 16
    planner.class_index = class_index
    planner.rare_class_ids = DINOV3_SAT_NIR_RARE_CLASS_IDS
    planner.hard_class_ids = DINOV3_SAT_NIR_HARD_CLASS_IDS
    planner._dinov3_sat_sampling_fractions = dict(scene=.5, rare=.3, hard=.2)
    planner.set_epoch(epoch)
    return planner


def dino_plan_audit(planner):
    blocks = planner._training_block_plan
    order = four_scene_block_order([b.scene_id for b in blocks])
    audit = sampling_order_audit(blocks, order, epoch=planner.epoch, physical_batch=4)
    if audit["arm_block_counts"] != dict(scene=960, rare=576, hard=384):
        raise ValueError("DINO must retain50/30/20 sampling")
    if set(b.scene_id for b in blocks) != set(planner.scene_ids):
        raise ValueError("DINO planner did not cover exact train96")
    return order, audit


class _MmapDinoReader:
    """Read-only replacement for TIFF decoding, with identical BGRN tensors."""
    def __init__(self, metadata, scene):
        self.image, self.labels = read_cached_scene(metadata, scene)
        self.height, self.width = self.labels.shape
        self.record = SimpleNamespace(scene_id=scene)
        self.normalization = metadata["normalization"]

    def _normalize_ms(self, raw):
        return normalize_bgrn(raw, self.normalization)

    def read_label_valid(self, window):
        y, x, h, w = map(int, (window.row_off, window.col_off, window.height, window.width))
        if x < 0 or y < 0 or x+w > self.width or y+h > self.height:
            raise ValueError("DINO training crop unexpectedly needs padding")
        label = np.asarray(self.labels[y:y+h, x:x+w], dtype=np.int64)
        valid = label != 255
        return np.where(valid, label, -1), valid

    def close(self):
        self.image, self.labels = None, None


class OOFDinoBase(DINOv3SATNIRDataset):
    def __init__(self, config, reference, epoch):
        data = check_fold_config(config)
        self.metadata = load_manifest(data["raw_manifest"])
        if (self.metadata["selection_train_ids"] != data["train_ids"]
                or self.metadata["held_ids"] != data["held_ids"]
                or self.metadata.get("oof_fold_id") != data["fold_id"]):
            raise ValueError("DINO source manifest differs from train96 fold")
        planner = dino_planner(self.metadata, load_dino_index(data, self.metadata), epoch)
        self.__dict__.update(planner.__dict__)
        self.config = copy.deepcopy(reference)
        norm = reference["data"]["normalization"]["ms"]
        if any(norm[k] != self.metadata["normalization"][k] for k in ("mean", "std")):
            raise ValueError("DINO reference normalization differs from sharedcache")
        self.patch_size, self.center_jitter = 512, 128
        self.sampling_attempts = int(reference["data"].get("sampling_attempts", 16))
        self.min_labeled_fraction = float(reference["data"].get("min_labeled_fraction", 0.0))
        self.horizontal_flip_probability = self.vertical_flip_probability = .5
        self._readers = OrderedDict()
        self.cache_scenes = int(data.get("cache_scenes", 2))

    def _reader(self, scene_id):
        if scene_id not in self.scene_ids:
            raise ValueError("held or official30 scene reached DINO reader")
        reader = self._readers.pop(scene_id, None)
        if reader is None:
            reader = _MmapDinoReader(self.metadata, scene_id)
        self._readers[scene_id] = reader
        while len(self._readers) > self.cache_scenes:
            self._readers.popitem(last=False)[1].close()
        return reader


class OOFDinoDataset(torch.utils.data.Dataset):
    def __init__(self, config, reference, epoch):
        self.base = OOFDinoBase(config, reference, epoch)
        self.scene_ids = self.base.scene_ids
        self.order, audit = dino_plan_audit(self.base)
        plans = json.loads(Path(config["data"]["plans_manifest"]).read_text())
        expected = plans["dino_epochs"][str(epoch)]
        if (plans["fold_id"] != config["fold_id"]
                or expected["ordered_plan_sha256"] != audit["ordered_plan_sha256"]):
            raise ValueError("sealed DINO50/30/20 plan changed")
        self.audit = dict(audit, role="native24_oof_dino_sampling", fold_id=config["fold_id"],
            train_scene_ids=self.scene_ids, held_scene_ids=config["data"]["held_ids"],
            num_classes=24, source_label_zero_supervised=False, artificial_padding_ignore=255,
            sampling_class_index_encoding="native24 semantic0..23; raw0 ignored255",
            official30_rasters_or_labels_read=False, mmap_shared_raw=True)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        sample = self.base[self.order[index]]
        target = sample["target"]
        if bool(((target < -1) | (target > 23)).any()):
            raise ValueError("DINO target encoding changed")
        return dict(ms=sample["ms"], label=torch.where(sample["ms_valid"] & (target >= 0), target, 255).long(),
                    index=index, scene_id=sample["scene_id"])

    def close(self):
        self.base.close()


def make_dino_dataset(config, reference, epoch):
    return OOFDinoDataset(config, reference, epoch)

class Full120SingleSampler:
    def __init__(self, *, updates, rank, world_size, start_update=0):
        if world_size!=1 or rank!=0 or not 0<=start_update<=updates: raise ValueError('single rank required')
        self.updates,self.start=updates,start_update
    def __iter__(self):return iter(range(self.start*64,self.updates*64))
    def __len__(self):return (self.updates-self.start)*64
