"""Train-only, label-side full120 data entry for the selected DINOv3-SAT expert.

The proven DINOv3-SAT local crop and 50/30/20 planner are reused unchanged.  The
constructor, provenance audit and receipt role are explicitly full120; no
fold96 alias, held24 sampling index or teacher prediction is substituted.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Sampler

from .data import FBPSamplingIndex, read_fbp_split, validate_fbp_splits
from .dinov3_sat_nir_data import (
    DINOv3SATNIRDataset,
    _validated_collate,
    exact_sampling_arm_schedule,
    four_scene_block_order,
    sampling_order_audit as _base_sampling_order_audit,
)
from .dinov3_sat_nir_full120_protocol import (
    DINOV3_SAT_NIR_FULL120_CLASS_INDEX,
    DINOV3_SAT_NIR_FULL120_CLASS_INDEX_SHA256,
    DINOV3_SAT_NIR_FULL120_SAMPLES_PER_EPOCH,
    DINOV3_SAT_NIR_FULL120_SPLIT,
    DINOV3_SAT_NIR_FULL120_STAGES,
)
from .dinov3_sat_nir_protocol import DINOV3_SAT_NIR_HARD_CLASS_IDS, DINOV3_SAT_NIR_RARE_CLASS_IDS
from .ms75_data import BalancedPointIndex, MS75MultiScaleDataset
from .runtime import atomic_json_dump


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_hash(scene_ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(scene_ids) + "\n").encode("utf-8")).hexdigest()


def _validate_data_config(config: Mapping[str, Any]) -> None:
    """Fail closed before generic scene discovery or any image is opened."""
    data = config["data"]
    stage = data.get("protocol_stage")
    if stage not in DINOV3_SAT_NIR_FULL120_STAGES:
        raise ValueError("DINOv3-SAT full120 data requires an explicit full120 stage")
    if config["experiment"].get("seed") != 1337:
        raise ValueError("DINOv3-SAT full120 data requires seed 1337")
    if data.get("dataset_format") != "fbp_ms" or data.get("dataset_variant") != "legacy":
        raise ValueError("DINOv3-SAT full120 data requires the pure-MS legacy interface")
    if data.get("official_120_30_protocol") is not True:
        raise ValueError("DINOv3-SAT full120 data must declare the official 120/30 protocol")
    if any(data.get(f"{mode}_split") != DINOV3_SAT_NIR_FULL120_SPLIT for mode in ("train", "val", "test")):
        raise ValueError("DINOv3-SAT full120 loader aliases must all be train")
    if data.get("load_sar") is not False or data.get("load_pan", False) is not False:
        raise ValueError("DINOv3-SAT full120 data forbids SAR and PAN")
    if config["model"].get("fusion_mode") != "ms_only":
        raise ValueError("DINOv3-SAT full120 data requires an MS-only model")
    if data.get("samples_per_epoch") != DINOV3_SAT_NIR_FULL120_SAMPLES_PER_EPOCH:
        raise ValueError("DINOv3-SAT full120 formal epoch must contain 38400 crops")
    if data.get("debug_scene_limit") is not None:
        raise ValueError("DINOv3-SAT full120 forbids scene truncation even in smoke/probe")
    expected_debug = {"dinov3_sat_nir_full120_train": None, "dinov3_sat_nir_full120_probe": 64, "dinov3_sat_nir_full120_smoke": 640}[str(stage)]
    if data.get("debug_samples_per_epoch") != expected_debug:
        raise ValueError("DINOv3-SAT full120 debug crop budget disagrees with its stage")
    dynamic = data["dynamic_sampling"]
    allowed = {"scene_fraction", "rare_fraction", "hard_fraction", "class_index", "class_index_sha256", "center_jitter", "max_scene_repeat"}
    if set(dynamic) != allowed:
        raise ValueError("DINOv3-SAT full120 rejects teacher/confusion/extra sampling fields")
    if tuple(dynamic[name] for name in ("scene_fraction", "rare_fraction", "hard_fraction")) != (0.5, 0.3, 0.2):
        raise ValueError("DINOv3-SAT full120 requires the original 50/30/20 sampling")
    if dynamic["class_index"] != DINOV3_SAT_NIR_FULL120_CLASS_INDEX or dynamic["class_index_sha256"] != DINOV3_SAT_NIR_FULL120_CLASS_INDEX_SHA256:
        raise ValueError("DINOv3-SAT full120 requires the pinned train120 label index")
    if dynamic["center_jitter"] != 128 or dynamic["max_scene_repeat"] != 16:
        raise ValueError("DINOv3-SAT full120 requires the original jitter/block policy")
    if data.get("rare_class_ids") != list(DINOV3_SAT_NIR_RARE_CLASS_IDS) or data.get("hard_class_ids") != list(DINOV3_SAT_NIR_HARD_CLASS_IDS):
        raise ValueError("DINOv3-SAT full120 rare/hard class lists changed")
    physical = config["train"].get("batch_size")
    if type(physical) is not int or physical != 8:
        raise ValueError("DINOv3-SAT full120 physical batch is invalid")
    if physical * config["train"].get("gradient_accumulation_steps", -1) != 64:
        raise ValueError("DINOv3-SAT full120 requires effective batch 64")


def _validate_index_provenance(class_index: FBPSamplingIndex, train_ids: Sequence[str]) -> None:
    if (class_index.source_split != "official_train" or class_index.scene_ids != list(train_ids)
            or class_index.seed != 1337 or class_index.stride != 16
            or any(not class_index.scenes_by_class[class_id] for class_id in range(24))):
        raise ValueError("DINOv3-SAT full120 class index lacks exact ordered train120 provenance")


def _validate_coordinate_labels(*, root: Path, train_ids: Sequence[str], class_index: FBPSamplingIndex) -> tuple[int, ...]:
    """Verify every point against a train120 label; never open test rasters."""
    per_class = [0] * 24
    for scene_id in train_ids:
        label_path = root / "labels" / f"{scene_id}.png"
        with Image.open(label_path) as source:
            if source.mode != "L":
                raise ValueError(f"FBP label must be indexed grayscale: {label_path}")
            label = np.asarray(source, dtype=np.uint8)
        height, width = label.shape
        for class_id, points in class_index.by_scene[scene_id].items():
            if not 0 < len(points) <= 64:
                raise ValueError("train120 class index exceeds its per-scene cap64")
            rows, columns = points[:, 0], points[:, 1]
            if (np.any(rows < 0) or np.any(columns < 0) or np.any(rows >= height)
                    or np.any(columns >= width) or np.any(label[rows, columns] != class_id + 1)):
                raise ValueError("class-index coordinate disagrees with train120 label")
            per_class[class_id] += len(points)
    if any(count <= 0 for count in per_class):
        raise ValueError("train120 label-side index must cover all 24 classes")
    return tuple(per_class)


def audit_dinov3_sat_nir_data_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    _validate_data_config(config)
    root = Path(str(config["data"]["root"])).expanduser().resolve()
    canonical = validate_fbp_splits(root)
    train_ids = read_fbp_split(root, DINOV3_SAT_NIR_FULL120_SPLIT)
    if len(train_ids) != 120 or train_ids != canonical["train"] or len(set(train_ids)) != 120:
        raise ValueError("DINOv3-SAT full120 requires all 120 official scenes in official order")
    if set(train_ids) & set(canonical["test"]):
        raise ValueError("DINOv3-SAT full120 training overlaps official30")
    class_path = (root / DINOV3_SAT_NIR_FULL120_CLASS_INDEX).resolve()
    if _sha256(class_path) != DINOV3_SAT_NIR_FULL120_CLASS_INDEX_SHA256:
        raise ValueError("DINOv3-SAT full120 class index SHA256 changed")
    class_index = FBPSamplingIndex(class_path)
    _validate_index_provenance(class_index, train_ids)
    per_class = _validate_coordinate_labels(root=root, train_ids=train_ids, class_index=class_index)
    return {
        "schema_version": 1,
        "role": "dinov3_sat_nir_full120_train_only_data_identity",
        "protocol_stage": str(config["data"]["protocol_stage"]),
        "data_root": str(root),
        "train_scene_count": 120,
        "heldout_scene_count": 0,
        "official_test_scene_count": len(canonical["test"]),
        "official30_split_metadata_read": True,
        "official30_rasters_or_labels_read": False,
        "full120_training_includes_former_held24": True,
        "validation_loader_enabled": False,
        "train_scene_ids_sha256": _scene_hash(train_ids),
        "split_file_sha256": {name: _sha256(root / "splits" / f"{name}.txt") for name in ("all", "train", "train_dev", "val", "test")},
        "class_index": str(class_path),
        "class_index_sha256": DINOV3_SAT_NIR_FULL120_CLASS_INDEX_SHA256,
        "class_index_all_24_classes": True,
        "sampling_index_teacher_predictions_used": False,
        "sampling_index_scene_ids_exact_ordered_train120": True,
        "sampling_coordinate_labels_verified": True,
        "rare_class_ids": list(DINOV3_SAT_NIR_RARE_CLASS_IDS),
        "hard_class_ids": list(DINOV3_SAT_NIR_HARD_CLASS_IDS),
        "rare_anchor_point_count": sum(per_class[index] for index in DINOV3_SAT_NIR_RARE_CLASS_IDS),
        "hard_anchor_point_count": sum(per_class[index] for index in DINOV3_SAT_NIR_HARD_CLASS_IDS),
        "class_anchor_point_counts": list(per_class),
        "sar_or_pan_rasters_read": False,
    }


def _validate_lightweight_training_identity(dataset: MS75MultiScaleDataset) -> None:
    if (len(dataset.scene_ids) != 120 or len(set(dataset.scene_ids)) != 120
            or tuple(dataset.scene_ids) != tuple(dataset.protocol.train_scene_ids)
            or set(dataset.scene_ids) & set(dataset.protocol.test_scene_ids)):
        raise ValueError("DINOv3-SAT full120 loader must contain exactly ordered official train120")
    if dataset.class_index is None or dataset.class_index.file_sha256 != DINOV3_SAT_NIR_FULL120_CLASS_INDEX_SHA256:
        raise ValueError("DINOv3-SAT full120 label-index identity changed")
    if getattr(dataset, "confusion_index", None) is not None:
        raise ValueError("DINOv3-SAT full120 rejects every teacher/confusion point index")
    if int(dataset.max_scene_repeat) != 16:
        raise ValueError("DINOv3-SAT full120 requires 16-crop source-local blocks")


class DINOv3SATNIRFull120Dataset(DINOv3SATNIRDataset):
    """Original per-index deterministic crop implementation, new full120 identity."""

    def __init__(self, config: dict, mode: str) -> None:
        if mode != "train":
            raise ValueError("DINOv3-SAT full120 fitting loader refuses every non-train mode")
        _validate_data_config(config)
        planner = copy.deepcopy(config)
        data = planner["data"]
        dynamic = copy.deepcopy(data["dynamic_sampling"])
        data["dataset_variant"] = "ms75_multiscale"
        # 'final' is the generic parent's real full120/train discovery mode,
        # not a train96 alias.  No full120 protocol validator sees this copy.
        data["protocol_stage"] = "final"
        data["short_context_size"] = 512
        data["short_input_size"] = 512
        data["global_input_size"] = 32
        data["horizontal_flip_probability"] = 0.5
        data["vertical_flip_probability"] = 0.5
        data["dynamic_sampling"] = {
            "scene_fraction": 1.0, "class_fraction": 0.0, "confusion_fraction": 0.0,
            "center_jitter": dynamic["center_jitter"], "max_scene_repeat": dynamic["max_scene_repeat"],
        }
        MS75MultiScaleDataset.__init__(self, planner, mode)
        # Keep the actual full120 configuration on every worker and receipt.
        self.config = copy.deepcopy(config)
        self.class_index = BalancedPointIndex.from_class_npz(
            self._resolve_index(dynamic["class_index"]),
            expected_sha256=DINOV3_SAT_NIR_FULL120_CLASS_INDEX_SHA256,
            active_scene_ids=self.scene_ids,
            official_train_scene_ids=self.protocol.train_scene_ids,
            official_test_scene_ids=self.protocol.test_scene_ids,
            expected_source_split="official_train",
            require_exact_active_scene_ids=True,
        )
        _validate_lightweight_training_identity(self)
        if set(self.class_index.keys) != set(range(24)):
            raise ValueError("DINOv3-SAT full120 target index must expose all 24 classes")
        self.rare_class_ids = DINOV3_SAT_NIR_RARE_CLASS_IDS
        self.hard_class_ids = DINOV3_SAT_NIR_HARD_CLASS_IDS
        self._dinov3_sat_sampling_fractions = {kind: float(dynamic[f"{kind}_fraction"]) for kind in ("scene", "rare", "hard")}
        self.sampling_fractions = dict(self._dinov3_sat_sampling_fractions)
        self.set_epoch(0)


def sampling_order_audit(blocks: Sequence[object], sample_order: Sequence[int], *, epoch: int, physical_batch: int = 1) -> dict[str, Any]:
    audit = _base_sampling_order_audit(blocks, sample_order, epoch=epoch, physical_batch=physical_batch)
    audit["role"] = "dinov3_sat_nir_full120_label_side_sampling_plan"
    del audit["sampling_coordinates_train96_labels_only"]
    audit["sampling_coordinates_train120_labels_only"] = True
    return audit


class _DynamicFull120FourSceneSampler(Sampler[int]):
    def __init__(self, dataset: DINOv3SATNIRFull120Dataset, *, start_index: int) -> None:
        self.dataset = dataset
        self.start_index = int(start_index)

    def __len__(self) -> int:
        return len(self.dataset) - self.start_index

    def __iter__(self) -> Iterator[int]:
        blocks = tuple(self.dataset._training_block_plan)
        order = four_scene_block_order([str(block.scene_id) for block in blocks])
        audit = sampling_order_audit(blocks, order, epoch=self.dataset.epoch, physical_batch=int(self.dataset.config["train"]["batch_size"]))
        audit.update({
            "train_scene_count": len(self.dataset.scene_ids),
            "train_scene_ids_sha256": _scene_hash(self.dataset.scene_ids),
            "class_index_sha256": self.dataset.class_index.file_sha256,
            "rare_class_ids": list(self.dataset.rare_class_ids),
            "hard_class_ids": list(self.dataset.hard_class_ids),
            "resume_start_index": self.start_index,
            "targeted_arm_class_block_counts": {
                kind: {str(class_id): sum(block.kind == kind and block.key == class_id for block in blocks)
                       for class_id in (self.dataset.rare_class_ids if kind == "rare" else self.dataset.hard_class_ids)}
                for kind in ("rare", "hard")
            },
        })
        output = Path(str(self.dataset.config["experiment"]["output_dir"]))
        atomic_json_dump(audit, output / "sampling_audits" / f"epoch_{self.dataset.epoch:04d}.json")
        return iter(order[self.start_index:])


def build_dinov3_sat_nir_dataloader(config: dict, mode: str, *, start_batch_index: int = 0,
                               io_batch_size: int | None = None,
                               num_workers_override: int | None = None) -> DataLoader:
    if mode != "train":
        raise ValueError("DINOv3-SAT full120 fitting loader refuses every non-train mode")
    if io_batch_size is not None:
        raise ValueError("DINOv3-SAT full120 does not use I/O super-batches")
    _validate_data_config(config)
    dataset = DINOv3SATNIRFull120Dataset(config, mode)
    batch_size = config["train"]["batch_size"]
    if type(start_batch_index) is not int or start_batch_index < 0:
        raise ValueError("invalid DINOv3-SAT full120 loader resume offset")
    start_index = start_batch_index * batch_size
    if start_index > len(dataset) or start_index % 64:
        raise ValueError("DINOv3-SAT full120 resume must lie on a 64-crop optimizer boundary")
    workers = config["data"]["num_workers"] if num_workers_override is None else num_workers_override
    if type(workers) is not int or workers < 0:
        raise ValueError("DINOv3-SAT full120 worker count must be a nonnegative integer")
    return DataLoader(dataset, batch_size=batch_size,
                      sampler=_DynamicFull120FourSceneSampler(dataset, start_index=start_index),
                      drop_last=True, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      # Recreate workers after set_epoch; per-index random.Random
                      # gives identical samples for zero/many workers and resume.
                      persistent_workers=False, collate_fn=_validated_collate)


__all__ = ["DINOv3SATNIRFull120Dataset", "audit_dinov3_sat_nir_data_identity", "build_dinov3_sat_nir_dataloader",
           "exact_sampling_arm_schedule", "four_scene_block_order", "sampling_order_audit"]


