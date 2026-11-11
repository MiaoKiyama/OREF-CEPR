"""Train96 optical/SAR crops without deleting labels where SAR is absent.

Every epoch has four distinct scenes per effective64 update and exact
50/30/10/5/5 scene/rare/hard/pond/water-boundary sampling. The plan depends
only on train96 labels, never held labels, SAR availability or predictions.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .light_sar_data import _availability, _checked_aux, _load_points, _sha, apply_flips, open_sar_scene
from .ms75_rebuild_data import (GlobalViews, HARD_CLASS_IDS, RARE_CLASS_IDS,
                               load_manifest, read_cached_scene, read_scaled_views, _check_paths)
from .ms75_sar_pilot_data import boundary_reservoir
from .target_dino_data import SPECTRAL_AUGMENTATION
from .target_ms75_data import MODEL_INPUT_KEYS, _augment_views, _flip_views

SCHEMA = "ms75_sar_joint_all96_sampling_v1_20260930"
KINDS = ("scene", "rare", "hard", "pond", "boundary")
EFFECTIVE_BATCH = 64
BLOCK_SIZE = 16


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prepare_joint_support(raw_manifest, output, *, workers=4, capacity=4096, progress=None):
    """Read train96 labels only; build each boundary reservoir once, not per worker."""
    metadata = load_manifest(raw_manifest)
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Joint support output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)

    def one(scene):
        row = metadata["scenes"][scene]
        _check_paths(metadata, row)
        labels = np.load(row["label"], mmap_mode="r", allow_pickle=False)
        if labels.dtype != np.uint8 or labels.shape != tuple(row["shape"]) or hashlib.sha256(memoryview(labels)).hexdigest() != row["label_sha256"]:
            raise ValueError("Original optical labels changed")
        points, population = boundary_reservoir(labels, capacity=capacity,
            seed=int(hashlib.sha256(scene.encode()).hexdigest()[:16], 16))
        path = out / (scene + ".boundary.npy")
        np.save(path, points, allow_pickle=False)
        result = dict(path=str(path), sha256=_sha(path), population=population, count=len(points), label_sha256=row["label_sha256"])
        if progress:
            progress(dict(event="joint_boundary_ready", scene=scene, **result))
        return scene, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = dict(pool.map(one, metadata["selection_train_ids"]))
    value = dict(schema=SCHEMA, raw_manifest_digest=digest(metadata),
                 scene_ids=metadata["selection_train_ids"], scenes=rows, held_labels_read=False)
    value["payload_sha256"] = digest(value)
    path = out / "manifest.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return path


def make_joint_plan(metadata, *, epoch=0, updates=480, seed=1337, boundary_scenes=None):
    """Stable epoch-local global indices; DDP must partition, never pad them."""
    scenes = tuple(metadata["selection_train_ids"])
    if type(epoch) is not int or epoch < 0 or type(updates) is not int or updates < 5 or updates % 5 or updates * 2 < len(scenes):
        raise ValueError("Need nonnegative epoch, whole five-update groups and full scene coverage")
    if len(scenes) < 4 or len(set(scenes)) != len(scenes) or set(scenes) & set(metadata["held_ids"]):
        raise ValueError("Need distinct train-only scenes")
    classes = {c: [s for s in scenes if metadata["scenes"][s]["histogram"][c] > 0] for c in range(24)}
    boundary_scenes = set(scenes if boundary_scenes is None else boundary_scenes)
    if not boundary_scenes.issubset(scenes):
        raise ValueError("Boundary support escapes training scenes")
    rng = random.Random(seed + epoch * 10000019)
    targeted = ["rare"] * (updates * 6 // 5) + ["hard"] * (updates * 2 // 5) + ["pond"] * (updates // 5) + ["boundary"] * (updates // 5)
    rng.shuffle(targeted)
    queue, blocks, coverage = deque(), [], set()
    for update in range(updates):
        chosen, current = set(), []
        for _ in range(2):
            if not queue:
                fresh = list(scenes)
                rng.shuffle(fresh)
                queue.extend(fresh)
            for _ in range(len(queue)):
                scene = queue.popleft()
                if scene not in chosen:
                    break
                queue.append(scene)
            else:
                scene = rng.choice([s for s in scenes if s not in chosen])
            chosen.add(scene)
            coverage.add(scene)
            current.append((scene, "scene", -1))
        for kind in targeted[update * 2:update * 2 + 2]:
            ids = RARE_CLASS_IDS if kind == "rare" else HARD_CLASS_IDS if kind == "hard" else (13,)
            if kind == "boundary":
                candidates = sorted(boundary_scenes - chosen)
                if not candidates:
                    raise ValueError("Cannot select a distinct scene with water-boundary support")
                scene, cls = rng.choice(candidates), -1
            else:
                eligible = {c: [s for s in classes[c] if s not in chosen] for c in ids}
                eligible = {c: ss for c, ss in eligible.items() if ss}
                if not eligible:
                    raise ValueError("Cannot select a distinct train-only target scene: " + kind)
                cls = rng.choice(sorted(eligible))
                scene = rng.choice(eligible[cls])
            chosen.add(scene)
            current.append((scene, kind, cls))
        rng.shuffle(current)
        blocks.extend(current)
    if coverage != set(scenes):
        raise AssertionError("Uniform sampling did not cover all train96")
    return tuple(blocks)


class MS75SARJointDataset(Dataset):
    """A deterministic raw-input dataset; missing SAR never removes optical GT."""
    def __init__(self, config, epoch=0, start_sample=0):
        data = config.get("data", config)
        self.metadata = load_manifest(data["raw_manifest"])
        self.scene_ids = tuple(self.metadata["selection_train_ids"])
        self.data_root, self.epoch = data["data_root"], int(epoch)
        self.seed = int(data.get("seed", config.get("seed", 1337)))
        self.n, self.sn = int(data.get("crop_size", 512)), int(data.get("sar_size", 166))
        self.short_size, self.global_size = int(data.get("short_input_size", 512)), int(data.get("global_input_size", 512))
        self.updates = int(data.get("updates_per_epoch", 480))
        self.samples_per_epoch = self.updates * EFFECTIVE_BATCH
        if type(start_sample) is not int or not 0 <= start_sample <= self.samples_per_epoch or start_sample % EFFECTIVE_BATCH:
            raise ValueError("Resume must be a complete effective64 update")
        self.start_sample, self.augmentation = start_sample, bool(data.get("augmentation", True))
        self.max_cache = int(data.get("cache_scenes", 4))
        if self.max_cache < 1:
            raise ValueError("Positive worker-local scene cache capacity required")
        self.availability, self.availability_path = _availability(self.data_root)
        if not set(self.scene_ids).issubset(self.availability):
            raise ValueError("SAR availability must enumerate all train96, including missing scenes")
        if self.metadata.get("strict_protocol", True):
            if (len(self.scene_ids), sum(self.availability[s] for s in self.scene_ids), self.n, self.sn, self.short_size, self.global_size) != (96, 83, 512, 166, 512, 512):
                raise ValueError("Expected original all96/83-paired, 512 optical views and SAR166")
            if not data.get("boundary_support"):
                raise ValueError("Formal training requires prebuilt train96 boundary support")
        self.support = None
        if data.get("boundary_support"):
            self.support = json.loads(Path(data["boundary_support"]).read_text(encoding="utf-8"))
            body = {k:v for k,v in self.support.items() if k != "payload_sha256"}
            if (self.support.get("schema") != SCHEMA or self.support.get("payload_sha256") != digest(body)
                    or self.support.get("raw_manifest_digest") != digest(self.metadata)
                    or self.support.get("scene_ids") != list(self.scene_ids)):
                raise ValueError("Boundary support source identity differs")
        boundary_scenes = [s for s in self.scene_ids if self.support["scenes"][s]["count"] > 0] if self.support else None
        self.blocks = make_joint_plan(self.metadata, epoch=self.epoch, updates=self.updates, seed=self.seed, boundary_scenes=boundary_scenes)
        self.plan_sha256 = digest(dict(epoch=self.epoch, seed=self.seed, blocks=self.blocks, schema=SCHEMA))
        self.global_views = GlobalViews(self.metadata, data.get("views_cache"), size=self.global_size)
        self._cache, self._points, self._boundaries, self._aux = (OrderedDict() for _ in range(4))

    def __len__(self):
        return self.samples_per_epoch - self.start_sample

    def _bounded(self, cache, key, factory):
        value = cache.pop(key, None)
        if value is None:
            value = factory()
        cache[key] = value
        while len(cache) > self.max_cache:
            old = cache.popitem(last=False)[1]
            if cache is self._aux:
                old.close()
        return value

    def _boundary(self, scene, labels):
        def get():
            if self.support:
                row = self.support["scenes"][scene]
                if _sha(row["path"]) != row["sha256"]:
                    raise ValueError("Boundary reservoir changed")
                return np.load(row["path"], allow_pickle=False)
            return boundary_reservoir(labels, capacity=4096, seed=int(hashlib.sha256(scene.encode()).hexdigest()[:16], 16))[0]
        return self._bounded(self._boundaries, scene, get)

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(self):
            raise IndexError(index)
        absolute = self.start_sample + int(index)
        scene, kind, cls = self.blocks[absolute // BLOCK_SIZE]
        rng = random.Random(self.seed + self.epoch * 10000019 + absolute * 65537)
        image, labels = self._bounded(self._cache, scene, lambda: read_cached_scene(self.metadata, scene))
        h, w = labels.shape
        if min(h, w) < self.n:
            raise ValueError("Scene smaller than optical crop")
        redraw_count, fallback_used = 0, False
        if kind == "scene":
            # Preserve the old sampler's distribution except impossible
            # all-ignore crops; a positive known count prevents NaN CE/DDP.
            for attempt in range(16):
                top, left = rng.randrange(h - self.n + 1), rng.randrange(w - self.n + 1)
                if np.any(labels[top:top + self.n, left:left + self.n] < 24):
                    break
                redraw_count += 1
            else:
                row = self.metadata["scenes"][scene]
                reservoirs = self._bounded(self._points, scene, lambda: _load_points(row["points"], row["points_sha256"]))
                eligible = [c for c, points in reservoirs.items() if len(points)]
                if not eligible:
                    raise ValueError("Training scene has no known optical labels")
                points = reservoirs[rng.choice(eligible)]
                cy, cx = map(int, points[rng.randrange(len(points))])
                top, left = int(np.clip(cy - self.n // 2, 0, h - self.n)), int(np.clip(cx - self.n // 2, 0, w - self.n))
                fallback_used = True
        else:
            if kind == "boundary":
                points = self._boundary(scene, labels)
            else:
                row = self.metadata["scenes"][scene]
                points = self._bounded(self._points, scene, lambda: _load_points(row["points"], row["points_sha256"]))[cls]
            if not len(points):
                raise ValueError("Chosen target scene has no target points")
            cy, cx = map(int, points[rng.randrange(len(points))])
            jitter = self.n // 4
            top = int(np.clip(cy - self.n // 2 + rng.randint(-jitter, jitter), 0, h - self.n))
            left = int(np.clip(cx - self.n // 2 + rng.randint(-jitter, jitter), 0, w - self.n))
        sample, target = read_scaled_views(image, labels, top=top, left=left, source_size=self.n,
            output_size=self.n, short_input_size=self.short_size, global_input_size=self.global_size,
            global_raw=self.global_views.get(scene, image), normalization=self.metadata["normalization"])
        known_pixels = int(np.count_nonzero(target < 24))
        if known_pixels < 1:
            raise ValueError("Crop must retain at least one known optical GT pixel")
        if self.availability[scene]:
            reader = self._bounded(self._aux, scene, lambda: open_sar_scene(self.data_root, scene))
            sar, native, optical = _checked_aux(reader.read_patch(top, left, self.n, modality="sar", out_size=self.sn), self.n, self.sn)
        else:
            sar, native, optical = np.zeros((2, self.sn, self.sn), np.float32), np.zeros((1, self.sn, self.sn), bool), np.zeros((1, self.n, self.n), bool)
        code = int(rng.random() < .5) + 2 * int(rng.random() < .5) if self.augmentation else 0
        sample = _flip_views(sample, bool(code & 1), bool(code & 2))
        if self.augmentation:
            sample = _augment_views(sample, rng=rng, augmentation=SPECTRAL_AUGMENTATION, normalization=self.metadata["normalization"])
        # SAR support controls auxiliary features and losses only. Every known
        # optical label remains available to the principal segmentation loss.
        sample.update(sar_db=apply_flips(sar, code), sar_valid=apply_flips(native, code),
                      optical_valid=apply_flips(optical, code), label=apply_flips(target, code))
        result = {k: torch.from_numpy(np.ascontiguousarray(v)) for k,v in sample.items()}
        result.update(index=absolute, scene_id=scene, top=top, left=left, sampling_kind=kind,
                      sampling_class=cls, flip_code=code, has_sar=self.availability[scene],
                      known_pixels=known_pixels, redraw_count=redraw_count, fallback_used=fallback_used)
        return result

    def audit(self):
        return dict(schema=SCHEMA, plan_sha256=self.plan_sha256, epoch=self.epoch,
            samples=self.samples_per_epoch, scene_count=len(self.scene_ids),
            paired_scene_count=sum(self.availability[s] for s in self.scene_ids),
            missing_scene_ids=[s for s in self.scene_ids if not self.availability[s]],
            block_counts=dict(Counter(kind for _,kind,_ in self.blocks)),
            scene_block_counts=dict(Counter(s for s,_,_ in self.blocks)),
            minimum_known_pixels_per_crop=1, empty_uniform_crop_max_attempts=16,
            empty_uniform_crop_fallback="same_scene_known_GT_point; independent_of_SAR",
            supervision="all_known_optical_GT; SAR support never removes principal labels")

    def close(self):
        for reader in getattr(self, "_aux", {}).values():
            reader.close()
        self._cache, self._points, self._boundaries, self._aux = (OrderedDict() for _ in range(4))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cache=OrderedDict(), _points=OrderedDict(), _boundaries=OrderedDict(), _aux=OrderedDict())
        return state

    def __del__(self):
        self.close()


__all__ = ["MS75SARJointDataset", "prepare_joint_support", "make_joint_plan", "MODEL_INPUT_KEYS", "SCHEMA"]
