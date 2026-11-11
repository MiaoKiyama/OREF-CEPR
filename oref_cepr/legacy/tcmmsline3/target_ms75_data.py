"""WHU native-seven MS75 views over the unchanged target DINO crop stream.

Historical MS75 geometry is local512, short1536->512 and WHOLE SCENE->512.
The base TargetDataset supplies the exact sampler, local crop and labels used
by the target DINO experts. Its deterministic RNG is replayed (and checked)
only to apply identical flips/radiometry to the extra aligned MS75 views.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Mapping

import numpy as np
from PIL import Image
import torch

from .target_dino_data import TargetDataset, load_manifest, normalize_bgrn, read_cached_scene

LOCAL_SIZE = 512
SHORT_CONTEXT_SIZE = 1536
SHORT_INPUT_SIZE = 512
GLOBAL_INPUT_SIZE = 512
VIEW_PROTOCOL = "whu_ms75_local512_short1536to512_whole512_v1"
MODEL_INPUT_KEYS = ("ms", "ms_short", "ms_short_valid", "ms_global",
                    "ms_global_valid", "local_box_short", "local_box_global")


def resize_raw(image: np.ndarray, size: int) -> np.ndarray:
    """Exact legacy per-band PIL uint8 bilinear resize (no color conversion)."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4 or size < 1:
        raise ValueError("Expected four-band uint8 image and positive size")
    return np.stack([np.asarray(Image.fromarray(image[..., c], mode="L").resize(
        (size, size), resample=Image.Resampling.BILINEAR), dtype=np.uint8)
        for c in range(4)], axis=-1)


def extract_boundless(image: np.ndarray, *, top: int, left: int, size: int):
    out = np.zeros((size, size, 4), dtype=np.uint8)
    valid = np.zeros((size, size), dtype=np.bool_)
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + size, image.shape[0]), min(left + size, image.shape[1])
    if y1 > y0 and x1 > x0:
        out[y0 - top:y1 - top, x0 - left:x1 - left] = image[y0:y1, x0:x1]
        valid[y0 - top:y1 - top, x0 - left:x1 - left] = True
    return out, valid


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _validate_whu(metadata):
    if metadata["dataset"] != "whu" or metadata["num_classes"] != 7:
        raise ValueError("This isolated MS75 run supports WHU native seven classes only")
    if metadata["band_order"] != ["B", "G", "R", "NIR"]:
        raise ValueError("MS75 target cache must use raw BGRN bands")


def prepare_view_cache(data_manifest, cache_root, *, workers=4) -> Path:
    """Build small read-only full-scene thumbnails, never touch test rasters.

    This creates a NEW directory outside the shared DINO cache. It does not
    rewrite any original image, label, sampling point, split or manifest.
    """
    metadata = load_manifest(data_manifest)
    _validate_whu(metadata)
    root = Path(cache_root).resolve()
    source_root = Path(metadata["cache_root"]).resolve()
    if root == source_root or source_root in root.parents or root in source_root.parents:
        raise ValueError("MS75 views cache must be separate from the original target cache")
    request = {"protocol": VIEW_PROTOCOL, "request_sha256": metadata["request_sha256"],
               "source_manifest_sha256": _digest(metadata), "scene_ids": metadata["train_ids"],
               "global_size": GLOBAL_INPUT_SIZE,
               "source_image_sha256": {s: metadata["scenes"][s]["image_sha256"] for s in metadata["train_ids"]}}
    root.mkdir(parents=True, exist_ok=True)
    request_file, output = root / "request.json", root / "manifest.json"
    if request_file.exists():
        if json.loads(request_file.read_text()) != request:
            raise ValueError("MS75 view cache identity mismatch")
    else:
        if list(root.iterdir()):
            raise ValueError("Refuse nonempty unowned views cache")
        _write_json(request_file, request)
    if output.exists():
        cached = GlobalViews(metadata, output)
        for scene in metadata["train_ids"]:
            cached.get(scene)
        return output

    def one(scene):
        path = root / f"{scene}.whole512.npy"
        # A completed manifest is the transaction marker; an incomplete owned
        # cache can be resumed by replacing only its own per-scene npy files.
        image, label = read_cached_scene(metadata, scene)
        del label
        if hashlib.sha256(memoryview(image)).hexdigest() != metadata["scenes"][scene]["image_sha256"]:
            raise ValueError("Original shared image cache checksum changed")
        raw = resize_raw(image, GLOBAL_INPUT_SIZE)
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("wb") as f:
            np.save(f, raw, allow_pickle=False)
        os.replace(temporary, path)
        path.chmod(0o444)
        return scene, {"path": str(path), "shape": list(raw.shape),
                       "sha256": hashlib.sha256(memoryview(raw)).hexdigest()}

    if type(workers) is not int or workers < 1:
        raise ValueError("positive thumbnail preparation workers required")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scenes = dict(pool.map(one, metadata["train_ids"]))
    _write_json(output, {**request, "cache_root": str(root), "scenes": scenes})
    return output


class GlobalViews:
    """Readonly shared thumbnails, or a bounded image-only in-memory fallback."""
    def __init__(self, data_manifest, views_cache=None, *, size=GLOBAL_INPUT_SIZE):
        self.metadata = load_manifest(data_manifest)
        _validate_whu(self.metadata)
        self.size, self._cache = int(size), OrderedDict()
        self.view_metadata = None
        if views_cache is not None:
            meta = json.loads(Path(views_cache).read_text(encoding="utf-8"))
            if (meta.get("protocol") != VIEW_PROTOCOL or meta.get("request_sha256") != self.metadata["request_sha256"]
                    or meta.get("source_manifest_sha256") != _digest(self.metadata)
                    or meta.get("scene_ids") != self.metadata["train_ids"]
                    or set(meta.get("scenes", {})) != set(self.metadata["train_ids"])
                    or meta.get("global_size") != self.size):
                raise ValueError("Views cache does not match this exact source training dataset")
            for scene in self.metadata["train_ids"]:
                if meta["source_image_sha256"][scene] != self.metadata["scenes"][scene]["image_sha256"]:
                    raise ValueError("Views cache source image identity differs")
            self.view_metadata = meta

    def get(self, scene_id, image=None):
        if scene_id not in self.metadata["train_ids"] or scene_id in self.metadata["test_ids"]:
            raise ValueError("Only official training/held scenes may supply global views")
        if scene_id in self._cache:
            self._cache.move_to_end(scene_id)
            return self._cache[scene_id]
        if self.view_metadata is not None:
            row = self.view_metadata["scenes"][scene_id]
            path = Path(row["path"]).resolve()
            if path.parent != Path(self.view_metadata["cache_root"]).resolve():
                raise ValueError("View path escapes its cache")
            raw = np.load(path, mmap_mode="r", allow_pickle=False)
            if raw.dtype != np.uint8 or raw.shape != (self.size, self.size, 4) or hashlib.sha256(memoryview(raw)).hexdigest() != row["sha256"]:
                raise ValueError("Global thumbnail bytes or layout changed")
        else:
            if image is None:
                image, label = read_cached_scene(self.metadata, scene_id)
                del label
            raw = resize_raw(image, self.size)
            raw.flags.writeable = False
        self._cache[scene_id] = raw
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return raw


def read_views(image, *, top: int, left: int, global_raw=None, normalization=None,
               local_size=LOCAL_SIZE, short_context_size=SHORT_CONTEXT_SIZE,
               short_input_size=SHORT_INPUT_SIZE, global_input_size=GLOBAL_INPUT_SIZE):
    """Original unaugmented view geometry; smaller constants are CPU-test seams."""
    h, w = image.shape[:2]
    if min(top, left) < 0 or top + local_size > h or left + local_size > w or short_context_size < local_size:
        raise ValueError("Local tile must be in bounds inside its short support")
    # Match legacy floor(center-context/2), including odd-size test seams.
    st = int(np.floor(top + local_size / 2 - short_context_size / 2))
    sl = int(np.floor(left + local_size / 2 - short_context_size / 2))
    raw_short, valid = extract_boundless(image, top=st, left=sl, size=short_context_size)
    short = resize_raw(raw_short, short_input_size)
    valid = np.asarray(Image.fromarray(valid.astype(np.uint8) * 255, mode="L").resize(
        (short_input_size, short_input_size), resample=Image.Resampling.NEAREST)) > 0
    global_raw = resize_raw(image, global_input_size) if global_raw is None else global_raw
    if global_raw.shape != (global_input_size, global_input_size, 4) or global_raw.dtype != np.uint8:
        raise ValueError("Full-scene raw thumbnail has incorrect geometry")
    return {"ms": normalize_bgrn(image[top:top + local_size, left:left + local_size], normalization),
            "ms_valid": np.ones((local_size, local_size), np.bool_),
            "ms_short": normalize_bgrn(short, normalization) * valid[None],
            "ms_short_valid": valid,
            "ms_global": normalize_bgrn(global_raw, normalization),
            "ms_global_valid": np.ones((global_input_size, global_input_size), np.bool_),
            "local_box_short": np.asarray([(top - st) / short_context_size, (left - sl) / short_context_size,
                                           (top + local_size - st) / short_context_size,
                                           (left + local_size - sl) / short_context_size], np.float32),
            "local_box_global": np.asarray([top / h, left / w, (top + local_size) / h,
                                            (left + local_size) / w], np.float32)}


def _flip_views(sample, horizontal, vertical):
    result = dict(sample)
    for key in ("ms", "ms_valid", "ms_short", "ms_short_valid", "ms_global", "ms_global_valid"):
        value = result[key]
        if horizontal:
            value = value[..., ::-1]
        if vertical:
            value = value[..., ::-1, :]
        result[key] = np.ascontiguousarray(value)
    for key in ("local_box_short", "local_box_global"):
        original, value = result[key], result[key].copy()
        if vertical:
            value[0], value[2] = 1.0 - float(original[2]), 1.0 - float(original[0])
        if horizontal:
            value[1], value[3] = 1.0 - float(original[3]), 1.0 - float(original[1])
        result[key] = value
    return result


def _augment_views(sample, *, rng, augmentation, normalization):
    aug = augmentation
    if not aug.get("enabled", False) or rng.random() >= float(aug.get("probability", 0)):
        return sample
    norm = normalization.get("ms", normalization)
    mean, std = np.asarray(norm["mean"], np.float32)[:, None, None], np.asarray(norm["std"], np.float32)[:, None, None]
    gain = rng.uniform(*aug.get("global_gain_range", [1, 1]))
    offset = rng.uniform(*aug.get("global_offset_range", [0, 0]))
    band_gain = np.array([rng.uniform(*aug.get("band_gain_range", [1, 1])) for _ in range(4)], np.float32)[:, None, None]
    band_offset = np.array([rng.uniform(*aug.get("band_offset_range", [0, 0])) for _ in range(4)], np.float32)[:, None, None]
    result = dict(sample)
    for key, valid in (("ms", "ms_valid"), ("ms_short", "ms_short_valid"), ("ms_global", "ms_global_valid")):
        original = sample[key]
        value = (np.clip((original * std + mean) * gain * band_gain + offset + band_offset, 0, 1.5) - mean) / std
        result[key] = np.where(sample[valid][None], value, original).astype(np.float32, copy=False)
    return result


class TargetMS75Dataset(TargetDataset):
    def __init__(self, manifest, *, views_cache=None, short_context_size=SHORT_CONTEXT_SIZE,
                 short_input_size=SHORT_INPUT_SIZE, global_input_size=GLOBAL_INPUT_SIZE, **kwargs):
        super().__init__(manifest, **kwargs)
        _validate_whu(self.metadata)
        if self.metadata.get("strict_protocol", True) and (
                self.crop_size, short_context_size, short_input_size, global_input_size) != (
                    LOCAL_SIZE, SHORT_CONTEXT_SIZE, SHORT_INPUT_SIZE, GLOBAL_INPUT_SIZE):
            raise ValueError("Formal MS75 must preserve historical512/1536to512/whole512 geometry")
        self.short_context_size = int(short_context_size)
        self.short_input_size = int(short_input_size)
        self.global_input_size = int(global_input_size)
        self.global_views = GlobalViews(self.metadata, views_cache, size=global_input_size)

    def __getitem__(self, index):
        base = super().__getitem__(index)
        absolute = base["index"]
        scene, kind, c = self.blocks[absolute // 16]
        image, label = self._scene(scene)
        h, w = label.shape
        rng = random.Random(self.seed + self.epoch * 10000019 + absolute * 65537)
        if kind == "class":
            points = self._points(scene, c)
            cy, cx = points[rng.randrange(len(points))]
            jitter = self.crop_size // 4
            top = int(np.clip(int(cy) - self.crop_size // 2 + rng.randint(-jitter, jitter), 0, h - self.crop_size))
            left = int(np.clip(int(cx) - self.crop_size // 2 + rng.randint(-jitter, jitter), 0, w - self.crop_size))
        else:
            top, left = rng.randrange(h - self.crop_size + 1), rng.randrange(w - self.crop_size + 1)
        if (scene, top, left) != (base["scene_id"], base["top"], base["left"]):
            raise RuntimeError("Base crop RNG contract changed; cannot claim matched target experiments")
        horizontal, vertical = rng.random() < .5, rng.random() < .5
        sample = read_views(image, top=top, left=left, global_raw=self.global_views.get(scene, image),
                            normalization=self.normalization, local_size=self.crop_size,
                            short_context_size=self.short_context_size, short_input_size=self.short_input_size,
                            global_input_size=self.global_input_size)
        sample = _flip_views(sample, horizontal, vertical)
        sample = _augment_views(sample, rng=rng, augmentation=self.spectral_augmentation,
                                normalization=self.normalization)
        # Check bitwise equivalence on every crop, not merely equal sample IDs.
        if not np.array_equal(sample["ms"], base["ms"].numpy()):
            raise RuntimeError("MS75 local augmentation differs from the frozen DINO crop stream")
        result = {**base, **{key: torch.from_numpy(np.ascontiguousarray(value))
                           for key, value in sample.items()}}
        result["label"] = base["label"]
        return result


__all__ = ["TargetMS75Dataset", "GlobalViews", "prepare_view_cache", "read_views", "resize_raw",
           "MODEL_INPUT_KEYS", "LOCAL_SIZE", "SHORT_CONTEXT_SIZE", "SHORT_INPUT_SIZE", "GLOBAL_INPUT_SIZE"]
