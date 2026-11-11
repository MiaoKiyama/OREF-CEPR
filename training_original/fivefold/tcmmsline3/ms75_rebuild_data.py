"""Isolated FBP24 data for the R/HR/robust MS75 rebuild experiments.

Only official training rasters are opened. Images may be borrowed READ-ONLY
from the authenticated GID5 BGRN cache, but its labels and point reservoirs are
never reused. All labels here come from the original FBP24 PNG files.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Mapping

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .target_dino_data import (NORMALIZATION, SPECTRAL_AUGMENTATION, _binding,
                               _points, normalize_bgrn)
from .target_ms75_data import (MODEL_INPUT_KEYS, _augment_views, _flip_views,
                               extract_boundless, read_views, resize_raw,
                               LOCAL_SIZE, SHORT_CONTEXT_SIZE, SHORT_INPUT_SIZE,
                               GLOBAL_INPUT_SIZE)

PROTOCOL = "fbp24_ms75_rebuild_train_only_v1_20260923"
VIEW_PROTOCOL = PROTOCOL + "_whole512"
IGNORE_INDEX = 255
TRAIN96_SPLIT = "ms75_oof_seed1337_fold_0_train"
HELD24_SPLIT = "ms75_oof_seed1337_fold_0_heldout"
SPLIT_SHA256 = {
    "train": "12422a667080adad95f87bc0397f3b70da9fdf339b1d71b9dc2ea310a947691e",
    "test": "abfbe27a0a88084034b0fbc9b8b9e4dfee7ebc2b5950a36c6f6a90b880cc923b",
    TRAIN96_SPLIT: "12138004d46797832e8a5352fad328159d3b78368f04c5c9c30a64e084de76e0",
    HELD24_SPLIT: "036a836c765d18f932b9d0512921160cec08453a0b0f244b6574e3967536280a",
}
CLASS_NAMES = ["industrial area", "paddy field", "irrigated field", "dry cropland",
               "garden land", "arbor forest", "shrub forest", "park", "natural meadow",
               "artificial meadow", "river", "urban residential", "lake", "pond",
               "fish pond", "snow", "bareland", "rural residential", "stadium", "square",
               "road", "overpass", "railway station", "airport"]
RARE_CLASS_IDS = (7, 9, 15, 18, 19, 22, 23)
HARD_CLASS_IDS = (0, 4, 5, 6, 8, 10, 11, 12, 13, 14, 17, 20, 21)
ROBUST_SCALES = (.8, 1.0, 1.25)
# Shared global radiometry plus very small independent band changes, never an
# RGB-only hue/saturation transform or NIR channel replication/permutation.
ROBUST_SPECTRAL_AUGMENTATION = {
    "enabled": True, "probability": .8, "global_gain_range": [.85, 1.15],
    "global_offset_range": [-.025, .025], "band_gain_range": [.96, 1.04],
    "band_offset_range": [-.01, .01],
}
REQUEST_KEYS = ("protocol", "dataset", "train_ids", "test_ids", "selection_train_ids",
                "held_ids", "seed", "normalization", "point_capacity", "bindings",
                "strict_protocol", "split_sha256", "borrowed_image_manifest_sha256")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _ids(values):
    result = list(map(str, values))
    if not result or len(set(result)) != len(result) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in result):
        raise ValueError("Scene IDs must be nonempty, unique and safe")
    return result


def make_splits(train_ids, test_ids, selection_train_ids, held_ids, *, strict_protocol=True):
    train, test, selection, held = map(_ids, (train_ids, test_ids, selection_train_ids, held_ids))
    if set(train) & set(test) or set(selection) & set(held) or set(selection + held) != set(train):
        raise ValueError("FBP official/internal partitions overlap or are incomplete")
    if strict_protocol and tuple(map(len, (train, test, selection, held))) != (120, 30, 96, 24):
        raise ValueError("FBP requires exact official120/30 and selection96/24")
    if len(selection) < 4:
        raise ValueError("At least four selection training scenes required")
    return dict(train_ids=train, test_ids=test, selection_train_ids=selection, held_ids=held)


def map_labels(raw):
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.ndim != 2 or not np.isin(raw, np.arange(25)).all():
        raise ValueError("FBP labels must be uint8 HW source codes 0..24")
    lut = np.full(256, 255, np.uint8)
    lut[1:25] = np.arange(24, dtype=np.uint8)
    return lut[raw]


def to_bgrn(raw):
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 4:
        raise ValueError("FBP source must be uint8 HWC NIR,R,G,B")
    return np.ascontiguousarray(raw[..., [3, 2, 1, 0]])


def load_manifest(manifest):
    m = dict(manifest) if isinstance(manifest, Mapping) else json.loads(Path(manifest).read_text(encoding="utf-8"))
    if (m.get("protocol") != PROTOCOL or m.get("dataset") != "fbp"
            or m.get("num_classes") != 24 or m.get("ignore_index") != 255
            or m.get("band_order") != ["B", "G", "R", "NIR"]):
        raise ValueError("Not the isolated FBP24 rebuild data protocol")
    make_splits(m["train_ids"], m["test_ids"], m["selection_train_ids"], m["held_ids"],
                strict_protocol=m.get("strict_protocol", True))
    if set(m["scenes"]) != set(m["train_ids"]) or set(m["bindings"]) != set(m["train_ids"]):
        raise ValueError("Only exact official training scenes may be cached")
    if m["seed"] != 1337 or _digest({k: m[k] for k in REQUEST_KEYS}) != m["request_sha256"]:
        raise ValueError("FBP request/seed identity changed")
    if m.get("strict_protocol", True) and m["split_sha256"] != SPLIT_SHA256:
        raise ValueError("Authenticated original fold0 split SHA256 differs")
    for row in m["scenes"].values():
        if len(row["histogram"]) != 24 or row["request_sha256"] != m["request_sha256"]:
            raise ValueError("FBP row has incorrect label taxonomy/identity")
    return m


def _check_paths(m, row):
    root = Path(m["cache_root"]).resolve()
    if any(Path(row[k]).resolve().parent != root for k in ("label", "points")):
        raise ValueError("FBP label/point path escapes dedicated cache")
    if Path(row["image"]).resolve().parent not in {Path(p).resolve() for p in m["image_cache_roots"]}:
        raise ValueError("Image path is outside authenticated read-only image roots")


def read_cached_scene(manifest, scene_id):
    m = manifest if isinstance(manifest, Mapping) else load_manifest(manifest)
    if scene_id not in m["train_ids"] or scene_id in m["test_ids"]:
        raise ValueError("Only official train/held scenes may be opened")
    row = m["scenes"][scene_id]
    _check_paths(m, row)
    image = np.load(row["image"], mmap_mode="r", allow_pickle=False)
    label = np.load(row["label"], mmap_mode="r", allow_pickle=False)
    if image.dtype != np.uint8 or label.dtype != np.uint8 or image.shape != (*row["shape"], 4) or label.shape != tuple(row["shape"]):
        raise ValueError("FBP mapped-cache shape/dtype mismatch")
    return image, label


def _validate_row(m, row):
    _check_paths(m, row)
    for key in ("image", "label"):
        array = np.load(row[key], mmap_mode="r", allow_pickle=False)
        shape = (*row["shape"], 4) if key == "image" else tuple(row["shape"])
        if array.dtype != np.uint8 or array.shape != shape or hashlib.sha256(memoryview(array)).hexdigest() != row[key + "_sha256"]:
            raise ValueError("FBP cache content changed: " + key)
    if hashlib.sha256(Path(row["points"]).read_bytes()).hexdigest() != row["points_sha256"]:
        raise ValueError("FBP point reservoir changed")


def prepare_data(*, data_root, cache_root, borrowed_image_manifest=None, workers=4,
                 point_capacity=4096, strict_protocol=True):
    """Prepare only FBP train120 labels/points, borrowing original images when possible.

    Test split TEXT is read, never its images/labels. Cache construction may
    read held24 labels for later evaluation, but their points/histograms are
    inaccessible to the selection training sampler.
    """
    root, out = Path(data_root).resolve(), Path(cache_root).resolve()
    if out == root or root in out.parents or out in root.parents:
        raise ValueError("Rebuild cache must be outside the source dataset")
    if type(workers) is not int or workers < 1 or point_capacity < 1:
        raise ValueError("Positive cache workers/point capacity required")
    split_names = {"train_ids": "train", "test_ids": "test", "selection_train_ids": TRAIN96_SPLIT, "held_ids": HELD24_SPLIT}
    split_bytes = {name: (root / "splits" / f"{name}.txt").read_bytes() for name in split_names.values()}
    split_hash = {name: hashlib.sha256(b).hexdigest() for name, b in split_bytes.items()}
    if strict_protocol and split_hash != SPLIT_SHA256:
        raise ValueError("Source splits differ from authenticated FBP fold0")
    splits = make_splits(**{k: split_bytes[v].decode().split() for k, v in split_names.items()}, strict_protocol=strict_protocol)
    borrowed = None
    if borrowed_image_manifest is not None:
        borrowed = json.loads(Path(borrowed_image_manifest).read_text(encoding="utf-8"))
        if (borrowed.get("dataset") != "gid5" or borrowed.get("band_order") != ["B", "G", "R", "NIR"]
                or set(borrowed["train_ids"]) != set(splits["train_ids"])
                or set(borrowed["test_ids"]) != set(splits["test_ids"])
                or set(borrowed["selection_train_ids"]) != set(splits["selection_train_ids"])
                or set(borrowed["held_ids"]) != set(splits["held_ids"])):
            raise ValueError("Borrowed image cache does not match FBP images/folds")
        borrowed_root = Path(borrowed["cache_root"]).resolve()
        if out == borrowed_root or out in borrowed_root.parents or borrowed_root in out.parents:
            raise ValueError("Dedicated FBP cache must not modify the shared GID cache")
    bindings = {}
    for scene in splits["train_ids"]:
        image_path = root / "images" / f"{scene}.tif"
        if borrowed is None:
            image_binding = _binding(image_path)
        else:
            image_binding = borrowed["bindings"][scene]["image"]
            # Source raw TIFF identity was authenticated by GID preparation.
            # Recheck its path/stat and the decoded BGRN hash below, not GID labels.
            st = image_path.stat()
            if (Path(image_binding["path"]).resolve() != image_path.resolve()
                    or (st.st_size, st.st_mtime_ns) != (image_binding["size"], image_binding["mtime_ns"])):
                raise ValueError("Borrowed original image source binding changed")
        bindings[scene] = {"image": image_binding, "label": _binding(root / "labels" / f"{scene}.png")}
    request = {"protocol": PROTOCOL, "dataset": "fbp", **splits, "seed": 1337,
               "normalization": NORMALIZATION, "point_capacity": int(point_capacity),
               "bindings": bindings, "strict_protocol": bool(strict_protocol),
               "split_sha256": split_hash,
               "borrowed_image_manifest_sha256": _digest(borrowed) if borrowed else None}
    identity = _digest(request)
    image_roots = [str(borrowed_root)] if borrowed else [str(out)]
    base = {**request, "request_sha256": identity, "cache_root": str(out), "image_cache_roots": image_roots}
    out.mkdir(parents=True, exist_ok=True)
    request_path, output = out / "request.json", out / "manifest.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("Rebuild cache request differs; use an isolated new directory")
    else:
        if list(out.iterdir()):
            raise ValueError("Refusing a nonempty unowned cache")
        _write_json(request_path, request)
    if output.exists():
        m = load_manifest(output)
        if m["request_sha256"] != identity:
            raise ValueError("Existing completed cache identity differs")
        for row in m["scenes"].values():
            _validate_row(m, row)
        return output

    def one(scene):
        receipt = out / f"{scene}.json"
        if receipt.exists():
            row = json.loads(receipt.read_text())
            if row["request_sha256"] != identity:
                raise ValueError("Partial scene belongs to another request")
            _validate_row(base, row)
            return scene, row
        if borrowed:
            br = borrowed["scenes"][scene]
            image_path = Path(br["image"])
            if image_path.resolve().parent != borrowed_root:
                raise ValueError("Borrowed image escapes authenticated cache")
            image = np.load(image_path, mmap_mode="r", allow_pickle=False)
            image_hash = hashlib.sha256(memoryview(image)).hexdigest()
            if image.dtype != np.uint8 or image.shape != (*br["shape"], 4) or image_hash != br["image_sha256"]:
                raise ValueError("Borrowed BGRN image content/shape changed")
        else:
            with Image.open(bindings[scene]["image"]["path"]) as f:
                image = to_bgrn(np.array(f))
            image_path = out / f"{scene}.image.npy"
            _save_npy(image_path, image)
            image_hash = hashlib.sha256(memoryview(image)).hexdigest()
        with Image.open(bindings[scene]["label"]["path"]) as f:
            label = map_labels(np.array(f))
        for binding in bindings[scene].values():
            st = Path(binding["path"]).stat()
            if (st.st_size, st.st_mtime_ns) != (binding["size"], binding["mtime_ns"]):
                raise ValueError("Authenticated original source changed during preparation")
        if image.shape[:2] != label.shape or (strict_protocol and label.shape != (6908, 7300)):
            raise ValueError("FBP image/label geometry mismatch")
        seed = int(hashlib.sha256(f"1337:{scene}".encode()).hexdigest()[:16], 16)
        hist, points = _points(label, 24, seed=seed, capacity=point_capacity)
        label_path, points_path = out / f"{scene}.label.npy", out / f"{scene}.points.npz"
        _save_npy(label_path, label)
        tmp = points_path.with_name(points_path.name + ".partial")
        with tmp.open("wb") as f:
            np.savez(f, **points)
        os.replace(tmp, points_path)
        points_path.chmod(0o444)
        row = {"image": str(image_path.resolve()), "label": str(label_path), "points": str(points_path),
               "shape": list(label.shape), "histogram": hist, "request_sha256": identity,
               "image_sha256": image_hash, "label_sha256": hashlib.sha256(memoryview(label)).hexdigest(),
               "points_sha256": hashlib.sha256(points_path.read_bytes()).hexdigest()}
        _write_json(receipt, row)
        print(json.dumps({"event": "cached_fbp24_train_scene", "scene": scene}), flush=True)
        return scene, row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = dict(pool.map(one, splits["train_ids"]))
    m = {**base, "num_classes": 24, "class_names": CLASS_NAMES, "band_order": ["B", "G", "R", "NIR"],
         "ignore_index": 255, "scenes": rows,
         "selection_histogram": np.sum([rows[s]["histogram"] for s in splits["selection_train_ids"]], axis=0).tolist(),
         "full_histogram": np.sum([rows[s]["histogram"] for s in splits["train_ids"]], axis=0).tolist(),
         "sampling": {"epoch_crops_per_train_scene_budget": 320, "block_size": 16,
                      "effective_batch_size": 64, "scene_fraction": .5, "rare_fraction": .3,
                      "hard_fraction": .2, "rare_class_ids": list(RARE_CLASS_IDS),
                      "hard_class_ids": list(HARD_CLASS_IDS), "test_data_read": False,
                      "selection_held_labels_used_for_sampling": False,
                      "teacher_predictions_used": False}}
    load_manifest(m)
    _write_json(output, m)
    return output


def _save_npy(path, array):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)
    path.chmod(0o444)


def prepare_view_cache(data_manifest, cache_root, *, workers=4, size=512):
    m, root = load_manifest(data_manifest), Path(cache_root).resolve()
    protected = [Path(m["cache_root"]).resolve(), *map(lambda p: Path(p).resolve(), m["image_cache_roots"])]
    if any(root == p or root in p.parents or p in root.parents for p in protected):
        raise ValueError("Views must use a separate dedicated cache")
    request = {"protocol": VIEW_PROTOCOL, "request_sha256": m["request_sha256"],
               "source_manifest_sha256": _digest(m), "scene_ids": m["train_ids"], "global_size": int(size)}
    root.mkdir(parents=True, exist_ok=True)
    request_path, output = root / "request.json", root / "manifest.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("Views cache identity mismatch")
    else:
        if list(root.iterdir()):
            raise ValueError("Refusing nonempty unowned views cache")
        _write_json(request_path, request)
    if output.exists():
        cached = GlobalViews(m, output, size=size)
        for scene in m["train_ids"]:
            cached.get(scene)
        return output
    def one(scene):
        image, _ = read_cached_scene(m, scene)
        if hashlib.sha256(memoryview(image)).hexdigest() != m["scenes"][scene]["image_sha256"]:
            raise ValueError("Source image cache changed")
        raw = resize_raw(image, size)
        path = root / f"{scene}.whole{size}.npy"
        _save_npy(path, raw)
        return scene, {"path": str(path), "shape": list(raw.shape), "sha256": hashlib.sha256(memoryview(raw)).hexdigest()}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scenes = dict(pool.map(one, m["train_ids"]))
    _write_json(output, {**request, "cache_root": str(root), "scenes": scenes})
    return output


class GlobalViews:
    def __init__(self, manifest, views_cache=None, *, size=512):
        self.metadata, self.size = load_manifest(manifest), int(size)
        self._cache, self.view_metadata = OrderedDict(), None
        if views_cache:
            m = json.loads(Path(views_cache).read_text(encoding="utf-8"))
            if (m.get("protocol") != VIEW_PROTOCOL or m.get("request_sha256") != self.metadata["request_sha256"]
                    or m.get("source_manifest_sha256") != _digest(self.metadata)
                    or m.get("global_size") != self.size
                    or m.get("scene_ids") != self.metadata["train_ids"]
                    or set(m.get("scenes", {})) != set(self.metadata["train_ids"])):
                raise ValueError("Views do not match FBP24 source identity")
            self.view_metadata = m

    def get(self, scene_id, image=None):
        if scene_id not in self.metadata["train_ids"] or scene_id in self.metadata["test_ids"]:
            raise ValueError("Test imagery is forbidden")
        if scene_id in self._cache:
            self._cache.move_to_end(scene_id)
            return self._cache[scene_id]
        if self.view_metadata:
            row = self.view_metadata["scenes"][scene_id]
            if Path(row["path"]).resolve().parent != Path(self.view_metadata["cache_root"]).resolve():
                raise ValueError("Thumbnail path escapes views cache")
            raw = np.load(row["path"], mmap_mode="r", allow_pickle=False)
            if raw.dtype != np.uint8 or raw.shape != (self.size, self.size, 4) or hashlib.sha256(memoryview(raw)).hexdigest() != row["sha256"]:
                raise ValueError("Thumbnail content changed")
        else:
            if image is None:
                image, _ = read_cached_scene(self.metadata, scene_id)
            raw = resize_raw(image, self.size)
            raw.flags.writeable = False
        self._cache[scene_id] = raw
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return raw


def read_scaled_views(image, label, *, top, left, source_size, output_size=512,
                      global_raw=None, normalization=None, short_input_size=512,
                      global_input_size=512):
    """One coherent physical support: local S, short3S and original whole scene.

    Output pixels always number512, but support can be410/512/640. Boxes remain
    in original scene coordinates; target uses nearest interpolation only.
    """
    h, w = image.shape[:2]
    if source_size < 1 or min(top, left) < 0 or top + source_size > h or left + source_size > w:
        raise ValueError("Scaled local crop must stay within the original scene")
    short_size = 3 * source_size
    st = math.floor(top + source_size / 2 - short_size / 2)
    sl = math.floor(left + source_size / 2 - short_size / 2)
    short, valid = extract_boundless(image, top=st, left=sl, size=short_size)
    short = resize_raw(short, short_input_size)
    valid = np.asarray(Image.fromarray(valid.astype(np.uint8)).resize((short_input_size, short_input_size), Image.Resampling.NEAREST)) > 0
    raw = image[top:top + source_size, left:left + source_size]
    target = np.array(label[top:top + source_size, left:left + source_size], copy=True)
    if source_size != output_size:
        raw = resize_raw(raw, output_size)
        target = np.asarray(Image.fromarray(target).resize((output_size, output_size), Image.Resampling.NEAREST))
    if global_raw is None:
        global_raw = resize_raw(image, global_input_size)
    if global_raw.dtype != np.uint8 or global_raw.shape != (global_input_size, global_input_size, 4):
        raise ValueError("Incorrect global thumbnail geometry")
    sample = {"ms": normalize_bgrn(raw, normalization), "ms_valid": np.ones((output_size, output_size), bool),
              "ms_short": normalize_bgrn(short, normalization) * valid[None], "ms_short_valid": valid,
              "ms_global": normalize_bgrn(global_raw, normalization),
              "ms_global_valid": np.ones((global_input_size, global_input_size), bool),
              "local_box_short": np.array([(top-st)/short_size, (left-sl)/short_size,
                                            (top+source_size-st)/short_size, (left+source_size-sl)/short_size], np.float32),
              "local_box_global": np.array([top/h, left/w, (top+source_size)/h, (left+source_size)/w], np.float32)}
    return sample, np.ascontiguousarray(target, dtype=np.int64)


class MS75RebuildDataset(Dataset):
    def __init__(self, manifest, *, stage="selection", variant="r", epoch=0, start_index=0,
                 views_cache=None, normalization=None, crop_size=512, short_input_size=512,
                 global_input_size=512, augmentation=True):
        self.metadata = load_manifest(manifest)
        if stage not in ("selection", "full") or variant not in ("r", "hr", "robust"):
            raise ValueError("Invalid FBP rebuild stage/variant")
        self.stage, self.variant, self.seed = stage, variant, int(self.metadata["seed"])
        self.scene_ids = tuple(self.metadata["selection_train_ids"] if stage == "selection" else self.metadata["train_ids"])
        self.samples_per_epoch, self.num_classes = len(self.scene_ids) * 320, 24
        self.crop_size, self.short_input_size, self.global_input_size = int(crop_size), int(short_input_size), int(global_input_size)
        if min(self.crop_size, self.short_input_size, self.global_input_size) < 1:
            raise ValueError("Positive input sizes required")
        if self.metadata.get("strict_protocol", True) and (crop_size, short_input_size, global_input_size) != (512, 512, 512):
            raise ValueError("Formal model inputs must each be512")
        self.normalization = normalization or self.metadata["normalization"]
        self.augmentation = bool(augmentation)
        self.spectral_augmentation = ROBUST_SPECTRAL_AUGMENTATION if variant == "robust" else SPECTRAL_AUGMENTATION
        self.global_views = GlobalViews(self.metadata, views_cache, size=global_input_size)
        self._cache, self._point_cache = {}, {}
        # Nothing below consults held24 labels/histograms during selection.
        self.class_scenes = {c: tuple(s for s in self.scene_ids if self.metadata["scenes"][s]["histogram"][c] > 0) for c in range(24)}
        self.class_scenes = {c: ss for c, ss in self.class_scenes.items() if ss}
        self.set_epoch(epoch, start_index=start_index)

    def set_epoch(self, epoch, start_index=0):
        self.epoch, self.start_index = int(epoch), int(start_index)
        if self.epoch < 0 or not 0 <= self.start_index <= self.samples_per_epoch or self.start_index % 64:
            raise ValueError("Resume offset must be an effective64 optimizer boundary")
        rng = random.Random(self.seed + self.epoch * 10000019)
        updates = self.samples_per_epoch // 64
        targeted = ["rare"] * (updates * 6 // 5) + ["hard"] * (updates * 4 // 5)
        if len(targeted) != updates * 2:
            raise AssertionError("Epoch budget must realize exact50/30/20")
        rng.shuffle(targeted)
        queue, blocks, coverage = deque(), [], set()
        for u in range(updates):
            chosen, current = set(), []
            for _ in range(2):
                if not queue:
                    fresh = list(self.scene_ids)
                    rng.shuffle(fresh)
                    queue.extend(fresh)
                for _ in range(len(queue)):
                    s = queue.popleft()
                    if s not in chosen:
                        break
                    queue.append(s)
                else:
                    s = rng.choice([s for s in self.scene_ids if s not in chosen])
                chosen.add(s)
                coverage.add(s)
                current.append((s, "scene", -1))
            for kind in targeted[u*2:u*2+2]:
                ids = RARE_CLASS_IDS if kind == "rare" else HARD_CLASS_IDS
                eligible = {c: [s for s in self.class_scenes.get(c, ()) if s not in chosen] for c in ids}
                eligible = {c: ss for c, ss in eligible.items() if ss}
                if not eligible:
                    raise ValueError("Cannot form four distinct train-only rare/hard source scenes")
                c = rng.choice(sorted(eligible))
                s = rng.choice(eligible[c])
                chosen.add(s)
                current.append((s, kind, c))
            rng.shuffle(current)
            blocks.extend(current)
        if coverage != set(self.scene_ids):
            raise AssertionError("Uniform arm lost training-scene coverage")
        self.blocks = tuple(blocks)

    def __len__(self):
        return self.samples_per_epoch - self.start_index

    def _scene(self, scene):
        if scene not in self._cache:
            self._cache[scene] = read_cached_scene(self.metadata, scene)
        return self._cache[scene]

    def _points(self, scene, c):
        key = (scene, c)
        if key not in self._point_cache:
            row = self.metadata["scenes"][scene]
            _check_paths(self.metadata, row)
            with np.load(row["points"], allow_pickle=False) as f:
                value = f[f"class_{c}"]
            if not len(value):
                raise ValueError("Class histogram contradicts point reservoir")
            self._point_cache[key] = value
        return self._point_cache[key]

    def __getitem__(self, index):
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        absolute = self.start_index + int(index)
        scene, kind, c = self.blocks[absolute // 16]
        rng = random.Random(self.seed + self.epoch * 10000019 + absolute * 65537)
        image, label = self._scene(scene)
        h, w = label.shape
        if min(h, w) < self.crop_size:
            raise ValueError("Scene smaller than local crop")
        anchor = None
        if kind != "scene":
            points = self._points(scene, c)
            cy, cx = points[rng.randrange(len(points))]
            anchor = (int(cy), int(cx))
            jitter = self.crop_size // 4
            top = int(np.clip(int(cy) - self.crop_size // 2 + rng.randint(-jitter, jitter), 0, h-self.crop_size))
            left = int(np.clip(int(cx) - self.crop_size // 2 + rng.randint(-jitter, jitter), 0, w-self.crop_size))
        else:
            top, left = rng.randrange(h-self.crop_size+1), rng.randrange(w-self.crop_size+1)
        source_size, scale = self.crop_size, 1.0
        if self.variant == "robust" and self.augmentation:
            # Independent augmentation RNG keeps sampling locations and flips
            # matched to HR, apart from the explicitly changed physical scale.
            scale_rng = random.Random(self.seed + self.epoch*10000019 + absolute*65537 + 917000003)
            scale = scale_rng.choice(ROBUST_SCALES)
            source_size = max(1, int(round(self.crop_size * scale)))
            if source_size > min(h, w):
                raise ValueError("Robust scaled source support exceeds scene")
            top = int(np.clip(round(top + (self.crop_size-source_size)/2), 0, h-source_size))
            left = int(np.clip(round(left + (self.crop_size-source_size)/2), 0, w-source_size))
            # A zoom-in at a scene edge must not silently discard the rare/hard
            # anchor that selected this crop. Keep it in the physical support.
            if anchor is not None:
                cy, cx = anchor
                top = int(np.clip(top, max(0, cy-source_size+1), min(cy, h-source_size)))
                left = int(np.clip(left, max(0, cx-source_size+1), min(cx, w-source_size)))
        sample, target = read_scaled_views(image, label, top=top, left=left, source_size=source_size,
            output_size=self.crop_size, short_input_size=self.short_input_size, global_input_size=self.global_input_size,
            global_raw=self.global_views.get(scene, image), normalization=self.normalization)
        if self.augmentation:
            horizontal, vertical = rng.random() < .5, rng.random() < .5
            sample = _flip_views(sample, horizontal, vertical)
            if horizontal:
                target = target[:, ::-1]
            if vertical:
                target = target[::-1]
            sample = _augment_views(sample, rng=rng, augmentation=self.spectral_augmentation, normalization=self.normalization)
        return {**{k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sample.items()},
                "label": torch.from_numpy(np.ascontiguousarray(target, dtype=np.int64)),
                "index": absolute, "scene_id": scene, "top": top, "left": left,
                "sampling_kind": kind, "sampling_class": c, "source_size": source_size,
                "augmentation_scale": scale}


__all__ = ["MS75RebuildDataset", "GlobalViews", "prepare_data", "prepare_view_cache", "load_manifest",
           "read_cached_scene", "read_views", "read_scaled_views", "MODEL_INPUT_KEYS", "NORMALIZATION",
           "CLASS_NAMES", "map_labels", "to_bgrn", "make_splits", "LOCAL_SIZE", "SHORT_CONTEXT_SIZE",
           "SHORT_INPUT_SIZE", "GLOBAL_INPUT_SIZE", "VIEW_PROTOCOL"]
