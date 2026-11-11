"""Train-only GID-5 / WHU data for fair, model-independent DINO fitting.

Cache preparation reads official TRAIN scenes only (including its internal held
fold). Selection sampling never consults held-fold counts or point reservoirs.
All cache images are uint8 B,G,R,NIR; labels are 0..C-1 with 255 ignored.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


PROTOCOL = "gid5_whu_native_dino_train_only_v1_20260922"
IGNORE_INDEX = 255
NORMALIZATION = {"divisor": 255.0, "mean": [.363, .390, .371, .506],
                 "std": [.231, .231, .244, .254]}
SPECTRAL_AUGMENTATION = {
    "enabled": True, "probability": .8, "global_gain_range": [.9, 1.1],
    "global_offset_range": [-.02, .02], "band_gain_range": [.97, 1.03],
    "band_offset_range": [-.008, .008],
}
CLASS_NAMES = {
    "gid5": ["built-up", "farmland", "forest", "meadow", "water"],
    "whu": ["farmland", "city", "village", "water", "forest", "road", "others"],
}


def _canonical(dataset: str) -> str:
    result = dataset.lower().replace("-", "").replace("_", "")
    if result == "whuoptsar":
        result = "whu"
    if result not in CLASS_NAMES:
        raise ValueError("dataset must be gid5 or whu")
    return result


def _ids(values: Sequence[str]) -> list[str]:
    result = [str(v) for v in values]
    if not result or len(set(result)) != len(result):
        raise ValueError("Scene IDs must be nonempty and unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in result):
        raise ValueError("Unsafe scene ID")
    return sorted(result)


def make_splits(dataset: str, train_ids: Sequence[str], test_ids: Sequence[str],
                *, selection_train_ids=None, held_ids=None, seed=1337,
                strict_protocol=True) -> dict:
    dataset = _canonical(dataset)
    train, test = _ids(train_ids), _ids(test_ids)
    if set(train) & set(test):
        raise ValueError("Official train/test split overlaps")
    if strict_protocol and (len(train), len(test)) != {"gid5": (120, 30), "whu": (80, 20)}[dataset]:
        raise ValueError("Wrong official scene counts")
    if (selection_train_ids is None) != (held_ids is None):
        raise ValueError("Provide both internal fold scene lists, or neither")
    if selection_train_ids is None:
        if dataset == "gid5":
            raise ValueError("GID-5 requires the authenticated existing FBP fold0 lists")
        order = list(train)
        random.Random(int(seed)).shuffle(order)
        split_at = len(order) * 4 // 5
        selection_train_ids, held_ids = order[:split_at], order[split_at:]
    selection, held = _ids(selection_train_ids), _ids(held_ids)
    if set(selection) & set(held) or set(selection + held) != set(train):
        raise ValueError("Internal folds must partition official train exactly")
    if strict_protocol and (len(selection), len(held)) != {"gid5": (96, 24), "whu": (64, 16)}[dataset]:
        raise ValueError("Wrong internal train/held scene counts")
    if len(selection) < 4 or len(train) < 4:
        raise ValueError("At least four training scenes required per update")
    return {"train_ids": train, "test_ids": test,
            "selection_train_ids": selection, "held_ids": held}


def to_bgrn(raw: np.ndarray, dataset: str) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 4:
        raise ValueError("Expected raw HWC four-band uint8 image")
    return np.ascontiguousarray(raw[..., [3, 2, 1, 0]] if _canonical(dataset) == "gid5" else raw)


def map_labels(raw: np.ndarray, dataset: str) -> np.ndarray:
    raw = np.asarray(raw)
    dataset = _canonical(dataset)
    if raw.dtype != np.uint8 or raw.ndim != 2:
        raise ValueError("Expected HW uint8 class labels")
    codes = list(range(6)) if dataset == "gid5" else list(range(0, 71, 10))
    actual = set(np.unique(raw).tolist())
    if not actual.issubset(codes):
        raise ValueError(f"Unexpected native label codes: {sorted(actual - set(codes))}")
    lookup = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    lookup[codes[1:]] = np.arange(len(codes) - 1, dtype=np.uint8)
    return lookup[raw]


def _digest_json(value: Mapping) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json_write(path: Path, value: Mapping) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _binding(path: Path) -> dict:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source changed while authenticating: " + str(path))
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def _validate_cached_row(row: Mapping, *, cache_root: Path, identity: str) -> None:
    if row.get("request_sha256") != identity:
        raise ValueError("Cached scene request identity mismatch")
    for key in ("image", "label", "points"):
        path = Path(row[key])
        if path.resolve().parent != cache_root.resolve() or not path.is_file():
            raise ValueError("Cached scene path invalid")
    for key in ("image", "label"):
        value = np.load(row[key], mmap_mode="r", allow_pickle=False)
        shape = (*row["shape"], 4) if key == "image" else tuple(row["shape"])
        if value.dtype != np.uint8 or value.shape != shape or hashlib.sha256(memoryview(value)).hexdigest() != row[key + "_sha256"]:
            raise ValueError("Cached image/label content identity mismatch")
        del value
    if hashlib.sha256(Path(row["points"]).read_bytes()).hexdigest() != row["points_sha256"]:
        raise ValueError("Cached point reservoir content identity mismatch")


def _points(label: np.ndarray, num_classes: int, *, seed: int, capacity: int) -> tuple[list[int], dict]:
    """Uniform class occurrence sampling; bounded-memory exact reservoir result."""
    counts = np.zeros(256, dtype=np.int64)
    for y in range(0, label.shape[0], 512):
        counts += np.bincount(label[y:y + 512].ravel(), minlength=256)
    result = {}
    for c in range(num_classes):
        count = int(counts[c])
        if count == 0:
            result[f"class_{c}"] = np.empty((0, 2), dtype=np.int32)
            continue
        rng = np.random.default_rng(seed + c * 104729)
        ranks = np.sort(rng.choice(count, min(capacity, count), replace=False))
        seen, gathered = 0, []
        for y in range(0, label.shape[0], 512):
            flat = np.flatnonzero(label[y:y + 512].ravel() == c)
            end = seen + len(flat)
            selected = ranks[(ranks >= seen) & (ranks < end)] - seen
            if len(selected):
                positions = flat[selected]
                gathered.append(np.stack((positions // label.shape[1] + y, positions % label.shape[1]), axis=1))
            seen = end
        result[f"class_{c}"] = np.concatenate(gathered).astype(np.int32)
    return counts[:num_classes].tolist(), result


def prepare_data(*, dataset: str, image_root, label_root, train_ids, test_ids,
                 selection_train_ids=None, held_ids=None, cache_root,
                 seed=1337, workers=4, point_capacity=4096,
                 strict_protocol=True) -> Path:
    """Create an isolated, reusable cache; never inspect official test files.

    Existing complete cache reuse requires identical protocol and source file
    bindings. Partial cache preparation can resume only from its own request.
    Test identities are metadata only; no path or stat call is made for them.
    """
    dataset = _canonical(dataset)
    splits = make_splits(dataset, train_ids, test_ids,
                         selection_train_ids=selection_train_ids, held_ids=held_ids,
                         seed=seed, strict_protocol=strict_protocol)
    if workers < 1 or point_capacity < 1:
        raise ValueError("Positive workers and point_capacity required")
    image_root, label_root, cache_root = Path(image_root), Path(label_root), Path(cache_root)
    bindings = {}
    for scene in splits["train_ids"]:
        image = image_root / f"{scene}.tif"
        label = label_root / (f"{scene}_5label.png" if dataset == "gid5" else f"{scene}.tif")
        bindings[scene] = {"image": _binding(image), "label": _binding(label)}
    request = {"protocol": PROTOCOL, "dataset": dataset, **splits, "seed": int(seed),
               "normalization": NORMALIZATION, "point_capacity": int(point_capacity),
               "bindings": bindings, "strict_protocol": bool(strict_protocol)}
    identity = _digest_json(request)
    cache_root.mkdir(parents=True, exist_ok=True)
    request_path, output = cache_root / "request.json", cache_root / "manifest.json"
    if request_path.exists():
        if json.loads(request_path.read_text(encoding="utf-8")) != request:
            raise ValueError("Cache request mismatch; use a new isolated cache directory")
    else:
        if list(cache_root.iterdir()):
            raise ValueError("Refusing nonempty cache without request identity")
        _json_write(request_path, request)
    if output.exists():
        manifest = load_manifest(output)
        if manifest["request_sha256"] != identity:
            raise ValueError("Existing cache manifest identity differs")
        for scene in splits["train_ids"]:
            _validate_cached_row(manifest["scenes"][scene], cache_root=cache_root, identity=identity)
        return output

    def prepare_scene(scene: str) -> tuple[str, dict]:
        image_path, label_path = cache_root / f"{scene}.image.npy", cache_root / f"{scene}.label.npy"
        point_path, receipt_path = cache_root / f"{scene}.points.npz", cache_root / f"{scene}.json"
        if receipt_path.exists():
            row = json.loads(receipt_path.read_text(encoding="utf-8"))
            if row.get("request_sha256") == identity and all(Path(row[k]).is_file() for k in ("image", "label", "points")):
                _validate_cached_row(row, cache_root=cache_root, identity=identity)
                return scene, row
            raise ValueError("Partial scene receipt identity or file mismatch")
        with Image.open(bindings[scene]["image"]["path"]) as handle:
            image = to_bgrn(np.array(handle), dataset)  # Never CMYK/RGBA convert.
        with Image.open(bindings[scene]["label"]["path"]) as handle:
            label = map_labels(np.array(handle), dataset)
        for binding in bindings[scene].values():
            stat = Path(binding["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (binding["size"], binding["mtime_ns"]):
                raise RuntimeError("Authenticated source changed before decoding")
        if image.shape[:2] != label.shape:
            raise ValueError(f"Image/label shape mismatch: {scene}")
        if strict_protocol:
            h, w = label.shape
            if ((dataset == "gid5" and (h, w) != (6908, 7300)) or
                    (dataset == "whu" and (h != 3704 or w not in (5555, 5556)))):
                raise ValueError(f"Unexpected original scene geometry: {scene} {label.shape}")
        seed_scene = int(hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest()[:16], 16)
        histogram, points = _points(label, len(CLASS_NAMES[dataset]), seed=seed_scene, capacity=point_capacity)
        for path, array in ((image_path, image), (label_path, label)):
            temporary = path.with_name(path.name + ".partial")
            with temporary.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            os.replace(temporary, path)
        temporary = point_path.with_name(point_path.name + ".partial")
        with temporary.open("wb") as handle:
            np.savez(handle, **points)
        os.replace(temporary, point_path)
        for path in (image_path, label_path, point_path):
            path.chmod(0o444)
        row = {"image": str(image_path.resolve()), "label": str(label_path.resolve()),
               "points": str(point_path.resolve()), "shape": list(label.shape),
               "histogram": histogram, "request_sha256": identity,
               "image_sha256": hashlib.sha256(memoryview(image)).hexdigest(),
               "label_sha256": hashlib.sha256(memoryview(label)).hexdigest(),
               "points_sha256": hashlib.sha256(point_path.read_bytes()).hexdigest()}
        _json_write(receipt_path, row)
        print(json.dumps({"event": "cached_train_scene", "dataset": dataset, "scene": scene}), flush=True)
        return scene, row

    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        rows = dict(pool.map(prepare_scene, splits["train_ids"]))
    selection_hist = np.sum([rows[s]["histogram"] for s in splits["selection_train_ids"]], axis=0).tolist()
    full_hist = np.sum([rows[s]["histogram"] for s in splits["train_ids"]], axis=0).tolist()
    held_hist = np.sum([rows[s]["histogram"] for s in splits["held_ids"]], axis=0).tolist()
    manifest = {**request, "request_sha256": identity, "cache_root": str(cache_root.resolve()),
                "num_classes": len(CLASS_NAMES[dataset]), "class_names": CLASS_NAMES[dataset],
                "band_order": ["B", "G", "R", "NIR"], "ignore_index": IGNORE_INDEX,
                "scenes": rows, "selection_histogram": selection_hist,
                "full_histogram": full_hist, "held_histogram_for_audit_only": held_hist,
                "sampling": {"epoch_crops_per_train_scene_budget": 320, "block_size": 16,
                             "effective_batch_size": 64, "scene_fraction": .5,
                             "class_fraction": .5, "test_data_read": False,
                             "selection_held_labels_used_for_sampling": False,
                             "scene_budget_is_average_not_exact_visit_count": True}}
    _json_write(output, manifest)
    return output


def load_manifest(manifest) -> dict:
    result = dict(manifest) if isinstance(manifest, Mapping) else json.loads(Path(manifest).read_text(encoding="utf-8"))
    if result.get("protocol") != PROTOCOL or result.get("ignore_index") != IGNORE_INDEX:
        raise ValueError("Unexpected target training cache protocol")
    if set(result["scenes"]) != set(result["train_ids"]):
        raise ValueError("Cache must contain official train scenes only")
    make_splits(result["dataset"], result["train_ids"], result["test_ids"],
                selection_train_ids=result["selection_train_ids"], held_ids=result["held_ids"],
                seed=result["seed"], strict_protocol=result.get("strict_protocol", True))
    if result["num_classes"] != len(CLASS_NAMES[_canonical(result["dataset"])]):
        raise ValueError("Unexpected native class count")
    request_keys = ("protocol", "dataset", "train_ids", "test_ids", "selection_train_ids",
                    "held_ids", "seed", "normalization", "point_capacity", "bindings", "strict_protocol")
    if _digest_json({k: result[k] for k in request_keys}) != result["request_sha256"]:
        raise ValueError("Manifest request identity changed")
    if set(result["bindings"]) != set(result["train_ids"]):
        raise ValueError("Source bindings must cover only official train")
    for pair in result["bindings"].values():
        for binding in pair.values():
            if not re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", ""))):
                raise ValueError("Missing source input SHA256")
    return result


def read_cached_scene(manifest, scene_id: str) -> tuple[np.ndarray, np.ndarray]:
    metadata = manifest if isinstance(manifest, Mapping) else load_manifest(manifest)
    if scene_id not in metadata["train_ids"] or scene_id in metadata["test_ids"]:
        raise ValueError("Only official train/held scenes may be opened")
    row = metadata["scenes"][scene_id]
    root = Path(metadata["cache_root"]).resolve()
    for key in ("image", "label"):
        if Path(row[key]).resolve().parent != root:
            raise ValueError("Cached scene path escapes dedicated cache")
    image, label = np.load(row["image"], mmap_mode="r", allow_pickle=False), np.load(row["label"], mmap_mode="r", allow_pickle=False)
    if image.dtype != np.uint8 or label.dtype != np.uint8 or image.shape != (*row["shape"], 4) or label.shape != tuple(row["shape"]):
        raise ValueError("Cached scene dtype/shape mismatch")
    return image, label


def normalize_bgrn(image: np.ndarray, normalization=None) -> np.ndarray:
    norm = normalization or NORMALIZATION
    norm = norm.get("ms", norm)
    mean, std = np.asarray(norm["mean"], dtype=np.float32), np.asarray(norm["std"], dtype=np.float32)
    divisor = float(norm.get("divisor", 255.0))
    if (mean.shape != (4,) or std.shape != (4,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)
            or not np.isfinite(divisor) or divisor <= 0):
        raise ValueError("Invalid four-band normalization")
    return np.ascontiguousarray((image.astype(np.float32).transpose(2, 0, 1) / divisor - mean[:, None, None]) / std[:, None, None])


class TargetDataset(Dataset):
    """Identical deterministic crop stream for N2/N3 and interrupted resumes."""
    def __init__(self, manifest, *, stage="selection", epoch=0, start_index=0,
                 crop_size=512, spectral_augmentation=None, normalization=None):
        self.metadata = load_manifest(manifest)
        if stage not in ("selection", "full"):
            raise ValueError("stage must be selection or full")
        self.stage = stage
        self.scene_ids = tuple(self.metadata["selection_train_ids"] if stage == "selection" else self.metadata["train_ids"])
        self.num_classes, self.seed = self.metadata["num_classes"], int(self.metadata["seed"])
        self.samples_per_epoch = len(self.scene_ids) * 320
        self.crop_size = int(crop_size)
        if self.crop_size < 1:
            raise ValueError("Positive crop size required")
        self.normalization = normalization or self.metadata["normalization"]
        self.spectral_augmentation = dict(SPECTRAL_AUGMENTATION if spectral_augmentation is None else spectral_augmentation)
        self._cache, self._point_cache = {}, {}
        # Crucially exclude held-fold class presence and point reservoirs here.
        self.class_scenes = {c: tuple(s for s in self.scene_ids if self.metadata["scenes"][s]["histogram"][c] > 0)
                             for c in range(self.num_classes)}
        self.class_scenes = {c: ss for c, ss in self.class_scenes.items() if ss}
        if not self.class_scenes:
            raise ValueError("No supervised training pixels")
        self.set_epoch(epoch, start_index=start_index)

    def set_epoch(self, epoch: int, start_index=0):
        self.epoch, self.start_index = int(epoch), int(start_index)
        if self.epoch < 0 or not 0 <= self.start_index <= self.samples_per_epoch or self.start_index % 64:
            raise ValueError("Resume offset must lie on an effective-update64 boundary")
        rng = random.Random(self.seed + self.epoch * 10000019)
        queue, blocks = deque(), []
        uniform_coverage = set()
        for _ in range(self.samples_per_epoch // 64):
            # Uniform first reserves scene coverage independently of labels.
            chosen, current = set(), []
            for _ in range(2):
                if not queue:
                    fresh = list(self.scene_ids)
                    rng.shuffle(fresh)
                    queue.extend(fresh)
                for _ in range(len(queue)):
                    scene = queue.popleft()
                    if scene not in chosen:
                        break
                    queue.append(scene)
                else:
                    fresh = [s for s in self.scene_ids if s not in chosen]
                    rng.shuffle(fresh)
                    scene = fresh[0]
                chosen.add(scene)
                uniform_coverage.add(scene)
                current.append((scene, "scene", -1))
            for _ in range(2):
                eligible = {c: [s for s in ss if s not in chosen] for c, ss in self.class_scenes.items()}
                eligible = {c: ss for c, ss in eligible.items() if ss}
                if not eligible:
                    raise ValueError("Cannot form four distinct supervised scenes")
                c = rng.choice(sorted(eligible))
                scene = rng.choice(eligible[c])
                chosen.add(scene)
                current.append((scene, "class", c))
            rng.shuffle(current)
            blocks.extend(current)
        if uniform_coverage != set(self.scene_ids):
            raise AssertionError("Uniform sampler failed complete training-scene coverage")
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
            path = Path(self.metadata["scenes"][scene]["points"])
            if path.resolve().parent != Path(self.metadata["cache_root"]).resolve():
                raise ValueError("Point cache escapes dedicated root")
            with np.load(path, allow_pickle=False) as data:
                value = data[f"class_{c}"]
            if not len(value):
                raise ValueError("Class histogram has no corresponding point reservoir")
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
            raise ValueError("Target scene smaller than crop; resizing/padding is not allowed")
        if kind == "class":
            points = self._points(scene, c)
            cy, cx = points[rng.randrange(len(points))]
            # Shift center within +/-quarter crop; selected anchor stays in crop.
            jitter = self.crop_size // 4
            y = int(np.clip(int(cy) - self.crop_size // 2 + rng.randint(-jitter, jitter), 0, h - self.crop_size))
            x = int(np.clip(int(cx) - self.crop_size // 2 + rng.randint(-jitter, jitter), 0, w - self.crop_size))
        else:
            y, x = rng.randrange(h - self.crop_size + 1), rng.randrange(w - self.crop_size + 1)
        crop = np.array(image[y:y + self.crop_size, x:x + self.crop_size], copy=True)
        target = np.array(label[y:y + self.crop_size, x:x + self.crop_size], dtype=np.int64, copy=True)
        if rng.random() < .5:
            crop, target = crop[:, ::-1], target[:, ::-1]
        if rng.random() < .5:
            crop, target = crop[::-1], target[::-1]
        ms = normalize_bgrn(crop, self.normalization)
        aug = self.spectral_augmentation
        if aug.get("enabled", False) and rng.random() < float(aug.get("probability", 0)):
            norm = self.normalization.get("ms", self.normalization)
            mean, std = np.asarray(norm["mean"], np.float32)[:, None, None], np.asarray(norm["std"], np.float32)[:, None, None]
            gain = rng.uniform(*aug.get("global_gain_range", [1, 1]))
            offset = rng.uniform(*aug.get("global_offset_range", [0, 0]))
            band_gain = np.array([rng.uniform(*aug.get("band_gain_range", [1, 1])) for _ in range(4)], np.float32)[:, None, None]
            band_offset = np.array([rng.uniform(*aug.get("band_offset_range", [0, 0])) for _ in range(4)], np.float32)[:, None, None]
            ms = (np.clip((ms * std + mean) * gain * band_gain + offset + band_offset, 0, 1.5) - mean) / std
        target = np.ascontiguousarray(target)
        return {"ms": torch.from_numpy(np.ascontiguousarray(ms)), "label": torch.from_numpy(target),
                "scene_id": scene, "index": absolute, "left": x, "top": y,
                "sampling_kind": kind, "sampling_class": c}
