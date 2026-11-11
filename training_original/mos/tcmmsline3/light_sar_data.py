"""Single-view, train96-only data for the lightweight SAR experiment.

The existing FBP24 cache is opened read-only (already B,G,R,NIR and 0..23/255).
Only this experiment's support points/statistics are prepared in a new folder.
No MS75 dataset or multiview image pipeline is instantiated. P/R/Z deliberately
share one ``adapt`` dataset; modality zeroing belongs to the model, not sampling.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Mapping
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from .ms75_rebuild_data import load_manifest as load_raw_manifest, read_cached_scene
from .target_dino_data import NORMALIZATION, SPECTRAL_AUGMENTATION, normalize_bgrn

SCHEMA = "fbp24_light_sar_train96_v2"
MS_SIZE, SAR_SIZE, EFFECTIVE_BATCH, BLOCK_SIZE = 512, 166, 32, 8


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_ms(raw_bgrn, normalization=None):
    """Normalize uint8 HWC BGRN (or a batch BHWC); no channel permutation."""
    raw = np.asarray(raw_bgrn)
    if raw.dtype != np.uint8 or raw.ndim not in (3, 4) or raw.shape[-1] != 4:
        raise ValueError("MS cache must be uint8 HWC/BHWC B,G,R,NIR")
    if raw.ndim == 4:
        return np.stack([normalize_bgrn(x, normalization) for x in raw])
    return normalize_bgrn(raw, normalization)


def load_scene_arrays(manifest, scene_id):
    """Return read-only BGRN uint8 HWC and mapped uint8 HW memmaps.

    Official train120 includes held24 for evaluation. Official30 is refused.
    This wrapper does not hash the complete image on every window read.
    """
    metadata = load_raw_manifest(manifest)
    return read_cached_scene(metadata, scene_id)


def open_sar_scene(data_root, scene_id):
    """The existing authenticated, label-free pixel-centre native reader."""
    from .semantic_aux_data import SemanticAuxScene
    return SemanticAuxScene(data_root, scene_id)


def native_valid_to_optical(mask, size=MS_SIZE):
    """Conservative bilinear support, exactly the native reader's convention.

    Pixel centres use (i+.5)*native/optical-.5, with border extension. This
    only transfers lattice support: AND with read_patch()['valid'] to retain
    the direct physical/registration support as well.
    """
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim not in (2, 3) or (mask.ndim == 3 and mask.shape[0] != 1):
        raise ValueError("native support must be bool HW or 1HW")
    if mask.shape[-2] != mask.shape[-1] or mask.shape[-1] < 1 or type(size) is not int or size < 1:
        raise ValueError("positive square native/optical supports required")
    plane = mask[0] if mask.ndim == 3 else mask
    n = plane.shape[0]
    x = np.clip((np.arange(size, dtype=np.float64) + .5) * n / size - .5, 0, n - 1)
    rounded = np.rint(x)
    x = np.where(np.abs(x - rounded) < 1e-7, rounded, x)
    lo, hi = np.floor(x).astype(np.int64), np.ceil(x).astype(np.int64)
    result = plane[lo[:, None], lo] & plane[lo[:, None], hi] & plane[hi[:, None], lo] & plane[hi[:, None], hi]
    return result[None] if mask.ndim == 3 else result


def apply_flips(value, code):
    """Apply the same horizontal/vertical reflection on both native lattices."""
    if type(code) is not int or not 0 <= code < 4:
        raise ValueError("flip code must be integer0..3")
    if code & 1:
        value = value[..., ::-1]
    if code & 2:
        value = value[..., ::-1, :]
    return np.ascontiguousarray(value)


def _availability(data_root):
    root = Path(data_root)
    path = (root if (root / "availability.csv").is_file() else root / "auxiliary") / "availability.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        scene = row["scene_stem"]
        flag = row["has_sar"].strip().lower()
        if scene in result or flag not in ("yes", "no", "true", "false", "1", "0"):
            raise ValueError("ambiguous scene/SAR availability")
        result[scene] = flag in ("yes", "true", "1")
    return result, path


def _load_points(path, expected_sha=None):
    if expected_sha is not None and _sha(path) != expected_sha:
        raise ValueError("class-point reservoir SHA changed")
    with np.load(path, allow_pickle=False) as archive:
        return {c: np.asarray(archive[f"class_{c}"], dtype=np.int64) for c in range(24)}


def _validate_points(points, labels):
    for c, values in points.items():
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("class points must have shape N2")
        if not len(values):
            continue
        y, x = values.T
        if ((y < 0).any() or (x < 0).any() or (y >= labels.shape[0]).any()
                or (x >= labels.shape[1]).any() or not np.all(labels[y, x] == c)):
            raise ValueError("class-point FBP24 label/coordinate binding changed")


def _checked_aux(aux, crop_size, sar_size):
    values = np.asarray(aux["aux"])
    native, optical = np.asarray(aux["native_valid"]), np.asarray(aux["valid"])
    if (values.shape != (2, sar_size, sar_size) or values.dtype != np.float32
            or native.shape != (1, sar_size, sar_size) or native.dtype != np.bool_
            or optical.shape != (1, crop_size, crop_size) or optical.dtype != np.bool_
            or not np.isfinite(values).all() or np.any(values[:, ~native[0]] != 0)
            or aux.get("quality", {}).get("label_usage") is not False):
        raise ValueError("SAR native values/support/label-free contract differs")
    if np.any(optical & ~native_valid_to_optical(native, crop_size)):
        raise ValueError("optical SAR support escaped conservative native support")
    return values, native, optical


def _label_integral(labels):
    """Bounded per-scene uint32 summed area table (FBP has <2**32 pixels)."""
    if labels.size >= 2**32:
        raise ValueError("label support exceeds uint32 integral capacity")
    result = np.zeros((labels.shape[0] + 1, labels.shape[1] + 1), np.uint32)
    np.cumsum(labels != 255, axis=0, dtype=np.uint32, out=result[1:, 1:])
    np.cumsum(result[1:, 1:], axis=1, dtype=np.uint32, out=result[1:, 1:])
    return result


def _label_window_count(integral, top, left, size):
    # Convert corners separately to Python int to avoid unsigned subtraction.
    return (int(integral[top+size, left+size]) - int(integral[top, left+size])
            - int(integral[top+size, left]) + int(integral[top, left]))


def _find_fallback(labels, integral, *, crop_size, sar_size, minimum_fraction,
                   class_id=-1, points=None, reader=None, anchor_limit=32, grid_axes=4):
    """Find a positive witness, never claim finite failure proves infeasibility.

    Class candidates contain the actual anchor at centre/corners/edge centres,
    allowing minority objects at the edge of a sufficiently labelled window.
    SAR warps are performed only after the inexpensive GT-area check passes.
    """
    h, w = labels.shape
    n = crop_size
    minimum = math.ceil(n*n*minimum_fraction)
    if min(h, w) < n:
        raise ValueError("training scene smaller than fallback support")
    if points is None:
        candidates = ((int(y), int(x), None) for y in np.unique(np.linspace(0, h-n, grid_axes).round().astype(int))
                      for x in np.unique(np.linspace(0, w-n, grid_axes).round().astype(int)))
    else:
        if not len(points):
            return None, {"reason": "empty_supported_point_reservoir", "candidate_count": 0, "sar_reads": 0}
        positions = ((n//2, n//2), (0, 0), (0, n-1), (n-1, 0), (n-1, n-1),
                     (0, n//2), (n-1, n//2), (n//2, 0), (n//2, n-1))
        selected = np.unique(np.linspace(0, len(points)-1, min(anchor_limit, len(points))).round().astype(int))
        def class_candidates():
            for index in selected:
                cy, cx = map(int, points[index])
                for oy, ox in positions:
                    top, left = int(np.clip(cy-oy, 0, h-n)), int(np.clip(cx-ox, 0, w-n))
                    yield top, left, (cy, cx)
        candidates = class_candidates()
    count, sar_reads, best_gt, best_supervised, seen = 0, 0, 0, 0, set()
    for top, left, anchor in candidates:
        key = (top, left, anchor)
        if key in seen:
            continue
        seen.add(key)
        count += 1
        labelled = _label_window_count(integral, top, left, n)
        best_gt = max(best_gt, labelled)
        if labelled < minimum:
            continue
        if anchor is not None and labels[anchor] != class_id:
            raise ValueError("fallback anchor no longer matches FBP24 class")
        if reader is None:
            supported = labelled
        else:
            aux = reader.read_patch(top, left, n, modality="sar", out_size=sar_size)
            _, _, optical = _checked_aux(aux, n, sar_size)
            sar_reads += 1
            supported = int(((labels[top:top+n, left:left+n] != 255) & optical[0]).sum())
            if anchor is not None and not optical[0, anchor[0]-top, anchor[1]-left]:
                best_supervised = max(best_supervised, supported)
                continue
        best_supervised = max(best_supervised, supported)
        if supported >= minimum:
            return {"top": top, "left": left, "anchor": list(anchor) if anchor is not None else [-1, -1],
                    "class_id": int(class_id), "supervised_count": supported}, {
                        "candidate_count": count, "sar_reads": sar_reads, "reason": "valid_witness"}
    return None, {"reason": "no_valid_window_in_predefined_candidate_set_not_proof_of_global_infeasibility",
                  "candidate_count": count, "sar_reads": sar_reads, "best_gt_count": best_gt,
                  "best_supervised_count": best_supervised, "anchor_limit": anchor_limit}


def prepare_training_support(raw_manifest, data_root, output_dir, *, stats_windows_per_axis=4,
                             crop_size=MS_SIZE, sar_size=SAR_SIZE,
                             min_supervised_fraction=.10, max_attempts=16):
    """Prepare ONLY train96 SAR support reservoirs and fixed raw-dB moments.

    Moments use a fixed label-independent grid of registered native samples,
    not deduplicated full-raster pixels. Spatial overlaps are counted again.
    Existing raw-cache arrays are never modified or re-created. Complete image
    hashes are inherited from the authenticated cache; points and their sampled
    mapped GT identity are rechecked here. The receipt states this limitation.
    """
    if type(stats_windows_per_axis) is not int or stats_windows_per_axis < 2:
        raise ValueError("statistics grid must have at least two positions per axis")
    raw_path, root, out = Path(raw_manifest).resolve(), Path(data_root).resolve(), Path(output_dir).resolve()
    metadata = load_raw_manifest(raw_path)
    strict = metadata.get("strict_protocol", True)
    if strict and (crop_size, sar_size) != (MS_SIZE, SAR_SIZE):
        raise ValueError("formal light SAR supports must be512/166")
    if min(crop_size, sar_size) < 1:
        raise ValueError("positive support sizes required")
    if not 0 < min_supervised_fraction <= 1 or type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("positive supervision threshold and finite redraw bound required")
    protected = [Path(metadata["cache_root"]).resolve(), root]
    if any(out == p or p in out.parents or out in p.parents for p in protected):
        raise ValueError("support output must be outside the source and existing cache")
    present, availability_path = _availability(root)
    train, held = list(metadata["selection_train_ids"]), list(metadata["held_ids"])
    if any(s not in present for s in train + held):
        raise ValueError("availability does not cover train96/held24")
    adapted = [s for s in train if present[s]]
    if len(adapted) < 4 or (strict and len(adapted) != 83):
        raise ValueError("adaptation requires the authenticated83 SAR training scenes")
    request = {"schema": SCHEMA, "raw_manifest": str(raw_path), "raw_manifest_sha256": _sha(raw_path),
               "source_request_sha256": metadata["request_sha256"], "data_root": str(root),
               "availability_sha256": _sha(availability_path), "train_ids": train, "held_ids": held,
               "adapt_scene_ids": adapted, "held_sar_scene_ids": [s for s in held if present[s]],
               "held_missing_sar_scene_ids": [s for s in held if not present[s]],
               "crop_size": crop_size, "sar_size": sar_size, "stats_windows_per_axis": stats_windows_per_axis,
               "min_supervised_fraction": float(min_supervised_fraction), "max_attempts": max_attempts,
               "fallback_protocol": "same_scene_class_prechecked_window_after_random_proposals_v2",
               "fallback_anchor_limit": 32,
               "seed": 1337, "strict_protocol": strict}
    destination = out / "prepared_manifest.json"
    if destination.exists():
        previous = load_prepared_manifest(destination)
        if any(previous.get(k) != v for k, v in request.items()):
            raise ValueError("existing prepared support belongs to a different request")
        return destination
    if out.exists() and any(out.iterdir()):
        raise ValueError("support output must be a new empty experiment directory")
    out.mkdir(parents=True, exist_ok=True)
    sums, squares, count = np.zeros(2, np.float64), np.zeros(2, np.float64), 0
    records, base_records = {}, {}
    class_scenes = {str(c): [] for c in range(24)}
    base_class_scenes = {str(c): [] for c in range(24)}
    begun = time.monotonic()
    for scene_index, scene in enumerate(train, start=1):
        _, labels = read_cached_scene(metadata, scene)
        row = metadata["scenes"][scene]
        points = _load_points(row["points"], row["points_sha256"])
        _validate_points(points, labels)
        combined = np.concatenate(list(points.values()), axis=0)
        if not len(combined):
            raise ValueError("training scene has no class points")
        integral = _label_integral(labels)
        base_fallback, base_audit, base_excluded = {}, {}, {}
        witness, audit = _find_fallback(labels, integral, crop_size=crop_size, sar_size=sar_size,
                                        minimum_fraction=min_supervised_fraction,
                                        grid_axes=stats_windows_per_axis)
        if witness is not None:
            base_fallback["scene"], base_audit["scene"] = witness, audit
        for c, values in points.items():
            if not len(values):
                continue
            witness, audit = _find_fallback(labels, integral, crop_size=crop_size, sar_size=sar_size,
                                            minimum_fraction=min_supervised_fraction, class_id=c, points=values)
            if witness is None:
                base_excluded[str(c)] = audit
            else:
                base_fallback[f"class_{c}"] = witness
                base_audit[f"class_{c}"] = audit
                base_class_scenes[str(c)].append(scene)
        if "scene" not in base_fallback and base_fallback:
            witness = next(iter(base_fallback.values()))
            base_fallback["scene"] = {**witness, "anchor": [-1, -1], "class_id": -1}
            base_audit["scene"] = {"reason": "reuse_valid_class_window_as_natural_witness"}
        if "scene" not in base_fallback:
            raise ValueError(f"no pre-audited natural base window for {scene}")
        base_records[scene] = {"fallbacks": base_fallback, "fallback_search": base_audit,
                               "excluded_target_classes": base_excluded}
        if scene not in adapted:
            del integral
            print(json.dumps({"event": "light_sar_prepare_scene", "index": scene_index, "total": len(train),
                              "scene": scene, "has_sar": False, "base_class_witnesses": len(base_fallback)-1,
                              "base_excluded": sorted(base_excluded), "elapsed_seconds": time.monotonic()-begun}), flush=True)
            continue
        reader = open_sar_scene(root, scene)
        try:
            if not reader.available["sar"]:
                raise ValueError("SAR availability changed during preparation")
            # This is the same raw sampling/mask/registration logic as read_patch.
            # Points are filtered by physical support, then crop-specific support
            # is checked again when drawing every targeted training example.
            _, valid_points, _ = reader._sample("sar", combined[:, 1].astype(np.float64), combined[:, 0].astype(np.float64))
            filtered, first = {}, 0
            for c, values in points.items():
                filtered[c] = values[valid_points[first:first + len(values)]]
                first += len(values)
            if not any(len(v) for v in filtered.values()):
                raise ValueError("declared SAR scene has no supervised support points")
            point_path = out / f"{scene}.sar_points.npz"
            np.savez_compressed(point_path, **{f"class_{c}": p for c, p in filtered.items()})
            h, w = labels.shape
            if min(h, w) < crop_size:
                raise ValueError("training scene smaller than crop")
            stats_count, windows, maximum_fraction = 0, [], 0.
            adapt_fallback, adapt_audit, adapt_excluded = {}, {}, {}
            tops = np.unique(np.linspace(0, h - crop_size, stats_windows_per_axis).round().astype(int))
            lefts = np.unique(np.linspace(0, w - crop_size, stats_windows_per_axis).round().astype(int))
            for top in tops:
                for left in lefts:
                    aux = reader.read_patch(int(top), int(left), crop_size, modality="sar", out_size=sar_size)
                    values, native, optical = _checked_aux(aux, crop_size, sar_size)
                    valid_values = values[:, native[0]].astype(np.float64)
                    n = valid_values.shape[1]
                    sums += valid_values.sum(axis=1)
                    squares += np.square(valid_values).sum(axis=1)
                    count += n
                    stats_count += n
                    labelled = labels[top:top+crop_size, left:left+crop_size] != 255
                    fraction = float((optical[0] & labelled).mean())
                    maximum_fraction = max(maximum_fraction, fraction)
                    windows.append([int(top), int(left), n, fraction])
                    if fraction >= min_supervised_fraction and "scene" not in adapt_fallback:
                        adapt_fallback["scene"] = {"top": int(top), "left": int(left), "anchor": [-1, -1],
                                                   "class_id": -1, "supervised_count": int((optical[0] & labelled).sum())}
                        adapt_audit["scene"] = {"reason": "valid_statistics_grid_witness", "candidate_count": len(windows)}
            for c, values in filtered.items():
                if not len(values):
                    if len(points[c]):
                        adapt_excluded[str(c)] = {"reason": "no_physical_SAR_supported_class_points"}
                    continue
                witness, audit = _find_fallback(labels, integral, crop_size=crop_size, sar_size=sar_size,
                                                minimum_fraction=min_supervised_fraction, class_id=c,
                                                points=values, reader=reader)
                if witness is None:
                    adapt_excluded[str(c)] = audit
                else:
                    adapt_fallback[f"class_{c}"] = witness
                    adapt_audit[f"class_{c}"] = audit
                    class_scenes[str(c)].append(scene)
            if "scene" not in adapt_fallback and adapt_fallback:
                witness = next(iter(adapt_fallback.values()))
                adapt_fallback["scene"] = {**witness, "anchor": [-1, -1], "class_id": -1}
                adapt_audit["scene"] = {"reason": "reuse_valid_class_window_as_natural_witness"}
            if "scene" not in adapt_fallback:
                raise ValueError(f"no pre-audited natural adaptation window for {scene}")
            records[scene] = {"points": str(point_path), "points_sha256": _sha(point_path),
                              "class_point_counts": [len(filtered[c]) for c in range(24)],
                              "registration_sha256": reader.registration_sha256,
                              "statistics_native_valid_count": stats_count, "statistics_windows": windows,
                              "max_grid_supervised_fraction": maximum_fraction,
                              "fallbacks": adapt_fallback, "fallback_search": adapt_audit,
                              "excluded_target_classes": adapt_excluded}
        finally:
            reader.close()
            del integral
        print(json.dumps({"event": "light_sar_prepare_scene", "index": scene_index, "total": len(train),
                          "scene": scene, "has_sar": True, "base_class_witnesses": len(base_fallback)-1,
                          "adapt_class_witnesses": len(adapt_fallback)-1,
                          "base_excluded": sorted(base_excluded), "adapt_excluded": sorted(adapt_excluded),
                          "elapsed_seconds": time.monotonic()-begun}), flush=True)
    if count == 0:
        raise ValueError("no valid train96 SAR statistics")
    missing_classes = [c for c, ids in class_scenes.items() if not ids]
    missing_base = [c for c, ids in base_class_scenes.items() if not ids]
    if strict and (missing_classes or missing_base):
        raise ValueError(f"missing globally viable classes: base={missing_base}, adapt={missing_classes}")
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    if not np.isfinite(std).all() or (std <= 1e-6).any():
        raise ValueError("degenerate train96 SAR statistics")
    result = {**request, "scenes": records, "class_scenes": class_scenes,
              "base_scenes": base_records, "base_class_scenes": base_class_scenes,
              "sar_mean": mean.tolist(), "sar_std": std.tolist(), "statistics_native_valid_count": count,
              "statistics_estimator": "fixed_label_independent_train96_registered_native_grid_samples_population_moments_overlap_not_deduplicated",
              "raw_content_hashes_recomputed": False, "source_point_sha_and_labels_verified": True,
              "held_labels_read_for_preparation": False, "official30_rasters_read": False,
              "band_order": ["B", "G", "R", "NIR"], "normalization": metadata["normalization"]}
    result["payload_sha256"] = _digest(result)
    temp = out / "prepared_manifest.tmp"
    temp.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(destination)
    return destination


def load_prepared_manifest(manifest):
    value = dict(manifest) if isinstance(manifest, Mapping) else json.loads(Path(manifest).read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("payload_sha256") != _digest({k: v for k, v in value.items() if k != "payload_sha256"}):
        raise ValueError("prepared data schema/payload identity changed")
    raw = load_raw_manifest(value["raw_manifest"])
    if (_sha(value["raw_manifest"]) != value["raw_manifest_sha256"]
            or raw["request_sha256"] != value["source_request_sha256"]
            or value["train_ids"] != raw["selection_train_ids"] or value["held_ids"] != raw["held_ids"]
            or not set(value["adapt_scene_ids"]).issubset(value["train_ids"])
            or set(value["scenes"]) != set(value["adapt_scene_ids"])
            or set(value["base_scenes"]) != set(value["train_ids"])):
        raise ValueError("prepared data escaped authenticated train96/held24")
    present, path = _availability(value["data_root"])
    if (_sha(path) != value["availability_sha256"]
            or value["adapt_scene_ids"] != [s for s in value["train_ids"] if present[s]]):
        raise ValueError("SAR availability binding changed")
    for key in ("sar_mean", "sar_std"):
        if len(value[key]) != 2 or not np.isfinite(value[key]).all():
            raise ValueError("invalid fixed SAR moments")
    if min(value["sar_std"]) <= 0:
        raise ValueError("SAR standard deviations must be positive")
    minimum = math.ceil(value["crop_size"]**2 * value["min_supervised_fraction"])
    for pool_key, records_key, ids_key in (("base_class_scenes", "base_scenes", "train_ids"),
                                          ("class_scenes", "scenes", "adapt_scene_ids")):
        pool, records = value[pool_key], value[records_key]
        if set(pool) != {str(c) for c in range(24)}:
            raise ValueError("class pool must explicitly account for all24 classes")
        for c, scenes in pool.items():
            if len(scenes) != len(set(scenes)) or not set(scenes).issubset(value[ids_key]):
                raise ValueError("class support includes an invalid training scene")
            if value["strict_protocol"] and not scenes:
                raise ValueError("prepared class pool lost global class coverage")
            expected = [s for s in value[ids_key] if f"class_{c}" in records[s]["fallbacks"]]
            if scenes != expected:
                raise ValueError("class pool and viable fallback witnesses disagree")
        for scene, record in records.items():
            if "scene" not in record["fallbacks"]:
                raise ValueError("training scene lost a natural fallback witness")
            h, w = raw["scenes"][scene]["shape"]
            for key, witness in record["fallbacks"].items():
                top, left, c = witness["top"], witness["left"], witness["class_id"]
                cy, cx = witness["anchor"]
                n = value["crop_size"]
                if (type(top) is not int or type(left) is not int or min(top, left) < 0
                        or top+n > h or left+n > w or not minimum <= witness["supervised_count"] <= n*n
                        or (key == "scene" and (c != -1 or [cy, cx] != [-1, -1]))
                        or (key != "scene" and (key != f"class_{c}" or not 0 <= c < 24
                                                or not top <= cy < top+n or not left <= cx < left+n))):
                    raise ValueError("invalid pre-audited fallback geometry/support")
    return value


def make_epoch_plan(scene_ids, class_scenes, *, epoch, samples_per_epoch, seed=1337):
    """Every effective32 update contains four distinct scenes, eight crops each.

    Half of blocks follow a shuffled scene cycle, half a shuffled class cycle.
    This is 50% scene / 50% class sampling, never teacher/error mining.
    """
    scenes = tuple(scene_ids)
    if len(set(scenes)) != len(scenes) or len(scenes) < 4:
        raise ValueError("at least four unique training scenes required")
    if type(epoch) is not int or epoch < 0 or samples_per_epoch % 32 or samples_per_epoch < 32:
        raise ValueError("nonnegative epoch and whole effective32 updates required")
    classes = {int(c): tuple(ss) for c, ss in class_scenes.items() if ss}
    if not classes or any(not set(ss).issubset(scenes) for ss in classes.values()):
        raise ValueError("class sampler escaped its training scene support")
    rng = random.Random(seed + epoch * 10000019)
    scene_queue, class_queue, blocks = deque(), deque(), []
    for _ in range(samples_per_epoch // 32):
        chosen, current = set(), []
        for _ in range(2):
            if not class_queue:
                values = sorted(classes)
                rng.shuffle(values)
                class_queue.extend(values)
            selected = None
            for _ in range(len(class_queue)):
                c = class_queue.popleft()
                eligible = [s for s in classes[c] if s not in chosen]
                if eligible:
                    selected = (rng.choice(eligible), "class", c)
                    break
                class_queue.append(c)
            if selected is None:
                eligible_classes = {c: [s for s in ss if s not in chosen] for c, ss in classes.items()}
                eligible_classes = {c: ss for c, ss in eligible_classes.items() if ss}
                if not eligible_classes:
                    raise ValueError("cannot form distinct-scene class blocks")
                c = rng.choice(sorted(eligible_classes))
                selected = (rng.choice(eligible_classes[c]), "class", c)
            chosen.add(selected[0])
            current.append(selected)
        for _ in range(2):
            if not scene_queue:
                values = list(scenes)
                rng.shuffle(values)
                scene_queue.extend(values)
            for _ in range(len(scene_queue)):
                s = scene_queue.popleft()
                if s not in chosen:
                    break
                scene_queue.append(s)
            else:
                s = rng.choice([s for s in scenes if s not in chosen])
            current.append((s, "scene", -1))
            chosen.add(s)
        rng.shuffle(current)
        blocks.extend(current)
    if set(s for s, _, _ in blocks) != set(scenes):
        raise ValueError("epoch budget lost scene coverage")
    if set(c for _, kind, c in blocks if kind == "class") != set(classes):
        raise ValueError("epoch budget lost targeted class coverage")
    return tuple(blocks)


def _spectral(ms, rng, normalization):
    aug = SPECTRAL_AUGMENTATION
    if rng.random() >= aug["probability"]:
        return ms
    mean = np.asarray(normalization["mean"], np.float32)[:, None, None]
    std = np.asarray(normalization["std"], np.float32)[:, None, None]
    gain, offset = rng.uniform(*aug["global_gain_range"]), rng.uniform(*aug["global_offset_range"])
    bg = np.asarray([rng.uniform(*aug["band_gain_range"]) for _ in range(4)], np.float32)[:, None, None]
    bo = np.asarray([rng.uniform(*aug["band_offset_range"]) for _ in range(4)], np.float32)[:, None, None]
    return np.ascontiguousarray((np.clip((ms * std + mean) * gain * bg + offset + bo, 0, 1.5) - mean) / std)


class LightSARDataset(Dataset):
    def __init__(self, config, *, stage="base", epoch=0, start_sample=0, augmentation=True):
        if stage not in ("base", "adapt"):
            raise ValueError("stage must be base or adapt; P/R/Z share adapt")
        self.config, self.stage = dict(config), stage
        self.prepared = load_prepared_manifest(config["prepared_manifest"])
        self.metadata = load_raw_manifest(self.prepared["raw_manifest"])
        if config.get("raw_manifest") is not None and Path(config["raw_manifest"]).resolve() != Path(self.prepared["raw_manifest"]):
            raise ValueError("config/raw data manifest mismatch")
        if config.get("data_root") is not None and Path(config["data_root"]).resolve() != Path(self.prepared["data_root"]):
            raise ValueError("config/SAR data root mismatch")
        self.scene_ids = tuple(self.prepared["train_ids"] if stage == "base" else self.prepared["adapt_scene_ids"])
        self.crop_size, self.sar_size = self.prepared["crop_size"], self.prepared["sar_size"]
        self.samples_per_epoch = len(self.scene_ids) * 64
        self.normalization = self.metadata["normalization"]
        self.seed, self.augmentation = self.prepared["seed"], bool(augmentation)
        self.min_supervised_fraction = float(config.get("min_supervised_fraction", self.prepared["min_supervised_fraction"]))
        self.max_attempts = config.get("max_attempts", self.prepared["max_attempts"])
        if (not 0 < self.min_supervised_fraction <= 1 or type(self.max_attempts) is not int
                or self.max_attempts < 1):
            raise ValueError("positive supervision threshold and finite redraw bound required")
        if (self.min_supervised_fraction != self.prepared["min_supervised_fraction"]
                or self.max_attempts != self.prepared["max_attempts"]):
            raise ValueError("supervision/redraw rules differ from prepared binding")
        if stage == "base":
            self.class_scenes = {int(c): tuple(ss) for c, ss in self.prepared["base_class_scenes"].items()}
        else:
            self.class_scenes = {int(c): tuple(ss) for c, ss in self.prepared["class_scenes"].items()}
        self._cache, self._point_cache, self._aux = OrderedDict(), {}, OrderedDict()
        self.set_epoch(epoch, start_sample=start_sample)

    def set_epoch(self, epoch, start_sample=0):
        if type(start_sample) is not int or not 0 <= start_sample <= self.samples_per_epoch or start_sample % 32:
            raise ValueError("resume must start at an effective32 sample boundary")
        self.blocks = make_epoch_plan(self.scene_ids, self.class_scenes, epoch=epoch,
                                     samples_per_epoch=self.samples_per_epoch, seed=self.seed)
        self.epoch, self.start_sample = epoch, start_sample
        self.sampling_audit = {"stage": self.stage, "epoch": epoch, "seed": self.seed,
                               "samples_per_epoch": self.samples_per_epoch, "resume_start_sample": start_sample,
                               "effective_batch": 32, "distinct_scenes_per_update": 4, "scene_block_size": 8,
                               "sampling": "scene0.5_class0.5", "plan_sha256": _digest(self.blocks),
                               "min_supervised_fraction": self.min_supervised_fraction,
                               "max_attempts": self.max_attempts, "scene_ids": list(self.scene_ids)}
        self.sampling_audit["fallback_protocol"] = self.prepared["fallback_protocol"]

    def __len__(self):
        return self.samples_per_epoch - self.start_sample

    def _scene(self, scene):
        value = self._cache.pop(scene, None)
        if value is None:
            value = read_cached_scene(self.metadata, scene)
        self._cache[scene] = value
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return value

    def _reader(self, scene):
        reader = self._aux.pop(scene, None)
        if reader is None:
            reader = open_sar_scene(self.prepared["data_root"], scene)
            expected = self.prepared["scenes"][scene]["registration_sha256"]
            if reader.registration_sha256 != expected:
                reader.close()
                raise ValueError("training SAR registration identity changed")
        self._aux[scene] = reader
        while len(self._aux) > 2:
            _, old = self._aux.popitem(last=False)
            old.close()
        return reader

    def _points(self, scene, c):
        if scene not in self._point_cache:
            row = self.metadata["scenes"][scene] if self.stage == "base" else self.prepared["scenes"][scene]
            self._point_cache[scene] = _load_points(row["points"], row["points_sha256"])
        return self._point_cache[scene][c]

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        absolute = self.start_sample + int(index)
        scene, kind, c = self.blocks[absolute // BLOCK_SIZE]
        rng = random.Random(self.seed + self.epoch * 10000019 + absolute * 65537)
        image, labels = self._scene(scene)
        h, w = labels.shape
        n, sn = self.crop_size, self.sar_size
        if min(h, w) < n:
            raise ValueError("training scene smaller than crop")
        fallback_used = False
        for attempt in range(self.max_attempts + 1):
            anchor = None
            fallback_used = attempt == self.max_attempts
            if fallback_used:
                records = self.prepared["base_scenes"] if self.stage == "base" else self.prepared["scenes"]
                witness = records[scene]["fallbacks"]["scene" if kind == "scene" else f"class_{c}"]
                top, left = witness["top"], witness["left"]
                if kind == "class":
                    cy, cx = witness["anchor"]
                    anchor = cy, cx
            elif kind == "class":
                points = self._points(scene, c)
                if not len(points):
                    raise ValueError("empty target-class point reservoir")
                cy, cx = map(int, points[rng.randrange(len(points))])
                anchor = cy, cx
                top = int(np.clip(cy - n//2 + rng.randint(-n//4, n//4), 0, h-n))
                left = int(np.clip(cx - n//2 + rng.randint(-n//4, n//4), 0, w-n))
            else:
                top, left = rng.randrange(h-n+1), rng.randrange(w-n+1)
            target = np.array(labels[top:top+n, left:left+n], dtype=np.int64, copy=True)
            if np.any((target != 255) & ((target < 0) | (target >= 24))):
                raise ValueError("invalid mapped FBP24 label")
            if self.stage == "adapt":
                aux = self._reader(scene).read_patch(top, left, n, modality="sar", out_size=sn)
                sar, native, optical = _checked_aux(aux, n, sn)
                support = (target != 255) & optical[0]
            else:
                sar, native, optical = np.zeros((2, sn, sn), np.float32), np.zeros((1, sn, sn), bool), np.zeros((1, n, n), bool)
                support = target != 255
            enough = int(support.sum()) >= math.ceil(n * n * self.min_supervised_fraction)
            anchor_ok = anchor is None or (0 <= cy-top < n and 0 <= cx-left < n and support[cy-top, cx-left] and target[cy-top, cx-left] == c)
            if enough and anchor_ok:
                break
        else:
            raise RuntimeError(f"insufficient supervised crop after{self.max_attempts} random attempts and pre-audited fallback: scene={scene} epoch={self.epoch} index={absolute} kind={kind} class={c}")
        target[~support] = 255
        ms = normalize_ms(image[top:top+n, left:left+n], self.normalization)
        code = int(rng.random() < .5) + 2 * int(rng.random() < .5) if self.augmentation else 0
        sample = {"ms": ms, "sar_db": sar, "sar_valid": native, "optical_valid": optical,
                  "ms_valid": np.ones((n, n), bool), "labels": target}
        sample = {k: apply_flips(v, code) for k, v in sample.items()}
        if self.augmentation:
            sample["ms"] = _spectral(sample["ms"], rng, self.normalization)
        tensors = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sample.items()}
        tensors.update(scene_id=scene, top=top, left=left, index=absolute, epoch=self.epoch,
                       flip_code=code, sampling_kind=kind, sampling_class=c, redraw_count=attempt,
                       supervised_count=int(support.sum()), stage=self.stage, fallback_used=fallback_used)
        return tensors

    def close(self):
        for reader in getattr(self, "_aux", {}).values():
            reader.close()
        self._aux, self._cache = OrderedDict(), OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_aux=OrderedDict(), _cache=OrderedDict())
        return state

    def __del__(self):
        self.close()


def worker_init_fn(_worker_id):
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass


__all__ = ["LightSARDataset", "prepare_training_support", "load_prepared_manifest", "load_scene_arrays",
           "normalize_ms", "open_sar_scene", "native_valid_to_optical", "make_epoch_plan", "worker_init_fn"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stats-windows-per-axis", type=int, default=4)
    options = parser.parse_args()
    print(prepare_training_support(options.raw_manifest, options.data_root, options.output_dir,
                                   stats_windows_per_axis=options.stats_windows_per_axis))
