"""Full held-fold target-task evaluation, called by the isolated trainer.

This is deliberately NOT an official-test evaluator. It only accepts the
complete held fold from an authenticated target cache manifest. Native logits
are Hann blended in FP32, matching the historical DINO held-scene recipe;
per-tile softmax, TTA, resizing and per-scene mIoU averaging are not used.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcmmsline3.target_dino_data import load_manifest, normalize_bgrn, read_cached_scene
from tcmmsline3.target_dino_model import MS_MEAN, MS_STD

PATCH = 512
STRIDE = 256
HANN_FLOOR = .05


def sliding_starts(length: int, *, patch: int = PATCH, stride: int = STRIDE) -> list[int]:
    if length < patch or not 0 < stride <= patch:
        raise ValueError("whole-scene tiles must be in bounds with stride <= crop size")
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def blend_window(patch: int = PATCH) -> np.ndarray:
    if patch < 2:
        raise ValueError("Hann patch size must be at least two")
    one = np.maximum(np.hanning(patch).astype(np.float32), np.float32(HANN_FLOOR))
    return np.outer(one, one).astype(np.float32)


def add_confusion(confusion: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    classes = confusion.shape[0]
    if confusion.shape != (classes, classes) or confusion.dtype != np.int64:
        raise ValueError("confusion must be an int64 CxC matrix")
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have matching HW geometry")
    valid = target != 255
    truth = target[valid].astype(np.int64, copy=False)
    pred = prediction[valid].astype(np.int64, copy=False)
    if np.any((truth < 0) | (truth >= classes)) or np.any((pred < 0) | (pred >= classes)):
        raise ValueError("native target class code outside 0..C-1 (255 alone is ignored)")
    confusion += np.bincount(truth * classes + pred, minlength=classes * classes).reshape(classes, classes)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(confusion)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2
            or not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0)):
        raise ValueError("nonnegative integer CxC confusion required")
    matrix = matrix.astype(np.int64, copy=False)
    target, predicted = matrix.sum(1), matrix.sum(0)
    tp = matrix.diagonal()
    union = target + predicted - tp
    if int(target.sum()) == 0:
        raise ValueError("no supervised pixels in evaluation")
    iou = np.divide(tp, union, out=np.zeros(len(tp), np.float64), where=union > 0)
    precision = np.divide(tp, predicted, out=np.zeros(len(tp), np.float64), where=predicted > 0)
    recall = np.divide(tp, target, out=np.zeros(len(tp), np.float64), where=target > 0)
    return {"miou": float(iou.mean()), "oa": float(tp.sum() / target.sum()),
            "class_iou": iou.tolist(), "per_class_iou": iou.tolist(),
            "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
            "confusion": matrix.tolist(), "confusion_matrix": matrix.tolist(),
            "target_pixels": target.tolist(), "predicted_pixels": predicted.tolist(),
            "valid_pixels": int(target.sum()), "target_present_classes": (target > 0).tolist(),
            "absent_target_classes": np.flatnonzero(target == 0).tolist(),
            "zero_union_classes": np.flatnonzero(union == 0).tolist(),
            "miou_target_present_only": float(iou[target > 0].mean()),
            "primary_miou_policy": "fixed_all_native_classes_zero_for_zero_union"}


def _prepared_batches(image, *, top, lefts, patch, batch_size, normalization, executor):
    """Bounded two-batch FIFO prefetch; overlap accumulation order is fixed."""
    def crop(left):
        raw = image[top:top + patch, left:left + patch]
        if raw.dtype != np.uint8 or raw.shape != (patch, patch, 4):
            raise ValueError("expected in-bounds HWC uint8 B/G/R/NIR crop")
        return normalize_bgrn(raw, normalization)

    chunks = iter([lefts[i:i + batch_size] for i in range(0, len(lefts), batch_size)])
    if executor is None:
        for chunk in chunks:
            yield chunk, np.stack([crop(left) for left in chunk])
        return
    pending = deque()
    for _ in range(2):
        chunk = next(chunks, None)
        if chunk is not None:
            pending.append((chunk, [executor.submit(crop, left) for left in chunk]))
    while pending:
        chunk, futures = pending.popleft()
        yield chunk, np.stack([future.result() for future in futures])
        upcoming = next(chunks, None)
        if upcoming is not None:
            pending.append((upcoming, [executor.submit(crop, left) for left in upcoming]))


@torch.inference_mode()
def evaluate_scene(model, image, label, *, num_classes, normalization, device,
                   batch_size=4, workers=4, progress_callback=None,
                   patch=PATCH, stride=STRIDE) -> tuple[np.ndarray, int]:
    """Rolling GPU stripe equals whole-image FP32 Hann native-logit blending.

    Smaller patch/stride are exposed only as a CPU unit-test seam. Production
    callers use evaluate_scene_set, which pins both to 512/256.
    """
    device = torch.device(device)
    if type(batch_size) is not int or batch_size < 1 or type(workers) is not int or not 0 <= workers <= 16:
        raise ValueError("positive batch and 0..16 prefetch workers required")
    if num_classes not in (5, 7):
        raise ValueError("only native five-/seven-class evaluation is supported")
    if image.dtype != np.uint8 or image.shape != (*label.shape, 4) or label.dtype != np.uint8:
        raise ValueError("expected aligned raw HWC uint8 image and HW uint8 native labels")
    height, width = label.shape
    ys, xs = sliding_starts(height, patch=patch, stride=stride), sliding_starts(width, patch=patch, stride=stride)
    weight = torch.from_numpy(blend_window(patch)).to(device)
    accumulated = torch.zeros((num_classes, patch, width), dtype=torch.float32, device=device)
    weights = torch.zeros((patch, width), dtype=torch.float32, device=device)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    windows = 0
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="target-eval") if workers else nullcontext(None)
    with pool as executor:
        for yi, top in enumerate(ys):
            for lefts, arrays in _prepared_batches(image, top=top, lefts=xs, patch=patch,
                                                   batch_size=batch_size, normalization=normalization,
                                                   executor=executor):
                ms = torch.from_numpy(arrays)
                if device.type == "cuda":
                    ms = ms.pin_memory()
                ms = ms.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda", cache_enabled=False):
                    logits = model(ms)["logits"]
                if tuple(logits.shape) != (len(lefts), num_classes, patch, patch):
                    raise ValueError("native output geometry/classifier changed")
                logits = logits.float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("nonfinite native logits")
                for i, left in enumerate(lefts):
                    accumulated[:, :, left:left + patch].add_(logits[i] * weight[None])
                    weights[:, left:left + patch].add_(weight)
                    windows += 1
            rows = ys[yi + 1] - top if yi + 1 < len(ys) else patch
            if not bool(torch.isfinite(weights[:rows]).all()) or float(weights[:rows].min()) <= 0:
                raise ValueError("uncovered/nonfinite Hann overlap")
            native = accumulated[:, :rows] / weights[None, :rows]
            if not bool(torch.isfinite(native).all()):
                raise ValueError("nonfinite fused logits")
            prediction = native.argmax(0).to(torch.uint8).cpu().numpy()
            add_confusion(confusion, prediction, label[top:top + rows])
            remaining = patch - rows
            if remaining:
                accumulated[:, :remaining].copy_(accumulated[:, rows:].clone())
                weights[:remaining].copy_(weights[rows:].clone())
            accumulated[:, remaining:].zero_()
            weights[remaining:].zero_()
            if progress_callback is not None:
                progress_callback(rows_completed=top + rows, rows_total=height,
                                  windows_completed=windows, windows_total=len(ys) * len(xs))
    if int(confusion.sum()) == 0:
        raise ValueError("held scene has no supervised pixels")
    return confusion, windows


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def evaluate_scene_set(model, data_manifest, ids: Sequence[str], *, normalization=None,
                       batch_size=4, workers=4, output_dir=None, device=None,
                       provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate the complete internal held fold; official test is refused."""
    metadata = load_manifest(data_manifest)
    ids = list(ids)
    if (not ids or len(set(ids)) != len(ids) or set(ids) != set(metadata["held_ids"])
            or set(ids) & set(metadata["test_ids"]) or set(ids) & set(metadata["selection_train_ids"])):
        raise ValueError("evaluation requires exactly the complete held fold, never official test/train")
    if metadata["band_order"] != ["B", "G", "R", "NIR"]:
        raise ValueError("target evaluation requires raw cached B/G/R/NIR order")
    norm = dict(normalization or metadata["normalization"])
    if "ms" in norm:
        norm = dict(norm["ms"])
    norm.setdefault("divisor", 255.0)
    if (tuple(norm["mean"]) != MS_MEAN or tuple(norm["std"]) != MS_STD
            or float(norm["divisor"]) != 255.0):
        raise ValueError("evaluation normalization must match the fixed training recipe")
    classes = int(metadata["num_classes"])
    if getattr(model, "num_classes", classes) != classes:
        raise ValueError("model native class count differs from target manifest")
    device = torch.device(device if device is not None else next(model.parameters()).device)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("target CUDA evaluation requires native BF16 support")
    out = Path(output_dir) if output_dir is not None else None
    identity = {"schema": "target_dino_held_fullscene_v1", "dataset": metadata["dataset"],
                "protocol": metadata["protocol"], "protocol_stage": "held_selection_only",
                "request_sha256": metadata["request_sha256"], "manifest_sha256": _json_digest(metadata),
                "scene_ids": ids, "num_classes": classes, "ignore_index": 255,
                "normalization": norm, "patch_size": PATCH, "stride": STRIDE,
                "hann_floor": HANN_FLOOR, "blend_value": "native_logits", "accumulator_dtype": "float32",
                "amp_dtype": "bfloat16" if device.type == "cuda" else "float32_cpu_test",
                "batch_size": batch_size, "tta": ["none"], "official_test_opened": False,
                "checkpoint_provenance": dict(provenance or {}),
                "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    start = time.monotonic()
    confusion = np.zeros((classes, classes), dtype=np.int64)
    scenes = []
    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for index, scene in enumerate(ids):
            image, label = read_cached_scene(metadata, scene)
            scene_start = time.monotonic()

            def report(**fields):
                print(json.dumps({"event": "target_held_scene_progress", "scene_id": scene,
                                  "scene_index": index + 1, "scene_count": len(ids), **fields}), flush=True)

            cm, windows = evaluate_scene(model, image, label, num_classes=classes, normalization=norm,
                                         device=device, batch_size=batch_size, workers=workers,
                                         progress_callback=report, patch=PATCH, stride=STRIDE)
            confusion += cm
            result = {"scene_id": scene, "windows": windows, "elapsed_seconds": time.monotonic() - scene_start,
                      **metrics_from_confusion(cm)}
            scenes.append(result)
            if out is not None:
                _write_json(out / "scenes" / f"{scene}.json", {"evaluation_identity": identity, **result})
    finally:
        for module, training in training_states:
            module.training = training
    result = {**identity, **metrics_from_confusion(confusion), "scenes": scenes,
              "completed_scenes": len(scenes), "elapsed_seconds": time.monotonic() - start,
              "metrics_aggregation": "sum_all_scene_confusion_then_compute_iou"}
    if out is not None:
        _write_json(out / "metrics.json", result)
    return result


__all__ = ["evaluate_scene_set", "evaluate_scene", "metrics_from_confusion", "add_confusion",
           "sliding_starts", "blend_window"]
