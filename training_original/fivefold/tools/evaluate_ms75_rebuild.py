"""Clean held24 selection for a rebuilt MS75 and two fixed train96 DINO experts.

Only rolling FP32 native-logit stripes are retained. Each expert is stitched
before softmax; equal-probability fusion is never performed on raw tile logits.
The primary score is the global 24-class mIoU of the three-expert ensemble.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_target_dino import (
    PATCH, STRIDE, HANN_FLOOR, _json_digest, _prepared_batches, _write_json,
    add_confusion, blend_window, metrics_from_confusion, sliding_starts,
)
from tcmmsline3.target_dino_data import NORMALIZATION

NUM_CLASSES = 24
TRAIN96 = "ms75_oof_seed1337_fold_0_train"
HELD24 = "ms75_oof_seed1337_fold_0_heldout"
RESULT_NAMES = ("ms75", "dino_pair_equal", "triple_equal")
PROBABILITY_CHUNK_ROWS = 64
DEFAULT_TEACHER_SPECS = (
    {"name": "n2_train96_e24", "variant": "v3l", "epoch": 24,
     "checkpoint": "/var/tmp/yeqi_dino_web_20260914_v1/v3l_gpu6/train96/checkpoints/ema_successful_updates_0011520.pt",
     "checkpoint_sha256": "7391d413d89573886bb0bd1a59593a793117c38f7ffc47ee8b10560698435b92",
     "config": "/var/tmp/yeqi_dino_web_20260914_v1/v3l_gpu6/train96/resolved_config.yaml",
     "config_sha256": "5d14277057079390a452746823bf651a10f86a195a46308913a4239eedf2a900",
     "config_fingerprint": "425524f18951337a2ed03faf4700cfd5a0a1e9e7d81d6756bb41047744a296cc"},
    {"name": "n3_train96_e18", "variant": "v2l", "epoch": 18,
     "checkpoint": "/var/tmp/yeqi_dino_web_20260914_v1/v2l_gpu7/train96/checkpoints/ema_successful_updates_0008640.pt",
     "checkpoint_sha256": "7b484467a8deebfd52bc8ae87856182240756bc64834ee78a5b42de7573a4e59",
     "config": "/var/tmp/yeqi_dino_web_20260914_v1/v2l_gpu7/train96/resolved_config.yaml",
     "config_sha256": "1f867b0ab5734b895bf4d492a5ea34047195d0823b41f2f15e0e490c608a229a",
     "config_fingerprint": "c7f2c9cbd8025ebaa2a1dacef26c3b0b6edf118800e66707da000717bc24f7e0"},
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _normalization(value):
    norm = dict(value.get("ms", value))
    norm.setdefault("divisor", 255.0)
    _require(float(norm["divisor"]) == 255.0
             and tuple(norm["mean"]) == tuple(NORMALIZATION["mean"])
             and tuple(norm["std"]) == tuple(NORMALIZATION["std"]),
             "All three experts require their original common BGRN normalization")
    return norm


def validate_held_request(metadata, ids):
    """Fail closed before reading any raster or constructing a GPU model."""
    groups = [list(metadata[k]) for k in ("selection_train_ids", "held_ids", "train_ids", "test_ids")]
    train96, held24, train120, official30 = groups
    _require(str(metadata["dataset"]).lower() == "fbp" and metadata["num_classes"] == 24,
             "Only FBP native24 is supported")
    _require(metadata["band_order"] == ["B", "G", "R", "NIR"], "Expected raw BGRN")
    _require(all(len(g) == size and len(set(g)) == size
                 for g, size in zip(groups, (96, 24, 120, 30))), "Wrong/duplicate split membership")
    _require(not set(train96) & set(held24) and set(train96) | set(held24) == set(train120)
             and not set(train120) & set(official30), "Train/held/test leakage")
    _require(list(ids) == held24, "Evaluate exactly the ordered complete held24, never train/test")
    return _normalization(metadata["normalization"])


def validate_teacher_payload(payload, config, spec, metadata=None):
    """Authenticate EMA role, exact milestone and clean data/source provenance."""
    variant, epoch = str(spec["variant"]), int(spec["epoch"])
    _require((variant, epoch) in (("v3l", 24), ("v2l", 18)), "Only the fixed clean96 DINO pair is allowed")
    _require(config["model"].get("variant") == variant
             and config["model"].get("architecture") == "dino_web_nir_dpt"
             and config["model"].get("num_classes") == 24,
             "Teacher architecture/variant/class taxonomy mismatch")
    _require(config["data"].get("protocol_stage") == "dino_web_train96"
             and config["data"].get("train_split") == TRAIN96
             and config["data"].get("val_split") == HELD24
             and config["data"].get("load_sar") is False,
             "Teacher must be pure-MS train96, never a full120 checkpoint")
    _normalization(config["data"]["normalization"])
    _require(_json_digest(config) == spec["config_fingerprint"] == payload.get("config_fingerprint"),
             "Teacher resolved-config fingerprint differs from training checkpoint")
    for key, expected in {"checkpoint_schema_version": 1,
                          "checkpoint_role": "ema_successful_optimizer_update_milestone",
                          "global_optimizer_step": epoch * 480,
                          "target_successful_optimizer_updates": epoch * 480,
                          "ema_n_averaged": epoch * 480, "budget_complete": False}.items():
        _require(payload.get(key) == expected, f"Teacher milestone/EMA mismatch: {key}")
    provenance = payload.get("data_provenance", {})
    binding = provenance.get("dino_web_runtime_binding", {})
    identity = binding.get("data_identity", {})
    _require(provenance.get("dataset") == "FBP" and provenance.get("source_split") == "official_train"
             and binding.get("stage") == "train96" and binding.get("variant") == variant
             and binding.get("config_fingerprint") == spec["config_fingerprint"]
             and identity.get("train_scene_count") == 96 and identity.get("heldout_scene_count") == 24
             and identity.get("official30_rasters_or_labels_read") is False
             and identity.get("full120_training_includes_former_held24") is False,
             "Teacher train-only source provenance mismatch")
    train, held = provenance.get("train_scene_ids", []), provenance.get("heldout_scene_ids", [])
    _require(len(train) == len(set(train)) == 96 and len(held) == len(set(held)) == 24
             and not set(train) & set(held), "Teacher train96/held24 are not disjoint")
    if metadata is not None:
        _require(train == list(metadata["selection_train_ids"]) and held == list(metadata["held_ids"]),
                 "Teacher fold is not this rebuilt-MS75 fold")
        _require(not (set(train) | set(held)) & set(metadata["test_ids"]), "Teacher saw official30")
    state = payload.get("model")
    _require(isinstance(state, Mapping) and state, "Missing teacher EMA state")
    return binding


def load_teachers(teacher_specs=None, *, device, metadata=None):
    """Strictly load retained EMA states; never downloads or reloads pretraining."""
    from tcmmsline3.dino_web_model import build_dino_web_model
    specs = list(DEFAULT_TEACHER_SPECS if teacher_specs is None else teacher_specs)
    _require([(s["variant"], int(s["epoch"])) for s in specs] == [("v3l", 24), ("v2l", 18)],
             "Expected ordered N2-e24 and N3-e18 clean train96 teachers")
    models, receipts = [], []
    try:
        for spec, pinned in zip(specs, DEFAULT_TEACHER_SPECS):
            # Paths may be relocated; identities and exact configurations may not.
            for key in ("checkpoint_sha256", "config_sha256", "config_fingerprint"):
                _require(spec.get(key) == pinned[key], f"Fixed teacher identity changed: {key}")
            cp, cfg_path = Path(spec["checkpoint"]).resolve(strict=True), Path(spec["config"]).resolve(strict=True)
            _require(_sha256(cp) == spec["checkpoint_sha256"], "Teacher checkpoint bytes changed")
            _require(_sha256(cfg_path) == spec["config_sha256"], "Teacher resolved config bytes changed")
            config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            payload = torch.load(cp, map_location="cpu", mmap=True, weights_only=True)
            binding = validate_teacher_payload(payload, config, spec, metadata)
            manifest_path = Path(spec.get("source_manifest", cfg_path.parent / "source_manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            body = {k: v for k, v in manifest.items() if k != "source_manifest_sha256"}
            source_sha = _json_digest(body)
            _require(source_sha == manifest.get("source_manifest_sha256") == binding["source_manifest_sha256"],
                     "Teacher original source manifest differs from checkpoint binding")
            sources = {row["path"]: row["sha256"] for row in manifest["files"]}
            runtime_sources = {}
            for relative in ("tcmmsline3/dino_web_model.py", "tcmmsline3/dinov2_nir_model.py"):
                observed = _sha256(ROOT / relative)
                _require(sources.get(relative) == observed, f"Teacher model implementation changed: {relative}")
                runtime_sources[relative] = observed
            # Constructing then overwriting a frozen model must not consume the
            # training run's CPU RNG stream at every held evaluation.
            with torch.random.fork_rng(devices=[]):
                model = build_dino_web_model(config, load_pretrained=False)
            model.load_state_dict(payload["model"], strict=True)
            for name, value in model.state_dict().items():
                _require(not value.is_floating_point() or bool(torch.isfinite(value).all()),
                         f"Nonfinite teacher weight: {name}")
            del payload
            model.requires_grad_(False).eval().to(device)
            models.append(model)
            receipts.append({**dict(spec), "source_manifest_sha256": source_sha,
                             "runtime_model_source_sha256": runtime_sources,
                             "split": "clean_train96_held24", "strict_state_load": True,
                             "normalization": _normalization(config["data"]["normalization"])})
    except BaseException:
        models.clear()
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        raise
    return models, receipts


class MultiViewSceneModel(torch.nn.Module):
    """Supply exact unaugmented MS75 context views in the tile scorer's order."""
    num_classes = NUM_CLASSES

    def __init__(self, model, image, *, global_raw, normalization, device, workers=4,
                 patch=PATCH, stride=STRIDE):
        super().__init__()
        self.model, self.image, self.global_raw = model, image, global_raw
        self.normalization, self.device, self.patch = normalization, torch.device(device), patch
        self.positions = [(top, left) for top in sliding_starts(image.shape[0], patch=patch, stride=stride)
                          for left in sliding_starts(image.shape[1], patch=patch, stride=stride)]
        self.cursor = 0
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers else None

    def _one(self, position):
        from tcmmsline3.ms75_rebuild_data import read_views
        return read_views(self.image, top=position[0], left=position[1], global_raw=self.global_raw,
                          normalization=self.normalization, local_size=self.patch)

    def forward(self, ms):
        from tcmmsline3.ms75_rebuild_data import MODEL_INPUT_KEYS
        positions = self.positions[self.cursor:self.cursor + len(ms)]
        _require(len(positions) == len(ms), "MS75 tile cursor exceeded scene geometry")
        samples = list(self.pool.map(self._one, positions)) if self.pool else [self._one(p) for p in positions]
        inputs = {"ms": ms}
        for key in MODEL_INPUT_KEYS:
            if key != "ms":
                value = torch.from_numpy(np.stack([row[key] for row in samples]))
                if self.device.type == "cuda":
                    value = value.pin_memory()
                inputs[key] = value.to(self.device, non_blocking=self.device.type == "cuda")
        self.cursor += len(ms)
        return self.model(**inputs)

    def close(self, *, require_complete=False):
        if self.pool:
            self.pool.shutdown(wait=True)
        if require_complete:
            _require(self.cursor == len(self.positions), "Incomplete MS75 context-view plan")


def predictions_from_stitched(logits: Sequence[torch.Tensor]):
    """Softmax each expert's already-stitched logits, then equal-probability fusion."""
    _require(len(logits) == 3 and all(x.shape == logits[0].shape for x in logits),
             "Three aligned CxHxW stitched expert logits required")
    _require(logits[0].ndim == 3 and logits[0].shape[0] == NUM_CLASSES,
             "Expected native 24-class stitched logits")
    probabilities = [x.float().softmax(0) for x in logits]
    for value in probabilities:
        _require(bool(torch.isfinite(value).all()), "Nonfinite stitched expert probabilities")
    pair_sum = probabilities[1] + probabilities[2]
    return {"ms75": probabilities[0].argmax(0), "dino_pair_equal": pair_sum.argmax(0),
            "triple_equal": (pair_sum + probabilities[0]).argmax(0)}


@torch.inference_mode()
def evaluate_scene(models, image, label, *, normalization, device, batch_size=4, workers=4,
                   progress_callback=None, patch=PATCH, stride=STRIDE):
    """Bounded three-expert rolling scorer; small geometry is a CPU-test seam."""
    _require(len(models) == 3, "Exactly one MS75 and two DINO models required")
    _require(type(batch_size) is int and batch_size > 0 and type(workers) is int and 0 <= workers <= 16,
             "Positive batch and 0..16 workers required")
    _require(image.dtype == np.uint8 and label.dtype == np.uint8
             and image.shape == (*label.shape, 4), "Expected aligned uint8 BGRN and native24 label")
    device = torch.device(device)
    height, width = label.shape
    ys, xs = sliding_starts(height, patch=patch, stride=stride), sliding_starts(width, patch=patch, stride=stride)
    weight = torch.from_numpy(blend_window(patch)).to(device)
    buffers = [torch.zeros((NUM_CLASSES, patch, width), dtype=torch.float32, device=device) for _ in models]
    weights = torch.zeros((patch, width), dtype=torch.float32, device=device)
    confusions = {key: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64) for key in RESULT_NAMES}
    changes = {"pair_wrong_triple_right": 0, "pair_right_triple_wrong": 0,
               "ms75_right_pair_wrong_triple_wrong": 0, "valid_pixels": 0}
    windows = 0
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ms75-held") if workers else nullcontext(None)
    with pool as executor:
        for yi, top in enumerate(ys):
            for lefts, arrays in _prepared_batches(image, top=top, lefts=xs, patch=patch,
                                                   batch_size=batch_size, normalization=normalization,
                                                   executor=executor):
                ms = torch.from_numpy(arrays)
                if device.type == "cuda":
                    ms = ms.pin_memory()
                ms = ms.to(device, non_blocking=device.type == "cuda")
                for model, accumulated in zip(models, buffers):
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=device.type == "cuda", cache_enabled=False):
                        native = model(ms)["logits"]
                    _require(tuple(native.shape) == (len(lefts), NUM_CLASSES, patch, patch),
                             "Expert native output geometry/class count changed")
                    native = native.float()
                    _require(bool(torch.isfinite(native).all()), "Nonfinite expert native logits")
                    for i, left in enumerate(lefts):
                        accumulated[:, :, left:left + patch].add_(native[i] * weight[None])
                    del native
                for left in lefts:
                    weights[:, left:left + patch].add_(weight)
                windows += len(lefts)
            rows = ys[yi + 1] - top if yi + 1 < len(ys) else patch
            _require(bool(torch.isfinite(weights[:rows]).all()) and float(weights[:rows].min()) > 0,
                     "Uncovered/nonfinite overlap weights")
            # Softmax is pixelwise, so chunking fully covered rows is exactly
            # equivalent but avoids 3 additional full-width 512-row probabilities.
            for begin in range(0, rows, PROBABILITY_CHUNK_ROWS):
                end = min(begin + PROBABILITY_CHUNK_ROWS, rows)
                stitched = [value[:, begin:end] / weights[None, begin:end] for value in buffers]
                predictions = {k: v.to(torch.uint8).cpu().numpy()
                               for k, v in predictions_from_stitched(stitched).items()}
                del stitched
                target = label[top + begin:top + end]
                for key, pred in predictions.items():
                    add_confusion(confusions[key], pred, target)
                valid = target != 255
                a, b, c = (predictions[key] == target for key in RESULT_NAMES)
                changes["pair_wrong_triple_right"] += int((valid & ~b & c).sum())
                changes["pair_right_triple_wrong"] += int((valid & b & ~c).sum())
                changes["ms75_right_pair_wrong_triple_wrong"] += int((valid & a & ~b & ~c).sum())
                changes["valid_pixels"] += int(valid.sum())
            remaining = patch - rows
            if remaining:
                for value in buffers:
                    value[:, :remaining].copy_(value[:, rows:].clone())
                weights[:remaining].copy_(weights[rows:].clone())
            for value in buffers:
                value[:, remaining:].zero_()
            weights[remaining:].zero_()
            if progress_callback:
                progress_callback(rows_completed=top + rows, rows_total=height, windows_completed=windows,
                                  windows_total=len(ys) * len(xs))
    _require(changes["valid_pixels"] > 0, "Held scene has no valid label pixels")
    return confusions, windows, changes


def evaluate_scene_set(model, data_manifest, ids, *, normalization=None, batch_size=4, workers=4,
                       output_dir=None, device=None, provenance: Mapping | None = None,
                       views_cache=None, teacher_specs=None):
    from tcmmsline3.ms75_rebuild_data import load_manifest, read_cached_scene, GlobalViews, VIEW_PROTOCOL
    metadata, ids = load_manifest(data_manifest), list(ids)
    norm = validate_held_request(metadata, ids)
    if normalization is not None:
        _normalization(normalization)
    _require(getattr(model, "num_classes", NUM_CLASSES) == NUM_CLASSES, "MS75 native classifier must have 24 classes")
    device = torch.device(device if device is not None else next(model.parameters()).device)
    if device.type == "cuda":
        _require(torch.cuda.is_bf16_supported(), "BF16-capable CUDA device required")
    out = Path(output_dir) if output_dir is not None else None
    modes = [(module, module.training) for module in model.modules()]
    teachers, teacher_receipts = load_teachers(teacher_specs, device=device, metadata=metadata)
    identity = {"schema": "ms75_rebuild_held24_three_expert_v1", "dataset": "fbp", "scene_ids": ids,
                "num_classes": NUM_CLASSES, "class_names": metadata.get("class_names"),
                "ignore_index": 255, "normalization": norm,
                "manifest_sha256": _json_digest(metadata), "protocol_stage": "held_selection_only",
                "patch_size": PATCH, "stride": STRIDE, "hann_floor": HANN_FLOOR,
                "view_protocol": VIEW_PROTOCOL, "global_support": "entire_original_scene",
                "blend_value": "per_expert_native_logits_then_softmax_then_equal_probability",
                "accumulator_dtype": "float32", "batch_size": batch_size, "tta": ["none"],
                "teacher_provenance": teacher_receipts, "checkpoint_provenance": dict(provenance or {}),
                "primary_score": "triple_equal_global_24_miou", "official_test_opened": False,
                "evaluator_sha256": _sha256(__file__), "probability_files_persisted": False}
    identity["probability_chunk_rows"] = PROBABILITY_CHUNK_ROWS
    global_views = GlobalViews(metadata, views_cache)
    confusions = {key: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64) for key in RESULT_NAMES}
    changes, scenes, start = {}, [], time.monotonic()
    model.eval()
    try:
        for index, scene in enumerate(ids):
            image, label = read_cached_scene(metadata, scene)
            scene_start = time.monotonic()
            wrapper = MultiViewSceneModel(model, image, global_raw=global_views.get(scene, image),
                                          normalization=norm, device=device, workers=workers)
            try:
                def report(**fields):
                    print(json.dumps({"event": "ms75_rebuild_held_scene_progress", "scene_id": scene,
                                      "scene_index": index + 1, "scene_count": len(ids), **fields}), flush=True)
                cms, windows, counts = evaluate_scene([wrapper, *teachers], image, label,
                    normalization=norm, device=device, batch_size=batch_size, workers=workers,
                    progress_callback=report)
                wrapper.close(require_complete=True)
            finally:
                wrapper.close()
            for key, cm in cms.items():
                confusions[key] += cm
            for key, value in counts.items():
                changes[key] = changes.get(key, 0) + value
            row = {"scene_id": scene, "windows": windows, "elapsed_seconds": time.monotonic() - scene_start,
                   "results": {key: metrics_from_confusion(cm) for key, cm in cms.items()}, "change_counts": counts}
            scenes.append(row)
            if out:
                _write_json(out / "scenes" / f"{scene}.json", {"evaluation_identity": identity, **row})
    finally:
        for module, mode in modes:
            module.training = mode
        teachers.clear()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results = {key: metrics_from_confusion(cm) for key, cm in confusions.items()}
    result = {**identity, **results["triple_equal"], "results": results, "change_counts": changes,
              "scenes": scenes, "completed_scenes": len(scenes), "elapsed_seconds": time.monotonic() - start,
              "metrics_aggregation": "sum_all_scene_confusion_then_compute_iou"}
    if out:
        _write_json(out / "metrics.json", result)
    return result


@torch.inference_mode()
def smoke_teacher_forward(model, sample, teacher_specs=None, *, device="cuda", metadata=None,
                          stripe_width=7300):
    """Real train-crop smoke while caller's optimizer/EMA remain resident.

    sample is an already normalized/augmented BGRN training batch with all MS75
    views. No held or test image/label is opened and labels are not consulted.
    CUDA allocation failures propagate: batch/protocol are never silently changed.
    """
    from tcmmsline3.ms75_rebuild_data import MODEL_INPUT_KEYS
    device = torch.device(device)
    teachers, receipts = load_teachers(teacher_specs, device=device, metadata=metadata)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    probabilities = []
    buffers = []
    try:
        _require(type(stripe_width) is int and stripe_width >= PATCH, "Smoke stripe must cover at least one tile")
        # Include real evaluator stripe residency while exercising forward; the
        # caller deliberately keeps training optimizer and EMA alive as well.
        buffers = [torch.zeros((NUM_CLASSES, PATCH, stripe_width), device=device, dtype=torch.float32)
                   for _ in range(3)]
        stripe_weights = torch.ones((PATCH, stripe_width), device=device, dtype=torch.float32)
        inputs = {key: sample[key].to(device) for key in MODEL_INPUT_KEYS}
        _require(inputs["ms"].ndim == 4 and tuple(inputs["ms"].shape[1:]) == (4, 512, 512),
                 "Smoke requires a batched training BGRN512 crop")
        for index, expert in enumerate([model, *teachers]):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda", cache_enabled=False):
                logits = expert(**inputs)["logits"] if index == 0 else expert(inputs["ms"])["logits"]
            _require(tuple(logits.shape) == (len(inputs["ms"]), 24, 512, 512), "Smoke native output mismatch")
            value = logits.float().softmax(1)
            _require(bool(torch.isfinite(value).all())
                     and bool(torch.allclose(value.sum(1), torch.ones_like(value[:, 0]), atol=2e-6, rtol=0)),
                     "Smoke expert probabilities are not finite normalized distributions")
            probabilities.append(value)
        fused = sum(probabilities) / 3
        _require(bool(torch.allclose(fused.sum(1), torch.ones_like(fused[:, 0]), atol=2e-6, rtol=0)),
                 "Smoke fused distribution is not normalized")
        result = {"teacher_provenance": receipts, "batch_size": len(inputs["ms"]),
                  "expert_probability_shapes": [list(p.shape) for p in probabilities],
                  "equal_fusion_shape": list(fused.shape), "labels_read": False,
                  "probability_sum_max_abs_error": float((fused.sum(1) - 1).abs().max()),
                  "resident_fp32_stripe_shape": [3, NUM_CLASSES, PATCH, stripe_width]}
        # Exercise the same bounded probability-finalization allocation too.
        stitched = [x[:, :PROBABILITY_CHUNK_ROWS] / stripe_weights[None, :PROBABILITY_CHUNK_ROWS]
                    for x in buffers]
        predictions_from_stitched(stitched)
        del stitched
        if device.type == "cuda":
            result.update(cuda_allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                          cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                          cuda_reserved_gib=torch.cuda.memory_reserved(device) / 2**30)
        return result
    finally:
        for module, mode in modes:
            module.training = mode
        teachers.clear()
        probabilities.clear()
        buffers.clear()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
