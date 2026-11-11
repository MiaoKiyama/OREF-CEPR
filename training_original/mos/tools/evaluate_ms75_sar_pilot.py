"""Fixed-endpoint all-held24 evaluation of the isolated P/R/Z SAR pilot.

Six rolling FP32 logit stripes, shared optical/SAR window reads and two fixed
DINO forwards produce nine confusion matrices without storing probabilities.
Every contributing window must support a pixel before ANY adapted expert is
used there. Unsupported probabilities are selected from the immutable base.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_ms75_rebuild import (
    DEFAULT_TEACHER_SPECS, MultiViewSceneModel, NUM_CLASSES, PATCH, STRIDE,
    HANN_FLOOR, _require, _sha256, _write_json, add_confusion, blend_window,
    load_teachers, metrics_from_confusion, sliding_starts, validate_held_request,
)
from tools.evaluate_target_dino import _prepared_batches

SCHEMA = "ms75_sar_pilot_allheld24_fixed_endpoint_v1"
ARMS = ("base", "P", "R", "Z")
RESULT_NAMES = (*ARMS, *(arm + "_triple" for arm in ARMS), "dual")
PROBABILITY_CHUNK_ROWS = 32
BASE_SHA256 = "f141aee97b45e56942529ad38c45fa3997d2dc89c5e42eb1359943e4bce46024"


def collect_aux_bindings(data_root, scene_ids):
    """Pin small geometry assets and stat large SAR rasters, without pixel reads."""
    from tcmmsline3.light_sar_data import _availability, open_sar_scene
    availability, path = _availability(data_root)
    files, scenes = {str(path): _sha256(path)}, {}
    for scene in scene_ids:
        _require(scene in availability, "Missing held scene in SAR availability")
        reader = open_sar_scene(data_root, scene)
        try:
            row = {"available": reader.available["sar"],
                   "registration_sha256": reader.registration_sha256, "assets": []}
            _require(row["available"] == availability[scene], "SAR availability disagrees")
            if reader.registration_path is not None:
                for candidate in (reader.registration_path, reader.registration_path.with_suffix(".json")):
                    files[str(candidate)] = _sha256(candidate)
            if reader.row.get("metadata_path"):
                from tcmmsline3.dual_aux_data import _asset_path
                candidate = _asset_path(reader.root, reader.row["metadata_path"])
                files[str(candidate)] = _sha256(candidate)
            if row["available"]:
                for candidate in reader._paths["sar"]:
                    stat = candidate.stat()
                    row["assets"].append({"path": str(candidate), "size": stat.st_size,
                                          "mtime_ns": stat.st_mtime_ns})
            scenes[scene] = row
        finally:
            reader.close()
    return {"files": files, "scenes": scenes,
            "large_asset_identity": "path_size_mtime_ns; not a full SAR content digest"}


class SceneInputs(MultiViewSceneModel):
    """One bounded three-view preparation and serial native reader per batch."""

    def __init__(self, image, *, reader, **kwargs):
        super().__init__(None, image, **kwargs)
        self.reader = reader
        self.has_sar = reader is not None and reader.available["sar"]

    def forward(self, ms):
        from tcmmsline3.light_sar_data import _checked_aux
        from tcmmsline3.ms75_rebuild_model import MODEL_INPUT_KEYS
        positions = self.positions[self.cursor:self.cursor + len(ms)]
        _require(len(positions) == len(ms), "Scene input cursor exhausted")
        self.cursor += len(ms)
        samples = list(self.pool.map(self._one, positions)) if self.pool else [self._one(p) for p in positions]
        inputs = {"ms": ms}
        for key in MODEL_INPUT_KEYS:
            if key != "ms":
                inputs[key] = self._device(np.stack([sample[key] for sample in samples]))
        aux = []
        for top, left in positions:
            if self.has_sar:
                aux.append(_checked_aux(self.reader.read_patch(top, left, self.patch,
                    modality="sar", out_size=166), self.patch, 166))
            else:
                aux.append((np.zeros((2, 166, 166), np.float32), np.zeros((1, 166, 166), bool),
                            np.zeros((1, self.patch, self.patch), bool)))
        for index, key in enumerate(("sar_db", "sar_valid", "optical_valid")):
            inputs[key] = self._device(np.stack([row[index] for row in aux]))
        return inputs

    def _device(self, array):
        value = torch.from_numpy(np.ascontiguousarray(array))
        if self.device.type == "cuda":
            value = value.pin_memory()
        return value.to(self.device, non_blocking=self.device.type == "cuda")


def probabilities_from_stitched(logits, support):
    """Softmax follows per-expert stitching; fallback is an exact selection."""
    _require(len(logits) == 6 and support.dtype == torch.bool, "Six experts and bool support required")
    shape = logits[0].shape
    _require(len(shape) == 3 and shape[0] == 24 and support.shape == shape[1:]
             and all(x.shape == shape and bool(torch.isfinite(x).all()) for x in logits),
             "Invalid/nonfinite stitched expert geometry")
    probabilities = [torch.softmax(x.float(), dim=0) for x in logits]
    base = probabilities[0]
    for index in (1, 2, 3):
        probabilities[index] = torch.where(support[None], probabilities[index], base)
    return probabilities


def predictions_from_stitched(logits, support):
    probabilities = probabilities_from_stitched(logits, support)
    pair = probabilities[4] + probabilities[5]
    result = {arm: probabilities[i].argmax(0) for i, arm in enumerate(ARMS)}
    result.update({arm + "_triple": (pair + probabilities[i]).argmax(0)
                   for i, arm in enumerate(ARMS)})
    result["dual"] = pair.argmax(0)
    return result


@torch.inference_mode()
def evaluate_scene(models, teachers, image, label, *, scene_inputs, normalization,
                   device, batch_size=1, progress_callback=None, patch=PATCH, stride=STRIDE):
    """Bounded rolling evaluator. Small geometry and input providers are test seams."""
    _require(tuple(models) == ARMS and len(teachers) == 2, "Expected ordered base/P/R/Z and two teachers")
    _require(type(batch_size) is int and 1 <= batch_size <= 2, "Evaluation batch must be one or two")
    _require(image.dtype == np.uint8 and label.dtype == np.uint8
             and image.shape == (*label.shape, 4), "Expected aligned uint8 BGRN/native24 input")
    device = torch.device(device)
    height, width = label.shape
    ys, xs = sliding_starts(height, patch=patch, stride=stride), sliding_starts(width, patch=patch, stride=stride)
    weight = torch.from_numpy(blend_window(patch)).to(device)
    buffers = [torch.zeros((24, patch, width), dtype=torch.float32, device=device) for _ in range(6)]
    weights = torch.zeros((patch, width), dtype=torch.float32, device=device)
    unsupported = torch.zeros((patch, width), dtype=torch.bool, device=device)
    cms = {key: np.zeros((24, 24), np.int64) for key in RESULT_NAMES}
    supported_cms = {key: np.zeros((24, 24), np.int64) for key in RESULT_NAMES}
    counts = dict(valid_pixels=0, supported_pixels=0, supported_valid_pixels=0,
                  total_pixels=height * width, window_supported_pixels=0, windows=0)
    comparisons = {"R_vs_" + arm: dict(corrected=0, damaged=0, changed=0) for arm in ("base", "P", "Z")}
    for yi, top in enumerate(ys):
        for lefts, arrays in _prepared_batches(image, top=top, lefts=xs, patch=patch,
                                               batch_size=batch_size, normalization=normalization, executor=None):
            ms = torch.from_numpy(arrays)
            if device.type == "cuda":
                ms = ms.pin_memory()
            ms = ms.to(device, non_blocking=device.type == "cuda")
            inputs = scene_inputs(ms)
            support = inputs["optical_valid"]
            _require(support.dtype == torch.bool and support.shape == (len(lefts), 1, patch, patch),
                     "Window support must be bool B1HW")
            optical = {k: v for k, v in inputs.items() if k not in ("sar_db", "sar_valid", "optical_valid")}
            batch_supported = bool(support.any())
            base_logits = None
            for index, model in enumerate([*models.values(), *teachers]):
                if index in (1, 2, 3) and not batch_supported:
                    native = base_logits
                else:
                    arguments = optical if index == 0 else inputs if index < 4 else {"ms": ms}
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=device.type == "cuda", cache_enabled=False):
                        native = model(**arguments)["logits"]
                    _require(native.shape == (len(lefts), 24, patch, patch), "Native expert geometry differs")
                    native = native.float()
                    _require(bool(torch.isfinite(native).all()), "Nonfinite window logits")
                    if index == 0:
                        base_logits = native
                for i, left in enumerate(lefts):
                    buffers[index][:, :, left:left + patch].add_(native[i] * weight[None])
                del native
            del base_logits, inputs, optical, ms
            for i, left in enumerate(lefts):
                weights[:, left:left + patch].add_(weight)
                unsupported[:, left:left + patch] |= ~support[i, 0]
            counts["window_supported_pixels"] += int(support.sum())
            counts["windows"] += len(lefts)
            del support
        rows = ys[yi + 1] - top if yi + 1 < len(ys) else patch
        _require(bool(torch.isfinite(weights[:rows]).all()) and float(weights[:rows].min()) > 0,
                 "Uncovered/nonfinite overlap weights")
        for begin in range(0, rows, PROBABILITY_CHUNK_ROWS):
            end = min(begin + PROBABILITY_CHUNK_ROWS, rows)
            supported = ~unsupported[begin:end]
            stitched = [value[:, begin:end] / weights[None, begin:end] for value in buffers]
            predictions = {k: v.to(torch.uint8).cpu().numpy()
                           for k, v in predictions_from_stitched(stitched, supported).items()}
            target = label[top + begin:top + end]
            valid = target != 255
            spatial = supported.cpu().numpy()
            counts["valid_pixels"] += int(valid.sum())
            counts["supported_pixels"] += int(spatial.sum())
            counts["supported_valid_pixels"] += int((spatial & valid).sum())
            supported_target = np.where(spatial, target, np.uint8(255))
            for key, pred in predictions.items():
                add_confusion(cms[key], pred, target)
                add_confusion(supported_cms[key], pred, supported_target)
            r = predictions["R"]
            for arm in ("base", "P", "Z"):
                original = predictions[arm]
                event = comparisons["R_vs_" + arm]
                event["corrected"] += int((valid & (r == target) & (original != target)).sum())
                event["damaged"] += int((valid & (r != target) & (original == target)).sum())
                event["changed"] += int((valid & (r != original)).sum())
            del stitched, predictions, supported
        remaining = patch - rows
        for value in [*buffers, weights, unsupported]:
            if remaining:
                value[..., :remaining, :].copy_(value[..., rows:, :].clone())
            value[..., remaining:, :].zero_()
        if progress_callback:
            progress_callback(rows_completed=top + rows, rows_total=height,
                              windows_completed=counts["windows"], windows_total=len(ys) * len(xs))
    _require(counts["valid_pixels"] > 0, "No supervised pixels in held scene")
    counts["fallback_pixels"] = counts["total_pixels"] - counts["supported_pixels"]
    counts["fallback_valid_pixels"] = counts["valid_pixels"] - counts["supported_valid_pixels"]
    return cms, counts, comparisons, supported_cms


def _bound(path, config):
    _require(str(path) in config["files"], "Unbound evaluation input: " + str(path))
    _require(_sha256(path) == config["files"][str(path)], "Changed evaluation input: " + str(path))


def validate_request(config):
    from tools import train_ms75_sar_pilot as train
    from tcmmsline3.ms75_rebuild_data import load_manifest
    _require(config.get("schema") == SCHEMA, "Wrong evaluator schema")
    _require(config.get("physical_gpu") in (0, 2) and config.get("device", "cuda") == "cuda",
             "Only one authorized physical GPU0/2 is permitted")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu_uuid"], "Evaluation GPU UUID binding differs")
    _require(config.get("batch_size", 1) in (1, 2) and 0 <= config.get("workers", 2) <= 4
             and 1 <= config.get("torch_threads", 4) <= 8, "Bounded evaluation resources required")
    _require(config.get("auto_full", False) is False and config.get("official_test_auto", False) is False,
             "No automatic full120/official30 evaluation")
    for path in config["files"]:
        _bound(path, config)
    for key in ("raw_manifest", "views_cache", "source_manifest", "scope_manifest"):
        _bound(config[key], config)
    _bound(str(Path(__file__).resolve()), config)
    metadata = load_manifest(config["raw_manifest"])
    validate_held_request(metadata, metadata["held_ids"])
    source = json.loads(Path(config["source_manifest"]).read_text())["scene_to_group"]
    scope = json.loads(Path(config["scope_manifest"]).read_text())
    _require(set(scope["held24"]) == set(metadata["held_ids"])
             and set(scope["train96"]) == set(metadata["selection_train_ids"]), "Source scope split differs")
    _require(set(scope["held_source_strata"]) == set(metadata["held_ids"]), "Incomplete held source strata")
    for scene in metadata["held_ids"]:
        _require(scene in source and source[scene] == scope["held_source_strata"][scene]["frozen_group_120"],
                 "Frozen source grouping differs")
    observed_aux = collect_aux_bindings(config["data_root"], metadata["held_ids"])
    _require(observed_aux == config["aux_binding"], "Held SAR assets/registration/availability changed")
    for path, value in observed_aux["files"].items():
        _require(config["files"].get(path) == value, "Unbound held auxiliary metadata")
    _require(sum(v["available"] for v in observed_aux["scenes"].values()) == 21, "Expected held21 SAR and 3 missing")
    for scene, value in observed_aux["scenes"].items():
        _require(value["available"] == scope["held_source_strata"][scene]["has_sar"], "Source/SAR scope differs")
    _require(set(config["train_configs"]) == set(config["delta_checkpoints"]) == {"P", "R", "Z"},
             "Need all three fixed endpoint arms")
    teacher_specs = config.get("teacher_specs", DEFAULT_TEACHER_SPECS)
    _require(len(teacher_specs) == 2, "Fixed teacher pair required")
    for actual, expected in zip(teacher_specs, DEFAULT_TEACHER_SPECS):
        _require(all(actual[key] == expected[key] for key in
                     ("variant", "epoch", "checkpoint_sha256", "config_sha256", "config_fingerprint")),
                 "Only the original authenticated train96 DINO pair is allowed")
    training = {}
    common = ("base_checkpoint", "base_binding", "base_epoch", "raw_manifest", "views_cache", "replay_manifest",
              "seed", "updates", "effective_batch", "batch_size", "sar_mean", "sar_std", "optical_lr", "sar_lr",
              "weight_decay", "warmup_updates", "ema_decay")
    for arm in ("P", "R", "Z"):
        _bound(config["train_configs"][arm], config)
        _bound(config["delta_checkpoints"][arm], config)
        row = json.loads(Path(config["train_configs"][arm]).read_text())
        _require(row["schema"] == train.SCHEMA and row["arm"] == arm
                 and row["updates"] == 1000 and row["base_epoch"] == 16
                 and row["files"][row["base_checkpoint"]] == BASE_SHA256,
                 "Expected authenticated R e16 base and 1000-update pilot")
        for path, value in row["files"].items():
            _require(config["files"].get(path) == value, "Training input identity not preserved: " + path)
        _require(row["raw_manifest"] == config["raw_manifest"] and row["views_cache"] == config["views_cache"],
                 "Evaluation data differs from training")
        if training:
            _require(all(row[key] == training["P"][key] for key in common), "Unmatched arm training protocol")
        training[arm] = row
    return metadata, source, scope, training


def load_pilot_models(config, training, device):
    from tools import train_ms75_sar_pilot as train
    models, receipts = {}, {}
    initial = train.make_model(training["P"], device="cpu")
    _require(initial.optical.variant == "r", "Immutable pilot base must be MS75-R")
    models["base"] = initial.optical.eval().requires_grad_(False).to(device)
    del initial
    for arm in ("P", "R", "Z"):
        row = training[arm]
        model = train.make_model(row, device="cpu")
        payload = torch.load(config["delta_checkpoints"][arm], map_location="cpu", weights_only=False)
        keys = set(train.mutable_keys(model))
        frozen = train.state_hash({k: v for k, v in model.state_dict().items() if k not in keys})
        _require(payload["schema"] == train.SCHEMA and payload["role"] == "fixed_endpoint_ema_delta"
                 and payload["updates"] == 1000 and payload["config"] == row
                 and payload["config_sha256"] == train.engine.canonical_hash(row)
                 and payload["base_sha256"] == BASE_SHA256 and payload["frozen_hash"] == frozen,
                 "Delta checkpoint role/endpoint/base/config/frozen identity mismatch")
        train.apply_delta(model, payload["state_dict"])
        receipts[arm] = {"checkpoint": config["delta_checkpoints"][arm],
                         "checkpoint_sha256": config["files"][config["delta_checkpoints"][arm]],
                         "config_sha256": payload["config_sha256"], "frozen_hash": frozen, "updates": 1000}
        models[arm] = model.eval().requires_grad_(False).to(device)
        del payload
    return models, receipts


def _check_array_binding(metadata, scene, image, label):
    for key, value in (("image", image), ("label", label)):
        _require(hashlib.sha256(memoryview(value)).hexdigest() == metadata["scenes"][scene][key + "_sha256"],
                 "Raw mapped array identity changed: " + scene + "/" + key)


def _raw_stats(metadata, scene):
    result = {}
    for key in ("image", "label"):
        path = Path(metadata["scenes"][scene][key])
        stat = path.stat()
        result[key] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def validate_scene_record(row, *, scene, config_sha256, metadata, source, scope, raw_stats):
    """Resume only a pinned complete scene; all class denominators are checked."""
    _require(row["config_sha256"] == config_sha256 and row["scene_id"] == scene
             and row["source_group"] == source[scene]
             and row["source_strata"] == scope["held_source_strata"][scene]
             and row["raw_asset_stats"] == raw_stats, "Completed scene identity changed")
    histogram = np.asarray(metadata["scenes"][scene]["histogram"], dtype=np.int64)
    shape = metadata["scenes"][scene]["shape"]
    counts = row["counts"]
    _require(counts["valid_pixels"] == int(histogram.sum()) and counts["total_pixels"] == int(np.prod(shape))
             and 0 <= counts["supported_valid_pixels"] <= min(counts["valid_pixels"], counts["supported_pixels"])
             and 0 <= counts["supported_pixels"] <= counts["total_pixels"]
             and counts["fallback_valid_pixels"] == counts["valid_pixels"] - counts["supported_valid_pixels"]
             and counts["fallback_pixels"] == counts["total_pixels"] - counts["supported_pixels"],
             "Completed scene support/GT denominator changed")
    _require(set(row["results"]) == set(row["supported_confusions"]) == set(RESULT_NAMES),
             "Incomplete scene confusion inventory")
    cms, supported_cms, support_histogram = {}, {}, None
    for key in RESULT_NAMES:
        cm = np.asarray(row["results"][key]["confusion"])
        supported = np.asarray(row["supported_confusions"][key])
        _require(cm.shape == supported.shape == (24, 24)
                 and np.issubdtype(cm.dtype, np.integer) and np.issubdtype(supported.dtype, np.integer)
                 and bool((cm >= 0).all()) and bool((supported >= 0).all()) and bool((supported <= cm).all())
                 and np.array_equal(cm.sum(1), histogram)
                 and int(supported.sum()) == counts["supported_valid_pixels"], "Invalid completed scene confusion")
        _require(row["results"][key] == metrics_from_confusion(cm), "Completed scene metrics differ from CM")
        if support_histogram is None:
            support_histogram = supported.sum(1)
        _require(np.array_equal(supported.sum(1), support_histogram), "Arms have unequal supported denominators")
        expected = metrics_from_confusion(supported) if supported.sum() else None
        _require(row["supported_results"][key] == expected, "Completed supported metrics differ from CM")
        cms[key], supported_cms[key] = cm.astype(np.int64), supported.astype(np.int64)
    for arm in ("P", "R", "Z"):
        for suffix in ("", "_triple"):
            _require(np.array_equal(cms[arm + suffix] - supported_cms[arm + suffix],
                                    cms["base" + suffix] - supported_cms["base" + suffix]),
                     "Unsupported scene pixels did not retain the immutable base confusion")
    return cms, supported_cms


def run(config, *, progress=None):
    from tcmmsline3.light_sar_data import open_sar_scene
    from tcmmsline3.ms75_rebuild_data import GlobalViews, read_cached_scene, VIEW_PROTOCOL
    from tools import train_target_dino as engine
    metadata, source, scope, training = validate_request(config)
    output = Path(config["output_dir"])
    identity_sha = engine.canonical_hash(config)
    if output.exists() and any(output.iterdir()):
        _require((output / "input_binding.json").is_file()
                 and json.loads((output / "input_binding.json").read_text()) == config,
                 "Cannot resume an output with a different or missing input binding")
        if (output / "completed.json").exists():
            completed = json.loads((output / "completed.json").read_text())
            _require(completed["status"] == "evaluation_completed"
                     and completed["config_sha256"] == identity_sha and completed["completed_scenes"] == 24,
                     "Completed evaluation identity differs")
            for path, value in completed["output_hashes"].items():
                _require(_sha256(output / path) == value, "Completed evaluation artifact changed")
            for scene in metadata["held_ids"]:
                row = json.loads((output / "scenes" / (scene + ".json")).read_text())
                validate_scene_record(row, scene=scene, config_sha256=identity_sha, metadata=metadata,
                                      source=source, scope=scope, raw_stats=_raw_stats(metadata, scene))
            return json.loads((output / "metrics.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    scene_index_path = output / "scene_index.json"
    scene_index = json.loads(scene_index_path.read_text()) if scene_index_path.exists() else {
        "config_sha256": identity_sha, "completed": {}}
    _require(scene_index["config_sha256"] == identity_sha
             and set(scene_index["completed"]) <= set(metadata["held_ids"]), "Completed scene index differs")
    def emit(**row):
        detail = {"config_sha256": identity_sha, "time_unix": time.time(), **row}
        _write_json(output / "progress.json", detail)
        if progress:
            progress(detail)
        print(json.dumps(detail), flush=True)
    _write_json(output / "input_binding.json", config)
    emit(status="loading_models")
    device = torch.device("cuda")
    _require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    torch.set_num_threads(config.get("torch_threads", 4))
    torch.cuda.set_per_process_memory_fraction(min(.75, 18 * 1024**3 / torch.cuda.get_device_properties(device).total_memory))
    torch.cuda.reset_peak_memory_stats()
    models, receipts = load_pilot_models(config, training, device)
    teachers, teacher_receipts = load_teachers(config.get("teacher_specs", DEFAULT_TEACHER_SPECS),
                                               device=device, metadata=metadata)
    views = GlobalViews(metadata, config["views_cache"])
    norm = validate_held_request(metadata, metadata["held_ids"])
    cms = {key: np.zeros((24, 24), np.int64) for key in RESULT_NAMES}
    supported_cms = {key: np.zeros((24, 24), np.int64) for key in RESULT_NAMES}
    scene_cms, scenes, group_cms, stratum_cms = [], [], {}, {}
    scene_supported_cms, group_supported_cms = [], {}
    started = time.monotonic()
    for index, scene in enumerate(metadata["held_ids"]):
        scene_path = output / "scenes" / (scene + ".json")
        asset_stats = _raw_stats(metadata, scene)
        expected = config["aux_binding"]["scenes"][scene]
        group = source[scene]
        strata = scope["held_source_strata"][scene]
        if scene in scene_index["completed"]:
            _require(_sha256(scene_path) == scene_index["completed"][scene], "Completed scene JSON digest changed")
            row = json.loads(scene_path.read_text())
            current, current_supported = validate_scene_record(row, scene=scene, config_sha256=identity_sha,
                metadata=metadata, source=source, scope=scope, raw_stats=asset_stats)
            counts = row["counts"]
            emit(status="scene_resumed", scene_id=scene, scene_index=index + 1, scene_count=24)
        else:
            emit(status="checking_scene", scene_id=scene, scene_index=index + 1, scene_count=24)
            image, label = read_cached_scene(metadata, scene)
            _check_array_binding(metadata, scene, image, label)
            reader = open_sar_scene(config["data_root"], scene)
            _require(reader.registration_sha256 == expected["registration_sha256"]
                     and reader.available["sar"] == expected["available"], "Scene auxiliary identity changed")
            if reader.available["sar"]:
                _require(tuple(reader.old_shape) == label.shape, "Registration and old image shape differ")
            inputs = SceneInputs(image, reader=reader, global_raw=views.get(scene, image),
                                 normalization=norm, device=device, workers=config.get("workers", 2))
            scene_start = time.monotonic()
            try:
                current, counts, changes, current_supported = evaluate_scene(models, teachers, image, label,
                    scene_inputs=inputs, normalization=norm, device=device, batch_size=config.get("batch_size", 1),
                    progress_callback=lambda **fields: emit(status="evaluating", scene_id=scene,
                        scene_index=index + 1, scene_count=24, **fields))
                inputs.close(require_complete=True)
            finally:
                inputs.close()
                reader.close()
            row = {"config_sha256": identity_sha, "scene_id": scene, "source_group": group, "source_strata": strata,
                   "raw_asset_stats": asset_stats, "elapsed_seconds": time.monotonic() - scene_start,
                   "counts": counts, "changes": changes,
                   "results": {key: metrics_from_confusion(value) for key, value in current.items()},
                   "supported_confusions": {key: value.tolist() for key, value in current_supported.items()},
                   "supported_results": {key: metrics_from_confusion(value) if value.sum() else None
                                         for key, value in current_supported.items()}}
            validate_scene_record(row, scene=scene, config_sha256=identity_sha, metadata=metadata,
                                  source=source, scope=scope, raw_stats=asset_stats)
            _write_json(scene_path, row)
            scene_index["completed"][scene] = _sha256(scene_path)
            _write_json(scene_index_path, scene_index)
            del inputs, reader, image, label
        tags = ["sar_available" if strata["has_sar"] else "no_sar",
                "known_train_association" if strata["known_train_association"] else "no_known_train_association"]
        for destination in (cms, group_cms.setdefault(group, {k: np.zeros((24, 24), np.int64) for k in RESULT_NAMES}),
                            *(stratum_cms.setdefault(tag, {k: np.zeros((24, 24), np.int64) for k in RESULT_NAMES}) for tag in tags)):
            for key in RESULT_NAMES:
                destination[key] += current[key]
        for destination in (supported_cms, group_supported_cms.setdefault(group,
                             {k: np.zeros((24, 24), np.int64) for k in RESULT_NAMES})):
            for key in RESULT_NAMES:
                destination[key] += current_supported[key]
        if not expected["available"]:
            _require(counts["supported_pixels"] == 0 and all(np.array_equal(current[arm], current["base"])
                     and np.array_equal(current[arm + "_triple"], current["base_triple"]) for arm in ("P", "R", "Z")),
                     "Missing-SAR scene did not preserve immutable base")
        scenes.append(row)
        scene_cms.append(np.stack([current[k] for k in RESULT_NAMES]))
        scene_supported_cms.append(np.stack([current_supported[k] for k in RESULT_NAMES]))
        gc.collect()
    results = {key: metrics_from_confusion(value) for key, value in cms.items()}
    report = dict(schema=SCHEMA, config_sha256=identity_sha, scene_ids=metadata["held_ids"],
        completed_scenes=len(scenes), results=results, scenes=scenes,
        supported_results={key: metrics_from_confusion(value) if value.sum() else None
                           for key, value in supported_cms.items()},
        supported_source_groups={g: {k: metrics_from_confusion(v) if v.sum() else None for k, v in values.items()}
                                 for g, values in group_supported_cms.items()},
        supported_metrics_role="descriptive common SAR-supported GT subset; full24 remains primary",
        source_groups={g: {k: metrics_from_confusion(v) for k, v in values.items()} for g, values in group_cms.items()},
        source_strata={g: {k: metrics_from_confusion(v) for k, v in values.items()} for g, values in stratum_cms.items()},
        primary_score="R_triple.miou; all24 pooled CM, fixed24 classes including zero-union",
        primary_comparisons={k: results["R_triple"]["miou"] - results[k + "_triple"]["miou"] for k in ("base", "P", "Z")},
        metrics_aggregation="sum all24 scene CMs before fixed24-class mIoU",
        support_rule="AND of label-free optical_valid from every positive-Hann participating window",
        fallback_rule="torch.where common support, adapted softmax, immutable base softmax after logit stitching",
        sar_geometry={"optical": 512, "native": 166}, view_protocol=VIEW_PROTOCOL,
        stitching={"patch": PATCH, "stride": STRIDE, "hann_floor": HANN_FLOOR, "dtype": "float32",
                   "softmax": "after per-expert logit stitching", "probability_chunk_rows": PROBABILITY_CHUNK_ROWS},
        checkpoint_provenance=receipts, base_checkpoint=training["P"]["base_checkpoint"],
        base_sha256=BASE_SHA256, teacher_provenance=teacher_receipts,
        scope="original train96/held24 system development; no source-OOD claim", official_test_opened=False,
        probability_files_persisted=False, elapsed_seconds=time.monotonic() - started,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    np.savez_compressed(output / "confusions.npz", scene_ids=np.asarray(metadata["held_ids"]),
        result_names=np.asarray(RESULT_NAMES), source_groups=np.asarray([source[s] for s in metadata["held_ids"]]),
        scene_cm=np.stack(scene_cms), pooled_cm=np.stack([cms[k] for k in RESULT_NAMES]),
        scene_supported_cm=np.stack(scene_supported_cms),
        pooled_supported_cm=np.stack([supported_cms[k] for k in RESULT_NAMES]))
    _write_json(output / "metrics.json", report)
    completed = dict(status="evaluation_completed", config_sha256=identity_sha, completed_scenes=24,
                     output_hashes={str(p.relative_to(output)): _sha256(p) for p in sorted(output.rglob("*"))
                                    if p.is_file() and p.name != "progress.json"})
    _write_json(output / "completed.json", completed)
    emit(status="evaluation_completed", completed_scenes=24, primary_results=report["primary_comparisons"])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
