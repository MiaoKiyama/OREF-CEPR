"""Full held24 scoring of joint-SAR milestones in one optical-teacher pass.

Each expert is Hann-stitched in FP32 before softmax and equal-probability
fusion. Missing SAR always runs the current joint model's optical pathway.
Scene shards can be resumed and merged only with exact held24 coverage.
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

from tools.evaluate_ms75_rebuild import (_require, _sha256, _write_json, add_confusion,
    blend_window, load_teachers, metrics_from_confusion, sliding_starts, validate_held_request,
    PATCH, STRIDE)
from tools.evaluate_ms75_sar_pilot import SceneInputs, collect_aux_bindings
from tools.evaluate_target_dino import _prepared_batches
from tcmmsline3.ms75_sar_joint_data import digest
from tcmmsline3.ms75_sar_joint_model import MODEL_INPUT_KEYS, JOINT_INPUT_KEYS

SCHEMA = "ms75_sar_joint_allheld24_multi_epoch_v1_20260930"
PROBABILITY_CHUNK_ROWS = 32


def expert_inputs(inputs):
    """Adapt the shared Rebuild reader to the original MS75 forward contract."""
    _require(set(JOINT_INPUT_KEYS).issubset(inputs), "Shared reader omitted a native joint input")
    return ({key:inputs[key] for key in MODEL_INPUT_KEYS},
            {key:inputs[key] for key in JOINT_INPUT_KEYS})


def result_names(arms):
    arms = tuple(arms)
    _require(len(arms) >= 2 and arms[0] == "base" and len(set(arms)) == len(arms), "Base followed by unique candidate names required")
    return (*arms, *(arm + "_triple" for arm in arms), "dual")


def predictions_from_stitched(logits, arms=("base", "real")):
    """No support-conditioned probability substitution, including missing SAR."""
    names = result_names(arms)
    _require(len(logits) == len(arms) + 2, "Expected each candidate plus two fixed DINO experts")
    shape = logits[0].shape
    _require(len(shape) == 3 and shape[0] == 24 and all(v.shape == shape and bool(torch.isfinite(v).all()) for v in logits), "Invalid stitched expert geometry")
    probs = [value.float().softmax(0) for value in logits]
    pair = probs[-2] + probs[-1]
    output = {arm: probs[i].argmax(0) for i, arm in enumerate(arms)}
    output.update({arm + "_triple": (pair + probs[i]).argmax(0) for i, arm in enumerate(arms)})
    output["dual"] = pair.argmax(0)
    _require(set(output) == set(names), "Result set differs")
    return output


@torch.inference_mode()
def evaluate_scene(models, teachers, image, label, *, scene_inputs, normalization, device,
                   batch_size=1, progress_callback=None, patch=PATCH, stride=STRIDE):
    """One read and one DINO pair forward per window for every candidate epoch."""
    arms = tuple(models)
    names = result_names(arms)
    _require(len(teachers) == 2, "Two fixed clean DINO teachers required")
    _require(type(batch_size) is int and 1 <= batch_size <= 4, "Evaluation batch must be 1..4")
    _require(image.dtype == np.uint8 and label.dtype == np.uint8 and image.shape == (*label.shape, 4), "Expected aligned BGRN/native24 arrays")
    device = torch.device(device)
    height, width = label.shape
    ys, xs = sliding_starts(height, patch=patch, stride=stride), sliding_starts(width, patch=patch, stride=stride)
    weight = torch.from_numpy(blend_window(patch)).to(device)
    experts = [*models.values(), *teachers]
    buffers = [torch.zeros((24, patch, width), dtype=torch.float32, device=device) for _ in experts]
    weights = torch.zeros((patch, width), dtype=torch.float32, device=device)
    cms = {key: np.zeros((24, 24), np.int64) for key in names}
    counts = dict(valid_pixels=0, total_pixels=height * width, windows=0, sar_supported_window_pixels=0)
    changes = {arm: dict(corrected=0, damaged=0, changed=0) for arm in arms[1:]}
    for yi, top in enumerate(ys):
        for lefts, arrays in _prepared_batches(image, top=top, lefts=xs, patch=patch,
                batch_size=batch_size, normalization=normalization, executor=None):
            ms = torch.from_numpy(arrays)
            if device.type == "cuda":
                ms = ms.pin_memory()
            ms = ms.to(device, non_blocking=device.type == "cuda")
            inputs = scene_inputs(ms)
            optical, joint = expert_inputs(inputs)
            support = inputs["optical_valid"]
            _require(support.dtype == torch.bool and support.shape == (len(lefts), 1, patch, patch), "Expected B1HW SAR support")
            for index, model in enumerate(experts):
                arguments = optical if index == 0 else joint if index < len(arms) else {"ms": ms}
                # Always call current candidate, even when support is all zero.
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda", cache_enabled=False):
                    native = model(**arguments)["logits"]
                _require(native.shape == (len(lefts), 24, patch, patch), "Expert native geometry differs")
                native = native.float()
                _require(bool(torch.isfinite(native).all()), "Nonfinite native logits")
                for i, left in enumerate(lefts):
                    buffers[index][:, :, left:left + patch].add_(native[i] * weight[None])
                del native
            for left in lefts:
                weights[:, left:left + patch].add_(weight)
            counts["windows"] += len(lefts)
            counts["sar_supported_window_pixels"] += int(support.sum())
            del inputs, optical, joint, support, ms
        rows = ys[yi + 1] - top if yi + 1 < len(ys) else patch
        _require(float(weights[:rows].min()) > 0 and bool(torch.isfinite(weights[:rows]).all()), "Uncovered or nonfinite overlap weights")
        for begin in range(0, rows, PROBABILITY_CHUNK_ROWS):
            end = min(begin + PROBABILITY_CHUNK_ROWS, rows)
            stitched = [value[:, begin:end] / weights[None, begin:end] for value in buffers]
            preds = {key:value.to(torch.uint8).cpu().numpy() for key,value in predictions_from_stitched(stitched, arms).items()}
            target = label[top + begin:top + end]
            valid = target != 255
            counts["valid_pixels"] += int(valid.sum())
            for key, pred in preds.items():
                add_confusion(cms[key], pred, target)
            original = preds["base_triple"]
            for arm in arms[1:]:
                candidate, event = preds[arm + "_triple"], changes[arm]
                event["corrected"] += int((valid & (candidate == target) & (original != target)).sum())
                event["damaged"] += int((valid & (candidate != target) & (original == target)).sum())
                event["changed"] += int((valid & (candidate != original)).sum())
            del stitched, preds
        remaining = patch - rows
        for value in [*buffers, weights]:
            if remaining:
                value[..., :remaining, :].copy_(value[..., rows:, :].clone())
            value[..., remaining:, :].zero_()
        if progress_callback:
            progress_callback(rows_completed=top + rows, rows_total=height,
                              windows_completed=counts["windows"], windows_total=len(ys) * len(xs))
    _require(counts["valid_pixels"] > 0, "No known held labels")
    for cm in cms.values():
        _require(int(cm.sum()) == counts["valid_pixels"], "Lost optical supervision pixels")
    return cms, counts, changes


def shard_scene_ids(held_ids, rank, world_size):
    _require(type(rank) is int and type(world_size) is int and 1 <= world_size <= len(held_ids) and 0 <= rank < world_size, "Invalid scene shard")
    _require(len(set(held_ids)) == len(held_ids), "Repeated held scene")
    # Stable round robin, no DistributedSampler padding/duplication.
    return list(held_ids[rank::world_size])


def merge_scene_records(records, held_ids, arms, *, config_sha256):
    """Completeness and pixel totals are checked before any global metric."""
    names = result_names(arms)
    _require(len(records) == len(held_ids), "Incomplete held scene coverage")
    rows = {row["scene"]:row for row in records}
    _require(len(rows) == len(records) and set(rows) == set(held_ids), "Duplicate, extra or missing held scene")
    totals = {key: np.zeros((24, 24), np.int64) for key in names}
    for scene in held_ids:
        row = rows[scene]
        _require(row.get("schema") == SCHEMA and row.get("config_sha256") == config_sha256, "Scene result belongs to a different evaluation")
        _require(tuple(row.get("arms", ())) == tuple(arms) and set(row["confusions"]) == set(names), "Scene expert set changed")
        count = row["counts"]["valid_pixels"]
        _require(type(count) is int and count > 0, "Invalid valid pixel count")
        for key in names:
            raw = np.asarray(row["confusions"][key])
            _require(raw.shape == (24, 24) and raw.dtype.kind in "iu" and bool((raw >= 0).all()) and int(raw.sum()) == count, "Malformed or incomplete confusion matrix")
            totals[key] += raw.astype(np.int64)
    return totals


def _validate_config(config):
    from tools.train_ms75_sar_joint import validate_config, verify_files
    from tcmmsline3.ms75_rebuild_data import load_manifest
    _require(config.get("schema") == SCHEMA, "Wrong evaluation schema")
    train_path = Path(config["training_config"])
    _require(_sha256(train_path) == config["training_config_sha256"], "Training config bytes changed")
    training = json.loads(train_path.read_text(encoding="utf-8"))
    validate_config(training, devices=False)
    verify_files(training)
    metadata = load_manifest(training["data"]["raw_manifest"])
    norm = validate_held_request(metadata, config["held_ids"])
    milestones = config["candidates"]
    _require([int(row["epoch"]) for row in milestones] == [2, 4, 8, 12], "Evaluate all four prespecified milestone epochs")
    arms = ("base", *(f"R_e{int(row['epoch']):02d}" for row in milestones))
    _require(config.get("patch", 512) == 512 and config.get("stride", 256) == 256, "Fixed 512/256 scoring protocol required")
    for row in milestones:
        _require(_sha256(row["checkpoint"]) == row["sha256"], "Candidate checkpoint changed")
    observed_aux = collect_aux_bindings(training["data"]["data_root"], config["held_ids"])
    _require(config.get("aux_bindings") == observed_aux, "Held SAR geometry/availability/assets changed")
    return training, metadata, norm, arms


def load_models(config, training, *, device):
    from tools.train_ms75_sar_joint import make_model, make_optical, apply_delta, state_hash, subset, SCHEMA as TRAIN_SCHEMA
    # The untouched initial optical model is reconstructed independently.
    base = make_optical(training, device=device)
    models = {"base": base.eval()}
    for row in config["candidates"]:
        payload = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
        _require(payload.get("schema") == TRAIN_SCHEMA and payload.get("role") == "candidate_ema_delta"
            and payload.get("epoch") == int(row["epoch"]) and payload.get("updates") == int(row["epoch"]) * 480
            and payload.get("config_sha256") == digest(training)
            and payload.get("base_sha256") == training["files"][training["base_checkpoint"]], "Wrong candidate checkpoint provenance")
        # Reconstruct frozen state with exactly the training initialization RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(training["seed"])
            model = make_model(training, device="cpu")
        frozen_keys = tuple(k for k in model.state_dict() if k not in set(model.mutable_keys()))
        _require(state_hash(subset(model, frozen_keys)) == payload.get("frozen_hash"), "Candidate frozen reconstruction differs")
        apply_delta(model, payload["state_dict"])
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        models[f"R_e{int(row['epoch']):02d}"] = model
        del payload
    base.requires_grad_(False)
    return models


def make_eval_config(training_config, output_root=None, *, batch_size=1, workers=4):
    """Create a reproducible post-training request with all candidate SHA bindings."""
    from tcmmsline3.ms75_rebuild_data import load_manifest
    path = Path(training_config).resolve(strict=True)
    training = json.loads(path.read_text(encoding="utf-8"))
    metadata = load_manifest(training["data"]["raw_manifest"])
    root = Path(training["output_dir"])
    candidates = []
    for epoch in (2, 4, 8, 12):
        candidate = root / "candidates" / f"epoch{epoch:03d}_ema_delta.pt"
        candidates.append(dict(epoch=epoch, checkpoint=str(candidate), sha256=_sha256(candidate)))
    return dict(schema=SCHEMA, training_config=str(path), training_config_sha256=_sha256(path),
        candidates=candidates, held_ids=metadata["held_ids"], patch=512, stride=256,
        output=str(Path(output_root).resolve()) if output_root else str(root / "held24"),
        device="cuda", batch_size=int(batch_size), workers=int(workers),
        aux_bindings=collect_aux_bindings(training["data"]["data_root"], metadata["held_ids"]))


def memory_smoke(training_config, *, output=None, batch_size=1, workers=4):
    """Seven independent models and widest-scene stripes, with a real SAR tile.

    This consumes the authenticated two-update smoke delta, never relabels it
    as an epoch candidate, and reports no validation metric or held GT score.
    """
    from tools.train_ms75_sar_joint import (validate_config, verify_files, make_model,
        make_optical, apply_delta, state_hash, subset, SCHEMA as TRAIN_SCHEMA)
    from tcmmsline3.ms75_rebuild_data import load_manifest, GlobalViews, _check_paths
    from tcmmsline3.light_sar_data import _availability, open_sar_scene
    started = time.time()
    train_path = Path(training_config).resolve(strict=True)
    training = json.loads(train_path.read_text(encoding="utf-8"))
    identity = validate_config(training, devices=False)
    verify_files(training)
    _require(batch_size in (1, 2, 4), "Memory smoke batch must be 1, 2 or4")
    metadata = load_manifest(training["data"]["raw_manifest"])
    normalization = validate_held_request(metadata, metadata["held_ids"])
    delta_path = Path(training["output_dir"]) / "smoke" / "smoke_delta.pt"
    smoke_receipt = json.loads((Path(training["output_dir"]) / "smoke.json").read_text(encoding="utf-8"))
    _require(smoke_receipt.get("status") == "passed" and smoke_receipt.get("config_sha256") == identity
             and smoke_receipt.get("formal_compatible") is True
             and smoke_receipt.get("smoke_delta") == str(delta_path)
             and smoke_receipt.get("smoke_delta_sha256") == _sha256(delta_path),
             "Memory smoke requires this training config's successful DDP smoke")
    payload = torch.load(delta_path, map_location="cpu", weights_only=True)
    _require(payload.get("schema") == TRAIN_SCHEMA and payload.get("role") == "smoke_delta"
        and payload.get("config_sha256") == identity and payload.get("updates") == 2
        and payload.get("world_size") == 2 and payload.get("micro_batch") == training["micro_batch"]
        and payload.get("base_sha256") == training["files"][training["base_checkpoint"]],
        "Only the real two-rank, two-stage smoke delta is eligible")
    availability, _ = _availability(training["data"]["data_root"])
    held = metadata["held_ids"]
    width = max(metadata["scenes"][s]["shape"][1] for s in held)
    scene = max((s for s in held if availability[s]), key=lambda s: metadata["scenes"][s]["shape"][1])
    source = metadata["scenes"][scene]
    _check_paths(metadata, source)
    image = np.load(source["image"], mmap_mode="r", allow_pickle=False)
    _require(image.dtype == np.uint8 and image.shape == (*source["shape"], 4), "Smoke raw optical geometry differs")
    # No held label raster is opened by this function.
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    models, teachers, buffers, reader, provider = {}, [], [], None, None
    try:
        models["base"] = make_optical(training, device=device).eval().requires_grad_(False)
        for replica in range(4):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(training["seed"])
                model = make_model(training, device="cpu")
            frozen = tuple(k for k in model.state_dict() if k not in set(model.mutable_keys()))
            _require(state_hash(subset(model, frozen)) == payload["frozen_hash"], "Smoke frozen model reconstruction differs")
            apply_delta(model, payload["state_dict"])
            models[f"smoke_replica{replica}"] = model.to(device).eval().requires_grad_(False)
        teachers, _ = load_teachers(device=device, metadata=metadata)
        model_bytes = torch.cuda.memory_allocated(device)
        buffers = [torch.zeros((24, 512, width), dtype=torch.float32, device=device) for _ in range(7)]
        weights = torch.ones((512, width), dtype=torch.float32, device=device)
        stripe_bytes = sum(v.numel() * v.element_size() for v in [*buffers, weights])
        views = GlobalViews(metadata, training["data"].get("views_cache"))
        reader = open_sar_scene(training["data"]["data_root"], scene)
        provider = SceneInputs(image, reader=reader, device=device, normalization=normalization,
            global_raw=views.get(scene, image), workers=workers)
        xs = sliding_starts(image.shape[1], patch=512, stride=256)
        lefts, arrays = next(_prepared_batches(image, top=0, lefts=xs, patch=512,
            batch_size=batch_size, normalization=normalization, executor=None))
        ms = torch.from_numpy(arrays).pin_memory().to(device, non_blocking=True)
        with torch.inference_mode():
            inputs = provider(ms)
            optical, joint = expert_inputs(inputs)
            forward_seconds = {}
            for index, (name, model) in enumerate([*models.items(), *(('teacher' + str(i), v) for i,v in enumerate(teachers))]):
                torch.cuda.synchronize(device)
                began = time.time()
                arguments = optical if index == 0 else joint if index < 5 else {"ms": ms}
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    native = model(**arguments)["logits"]
                _require(native.shape == (len(lefts), 24, 512, 512) and bool(torch.isfinite(native).all()), "Smoke expert output differs")
                native = native.float()
                for j, left in enumerate(lefts):
                    buffers[index][:, :, left:left+512].add_(native[j])
                torch.cuda.synchronize(device)
                forward_seconds[name] = time.time() - began
                del native
            # The real evaluator forms seven FP32 normalized stripes then
            # seven probability stripes at once for a32-row full-width chunk.
            chunk = [value[:, :32] / weights[None, :32] for value in buffers]
            predictions_from_stitched(chunk, tuple(models))
            del chunk
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        result = dict(schema=SCHEMA, role="evaluation_memory_smoke", status="passed",
            training_config_sha256=identity, training_config_file_sha256=_sha256(train_path),
            smoke_delta=str(delta_path), smoke_delta_sha256=_sha256(delta_path),
            candidate_model_count=4, candidates_are_smoke_replicas=True,
            formal_epochs_evaluated=[], formal_metrics=False, held_GT_read=False,
            image_scene=scene, maximum_held_width=width, batch_size=batch_size,
            model_allocated_bytes=model_bytes, rolling_stripe_bytes=stripe_bytes,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            free_bytes_after_forward=free, total_device_bytes=total,
            native_SAR_valid_pixels=int(inputs["sar_valid"].sum()),
            forward_seconds=forward_seconds, duration_seconds=time.time()-started,
            interpretation="Memory/forward mechanics only; smoke replicas do not measure epoch quality.")
        result["payload_sha256"] = digest(result)
        destination = Path(output) if output else Path(training["output_dir"]) / f"eval_memory_smoke_b{batch_size}.json"
        _write_json(destination, result)
        return result
    finally:
        if provider is not None:
            provider.close()
        if reader is not None:
            reader.close()
        models.clear()
        teachers.clear()
        buffers.clear()
        gc.collect()
        torch.cuda.empty_cache()


def run(config, *, rank=0, world_size=1, progress=None):
    from tcmmsline3.ms75_rebuild_data import GlobalViews, read_cached_scene
    from tcmmsline3.light_sar_data import open_sar_scene
    training, metadata, normalization, arms = _validate_config(config)
    fingerprint = digest(config)
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    scenes = shard_scene_ids(config["held_ids"], rank, world_size)
    device = torch.device(config.get("device", "cuda"))
    models = load_models(config, training, device=device)
    teachers, _ = load_teachers(config.get("teachers"), device=device, metadata=metadata)
    views = GlobalViews(metadata, training["data"].get("views_cache"))
    records = []
    try:
        for scene in scenes:
            path = out / (scene + ".json")
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                # Use the same validator for resume and complete-shard merging.
                merge_scene_records([row], [scene], arms, config_sha256=fingerprint)
                records.append(row)
                continue
            image, label = read_cached_scene(metadata, scene)
            source = metadata["scenes"][scene]
            for array, key in ((image, "image_sha256"), (label, "label_sha256")):
                _require(hashlib.sha256(memoryview(array)).hexdigest() == source[key], "Held input payload changed")
            reader = open_sar_scene(training["data"]["data_root"], scene)
            provider = SceneInputs(image, reader=reader, device=device, normalization=normalization,
                global_raw=views.get(scene, image), workers=int(config.get("workers", 4)))
            started = time.time()
            try:
                cms, counts, changes = evaluate_scene(models, teachers, image, label,
                    scene_inputs=provider, normalization=normalization, device=device,
                    batch_size=int(config.get("batch_size", 1)),
                    progress_callback=(lambda **row: progress(dict(event="scene_progress", rank=rank, scene=scene, **row))) if progress else None)
                provider.close(require_complete=True)
            finally:
                reader.close()
                provider.close()
            row = dict(schema=SCHEMA, config_sha256=fingerprint, scene=scene, arms=list(arms),
                counts=counts, changes=changes, has_sar=reader.available["sar"],
                input_image_sha256=source["image_sha256"], input_label_sha256=source["label_sha256"],
                duration_seconds=time.time() - started, confusions={k:v.tolist() for k,v in cms.items()})
            _write_json(path, row)
            records.append(row)
            if progress:
                progress(dict(event="scene_complete", rank=rank, scene=scene, seconds=row["duration_seconds"],
                    scores={k:metrics_from_confusion(v)["miou"] for k,v in cms.items()}))
            del image, label, provider, reader, cms
            gc.collect()
        _write_json(out / f"shard_{rank:02d}.json", dict(schema=SCHEMA, config_sha256=fingerprint,
            rank=rank, world_size=world_size, scene_ids=scenes, complete=True))
    finally:
        del models, teachers
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if world_size == 1:
        return merge(config)
    return dict(rank=rank, scene_ids=scenes, complete=True)


def merge(config):
    training, metadata, normalization, arms = _validate_config(config)
    out = Path(config["output"])
    records = [json.loads((out / (s + ".json")).read_text(encoding="utf-8")) for s in config["held_ids"]]
    totals = merge_scene_records(records, config["held_ids"], arms, config_sha256=digest(config))
    np.savez_compressed(out / "confusions.npz", **totals)
    value = dict(schema=SCHEMA, config_sha256=digest(config), scene_ids=config["held_ids"],
        scores={key:metrics_from_confusion(cm) for key,cm in totals.items()},
        selection_metric="complete_held24_global_miou_equal_probability_triple",
        missing_sar="current_joint_optical_path; no immutable_base_probability_fallback")
    _write_json(out / "metrics.json", value)
    _write_json(out / "completed.json", dict(schema=SCHEMA, config_sha256=digest(config),
        scene_count=len(records), metrics_sha256=_sha256(out / "metrics.json"),
        confusions_sha256=_sha256(out / "confusions.npz")))
    return value


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Existing pinned evaluation request")
    group.add_argument("--training-config", help="Build request from completed training config")
    parser.add_argument("--output-root")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare", action="store_true", help="Only create held24/eval_config.json")
    parser.add_argument("--smoke", action="store_true", help="Seven-model memory/forward check using authenticated smoke delta")
    parser.add_argument("--smoke-output")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        _require(bool(args.training_config) and not args.prepare and not args.merge, "Smoke requires --training-config only")
        print(json.dumps(memory_smoke(args.training_config, output=args.smoke_output,
            batch_size=args.batch_size, workers=args.workers), sort_keys=True), flush=True)
        return
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else make_eval_config(
        args.training_config, args.output_root, batch_size=args.batch_size, workers=args.workers)
    if args.prepare or args.training_config:
        destination = Path(config["output"]) / "eval_config.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            _require(json.loads(destination.read_text(encoding="utf-8")) == config, "Existing evaluation request differs")
        else:
            _write_json(destination, config)
        if args.prepare:
            print(json.dumps(dict(config=str(destination), config_sha256=digest(config))), flush=True)
            return
    value = merge(config) if args.merge else run(config, rank=args.rank, world_size=args.world_size,
        progress=lambda row: print(json.dumps(row, sort_keys=True), flush=True))
    print(json.dumps(value, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
