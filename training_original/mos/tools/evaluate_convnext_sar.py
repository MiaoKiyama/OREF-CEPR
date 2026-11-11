"""Full held24 scoring of direction3 ConvNeXt-L + SAR milestones.

Each expert is Hann-stitched in FP32 before softmax and equal-probability
fusion. Missing SAR always runs the current joint model's optical pathway.
Scene shards can be resumed and merged only with exact held24 coverage.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_ms75_rebuild import (_require, _sha256, _write_json,
    load_teachers, metrics_from_confusion, sliding_starts, validate_held_request,
    DEFAULT_TEACHER_SPECS)
from tools.evaluate_ms75_sar_pilot import SceneInputs, collect_aux_bindings
from tools.evaluate_target_dino import _prepared_batches
from tcmmsline3.ms75_sar_joint_data import digest

SCHEMA = "convnext_sar_allheld24_multi_epoch_v1_20260930"
CHECKPOINT_EPOCHS = (4, 8, 16, 24, 32)
BASE_SHA256 = "3d9012ebaea662c7577a309cb92fcc03e4b2f2e7bed84e507e49548d62a569ec"
from tools.evaluate_ms75_sar_joint import (expert_inputs, result_names,
    predictions_from_stitched, evaluate_scene, shard_scene_ids,
    merge_scene_records as _joint_merge_scene_records, SCHEMA as JOINT_EVAL_SCHEMA)


def merge_scene_records(records, held_ids, arms, *, config_sha256):
    """Require D3 provenance, then reuse D1's strict mathematical merge checks."""
    for row in records:
        _require(row.get("schema") == SCHEMA, "Scene result belongs to another direction")
    canonical = [dict(row, schema=JOINT_EVAL_SCHEMA) for row in records]
    return _joint_merge_scene_records(canonical, held_ids, arms, config_sha256=config_sha256)


def validate_delta(payload, training, *, epoch=None, smoke=False):
    """D3 artifacts cannot be substituted with MS75 weights or fake milestone smoke."""
    from tools.train_convnext_sar import SCHEMA as TRAIN_SCHEMA, pretraining_identity
    _require(type(smoke) is bool and ((epoch is None) == smoke), "Specify exactly smoke or a candidate epoch")
    identity = digest(training)
    _require(payload.get("schema") == TRAIN_SCHEMA
        and payload.get("config_sha256") == identity
        and payload.get("convnext_checkpoint_sha256") == training["files"][training["convnext_checkpoint"]]
        and payload.get("croma_checkpoint_sha256") == training["files"][training["croma_checkpoint"]]
        and payload.get("pretraining_identity") == pretraining_identity(training),
        "D3 checkpoint training provenance differs")
    if smoke:
        _require(payload.get("role") == "smoke_delta" and payload.get("updates") == 3
            and payload.get("world_size") == 2 and payload.get("micro_batch") == training["micro_batch"],
            "Only the actual two-rank, three-update D3 smoke delta is eligible")
    else:
        _require(type(epoch) is int and epoch in CHECKPOINT_EPOCHS
            and payload.get("role") == "candidate_ema_delta"
            and payload.get("epoch") == epoch and payload.get("updates") == epoch * 480,
            "Wrong D3 candidate milestone/role")
    return identity


def validate_teacher_specs(training):
    """Teachers are evaluation-only; bind their exact clean96 assets and source."""
    specs = training.get("teachers")
    _require(isinstance(specs, list) and len(specs) == 2, "Two evaluation-only clean96 DINO specs required")
    for row, pinned in zip(specs, DEFAULT_TEACHER_SPECS):
        for key in ("name", "variant", "epoch", "checkpoint_sha256", "config_sha256", "config_fingerprint"):
            _require(row.get(key) == pinned[key], "Evaluation-only teacher identity changed: " + key)
        for key in ("checkpoint", "config", "source_manifest"):
            _require(bool(row.get(key)) and row[key] in training["files"], "Teacher asset/source is not SHA-bound: " + key)
        for key in ("checkpoint", "config"):
            _require(training["files"][row[key]] == row[key + "_sha256"], "Teacher SHA disagrees with the retained clean checkpoint")
    return specs


def validate_evaluation_assets(training):
    from tools.train_ms75_sar_joint import BASE_FINGERPRINT
    _require(training.get("base_config_fingerprint") == BASE_FINGERPRINT
        and training["files"].get(training.get("base_checkpoint")) == BASE_SHA256
        and bool(training.get("base_config")) and training["base_config"] in training["files"],
        "Evaluation baseline must be the retained clean train96 MS75, never full120")
    _require(training.get("checkpoint_epochs") == list(CHECKPOINT_EPOCHS), "D3 candidate schedule changed")
    return validate_teacher_specs(training)


def _validate_config(config):
    from tools.train_convnext_sar import validate_config, verify_files
    from tcmmsline3.ms75_rebuild_data import load_manifest
    _require(config.get("schema") == SCHEMA, "Wrong evaluation schema")
    train_path = Path(config["training_config"])
    _require(_sha256(train_path) == config["training_config_sha256"], "Training config bytes changed")
    training = json.loads(train_path.read_text(encoding="utf-8"))
    validate_config(training, devices=False)
    verify_files(training)
    validate_evaluation_assets(training)
    _require(config.get("teachers") == training["teachers"], "Evaluation-only teacher specs differ from the bound request")
    metadata = load_manifest(training["data"]["raw_manifest"])
    norm = validate_held_request(metadata, config["held_ids"])
    milestones = config["candidates"]
    _require([int(row["epoch"]) for row in milestones] == list(CHECKPOINT_EPOCHS), "Evaluate all five prespecified milestone epochs")
    arms = ("base", *(f"R_e{int(row['epoch']):02d}" for row in milestones))
    _require(config.get("patch", 512) == 512 and config.get("stride", 256) == 256, "Fixed 512/256 scoring protocol required")
    for row in milestones:
        _require(_sha256(row["checkpoint"]) == row["sha256"], "Candidate checkpoint changed")
    observed_aux = collect_aux_bindings(training["data"]["data_root"], config["held_ids"])
    _require(config.get("aux_bindings") == observed_aux, "Held SAR geometry/availability/assets changed")
    return training, metadata, norm, arms


def load_models(config, training, *, device):
    from tools.train_convnext_sar import make_model, apply_delta, state_hash, subset
    from tools.train_ms75_sar_joint import make_optical
    # The untouched initial optical model is reconstructed independently.
    base = make_optical(training, device=device)
    models = {"base": base.eval()}
    for row in config["candidates"]:
        payload = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
        validate_delta(payload, training, epoch=int(row["epoch"]))
        # Reconstruct frozen state with exactly the training initialization RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(training["seed"])
            model = make_model(training, device="cpu")
        frozen_keys = tuple(k for k in model.state_dict() if k not in set(model.mutable_keys()))
        _require(state_hash(subset(model, frozen_keys)) == payload.get("frozen_hash"), "Candidate frozen reconstruction differs")
        apply_delta(model, payload["state_dict"])
        model.set_stage("joint")
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
    from tools.train_convnext_sar import validate_config, verify_files
    path = Path(training_config).resolve(strict=True)
    training = json.loads(path.read_text(encoding="utf-8"))
    validate_config(training, devices=False)
    verify_files(training)
    validate_evaluation_assets(training)
    _require(type(batch_size) is int and batch_size in (1, 2, 4), "Evaluation batch must be 1, 2 or 4")
    _require(type(workers) is int and 1 <= workers <= 6, "Evaluation workers must be 1..6")
    metadata = load_manifest(training["data"]["raw_manifest"])
    root = Path(training["output_dir"])
    candidates = []
    for epoch in CHECKPOINT_EPOCHS:
        candidate = root / "candidates" / f"epoch{epoch:03d}_ema_delta.pt"
        candidates.append(dict(epoch=epoch, checkpoint=str(candidate), sha256=_sha256(candidate)))
    return dict(schema=SCHEMA, training_config=str(path), training_config_sha256=_sha256(path),
        candidates=candidates, teachers=training["teachers"], held_ids=metadata["held_ids"], patch=512, stride=256,
        output=str(Path(output_root).resolve()) if output_root else str(root / "held24"),
        device="cuda", batch_size=int(batch_size), workers=int(workers),
        aux_bindings=collect_aux_bindings(training["data"]["data_root"], metadata["held_ids"]))


def memory_smoke(training_config, *, output=None, batch_size=1, workers=4):
    """Eight independent models and widest-scene stripes, with a real SAR tile.

    This consumes the authenticated three-update smoke delta, never relabels it
    as an epoch candidate, and reports no validation metric or held GT score.
    """
    from tools.train_convnext_sar import (validate_config, verify_files, make_model,
        apply_delta, state_hash, subset)
    from tools.train_ms75_sar_joint import make_optical
    from tcmmsline3.ms75_rebuild_data import load_manifest, GlobalViews, _check_paths
    from tcmmsline3.light_sar_data import _availability, open_sar_scene
    started = time.time()
    train_path = Path(training_config).resolve(strict=True)
    training = json.loads(train_path.read_text(encoding="utf-8"))
    identity = validate_config(training, devices=False)
    verify_files(training)
    validate_evaluation_assets(training)
    _require(type(batch_size) is int and batch_size in (1, 2, 4), "Memory smoke batch must be 1, 2 or 4")
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
    validate_delta(payload, training, smoke=True)
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
        for replica in range(len(CHECKPOINT_EPOCHS)):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(training["seed"])
                model = make_model(training, device="cpu")
            frozen = tuple(k for k in model.state_dict() if k not in set(model.mutable_keys()))
            _require(state_hash(subset(model, frozen)) == payload["frozen_hash"], "Smoke frozen model reconstruction differs")
            apply_delta(model, payload["state_dict"])
            model.set_stage("joint")
            models[f"smoke_replica{replica}"] = model.to(device).eval().requires_grad_(False)
        teachers, teacher_receipts = load_teachers(training["teachers"], device=device, metadata=metadata)
        teacher_hashes = {spec["name"]:state_hash(model.state_dict()) for spec,model in zip(training["teachers"], teachers)}
        model_bytes = torch.cuda.memory_allocated(device)
        buffers = [torch.zeros((24, 512, width), dtype=torch.float32, device=device) for _ in range(len(models) + len(teachers))]
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
            _require(bool(inputs["sar_valid"].any()), "Memory smoke tile must exercise the observed real-SAR branch")
            optical, joint = expert_inputs(inputs)
            forward_seconds = {}
            for index, (name, model) in enumerate([*models.items(), *(('teacher' + str(i), v) for i,v in enumerate(teachers))]):
                torch.cuda.synchronize(device)
                began = time.time()
                arguments = optical if index == 0 else joint if index < len(models) else {"ms": ms}
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    native = model(**arguments)["logits"]
                _require(native.shape == (len(lefts), 24, 512, 512) and bool(torch.isfinite(native).all()), "Smoke expert output differs")
                native = native.float()
                for j, left in enumerate(lefts):
                    buffers[index][:, :, left:left+512].add_(native[j])
                torch.cuda.synchronize(device)
                forward_seconds[name] = time.time() - began
                del native
            # The real evaluator forms eight FP32 normalized stripes then
            # eight probability stripes at once for a32-row full-width chunk.
            chunk = [value[:, :32] / weights[None, :32] for value in buffers]
            predictions_from_stitched(chunk, tuple(models))
            del chunk
            # Cover the extra half-stripe clone retained during row rollover.
            for value in [*buffers, weights]:
                value[..., :256, :].copy_(value[..., 256:, :].clone())
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        result = dict(schema=SCHEMA, role="evaluation_memory_smoke", status="passed",
            training_config_sha256=identity, training_config_file_sha256=_sha256(train_path),
            smoke_delta=str(delta_path), smoke_delta_sha256=_sha256(delta_path),
            evaluation_only_teacher_specs=training["teachers"], teacher_frozen_hashes=teacher_hashes,
            teacher_provenance=teacher_receipts, candidate_model_count=len(CHECKPOINT_EPOCHS),
            total_model_count=len(models) + len(teachers), candidates_are_smoke_replicas=True,
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
    teachers, teacher_receipts = load_teachers(training["teachers"], device=device, metadata=metadata)
    from tools.train_convnext_sar import state_hash
    teacher_hashes = {spec["name"]:state_hash(model.state_dict()) for spec,model in zip(training["teachers"], teachers)}
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
                teacher_frozen_hashes=teacher_hashes,
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


def comparison_report(records, held_ids, arms, *, config_sha256):
    """All24 remains primary; group and scene summaries explain where a gain came from."""
    totals = merge_scene_records(records, held_ids, arms, config_sha256=config_sha256)
    scores = {key:metrics_from_confusion(value) for key,value in totals.items()}
    by_scene = {row["scene"]:row for row in records}
    scene_scores = {}
    for scene in held_ids:
        row = by_scene[scene]
        _require(type(row.get("has_sar")) is bool, "SAR availability group must be explicit")
        scene_scores[scene] = {key:metrics_from_confusion(np.asarray(value, dtype=np.int64))["miou"]
                              for key,value in row["confusions"].items()}
    candidates = {}
    for arm in arms[1:]:
        key = arm + "_triple"
        delta = {s:100 * (scene_scores[s][key] - scene_scores[s]["base_triple"]) for s in held_ids}
        candidates[arm] = dict(
            triple_miou=scores[key]["miou"],
            triple_delta_vs_base_pp=100 * (scores[key]["miou"] - scores["base_triple"]["miou"]),
            triple_delta_vs_dual_pp=100 * (scores[key]["miou"] - scores["dual"]["miou"]),
            per_class_triple_delta_vs_base_pp=(100 * (np.asarray(scores[key]["class_iou"])
                - np.asarray(scores["base_triple"]["class_iou"]))).tolist(),
            scene_delta_vs_base_pp=delta,
            scenes_improved=sum(v > 0 for v in delta.values()),
            scenes_degraded=sum(v < 0 for v in delta.values()),
            corrected_pixels=sum(row["changes"][arm]["corrected"] for row in records),
            damaged_pixels=sum(row["changes"][arm]["damaged"] for row in records))
    groups = {}
    for name, present in (("with_sar", True), ("without_sar", False)):
        ids = [s for s in held_ids if by_scene[s]["has_sar"] is present]
        if not ids:
            groups[name] = dict(scene_ids=[], scores=None, candidate_triple_delta_vs_base_pp=None)
            continue
        group_total = merge_scene_records([by_scene[s] for s in ids], ids, arms, config_sha256=config_sha256)
        group_scores = {k:metrics_from_confusion(v) for k,v in group_total.items()}
        groups[name] = dict(scene_ids=ids, scores=group_scores,
            candidate_triple_delta_vs_base_pp={arm:100 * (group_scores[arm + "_triple"]["miou"]
                - group_scores["base_triple"]["miou"]) for arm in arms[1:]})
    best = max(arms[1:], key=lambda arm:scores[arm + "_triple"]["miou"])
    return dict(scores=scores, comparisons=candidates, sar_availability_groups=groups,
        scene_miou=scene_scores, selected_arm=best, selected_epoch=int(best.rsplit("_e", 1)[1]),
        selection_note="Prespecified held24 screening only; earliest epoch breaks exact ties. No automatic full120 or official30.",
        interpretation="Group and scene mIoU retain all24 zero-union policy; inspect target-present classes before interpreting rare-class subgroup changes.")


def merge(config):
    training, metadata, normalization, arms = _validate_config(config)
    out = Path(config["output"])
    records = [json.loads((out / (s + ".json")).read_text(encoding="utf-8")) for s in config["held_ids"]]
    totals = merge_scene_records(records, config["held_ids"], arms, config_sha256=digest(config))
    np.savez_compressed(out / "confusions.npz", **totals)
    report = comparison_report(records, config["held_ids"], arms, config_sha256=digest(config))
    value = dict(schema=SCHEMA, config_sha256=digest(config), scene_ids=config["held_ids"],
        **report,
        selection_metric="complete_held24_global_miou_equal_probability_triple",
        baseline="retained_clean_train96_MS75; same_fixed_clean_DINO_pair",
        teachers="evaluation_only; no_DINO_or_MS75_training_teacher",
        missing_sar="current_ConvNeXt_optical_path; no immutable_base_probability_fallback")
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
    parser.add_argument("--smoke", action="store_true", help="Eight-model memory/forward check using authenticated smoke delta")
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
