"""Bounded rerun of the original native24 trainers, preserving their schedules.

This entry point changes device authorization and the terminal update only.
The historical model, loss, sampler, optimizer, EMA and recovery implementations
remain in the copied historical project. The original trainer files are never
edited. Run once with --smoke before the formal run, just as in those trainers.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "native24_retrain_limited_v1_20261015"
JOBS = {
    "pyramid": dict(module="tools.train_convnext_upgrade_full120", gpus=[3, 4],
                    epochs=30, scheduler_epochs=30, stop=12, candidate_epochs=[10, 11, 12]),
    "convnext": dict(module="tools.train_convnext_sar_full120", gpus=[5, 6],
                    epochs=24, scheduler_epochs=32, stop=14, candidate_epochs=[12, 13, 14]),
}
GPU_UUID_BY_INDEX = {
    3: "GPU-183f3cce-cb37-5f1c-49dc-f1db79331e11",
    4: "GPU-64ff02fb-e035-2e17-9e55-961de4dd9f75",
    5: "GPU-3460188f-cb0e-f065-26d0-348e74ebe26c",
    6: "GPU-c8faf4b1-5638-90c6-fd97-fcc8eaa3ccbb",
}


def validate_contract(config, *, wrapper_path=None, trainer_path=None):
    extension = config.get("native24_retrain", {})
    if extension.get("schema") != SCHEMA or extension.get("model_kind") not in JOBS:
        raise ValueError("missing explicit native24 limited-retrain contract")
    if type(extension.get("checkpoint_reserve_bytes")) is not int or extension["checkpoint_reserve_bytes"] != 0:
        raise ValueError("this rerun requires the authorized zero checkpoint reserve")
    kind = extension["model_kind"]
    job = JOBS[kind]
    for key in ("epochs", "scheduler_epochs"):
        if config.get(key) != job[key]:
            raise ValueError("original learning-rate/training horizon changed: " + key)
    if type(config.get("stop_after_epoch")) is not int or config["stop_after_epoch"] != job["stop"]:
        raise ValueError("stop_after_epoch differs from the authorized job")
    if "stop_after_epoch" in extension and extension["stop_after_epoch"] != config["stop_after_epoch"]:
        raise ValueError("launcher and trainer stop_after_epoch values disagree")
    if config.get("physical_gpus") != job["gpus"]:
        raise ValueError("physical GPU pair differs from the authorized job")
    if config.get("gpu_uuids") != [GPU_UUID_BY_INDEX[i] for i in job["gpus"]]:
        raise ValueError("GPU UUIDs differ from the authorized physical pair")
    if kind == "pyramid" and config.get("variant") != "pyramid":
        raise ValueError("only the native24 Pyramid upgrade variant is authorized")
    if kind == "convnext" and config.get("variant") not in (None, "convnext"):
        raise ValueError("unexpected ordinary ConvNeXt variant")
    if config.get("auto_full") is not False or config.get("official_test_auto") is not False:
        raise ValueError("automatic evaluation must remain disabled")
    if config.get("updates_per_epoch") != 600 or config.get("world_size") != 2:
        raise ValueError("native24 update budget/rank partition changed")
    candidates = config.get("checkpoint_epochs", [])
    if candidates != job["candidate_epochs"]:
        raise ValueError("EMA candidate epochs differ from the authorized save list")
    output = Path(config.get("output_dir", "")).resolve()
    if "native24_retrain_20261015_v1" not in output.parts or output.name != kind:
        raise ValueError("output must be this isolated native24 rerun's model directory")
    # In particular, do not let historical storage overrides write old results.
    for key, expected in (("checkpoint_dir", output), ("candidate_dir", output / "candidates")):
        if key in config and Path(config[key]).resolve() != expected:
            raise ValueError("storage override escapes this isolated run: " + key)
    affinities = config.get("cpu_affinity")
    if (not isinstance(affinities, list) or len(affinities) != 2
            or any(not isinstance(cpus, list) or not cpus for cpus in affinities)
            or any(type(cpu) is not int or cpu < 0 for cpus in affinities for cpu in cpus)
            or set(affinities[0]) & set(affinities[1])):
        raise ValueError("two nonempty disjoint CPU affinity lists are required")
    for path in (wrapper_path, trainer_path):
        if path is not None and str(Path(path).resolve()) not in config.get("files", {}):
            raise ValueError("runtime source must be SHA-bound: " + str(path))
    return kind


def replace_once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f"historical training source anchor changed: expected 1, found {count}: {old!r}")
    return source.replace(old, new, 1)


def patch_training_loop_source(source):
    """Change only the terminal budget, terminal inventory and final receipt."""
    source = textwrap.dedent(source)
    source = replace_once(
        source,
        'target = (benchmark_updates or SMOKE_UPDATES) if smoke else config["epochs"] * config["updates_per_epoch"]',
        'target = (benchmark_updates or SMOKE_UPDATES) if smoke else config["stop_after_epoch"] * config["updates_per_epoch"]',
    )
    source = replace_once(
        source,
        'for epoch_number in config["checkpoint_epochs"]:',
        'for epoch_number in (e for e in config["checkpoint_epochs"] if e <= config["stop_after_epoch"]):',
    )
    source = replace_once(
        source,
        'engine.atomic_json(output / "completed.json", receipt)',
        '_native24_verify_terminal(config, receipt, model, identity, frozen_hash)\n'
        '        engine.atomic_json(output / "completed.json", receipt)',
    )
    ast.parse(source)
    return source


def verify_terminal(module, config, receipt, model, identity, frozen_hash):
    """Read the written files and check final EMA and full recovery agree."""
    target = config["stop_after_epoch"] * config["updates_per_epoch"]
    output = Path(config["output_dir"])
    recovery_path = output / "last.pt"
    candidate_path = output / "candidates" / f'epoch{config["stop_after_epoch"]:03d}_ema_delta.pt'
    if receipt.get("updates") != target:
        raise ValueError("terminal receipt does not match the requested stop update")
    # mmap reads the on-disk metadata without allocating another multi-GB copy;
    # state_hash below also verifies the actual serialized raw and EMA tensors.
    recovery = module.torch.load(recovery_path, map_location="cpu", weights_only=False, mmap=True)
    module.validate_recovery(recovery, model, config, identity, frozen_hash)
    if recovery.get("updates") != target:
        raise ValueError("final recovery checkpoint is not at the stop update")
    if not recovery.get("optimizer", {}).get("state"):
        raise ValueError("final recovery checkpoint lacks optimizer state")
    if module.state_hash(recovery["model"]) != receipt["raw_sha256"]:
        raise ValueError("final recovery raw weights differ from the completed training state")
    if module.state_hash(recovery["ema"]) != receipt["ema_sha256"]:
        raise ValueError("final recovery EMA differs from the completed training state")
    del recovery
    candidate = module.torch.load(candidate_path, map_location="cpu", weights_only=False, mmap=True)
    if (candidate.get("updates") != target
            or candidate.get("epoch") != config["stop_after_epoch"]
            or candidate.get("config_sha256") != identity
            or candidate.get("role") != "candidate_ema_delta"
            or candidate.get("frozen_hash") != frozen_hash
            or candidate.get("pretraining_identity") != module.pretraining_identity(config)
            or module.state_hash(candidate["state_dict"]) != receipt["ema_sha256"]):
        raise ValueError("final candidate does not match the full recovery/EMA at the stop update")
    del candidate
    receipt.update(
        status="training_completed_no_evaluation",
        native24_retrain=config["native24_retrain"],
        stop_after_epoch=config["stop_after_epoch"],
        original_epochs=config["epochs"],
        scheduler_epochs=config["scheduler_epochs"],
        recovery_checkpoint=str(recovery_path),
        recovery_bytes=recovery_path.stat().st_size,
        final_candidate=str(candidate_path),
        final_candidate_bytes=candidate_path.stat().st_size,
        final_candidate_recovery_same_update=True,
        final_recovery_raw_ema_verified=True,
        checkpoint_reserve_bytes=0,
        evaluation_started=False,
    )


def install_zero_checkpoint_reserve(engine):
    """Remove only the discretionary reserve; retain the original atomic writer."""
    original = engine.atomic_checkpoint

    def atomic_checkpoint(path, payload, *, reserve_bytes=0):
        # The original writer still checks tensor bytes + 64 MiB, validates a
        # reloaded temporary, fsyncs, and atomically replaces the owned target.
        return original(path, payload, reserve_bytes=0)

    engine.atomic_checkpoint = atomic_checkpoint


def install_runtime(config):
    kind = validate_contract(config)
    module = importlib.import_module(JOBS[kind]["module"])
    validate_contract(config, wrapper_path=__file__, trainer_path=module.__file__)
    if getattr(module, "_native24_limited_installed", False):
        raise RuntimeError("limited native24 runtime must be installed once per process")
    module.FIXED = dict(module.FIXED, checkpoint_epochs=JOBS[kind]["candidate_epochs"])
    if kind == "convnext":
        module.FIXED = dict(module.FIXED, physical_gpus=JOBS[kind]["gpus"])
        module.GPU_UUIDS = tuple(config["gpu_uuids"])
        # Ordinary ConvNeXt's historical physical-device audit hard-codes 6/7.
        physical_source = textwrap.dedent(inspect.getsource(module.verify_physical_devices))
        physical_source = replace_once(physical_source, '(6,7)', 'config["physical_gpus"]')
        exec(compile(physical_source, module.__file__ + ":native24_devices", "exec"), module.__dict__)
    else:
        module.VARIANT_GPUS = {"pyramid": JOBS[kind]["gpus"]}
        module.GPU_UUID_BY_INDEX = dict(GPU_UUID_BY_INDEX)
    original_validate = module.validate_config
    original_recovery_validate = module.validate_recovery

    def validate(config, *, devices=True):
        validate_contract(config, wrapper_path=__file__, trainer_path=module.__file__)
        return original_validate(config, devices=devices)

    def validate_recovery(payload, model, config, identity, frozen_hash):
        original_recovery_validate(payload, model, config, identity, frozen_hash)
        if payload["updates"] > config["stop_after_epoch"] * config["updates_per_epoch"]:
            raise ValueError("recovery checkpoint is beyond the authorized stop update")

    module.validate_config = validate
    module.validate_recovery = validate_recovery
    module._native24_verify_terminal = lambda c, r, m, i, f: verify_terminal(module, c, r, m, i, f)
    install_zero_checkpoint_reserve(module.engine)
    source = patch_training_loop_source(inspect.getsource(module._run_initialized))
    exec(compile(source, module.__file__ + ":native24_limited", "exec"), module.__dict__)
    module._native24_limited_installed = True
    module.validate_config(config, devices=False)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", nargs="?", const="last")
    parser.add_argument("--benchmark-updates", type=int)
    parser.add_argument("--smoke-output", default="smoke.json")
    parser.add_argument("--smoke-micro-batch", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    module = install_runtime(config)
    if args.validate_only:
        module.verify_files(config)
        module.verify_physical_devices(config)
        print(json.dumps(dict(status="validated", model_kind=config["native24_retrain"]["model_kind"],
                              stop_after_epoch=config["stop_after_epoch"],
                              original_epochs=config["epochs"], scheduler_epochs=config["scheduler_epochs"])))
        return
    module.run(config, smoke=args.smoke, resume=args.resume,
               benchmark_updates=args.benchmark_updates, smoke_output=args.smoke_output,
               smoke_micro_batch=args.smoke_micro_batch)


if __name__ == "__main__":
    main()
