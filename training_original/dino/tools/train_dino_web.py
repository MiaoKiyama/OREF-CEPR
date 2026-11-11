"""DINO WEB construction/provenance hooks over the unchanged mature trainer.

CLI: python tools/train_dino_web.py --config resolved.yaml
All optimization, AMP/FP32 loss, EMA, update budgets, signal handling, and
rolling recovery remain in train.py. Importing this module changes no hooks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train as legacy  # noqa: E402
from tcmmsline3.dino_web_data import (  # noqa: E402
    audit_dino_web_data_identity,
    build_dino_web_dataloader,
)
from tcmmsline3.dino_web_protocol import (  # noqa: E402
    RUNTIME_BINDING_KEY,
    build_dino_web_source_manifest,
    load_dino_web_config,
    resolve_dino_web_runtime_binding,
    validate_dino_web_checkpoint_runtime_binding,
)
from tcmmsline3.runtime import atomic_json_dump  # noqa: E402

_RUNTIME_BINDING: dict[str, Any] | None = None
_CONFIG: dict[str, Any] | None = None
_ORIGINALS = {
    name: getattr(legacy, name)
    for name in (
        "load_config",
        "build_model",
        "build_dataloader",
        "_build_optimizer_and_scheduler",
        "_checkpoint_data_provenance",
        "_validate_resume_checkpoint_role",
        "_validate_existing_successful_budget_completion",
        "_atomic_save_rolling_recovery",
        "atomic_torch_save",
    )
}
_LAST_SAVE: tuple[Path, tuple[Any, ...], tuple[Any, ...]] | None = None


def load_bound_dino_web_config(path: str | Path) -> dict[str, Any]:
    global _RUNTIME_BINDING, _CONFIG
    config = load_dino_web_config(path, require_pretrained_file=True)
    audit = audit_dino_web_data_identity(config)
    binding = resolve_dino_web_runtime_binding(
        config, project_root=PROJECT_ROOT, data_identity=audit
    )
    output = Path(str(config["experiment"]["output_dir"])).resolve()
    previous = output / "runtime_binding.json"
    if (
        previous.is_file()
        and json.loads(previous.read_text(encoding="utf-8")) != binding
    ):
        raise RuntimeError(
            "DINO WEB output already belongs to a different source/config/pretrain/split/mode"
        )
    _CONFIG, _RUNTIME_BINDING = config, binding
    atomic_json_dump(audit, output / "data_identity.json")
    atomic_json_dump(
        build_dino_web_source_manifest(PROJECT_ROOT), output / "source_manifest.json"
    )
    atomic_json_dump(binding, previous)
    threads = config.get("pipeline", {}).get("torch_threads", 2)
    if type(threads) is not int or threads < 1:
        raise ValueError("pipeline.torch_threads must be a positive integer")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return config


def _required_binding() -> dict[str, Any]:
    if _RUNTIME_BINDING is None:
        raise RuntimeError("DINO WEB runtime binding has not been initialized")
    return copy.deepcopy(_RUNTIME_BINDING)


def checkpoint_data_provenance_with_runtime_binding(config: dict) -> dict[str, Any]:
    provenance = _ORIGINALS["_checkpoint_data_provenance"](config)
    if not isinstance(provenance, dict) or RUNTIME_BINDING_KEY in provenance:
        raise RuntimeError("DINO WEB requires an unbound FBP data provenance mapping")
    provenance[RUNTIME_BINDING_KEY] = _required_binding()
    return provenance


def validate_resume_checkpoint_role_and_runtime(checkpoint: Mapping[str, Any]) -> None:
    _ORIGINALS["_validate_resume_checkpoint_role"](checkpoint)
    validate_dino_web_checkpoint_runtime_binding(checkpoint, _required_binding())


def validate_existing_budget_completion_and_runtime(
    config: Mapping[str, Any], output: Path
) -> dict | None:
    completion = _ORIGINALS["_validate_existing_successful_budget_completion"](
        config, output
    )
    if completion is not None:
        validate_dino_web_checkpoint_runtime_binding(completion, _required_binding())
    return completion


def build_dino_web_model(config: Mapping[str, Any]) -> torch.nn.Module:
    from tcmmsline3.dino_web_model import build_dino_web_model as build

    return build(config)


def build_dino_web_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    *,
    staged_parameters: list[torch.nn.Parameter],
) -> tuple:
    from tcmmsline3.dino_web_model import dino_web_parameter_groups

    groups = dino_web_parameter_groups(
        model, config, staged_parameters=staged_parameters
    )
    grouped = {id(parameter) for group in groups for parameter in group["params"]}
    if not {id(parameter) for parameter in staged_parameters}.issubset(grouped):
        raise RuntimeError("DINO WEB optimizer omitted staged parameters")
    if config["model"]["variant"] == "v2l" and staged_parameters:
        raise ValueError("DINOv2 WEB full fine-tuning does not use staged unfreezing")
    if config["model"]["variant"] != "v2l" and not staged_parameters:
        raise ValueError(
            "DINOv3 WEB requires optimizer-bound upper-half staged parameters"
        )
    optimizer_groups = [
        {key: value for key, value in group.items() if key != "parameter_names"}
        for group in groups
    ]
    train = config["train"]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]),
    )
    samples = (
        config["data"].get("debug_samples_per_epoch")
        or config["data"]["samples_per_epoch"]
    )
    steps_per_epoch = math.ceil(
        (int(samples) // int(train["batch_size"]))
        / int(train["gradient_accumulation_steps"])
    )
    horizon = int(train["scheduler_total_successful_optimizer_updates"])
    scheduler = legacy._scheduler(
        optimizer,
        horizon,
        int(train["warmup_steps"]),
        minimum_lr_ratio=float(train.get("minimum_lr_ratio", 0.0)),
    )
    return optimizer, scheduler, steps_per_epoch, horizon


def _free_space_guard(config: Mapping[str, Any], checkpoint_dir: Path) -> None:
    configured = Path(str(config["experiment"]["output_dir"])).resolve() / "checkpoints"
    if checkpoint_dir.resolve() != configured:
        raise RuntimeError(
            "DINO WEB recovery write escaped the run's checkpoints directory"
        )
    minimum = float(config.get("pipeline", {}).get("minimum_free_gib", 0.0))
    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError("pipeline.minimum_free_gib must be finite and nonnegative")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(checkpoint_dir).free < minimum * (1024**3):
        raise RuntimeError(
            "DINO WEB checkpoint paused: insufficient free disk; existing recovery remains intact"
        )


def atomic_save_rolling_recovery(
    state: Mapping[str, Any], *, checkpoint_dir: Path, config: Mapping[str, Any]
) -> None:
    _free_space_guard(config, checkpoint_dir)
    _ORIGINALS["_atomic_save_rolling_recovery"](
        state, checkpoint_dir=checkpoint_dir, config=config
    )


def _state_snapshot(value: Any) -> tuple[Any, ...]:
    """Content evidence for *every* tensor, including optimizer/EMA/RNG.

    Version counters cannot detect a ``tensor.data`` write. Hashing the full
    payload avoids that hole without serializing another multi-GiB best.pt.
    At most one tensor's CPU representation is live during each hash.
    """
    if torch.is_tensor(value):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return (
            "tensor",
            id(value),
            value._version,
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
            value.data_ptr(),
            hashlib.sha256(raw).hexdigest(),
        )
    if isinstance(value, np.ndarray):
        return (
            "numpy",
            str(value.dtype),
            tuple(value.shape),
            hashlib.sha256(value.tobytes()).hexdigest(),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (_state_snapshot(key), _state_snapshot(item))
                for key, item in sorted(
                    value.items(), key=lambda row: (type(row[0]).__name__, repr(row[0]))
                )
            ),
            _state_snapshot(getattr(value, "_metadata", None)),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_state_snapshot(item) for item in value))
    if value is None or isinstance(value, (str, int, float, bool, bytes, np.generic)):
        return (type(value).__name__, repr(value))
    # Unknown objects disable dedup rather than inferring their equality.
    raise TypeError(f"unrecognized checkpoint metadata type: {type(value).__name__}")


def _file_token(path: Path) -> tuple[Any, ...]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def atomic_torch_save_with_identical_best_link(state: Any, path: str | Path) -> None:
    """Deduplicate only an unchanged full recovery just written as last.pt.

    Atomic replacement of rolling last.pt leaves best.pt's old inode intact.
    EMA-only payloads and reconciliation without same-state evidence follow
    the original writer. No older run or checkpoint path can be deduplicated.
    """
    global _LAST_SAVE
    destination = Path(path).resolve()
    checkpoint_dir = (
        Path(str(_CONFIG["experiment"]["output_dir"])).resolve() / "checkpoints"
        if _CONFIG is not None
        else None
    )
    eligible = (
        checkpoint_dir is not None
        and destination.parent == checkpoint_dir
        and isinstance(state, Mapping)
        and state.get("selection_metric") == "last_epoch"
        and all(
            isinstance(state.get(key), Mapping)
            for key in ("model", "training_model", "optimizer", "rng_state")
        )
    )
    if eligible and destination.name == "best.pt" and _LAST_SAVE is not None:
        last, snapshot, file_token = _LAST_SAVE
        try:
            identical = (
                last == checkpoint_dir / "last.pt"
                and _file_token(last) == file_token
                and _state_snapshot(state) == snapshot
            )
        except (OSError, TypeError):
            identical = False
        if identical:
            temporary = destination.with_name(f".best.dino_web.{os.getpid()}.tmp")
            try:
                os.link(last, temporary)
                os.replace(temporary, destination)
                # Match the mature atomic writer's directory durability where
                # supported, without ever opening a checkpoint for mutation.
                if os.name != "nt":
                    descriptor = os.open(
                        destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    )
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                return
            except OSError:
                # A filesystem without hardlinks remains correct, albeit
                # with the original disk requirement.
                pass
            finally:
                temporary.unlink(missing_ok=True)
    _ORIGINALS["atomic_torch_save"](state, destination)
    if eligible and destination.name == "last.pt":
        try:
            _LAST_SAVE = (destination, _state_snapshot(state), _file_token(destination))
        except TypeError:
            _LAST_SAVE = None


@contextmanager
def installed_training_hooks() -> Iterator[None]:
    hooks = {
        "load_config": load_bound_dino_web_config,
        "build_model": build_dino_web_model,
        "build_dataloader": build_dino_web_dataloader,
        "_build_optimizer_and_scheduler": build_dino_web_optimizer_and_scheduler,
        "_checkpoint_data_provenance": checkpoint_data_provenance_with_runtime_binding,
        "_validate_resume_checkpoint_role": validate_resume_checkpoint_role_and_runtime,
        "_validate_existing_successful_budget_completion": validate_existing_budget_completion_and_runtime,
        "_atomic_save_rolling_recovery": atomic_save_rolling_recovery,
        "atomic_torch_save": atomic_torch_save_with_identical_best_link,
    }
    previous = {name: getattr(legacy, name) for name in hooks}
    try:
        for name, hook in hooks.items():
            setattr(legacy, name, hook)
        yield
    finally:
        for name, hook in previous.items():
            setattr(legacy, name, hook)


def _write_memory_receipt(path: str | Path) -> None:
    if not torch.cuda.is_available() or _CONFIG is None:
        raise RuntimeError("DINO WEB memory probe requires an initialized CUDA run")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    reserved, peak = (
        int(torch.cuda.memory_reserved()),
        int(torch.cuda.max_memory_reserved()),
    )
    external = max(int(total) - int(free) - reserved, 0)
    completion = json.loads(
        (Path(_CONFIG["experiment"]["output_dir"]) / "completed.json").read_text(
            encoding="utf-8"
        )
    )
    atomic_json_dump(
        {
            "status": "passed",
            "role": "dino_web_real_adamw_batch_probe",
            "variant": _CONFIG["model"]["variant"],
            "runtime_binding": _required_binding(),
            "physical_batch": int(_CONFIG["train"]["batch_size"]),
            "gradient_accumulation_steps": int(
                _CONFIG["train"]["gradient_accumulation_steps"]
            ),
            "effective_crops_per_update": 64,
            "successful_optimizer_updates": int(
                completion["terminal_successful_updates"]
            ),
            "optimizer_attempt_count": int(completion["optimizer_attempt_count"]),
            "amp_skips": int(completion["skipped_optimizer_attempt_count"]),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": peak,
            "current_memory_reserved_bytes": reserved,
            "estimated_driver_or_external_bytes": external,
            "device_total_bytes": int(total),
            "estimated_peak_headroom_bytes": max(int(total) - external - peak, 0),
        },
        path,
    )


def main() -> None:
    memory = os.environ.get("DINOWEB_MEMORY_RECEIPT")
    if memory and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with installed_training_hooks():
        legacy.main()
    if memory:
        _write_memory_receipt(memory)


if __name__ == "__main__":
    main()
