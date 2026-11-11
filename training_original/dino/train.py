from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from tcmmsline3.config import load_config
from tcmmsline3.data import build_dataloader, read_fbp_split
from tcmmsline3.losses import (
    grouped_segmentation_loss,
    native_sar_targets,
    segmentation_loss,
)
from tcmmsline3.metrics import ConfusionMetrics
from tcmmsline3.model import MSGuidedRegionFusion, SymmSafeFusion, build_model
from tcmmsline3.ms75_checkpoint import (
    load_fbponlyms_complete_anchor,
    load_fbponlyms_local_anchor,
    sha256_file,
)
from tcmmsline3.ms75_paired_protocol import verify_runtime_pair_binding
from tcmmsline3.runtime import (
    atomic_json_dump,
    atomic_torch_save,
    config_fingerprint,
    prepare_output,
    resolve_device,
    seed_everything,
)


RECOVERY_CHECKPOINT_VERSION = 1
PAIRED_CONTINUATION_CHECKPOINT_ROLE = "paired_continuation_training"
SUCCESSFUL_UPDATE_BUDGET_CHECKPOINT_ROLE = (
    "successful_optimizer_update_budget_training"
)
SUCCESSFUL_UPDATE_EMA_CHECKPOINT_ROLE = "ema_successful_optimizer_updates"
SUCCESSFUL_UPDATE_MODEL_CHECKPOINT_ROLE = "model_successful_optimizer_updates"


class _SuccessfulUpdateBudgetComplete(RuntimeError):
    """Internal control flow raised only after a durable terminal checkpoint."""


def _is_primary_successful_update(
    value: Mapping[str, Any], successful_update: int
) -> bool:
    primary = value.get("primary_successful_optimizer_update")
    return primary is not None and int(primary) == int(successful_update)


def _successful_optimizer_update_audit(
    *,
    successful_updates: int,
    optimizer_attempts: int | None = None,
    skipped_attempt_indices: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Describe every optimizer attempt for an ordinary strict-budget run.

    Unlike the historical paired-continuation audit, this schema stores the
    skip count explicitly.  The separate helper deliberately leaves the
    authenticated paired schema and its existing hashes byte-for-byte stable.
    """

    successful_updates = int(successful_updates)
    if successful_updates < 0:
        raise ValueError("successful optimizer updates must be non-negative")
    if optimizer_attempts is None:
        optimizer_attempts = successful_updates
    optimizer_attempts = int(optimizer_attempts)
    if optimizer_attempts < successful_updates:
        raise ValueError("optimizer attempts cannot be fewer than updates")
    skipped = [int(value) for value in (skipped_attempt_indices or [])]
    if skipped != sorted(set(skipped)):
        raise ValueError(
            "skipped optimizer attempts must be sorted and unique"
        )
    if any(value <= 0 or value > optimizer_attempts for value in skipped):
        raise ValueError("skipped optimizer attempt index is out of range")
    if optimizer_attempts - len(skipped) != successful_updates:
        raise ValueError(
            "optimizer attempts minus AMP skips must equal successful updates"
        )
    body: dict[str, Any] = {
        "schema_version": 1,
        "attempt_index_base": 1,
        "optimizer_attempt_count": optimizer_attempts,
        "successful_optimizer_update_count": successful_updates,
        "skipped_optimizer_attempt_count": len(skipped),
        "skipped_optimizer_attempt_indices": skipped,
    }
    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {**body, "update_mask_sha256": digest}


def _validate_successful_optimizer_update_audit(
    value: object, *, successful_updates: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(
            "successful-update checkpoint has no optimizer update audit"
        )
    try:
        expected = _successful_optimizer_update_audit(
            successful_updates=successful_updates,
            optimizer_attempts=int(value.get("optimizer_attempt_count", -1)),
            skipped_attempt_indices=list(
                value.get("skipped_optimizer_attempt_indices", [])
            ),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "successful-update optimizer update audit is malformed"
        ) from error
    if dict(value) != expected:
        raise RuntimeError(
            "successful-update optimizer update audit identity mismatch"
        )
    return expected


def _paired_optimizer_update_audit(
    *,
    successful_updates: int,
    optimizer_attempts: int | None = None,
    skipped_attempt_indices: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Content-address the exact AMP optimizer-update mask.

    The paired experiment stops after *successful* updates.  If GradScaler
    skips different batches in A0 and A1, equal terminal update counts alone
    no longer imply that both arms optimized on the same non-class examples.
    Persisting the one-based skipped-attempt indices closes that ambiguity and
    remains compact because skips are exceptional.
    """

    successful_updates = int(successful_updates)
    if successful_updates < 0:
        raise ValueError("successful optimizer updates must be non-negative")
    if optimizer_attempts is None:
        optimizer_attempts = successful_updates
    optimizer_attempts = int(optimizer_attempts)
    if optimizer_attempts < successful_updates:
        raise ValueError("optimizer attempts cannot be fewer than updates")
    skipped = [int(value) for value in (skipped_attempt_indices or [])]
    if skipped != sorted(set(skipped)):
        raise ValueError(
            "skipped optimizer attempts must be sorted and unique"
        )
    if any(value <= 0 or value > optimizer_attempts for value in skipped):
        raise ValueError("skipped optimizer attempt index is out of range")
    if optimizer_attempts - len(skipped) != successful_updates:
        raise ValueError(
            "optimizer attempts minus AMP skips must equal successful updates"
        )
    body: dict[str, Any] = {
        "schema_version": 1,
        "attempt_index_base": 1,
        "optimizer_attempt_count": optimizer_attempts,
        "successful_optimizer_update_count": successful_updates,
        "skipped_optimizer_attempt_indices": skipped,
    }
    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {**body, "update_mask_sha256": digest}


def _validate_paired_optimizer_update_audit(
    value: object, *, successful_updates: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(
            "paired continuation checkpoint has no optimizer update audit"
        )
    try:
        expected = _paired_optimizer_update_audit(
            successful_updates=successful_updates,
            optimizer_attempts=int(value.get("optimizer_attempt_count", -1)),
            skipped_attempt_indices=list(
                value.get("skipped_optimizer_attempt_indices", [])
            ),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "paired continuation optimizer update audit is malformed"
        ) from error
    if dict(value) != expected:
        raise RuntimeError(
            "paired continuation optimizer update audit identity mismatch"
        )
    return expected


def _capture_rng_state() -> dict[str, Any]:
    """Capture every process-local RNG that can affect a training update."""

    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        # Keep the checkpoint compatible with torch.load(weights_only=True):
        # a raw NumPy ndarray asks pickle to reconstruct a NumPy object, while
        # a CPU tensor plus primitives remains in the safe checkpoint subset.
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            # NumPy's MT19937 array is uint32.  PyTorch 2.4 can construct that
            # tensor but cannot serialize its storage, so an otherwise valid
            # recovery checkpoint fails on the production server.  Every
            # uint32 value is represented exactly by int64 and converted back
            # during restore; int64 is also accepted by weights_only loading.
            "state": torch.from_numpy(numpy_state[1].copy()).to(torch.int64),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a checkpoint RNG payload, rejecting incomplete new payloads."""

    required = {"python", "numpy", "torch_cpu"}
    missing = required - set(state)
    if missing:
        raise RuntimeError(
            f"recovery checkpoint RNG state is missing {sorted(missing)}"
        )
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "bit_generator",
        "state",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise RuntimeError("recovery checkpoint NumPy RNG state is malformed")
    numpy_values = (
        torch.as_tensor(numpy_state["state"], dtype=torch.int64)
        .cpu()
        .numpy()
        .astype(np.uint32, copy=False)
    )
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_values,
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(
        torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu()
    )
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA RNG state is present but CUDA is unavailable during resume"
            )
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from the recovery checkpoint"
            )
        torch.cuda.set_rng_state_all(
            [
                torch.as_tensor(value, dtype=torch.uint8).cpu()
                for value in cuda_state
            ]
        )


def _validated_epoch_progress(
    value: object,
    *,
    epoch: int,
    next_batch_index: int,
    num_classes: int,
    total_batches: int,
    accumulation_steps: int,
) -> dict[str, Any]:
    """Validate the metric accumulator embedded in a mid-epoch checkpoint."""

    if not isinstance(value, Mapping):
        raise RuntimeError("mid-epoch checkpoint is missing epoch_progress")
    if int(value.get("epoch", -1)) != int(epoch):
        raise RuntimeError("recovery epoch_progress belongs to another epoch")
    if int(value.get("next_batch_index", -1)) != int(next_batch_index):
        raise RuntimeError(
            "recovery batch position disagrees with epoch_progress"
        )
    stored_total_batches = int(value.get("total_batches", -1))
    if stored_total_batches != int(total_batches) or not (
        0 < int(next_batch_index) <= stored_total_batches
    ):
        raise RuntimeError(
            "recovery epoch length or batch position disagrees with the loader"
        )
    if (
        int(next_batch_index) % int(accumulation_steps) != 0
        and int(next_batch_index) != stored_total_batches
    ):
        raise RuntimeError(
            "recovery batch position is not an optimizer/accumulation boundary"
        )
    batches = int(value.get("batches", -1))
    if batches != int(next_batch_index):
        raise RuntimeError(
            "recovery accumulator must contain every completed batch exactly once"
        )
    totals = value.get("loss_totals")
    if not isinstance(totals, Mapping) or any(
        not isinstance(key, str) or not math.isfinite(float(item))
        for key, item in totals.items()
    ):
        raise RuntimeError("recovery loss_totals is malformed")
    confusion = np.asarray(value.get("confusion_matrix"), dtype=np.int64)
    if confusion.shape != (num_classes, num_classes) or bool(
        (confusion < 0).any()
    ):
        raise RuntimeError("recovery confusion_matrix is malformed")
    native_value = value.get("native_confusion_matrix")
    native_confusion = None
    if native_value is not None:
        native_confusion = np.asarray(native_value, dtype=np.int64)
        if native_confusion.shape != (num_classes, num_classes) or bool(
            (native_confusion < 0).any()
        ):
            raise RuntimeError("recovery native_confusion_matrix is malformed")
    return {
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "batches": batches,
        "total_batches": stored_total_batches,
        "loss_totals": {key: float(item) for key, item in totals.items()},
        "confusion_matrix": confusion,
        "native_confusion_matrix": native_confusion,
    }


def _validate_finite_checkpoint_payload(value: object) -> None:
    """Reject a recovery payload before it can replace the last good state."""

    def visit(item: object, path: str) -> None:
        if torch.is_tensor(item):
            if (item.is_floating_point() or item.is_complex()) and not bool(
                torch.isfinite(item.detach()).all().item()
            ):
                raise FloatingPointError(
                    f"checkpoint contains a non-finite tensor at {path}"
                )
            return
        if isinstance(item, np.ndarray):
            if np.issubdtype(item.dtype, np.inexact) and not bool(
                np.isfinite(item).all()
            ):
                raise FloatingPointError(
                    f"checkpoint contains a non-finite array at {path}"
                )
            return
        if isinstance(item, (float, np.floating)):
            if not math.isfinite(float(item)):
                raise FloatingPointError(
                    f"checkpoint contains a non-finite scalar at {path}"
                )
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "checkpoint")


def _paired_checkpoint_epoch_progress(
    value: Mapping[str, Any],
    *,
    local_epoch: int,
    effective_epoch: int,
) -> dict[str, Any]:
    """Translate phase/data epoch bookkeeping to resumable local epoch."""

    observed = int(value.get("epoch", -1))
    if observed != int(effective_epoch):
        raise RuntimeError(
            "paired continuation epoch progress disagrees with effective epoch"
        )
    result = dict(value)
    result["effective_epoch"] = int(effective_epoch)
    result["epoch"] = int(local_epoch)
    return result


def _load_model_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    strict: bool,
    exclude_prefixes: tuple[str, ...] = (),
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if exclude_prefixes:
        if strict:
            raise ValueError(
                "excluded checkpoint prefixes require strict=False"
            )
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith(exclude_prefixes)
        }
    incompatible = model.load_state_dict(state, strict=strict)
    if exclude_prefixes:
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(exclude_prefixes)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "warm-start checkpoint is incompatible outside excluded prefixes: "
                f"missing={invalid_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    if not strict and (
        incompatible.missing_keys or incompatible.unexpected_keys
    ):
        print(
            json.dumps(
                {
                    "checkpoint": str(path),
                    "excluded_prefixes": list(exclude_prefixes),
                    "missing_keys": incompatible.missing_keys,
                    "unexpected_keys": incompatible.unexpected_keys,
                }
            ),
            flush=True,
        )
    return checkpoint


def _model_tensor_digest(state: Mapping[str, Any]) -> str:
    """Hash canonical tensor names, metadata, and bytes in a model state."""

    if not state:
        raise ValueError("model state is empty")
    digest = hashlib.sha256()
    for name in sorted(str(key) for key in state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state value is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _seed_paired_ema_from_evaluated_base(
    ema_model: AveragedModel, model: torch.nn.Module
) -> None:
    """Keep the evaluated Base in EMA instead of dropping it at step one."""

    if int(ema_model.n_averaged.item()) != 0:
        raise RuntimeError(
            "fresh paired continuation EMA seed count must be zero"
        )
    model_state = model.state_dict()
    ema_state = ema_model.module.state_dict()
    if model_state.keys() != ema_state.keys() or any(
        not torch.equal(model_state[name], ema_state[name])
        for name in model_state
    ):
        raise RuntimeError(
            "fresh paired continuation EMA module differs from the loaded "
            "evaluation model"
        )
    # n_averaged==0 triggers an unconditional copy in AveragedModel. Mark the
    # exact evaluated checkpoint as the seed so update one already applies
    # ema_decay and does not discard the 71.322573% model.
    ema_model.n_averaged.copy_(torch.ones_like(ema_model.n_averaged))


def _paired_continuation_metadata_from_config(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Authenticate immutable pair/base inputs and return checkpoint metadata."""

    inference = config.get("inference", {})
    if (
        not isinstance(inference, Mapping)
        or str(inference.get("protocol_role", ""))
        != "paired_continuation_train"
    ):
        return None
    train = config.get("train")
    if not isinstance(train, Mapping):
        raise ValueError("paired continuation has no train mapping")
    base_path = Path(str(train["continuation_initialization_checkpoint"]))
    expected_base = str(
        train["continuation_initialization_expected_sha256"]
    ).lower()
    actual_base = sha256_file(base_path)
    if actual_base != expected_base:
        raise RuntimeError(
            "paired continuation base checkpoint SHA256 mismatch: "
            f"expected={expected_base}, actual={actual_base}"
        )
    manifest_path = Path(str(train["paired_protocol_manifest"]))
    expected_manifest = str(
        train["paired_protocol_manifest_expected_sha256"]
    ).lower()
    actual_manifest = sha256_file(manifest_path)
    if actual_manifest != expected_manifest:
        raise RuntimeError(
            "paired continuation protocol manifest SHA256 mismatch: "
            f"expected={expected_manifest}, actual={actual_manifest}"
        )
    pair_binding = verify_runtime_pair_binding(
        config=config,
        pair_manifest_path=manifest_path,
        expected_pair_file_sha256=expected_manifest,
        expected_base_checkpoint_sha256=actual_base,
    )
    return {
        "schema_version": 1,
        "checkpoint_role": PAIRED_CONTINUATION_CHECKPOINT_ROLE,
        "arm": str(train["paired_continuation_arm"]),
        "base_checkpoint_path": str(base_path),
        "base_checkpoint_sha256": actual_base,
        "paired_protocol_manifest_path": str(manifest_path),
        "paired_protocol_manifest_sha256": actual_manifest,
        "paired_protocol_manifest_inner_sha256": pair_binding[
            "pair_manifest_sha256"
        ],
        "source_revision": pair_binding["source_revision"],
        "paired_arm_config_fingerprint": pair_binding[
            "arm_config_fingerprint"
        ],
        "class_probability_manifest_path": pair_binding[
            "class_probability_manifest_path"
        ],
        "class_probability_manifest_file_sha256": pair_binding[
            "class_probability_manifest_file_sha256"
        ],
        "class_probability_manifest_sha256": pair_binding[
            "class_probability_manifest_sha256"
        ],
        "difficulty_manifest_sha256": pair_binding[
            "difficulty_manifest_sha256"
        ],
        "common_class_index_path": pair_binding["common_class_index_path"],
        "common_class_index_sha256": pair_binding["common_class_index_sha256"],
        "common_clean_confusion_index": pair_binding[
            "common_clean_confusion_index"
        ],
        "max_successful_optimizer_updates": int(
            train["max_successful_optimizer_updates"]
        ),
        "continuation_epoch_offset": int(train["continuation_epoch_offset"]),
    }


def _load_paired_continuation_initialization(
    model: torch.nn.Module,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Load exactly checkpoint['model']; never raw training/optimizer state."""

    metadata = _paired_continuation_metadata_from_config(config)
    if metadata is None:
        raise ValueError(
            "paired continuation initialization requires its role"
        )
    checkpoint = torch.load(
        metadata["base_checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            "paired continuation base checkpoint is not a mapping"
        )
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(
            "paired continuation base must explicitly contain checkpoint['model']"
        )
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError(
            "paired continuation model state contains non-tensors"
        )
    model.load_state_dict(state, strict=True)
    loaded_digest = _model_tensor_digest(model.state_dict())
    source_digest = _model_tensor_digest(state)
    if loaded_digest != source_digest:
        raise RuntimeError(
            "paired continuation strict load changed the base model tensors"
        )
    return {
        **metadata,
        "base_model_tensor_sha256": loaded_digest,
        "base_model_tensor_count": len(state),
        "source_state_key": "model",
        "optimizer_restored": False,
        "scheduler_restored": False,
        "scaler_restored": False,
        "ema_reinitialized_from_base": bool(
            config["train"].get("use_ema", False)  # type: ignore[index]
        ),
    }


def _load_sar_backbone_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
) -> dict:
    """Load only a standalone or fusion checkpoint's SAR backbone."""
    sar_backbone = getattr(model, "sar_backbone", None)
    if not isinstance(sar_backbone, torch.nn.Module):
        raise ValueError("target model does not expose a sar_backbone")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    prefix = "sar_backbone."
    sar_state = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not sar_state:
        raise ValueError(f"checkpoint {path} does not contain a sar_backbone")
    sar_backbone.load_state_dict(sar_state, strict=True)
    print(
        json.dumps(
            {
                "sar_initialization_checkpoint": str(path),
                "loaded_sar_backbone_tensors": len(sar_state),
            }
        ),
        flush=True,
    )
    return checkpoint


def _load_ms_anchor_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
) -> dict:
    """Load only the MS backbone and decoder from a Line2 anchor."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    prefixes = ("ms_backbone.", "decoder.")
    selected = {
        key: value for key, value in state.items() if key.startswith(prefixes)
    }
    if not selected:
        raise ValueError(
            f"checkpoint {path} does not contain an MS backbone and decoder"
        )
    target = model.state_dict()
    unexpected = sorted(set(selected) - set(target))
    mismatched = sorted(
        key
        for key, value in selected.items()
        if key in target and target[key].shape != value.shape
    )
    if unexpected or mismatched:
        raise ValueError(
            "MS anchor is incompatible with the Line3 model: "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    missing_prefixes = [
        prefix
        for prefix in prefixes
        if not any(key.startswith(prefix) for key in selected)
    ]
    if missing_prefixes:
        raise ValueError(
            f"MS anchor is missing required prefixes: {missing_prefixes}"
        )
    incompatible = model.load_state_dict(selected, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected MS initialization keys: {incompatible.unexpected_keys}"
        )
    print(
        json.dumps(
            {
                "ms_initialization_checkpoint": str(path),
                "loaded_ms_tensors": len(selected),
            }
        ),
        flush=True,
    )
    return checkpoint


def _resolve_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    value: object = model
    for part in str(name).split("."):
        if not isinstance(value, torch.nn.Module) or not hasattr(value, part):
            raise ValueError(f"unknown module path: {name}")
        value = getattr(value, part)
    if not isinstance(value, torch.nn.Module):
        raise ValueError(f"module path does not resolve to a module: {name}")
    return value


def _configure_trainable_modules(
    model: torch.nn.Module,
    module_names: list[str],
) -> None:
    """Freeze the complete model and enable only an explicit module allowlist."""
    if not module_names:
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module_name in module_names:
        module = _resolve_module(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def _set_fully_frozen_submodules_eval(module: torch.nn.Module) -> None:
    """Disable stochastic/dropout behavior in completely frozen subtrees."""

    for child in module.children():
        _set_fully_frozen_submodules_eval(child)
    parameters = list(module.parameters())
    if parameters and not any(
        parameter.requires_grad for parameter in parameters
    ):
        module.eval()


def _forward_model(model: torch.nn.Module, batch: dict) -> dict:
    sar = batch.get("sar")
    sar_valid = batch.get("sar_valid")
    if sar is None or sar_valid is None:
        # MS75 datasets are intentionally pure MS and never fabricate a SAR
        # sample at the data boundary.  Legacy model call compatibility is
        # handled here with an inert tensor that cannot carry information.
        height = max(int(batch["ms"].shape[-2]) // 4, 1)
        width = max(int(batch["ms"].shape[-1]) // 4, 1)
        sar = batch["ms"].new_zeros((batch["ms"].shape[0], 2, height, width))
        sar_valid = torch.zeros(
            (batch["ms"].shape[0], height, width),
            dtype=torch.bool,
            device=batch["ms"].device,
        )
    optional = {
        key: batch[key]
        for key in (
            "ms_short",
            "ms_short_valid",
            "ms_global",
            "ms_global_valid",
            "local_box_short",
            "local_box_global",
        )
        if key in batch
    }
    return model(
        batch["ms"],
        sar,
        sar_valid,
        time_gap_days=batch.get("time_gap_days"),
        ms_context=batch.get("ms_context"),
        ms_context_valid=batch.get("ms_context_valid"),
        **optional,
    )


def _build_teacher(
    config: dict, checkpoint_path: str, device: torch.device
) -> MSGuidedRegionFusion:
    teacher_config = copy.deepcopy(config)
    teacher_config["model"]["fusion_mode"] = "ms_only"
    teacher = MSGuidedRegionFusion.from_config(teacher_config)
    _load_model_checkpoint(teacher, checkpoint_path, strict=True)
    teacher.set_fusion_mode("ms_only")
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


class _MS75CompleteAnchorTeacher(torch.nn.Module):
    """Expose only an immutable authenticated CASTR anchor as a teacher."""

    def __init__(self, anchor_model: torch.nn.Module) -> None:
        super().__init__()
        if not hasattr(
            anchor_model, "forward_complete_anchor_from_multiscale_inputs"
        ):
            raise TypeError(
                "MS75 anchor model has no complete-anchor batch helper"
            )
        self.anchor_model = anchor_model

    def forward(
        self,
        ms: torch.Tensor,
        sar: torch.Tensor | None = None,
        sar_valid: torch.Tensor | None = None,
        time_gap_days: torch.Tensor | None = None,
        ms_context: torch.Tensor | None = None,
        ms_context_valid: torch.Tensor | None = None,
        ms_short: torch.Tensor | None = None,
        ms_short_valid: torch.Tensor | None = None,
        ms_global: torch.Tensor | None = None,
        ms_global_valid: torch.Tensor | None = None,
        local_box_short: torch.Tensor | None = None,
        local_box_global: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del sar, sar_valid, time_gap_days, local_box_short
        return (
            self.anchor_model.forward_complete_anchor_from_multiscale_inputs(
                ms,
                ms_context=ms_context,
                ms_context_valid=ms_context_valid,
                ms_short=ms_short,
                ms_short_valid=ms_short_valid,
                ms_global=ms_global,
                ms_global_valid=ms_global_valid,
                local_box_global=local_box_global,
            )
        )


def _build_ms75_fixed_anchor_teacher(
    config: dict, device: torch.device
) -> torch.nn.Module:
    """Rebuild the fixed teacher from the immutable 70.1564 source every run."""

    if str(config["model"].get("architecture")) != "ms75_multiscale":
        raise ValueError("fixed MS75 anchor teacher requires ms75_multiscale")
    checkpoint_path = config["train"].get("ms75_anchor_checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "fixed MS75 anchor teacher requires ms75_anchor_checkpoint"
        )
    anchor_model = build_model(config)
    report = load_fbponlyms_complete_anchor(
        anchor_model,
        checkpoint_path,
        expected_sha256=config["train"].get("ms75_anchor_destination_sha256"),
        expected_source_sha256=config["train"].get(
            "ms75_anchor_source_sha256"
        ),
    )
    if not bool(report.get("complete_anchor", False)):
        raise RuntimeError(
            "fixed MS75 teacher did not load the complete anchor"
        )
    teacher = _MS75CompleteAnchorTeacher(anchor_model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    print(
        json.dumps(
            {
                "fixed_anchor_teacher": str(checkpoint_path),
                "source_sha256": report["source_sha256"],
                "loaded_anchor_tensor_count": report[
                    "loaded_anchor_tensor_count"
                ],
            }
        ),
        flush=True,
    )
    return teacher


def _resolve_resume_checkpoint(
    config: dict,
    checkpoint_dir: Path,
    output: Path,
) -> str | None:
    """Resolve resume state before any one-time initialization is attempted."""

    requested = config["train"].get("resume_checkpoint")
    if requested:
        return str(requested)
    automatic_last = checkpoint_dir / "last.pt"
    automatic_previous = checkpoint_dir / "previous.pt"
    if (
        bool(config["train"].get("auto_resume", True))
        and not (output / "completed.json").is_file()
    ):
        if automatic_last.is_file():
            return str(automatic_last)
        if (
            bool(
                config["train"].get(
                    "retain_previous_recovery_checkpoint", False
                )
            )
            and automatic_previous.is_file()
        ):
            return str(automatic_previous)
    return None


def _validate_recovery_candidate_for_config(
    candidate: object, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not isinstance(candidate, Mapping):
        raise RuntimeError("resume checkpoint must be a mapping")
    _validate_resume_checkpoint_role(candidate)
    if candidate.get("config_fingerprint") != config_fingerprint(dict(config)):
        raise RuntimeError("resume checkpoint config fingerprint mismatch")
    if bool(config["train"].get("validate_checkpoint_finiteness", False)):
        _validate_finite_checkpoint_payload(candidate)
    if bool(
        config["train"].get("retain_previous_recovery_checkpoint", False)
    ):
        for key in ("model", "training_model", "optimizer", "scheduler", "scaler", "rng_state"):
            if not isinstance(candidate.get(key), Mapping):
                raise RuntimeError(
                    f"rolling recovery checkpoint lacks mapping state: {key}"
                )
        if not candidate["model"] or not candidate["training_model"]:
            raise RuntimeError("rolling recovery checkpoint lacks model tensors")
        if not candidate["optimizer"] or not candidate["scheduler"]:
            raise RuntimeError("rolling recovery checkpoint lacks optimizer state")
        if str(config["train"].get("amp_dtype", "float16")).lower() in {
            "bfloat16",
            "bf16",
        } and candidate["scaler"] != {}:
            raise RuntimeError("BF16 rolling recovery must have an empty scaler state")
        rng = candidate["rng_state"]
        if not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(rng):
            raise RuntimeError("rolling recovery checkpoint lacks complete RNG state")
        for key in (
            "epoch",
            "epoch_complete",
            "next_batch_index",
            "global_optimizer_step",
            "ema_n_averaged",
        ):
            if key not in candidate:
                raise RuntimeError(
                    f"rolling recovery checkpoint lacks required field: {key}"
                )
        if not bool(candidate["epoch_complete"]):
            if int(candidate.get("recovery_checkpoint_version", -1)) != (
                RECOVERY_CHECKPOINT_VERSION
            ) or not isinstance(candidate.get("epoch_progress"), Mapping):
                raise RuntimeError(
                    "mid-epoch rolling recovery lacks resumable progress"
                )
    target = _ordinary_successful_update_budget(config)
    if target is not None:
        _validate_successful_update_budget_resume(candidate, target=target)
    return candidate


def _load_checkpoint_cpu_mmap(path: str | Path) -> object:
    """Load through mmap where supported, including PyTorch 2.1's string rule."""

    try:
        return torch.load(
            str(path), map_location="cpu", weights_only=False, mmap=True
        )
    except (TypeError, ValueError):
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _atomic_save_rolling_recovery(
    state: Mapping[str, Any], *, checkpoint_dir: Path, config: Mapping[str, Any]
) -> None:
    """Keep the preceding durable recovery inode before replacing ``last.pt``."""

    last = checkpoint_dir / "last.pt"
    if bool(
        config["train"].get("retain_previous_recovery_checkpoint", False)
    ) and last.is_file():
        try:
            old_last = _load_checkpoint_cpu_mmap(last)
            _validate_recovery_candidate_for_config(old_last, config)
        except Exception as error:
            print(
                json.dumps(
                    {
                        "event": "rolling_recovery_rotation_skipped",
                        "checkpoint": str(last),
                        "error": repr(error),
                    }
                ),
                flush=True,
            )
        else:
            previous = checkpoint_dir / "previous.pt"
            temporary = checkpoint_dir / f".previous.pt.{os.getpid()}.tmp"
            temporary.unlink(missing_ok=True)
            try:
                os.link(last, temporary)
                os.replace(temporary, previous)
                directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                try:
                    descriptor = os.open(checkpoint_dir, directory_flags)
                except OSError:
                    descriptor = None
                if descriptor is not None:
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            finally:
                temporary.unlink(missing_ok=True)
    atomic_torch_save(dict(state), last)
    if bool(config["train"].get("validate_checkpoint_finiteness", False)):
        written = _load_checkpoint_cpu_mmap(last)
        _validate_recovery_candidate_for_config(written, config)


def _ordinary_successful_update_budget(
    config: Mapping[str, Any],
) -> int | None:
    """Return the strict budget for a non-paired training configuration."""

    train = config.get("train")
    if not isinstance(train, Mapping):
        return None
    value = train.get("max_successful_optimizer_updates")
    if value is None:
        return None
    inference = config.get("inference", {})
    if (
        isinstance(inference, Mapping)
        and str(inference.get("protocol_role", ""))
        == "paired_continuation_train"
    ):
        return None
    target = int(value)
    if target <= 0:
        raise ValueError(
            "train.max_successful_optimizer_updates must be positive"
        )
    return target


def _validate_existing_paired_completion(
    config: Mapping[str, Any],
    output: Path,
) -> dict[str, Any] | None:
    """Verify an already-complete pair arm before any output files are touched."""

    inference = config.get("inference", {})
    if (
        not isinstance(inference, Mapping)
        or str(inference.get("protocol_role", ""))
        != "paired_continuation_train"
    ):
        return None
    completion_path = output / "completed.json"
    if not completion_path.is_file():
        return None
    expected = _paired_continuation_metadata_from_config(config)
    assert expected is not None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "paired continuation completed.json is unreadable"
        ) from error
    if not isinstance(completion, Mapping):
        raise RuntimeError("paired continuation completion must be a mapping")
    required_equal = {
        "status": "completed",
        "checkpoint_role": PAIRED_CONTINUATION_CHECKPOINT_ROLE,
        "completion_kind": "successful_optimizer_update_budget",
        "budget_complete": True,
        "terminal_successful_updates": expected[
            "max_successful_optimizer_updates"
        ],
        "arm": expected["arm"],
        "base_checkpoint_sha256": expected["base_checkpoint_sha256"],
        "paired_protocol_manifest_sha256": expected[
            "paired_protocol_manifest_sha256"
        ],
        "common_class_index_path": expected["common_class_index_path"],
        "common_class_index_sha256": expected["common_class_index_sha256"],
        "common_clean_confusion_index": expected[
            "common_clean_confusion_index"
        ],
        "config_fingerprint": config_fingerprint(dict(config)),
    }
    for key, value in required_equal.items():
        if completion.get(key) != value:
            raise RuntimeError(
                f"paired continuation completion {key} mismatch: "
                f"expected={value!r}, found={completion.get(key)!r}"
            )
    completion_update_audit = _validate_paired_optimizer_update_audit(
        completion.get("optimizer_update_audit"),
        successful_updates=int(expected["max_successful_optimizer_updates"]),
    )
    for name in ("last", "final"):
        path_value = completion.get(f"{name}_checkpoint")
        expected_sha = str(
            completion.get(f"{name}_checkpoint_sha256", "")
        ).lower()
        if not isinstance(path_value, str) or not path_value:
            raise RuntimeError(
                f"paired completion has no {name}_checkpoint path"
            )
        path = Path(path_value)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(
                f"paired completion {name} checkpoint is missing or changed"
            )
        try:
            checkpoint = torch.load(
                str(path),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:  # pragma: no cover - older supported PyTorch
            checkpoint = torch.load(
                str(path), map_location="cpu", weights_only=False
            )
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(
                f"paired completion {name} checkpoint is not a mapping"
            )
        _validate_paired_continuation_resume(checkpoint, expected)
        if checkpoint.get("config_fingerprint") != completion[
            "config_fingerprint"
        ] or not bool(checkpoint.get("budget_complete", False)):
            raise RuntimeError(
                f"paired completion {name} checkpoint identity mismatch"
            )
        if checkpoint.get("optimizer_update_audit") != (
            completion_update_audit
        ):
            raise RuntimeError(
                f"paired completion {name} optimizer update audit mismatch"
            )
    return dict(completion)


def _validate_existing_successful_budget_completion(
    config: Mapping[str, Any],
    output: Path,
) -> dict[str, Any] | None:
    """Validate an already-published ordinary strict-budget completion."""

    target = _ordinary_successful_update_budget(config)
    if target is None:
        return None
    completion_path = output / "completed.json"
    if not completion_path.is_file():
        return None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "successful-update completed.json is unreadable"
        ) from error
    if not isinstance(completion, Mapping):
        raise RuntimeError("successful-update completion must be a mapping")
    required_equal = {
        "status": "completed",
        "checkpoint_role": SUCCESSFUL_UPDATE_BUDGET_CHECKPOINT_ROLE,
        "completion_kind": "successful_optimizer_update_budget",
        "budget_complete": True,
        "target_successful_optimizer_updates": target,
        "terminal_successful_updates": target,
        "config_fingerprint": config_fingerprint(dict(config)),
    }
    for key, expected in required_equal.items():
        if completion.get(key) != expected:
            raise RuntimeError(
                f"successful-update completion {key} mismatch: "
                f"expected={expected!r}, found={completion.get(key)!r}"
            )
    completion_audit = _validate_successful_optimizer_update_audit(
        completion.get("optimizer_update_audit"),
        successful_updates=target,
    )
    for name in ("last", "best", "evaluation"):
        path_value = completion.get(f"{name}_checkpoint")
        expected_sha = str(
            completion.get(f"{name}_checkpoint_sha256", "")
        ).lower()
        if not isinstance(path_value, str) or not path_value:
            raise RuntimeError(
                f"successful-update completion has no {name} checkpoint"
            )
        path = Path(path_value)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(
                f"successful-update completion {name} checkpoint is missing or changed"
            )
    last_path = Path(str(completion["last_checkpoint"]))
    try:
        last_state = torch.load(
            str(last_path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:  # pragma: no cover - older supported PyTorch
        last_state = torch.load(
            str(last_path), map_location="cpu", weights_only=False
        )
    if not isinstance(last_state, Mapping):
        raise RuntimeError(
            "successful-update last checkpoint is not a mapping"
        )
    validated_audit = _validate_successful_update_budget_resume(
        last_state, target=target
    )
    if validated_audit != completion_audit:
        raise RuntimeError(
            "successful-update completion and last checkpoint audits differ"
        )
    if not bool(last_state.get("budget_complete", False)):
        raise RuntimeError("successful-update last checkpoint is not terminal")
    return dict(completion)


def _validate_resume_checkpoint_role(checkpoint: Mapping[str, Any]) -> None:
    """Reject evaluation-only artifacts before optimizer restoration."""

    role = checkpoint.get("checkpoint_role")
    if role in {
        "ema_milestone",
        SUCCESSFUL_UPDATE_EMA_CHECKPOINT_ROLE,
        SUCCESSFUL_UPDATE_MODEL_CHECKPOINT_ROLE,
    }:
        raise RuntimeError(
            "evaluation-only checkpoints cannot resume "
            "training; use checkpoints/last.pt"
        )


def _validate_paired_continuation_resume(
    checkpoint: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the resumed state to belong to this exact arm and pair."""

    if (
        checkpoint.get("checkpoint_role")
        != PAIRED_CONTINUATION_CHECKPOINT_ROLE
    ):
        raise RuntimeError(
            "paired continuation can resume only its own training checkpoint"
        )
    stored = checkpoint.get("paired_continuation")
    if not isinstance(stored, Mapping):
        raise RuntimeError("paired continuation checkpoint has no provenance")
    required = (
        "arm",
        "base_checkpoint_sha256",
        "paired_protocol_manifest_sha256",
        "paired_protocol_manifest_inner_sha256",
        "paired_arm_config_fingerprint",
        "class_probability_manifest_file_sha256",
        "class_probability_manifest_sha256",
        "difficulty_manifest_sha256",
        "common_class_index_path",
        "common_class_index_sha256",
        "common_clean_confusion_index",
        "max_successful_optimizer_updates",
        "continuation_epoch_offset",
        "source_revision",
    )
    for key in required:
        if stored.get(key) != expected.get(key):
            raise RuntimeError(
                f"paired continuation resume {key} mismatch: "
                f"expected={expected.get(key)!r}, found={stored.get(key)!r}"
            )
    sampling_audits = checkpoint.get("sampling_plan_audits")
    if not isinstance(sampling_audits, list) or not sampling_audits:
        raise RuntimeError(
            "paired continuation checkpoint has no sampling-index audit"
        )
    expected_confusion_sha = expected["common_clean_confusion_index"][
        "index_file_sha256"
    ]
    for audit in sampling_audits:
        if (
            not isinstance(audit, Mapping)
            or audit.get("confusion_index_file_sha256")
            != expected_confusion_sha
        ):
            raise RuntimeError(
                "paired continuation checkpoint confusion-index audit mismatch"
            )
        if (
            audit.get("class_index_file_sha256")
            != expected["common_class_index_sha256"]
        ):
            raise RuntimeError(
                "paired continuation checkpoint class-index audit mismatch"
            )
    tensor_digest = str(stored.get("base_model_tensor_sha256", ""))
    if len(tensor_digest) != 64 or any(
        character not in "0123456789abcdef" for character in tensor_digest
    ):
        raise RuntimeError(
            "paired continuation checkpoint has no valid base model tensor digest"
        )
    if (
        int(stored.get("initial_ema_n_averaged", -1)) != 1
        or stored.get("ema_seed_is_evaluated_base") is not True
    ):
        raise RuntimeError(
            "paired continuation checkpoint did not seed EMA from the evaluated Base"
        )
    successful_updates = int(checkpoint.get("global_optimizer_step", -1))
    target = int(expected["max_successful_optimizer_updates"])
    if successful_updates < 0 or successful_updates > target:
        raise RuntimeError(
            "paired continuation checkpoint successful-update count is invalid"
        )
    _validate_paired_optimizer_update_audit(
        checkpoint.get("optimizer_update_audit"),
        successful_updates=successful_updates,
    )
    if int(checkpoint.get("ema_n_averaged", -1)) != successful_updates + 1:
        raise RuntimeError(
            "paired continuation checkpoint EMA count must equal the Base seed "
            "plus successful updates"
        )
    budget_complete = bool(checkpoint.get("budget_complete", False))
    terminal_updates = checkpoint.get("terminal_successful_updates")
    if budget_complete:
        if (
            successful_updates != target
            or int(terminal_updates or -1) != target
        ):
            raise RuntimeError(
                "paired continuation terminal checkpoint does not end exactly "
                "at its successful-update budget"
            )
    elif terminal_updates is not None:
        raise RuntimeError(
            "non-terminal paired checkpoint cannot declare terminal updates"
        )
    return dict(stored)


def _successful_update_ema_seed_offset(
    checkpoint: Mapping[str, Any],
) -> int:
    seeded = checkpoint.get("ema_seed_is_evaluated_initialization")
    if seeded is None:
        return 0
    if seeded is not True or checkpoint.get("initial_ema_n_averaged") != 1:
        raise RuntimeError(
            "successful-update EMA initialization seed contract is malformed"
        )
    provenance = checkpoint.get("fusionaware_a1_initialization")
    if not isinstance(provenance, Mapping):
        raise RuntimeError(
            "seeded FusionAware A1 checkpoint lacks initialization provenance"
        )
    return 1


def _restore_fusionaware_a1_resume_provenance(
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    output: Path,
) -> None:
    """Restore the authenticated fresh-run A0-EMA seed on A1 resume.

    The initializer is deliberately not executed again on resume. Its
    immutable provenance must nevertheless remain attached to ``model`` so
    every subsequent recovery checkpoint repeats the same seed identity.
    """

    if str(config["data"].get("protocol_stage", "")) != (
        "local_detail_cnxb256_fusionaware_a1_full120_10k"
    ):
        return
    if _successful_update_ema_seed_offset(checkpoint) != 1:
        raise RuntimeError("FusionAware A1 resume did not preserve its EMA seed")
    provenance = checkpoint.get("fusionaware_a1_initialization")
    assert isinstance(provenance, Mapping)
    expected_path = Path(
        str(config["train"]["fusionaware_a1_initialization_checkpoint"])
    ).resolve()
    expected_sha256 = str(
        config["train"]["fusionaware_a1_initialization_checkpoint_sha256"]
    )
    required = {
        "initialization_role": "fusionaware_a1_from_terminal_a0_ema",
        "checkpoint": str(expected_path),
        "checkpoint_sha256": expected_sha256,
        "state_container": "model",
        "strict_load": True,
        "initial_ema_n_averaged": 1,
        "ema_seed_is_evaluated_initialization": True,
    }
    changed = {
        key: {"expected": expected, "observed": provenance.get(key)}
        for key, expected in required.items()
        if provenance.get(key) != expected
    }
    seed_digest = str(provenance.get("initial_ema_model_tensor_sha256", ""))
    if len(seed_digest) != 64 or any(
        character not in "0123456789abcdef" for character in seed_digest
    ):
        changed["initial_ema_model_tensor_sha256"] = {
            "expected": "lowercase SHA256",
            "observed": seed_digest,
        }
    if changed:
        raise RuntimeError(
            "FusionAware A1 resume initialization provenance changed: "
            f"{changed}"
        )
    restored = dict(provenance)
    model.fusionaware_a1_initialization_provenance = restored
    atomic_json_dump(restored, output / "fusionaware_a1_initialization.json")


def _validate_successful_update_budget_resume(
    checkpoint: Mapping[str, Any], *, target: int
) -> dict[str, Any]:
    """Validate an ordinary strict-budget recovery or terminal checkpoint."""

    target = int(target)
    if target <= 0:
        raise ValueError("successful optimizer-update target must be positive")
    if checkpoint.get("optimizer_budget_checkpoint_role") != (
        SUCCESSFUL_UPDATE_BUDGET_CHECKPOINT_ROLE
    ):
        raise RuntimeError(
            "successful-update run can resume only its own budget checkpoint"
        )
    if int(checkpoint.get("successful_update_budget", -1)) != target:
        raise RuntimeError("successful-update resume target mismatch")
    successful_updates = int(checkpoint.get("global_optimizer_step", -1))
    if successful_updates < 0 or successful_updates > target:
        raise RuntimeError(
            "successful-update checkpoint update count is out of range"
        )
    audit = _validate_successful_optimizer_update_audit(
        checkpoint.get("optimizer_update_audit"),
        successful_updates=successful_updates,
    )
    audit_mirrors = {
        "optimizer_attempt_count": audit["optimizer_attempt_count"],
        "skipped_optimizer_attempt_count": audit[
            "skipped_optimizer_attempt_count"
        ],
        "skipped_optimizer_attempt_indices": audit[
            "skipped_optimizer_attempt_indices"
        ],
    }
    for key, expected in audit_mirrors.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(
                f"successful-update checkpoint {key} disagrees with its audit"
            )
    scheduler_state = checkpoint.get("scheduler")
    if (
        not isinstance(scheduler_state, Mapping)
        or int(scheduler_state.get("last_epoch", -1)) != successful_updates
    ):
        raise RuntimeError(
            "successful-update checkpoint scheduler does not match successful "
            "optimizer updates"
        )
    ema_count = checkpoint.get("ema_n_averaged")
    ema_seed_offset = _successful_update_ema_seed_offset(checkpoint)
    if ema_count is not None and int(ema_count) != (
        successful_updates + ema_seed_offset
    ):
        raise RuntimeError(
            "ordinary successful-update EMA count must equal successful updates "
            "plus its authenticated initialization seed"
        )
    budget_complete = bool(checkpoint.get("budget_complete", False))
    terminal_updates = checkpoint.get("terminal_successful_updates")
    if budget_complete:
        if (
            successful_updates != target
            or int(terminal_updates or -1) != target
        ):
            raise RuntimeError(
                "successful-update terminal checkpoint does not end exactly at "
                "its target"
            )
    elif successful_updates == target:
        raise RuntimeError(
            "successful-update checkpoint reached its target without a terminal marker"
        )
    elif terminal_updates is not None:
        raise RuntimeError(
            "non-terminal successful-update checkpoint declares terminal updates"
        )
    return audit


def _initialize_model_from_config(
    model: torch.nn.Module,
    config: dict,
    output: Path,
) -> dict[str, Any] | None:
    """Apply fresh-run initializers; resumed checkpoints are self-contained."""

    a1_initialization = config["train"].get(
        "fusionaware_a1_initialization_checkpoint"
    )
    if a1_initialization:
        if str(config["data"].get("protocol_stage", "")) != (
            "local_detail_cnxb256_fusionaware_a1_full120_10k"
        ) or str(config["model"].get("architecture", "")) != (
            "local_detail_cnxb"
        ):
            raise ValueError(
                "dedicated FusionAware A1 initialization used outside its stage"
            )
        conflicting = {
            key: config["train"].get(key)
            for key in (
                "initialization_checkpoint",
                "continuation_initialization_checkpoint",
                "ms_initialization_checkpoint",
                "sar_initialization_checkpoint",
                "ms75_anchor_checkpoint",
            )
            if config["train"].get(key)
        }
        if conflicting:
            raise ValueError(
                "FusionAware A1 warm-start conflicts with legacy initializers: "
                f"{conflicting}"
            )
        initialization_path = Path(str(a1_initialization)).resolve()
        if not initialization_path.is_file():
            raise FileNotFoundError(
                f"missing FusionAware A1 initialization: {initialization_path}"
            )
        expected_sha256 = str(
            config["train"].get(
                "fusionaware_a1_initialization_checkpoint_sha256", ""
            )
        )
        observed_sha256 = sha256_file(initialization_path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                "FusionAware A1 initialization checkpoint SHA256 mismatch: "
                f"expected={expected_sha256}, observed={observed_sha256}"
            )
        checkpoint = torch.load(
            initialization_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(
                "FusionAware A1 initialization checkpoint must be a mapping"
            )
        source_contract = {
            "checkpoint_schema_version": 1,
            "checkpoint_role": SUCCESSFUL_UPDATE_EMA_CHECKPOINT_ROLE,
            "completion_kind": "successful_optimizer_update_budget",
            "budget_complete": True,
            "target_successful_optimizer_updates": 23_999,
            "global_optimizer_step": 23_999,
            "ema_n_averaged": 23_999,
        }
        changed = {
            key: {"expected": expected, "observed": checkpoint.get(key)}
            for key, expected in source_contract.items()
            if checkpoint.get(key) != expected
        }
        if changed:
            raise RuntimeError(
                "FusionAware A1 source A0 EMA contract changed: "
                f"{changed}"
            )
        forbidden_training_state = {
            key
            for key in (
                "training_model",
                "optimizer",
                "scheduler",
                "scaler",
                "rng_state",
            )
            if key in checkpoint
        }
        if forbidden_training_state:
            raise RuntimeError(
                "FusionAware A1 requires compact EMA-only initialization; found "
                f"{sorted(forbidden_training_state)}"
            )
        state = checkpoint.get("model")
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError(
                "FusionAware A1 source checkpoint has no model state"
            )
        expected_state = model.state_dict()
        observed_schema = [
            {
                "name": str(name),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in state.items()
            if torch.is_tensor(value)
        ]
        expected_schema = [
            {
                "name": str(name),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in expected_state.items()
        ]
        if len(observed_schema) != len(state) or observed_schema != expected_schema:
            raise RuntimeError(
                "FusionAware A1 source model architecture/state schema mismatch"
            )
        model.load_state_dict(state, strict=True)
        architecture_schema_sha256 = hashlib.sha256(
            json.dumps(
                expected_schema,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        provenance = {
            "schema_version": 1,
            "initialization_role": "fusionaware_a1_from_terminal_a0_ema",
            "checkpoint": str(initialization_path),
            "checkpoint_sha256": observed_sha256,
            "source_checkpoint_role": checkpoint["checkpoint_role"],
            "source_successful_optimizer_updates": 23_999,
            "state_container": "model",
            "strict_load": True,
            "model_architecture": "local_detail_cnxb",
            "architecture_schema_sha256": architecture_schema_sha256,
            "loaded_tensor_count": len(expected_schema),
            "optimizer_restored": False,
            "scheduler_restored": False,
            "scaler_restored": False,
            "rng_restored": False,
        }
        model.fusionaware_a1_initialization_provenance = dict(provenance)
        atomic_json_dump(
            provenance, output / "fusionaware_a1_initialization.json"
        )
        print(
            json.dumps(
                {"event": "fusionaware_a1_initialized", **provenance}
            ),
            flush=True,
        )
        return None

    continuation_initialization = config["train"].get(
        "continuation_initialization_checkpoint"
    )
    continuation_provenance = None
    if continuation_initialization:
        continuation_provenance = _load_paired_continuation_initialization(
            model, config
        )
        atomic_json_dump(
            continuation_provenance,
            output / "paired_continuation_initialization.json",
        )
        print(
            json.dumps(
                {
                    "event": "paired_continuation_initialized",
                    **continuation_provenance,
                }
            ),
            flush=True,
        )
        # A strict paired continuation is a complete-model initialization.
        # Returning here is deliberate defense in depth: no legacy full,
        # anchor, MS-backbone, or SAR-backbone initializer may mutate the
        # authenticated evaluation state after this point.  The fixed 70.1564
        # teacher is constructed independently later in main().
        return continuation_provenance
    initialization = config["train"].get("initialization_checkpoint")
    if initialization:
        initialization_excludes = tuple(
            str(value)
            for value in config["train"].get(
                "initialization_exclude_prefixes", []
            )
        )
        _load_model_checkpoint(
            model,
            initialization,
            strict=not initialization_excludes,
            exclude_prefixes=initialization_excludes,
        )
    ms75_anchor = config["train"].get("ms75_anchor_checkpoint")
    # In a paired continuation the anchor path is retained solely to build the
    # independent fixed 70.1564 teacher below.  Injecting it into the freshly
    # loaded 71-model would silently clobber the experimental starting point.
    if ms75_anchor and continuation_initialization is None:
        anchor_report = load_fbponlyms_local_anchor(
            model,
            ms75_anchor,
            expected_sha256=config["train"].get(
                "ms75_anchor_destination_sha256"
            ),
            expected_source_sha256=config["train"].get(
                "ms75_anchor_source_sha256"
            ),
        )
        atomic_json_dump(
            anchor_report, output / "ms75_anchor_initialization.json"
        )
        print(
            json.dumps(
                {
                    "ms75_anchor_checkpoint": str(ms75_anchor),
                    "source_sha256": anchor_report["source_sha256"],
                    "original_source_sha256": anchor_report[
                        "original_source_sha256"
                    ],
                    "loaded_tensor_count": anchor_report[
                        "loaded_tensor_count"
                    ],
                    "ignored_source_tensor_count": anchor_report[
                        "ignored_source_tensor_count"
                    ],
                }
            ),
            flush=True,
        )
    ms_initialization = config["train"].get("ms_initialization_checkpoint")
    if ms_initialization:
        _load_ms_anchor_checkpoint(model, ms_initialization)
    sar_initialization = config["train"].get("sar_initialization_checkpoint")
    if sar_initialization:
        _load_sar_backbone_checkpoint(model, sar_initialization)
    return continuation_provenance


def _scaled_optimizer_step(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    *,
    minimum_scale: float = 0.0,
) -> bool:
    """Step through GradScaler and report whether parameters were updated.

    GradScaler lowers its scale when non-finite gradients make it skip
    ``optimizer.step``.  Scheduler and EMA state must advance only alongside a
    real optimizer update or resume will no longer represent the same
    trajectory.
    """

    minimum_scale = float(minimum_scale)
    if not math.isfinite(minimum_scale) or minimum_scale < 0.0:
        raise ValueError("minimum AMP scale must be finite and non-negative")
    scale_before = float(scaler.get_scale())
    if scaler.is_enabled() and (
        not math.isfinite(scale_before) or scale_before <= minimum_scale
    ):
        raise FloatingPointError(
            "AMP scale is non-finite or already at/below its fail-closed floor"
        )
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if scaler.is_enabled() and (
        not math.isfinite(scale_after) or scale_after <= minimum_scale
    ):
        raise FloatingPointError(
            "AMP scale became non-finite or reached its fail-closed floor"
        )
    return not scaler.is_enabled() or scale_after >= scale_before


def _amp_autocast_dtype(
    config: Mapping[str, Any], device: torch.device
) -> torch.dtype:
    """Resolve the explicit mixed-precision dtype without changing old runs."""

    value = str(config["train"].get("amp_dtype", "float16")).strip().lower()
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value not in aliases:
        raise ValueError("train.amp_dtype must be float16 or bfloat16")
    dtype = aliases[value]
    if (
        bool(config["train"].get("amp", False))
        and device.type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("configured CUDA device does not support bfloat16 AMP")
    return dtype


def _build_grad_scaler(
    config: Mapping[str, Any], device: torch.device
) -> torch.amp.GradScaler:
    """Construct GradScaler from the fully resolved AMP trajectory contract."""

    train = config["train"]
    amp_enabled = bool(
        train["amp"]
        and device.type == "cuda"
        and _amp_autocast_dtype(config, device) == torch.float16
    )
    return torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
        init_scale=float(train.get("amp_initial_scale", 65536.0)),
        growth_factor=float(train.get("amp_growth_factor", 2.0)),
        backoff_factor=float(train.get("amp_backoff_factor", 0.5)),
        growth_interval=int(train.get("amp_growth_interval", 2000)),
    )


@torch.no_grad()
def _project_model_constraints(model: torch.nn.Module) -> None:
    """Project explicitly bounded raw parameters after a real optimizer step.

    A boundary-aware gradient prevents a gate from accumulating outward debt
    while it is exactly saturated, but Adam can still overshoot a boundary
    from a small positive value in one update.  Modules may expose an in-place
    ``project_constrained_parameters_`` hook to close that final gap.  The hook
    runs before EMA so the training and evaluation states remain identical.
    """

    for module in model.modules():
        hook = getattr(module, "project_constrained_parameters_", None)
        if callable(hook):
            hook()


@torch.no_grad()
def _validate_module_finiteness(
    module: torch.nn.Module, *, label: str
) -> None:
    """Check all floating parameters/buffers with one synchronization per device."""

    aggregates: dict[torch.device, torch.Tensor] = {}
    floating: list[tuple[str, torch.Tensor]] = []
    for name, tensor in module.state_dict().items():
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        floating.append((name, tensor))
        finite = torch.isfinite(tensor.detach()).all()
        current = aggregates.get(tensor.device)
        aggregates[tensor.device] = finite if current is None else current & finite
    if all(bool(value.item()) for value in aggregates.values()):
        return
    for name, tensor in floating:
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise FloatingPointError(
                f"{label} contains a non-finite tensor after optimizer step: {name}"
            )
    raise FloatingPointError(f"{label} failed its finiteness audit")


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    config: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    scaler: torch.amp.GradScaler,
    teacher: torch.nn.Module | None,
    teacher_amp_enabled: bool = False,
    ema_model: AveragedModel | None = None,
    phase: str = "joint",
    epoch_index: int | None = None,
    checkpoint_epoch_index: int | None = None,
    progress_label: str | None = None,
    start_batch_index: int = 0,
    initial_epoch_progress: Mapping[str, Any] | None = None,
    resume_rng_state: Mapping[str, Any] | None = None,
    recovery_boundary_callback: (
        Callable[[dict[str, Any], bool], None] | None
    ) = None,
) -> tuple[dict, dict]:
    training = optimizer is not None
    model.train(training)
    if training:
        # Parameter freezing alone does not stop ConvNeXt stochastic depth.
        # FCR fine-tuning needs a deterministic frozen anchor, while staged
        # decoder modules that currently require gradients remain in train mode.
        _set_fully_frozen_submodules_eval(model)
    metrics = ConfusionMetrics(int(config["model"]["num_classes"]))
    native_metrics = (
        ConfusionMetrics(int(config["model"]["num_classes"]))
        if str(config["model"].get("architecture", "ms_sar_fusion"))
        == "sar_native"
        else None
    )
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    start_batch_index = int(start_batch_index)
    if start_batch_index < 0:
        raise ValueError("start_batch_index must be non-negative")
    if not training and (
        start_batch_index or initial_epoch_progress is not None
    ):
        raise ValueError("validation cannot resume from a training batch")
    amp_enabled = bool(config["train"]["amp"] and device.type == "cuda")
    autocast_dtype = _amp_autocast_dtype(config, device)
    strict_step_finiteness = bool(
        config["train"].get("validate_step_finiteness", False)
    )
    accumulation_steps = (
        int(config["train"].get("gradient_accumulation_steps", 1))
        if training
        else 1
    )
    # A resumed loader begins at ``start_batch_index`` through an offset
    # sampler, so its length contains only the remaining batches.
    total_batches = start_batch_index + len(loader)
    if start_batch_index > total_batches:
        raise RuntimeError("training recovery starts beyond the epoch")
    if initial_epoch_progress is not None:
        if epoch_index is None:
            raise ValueError("epoch_index is required for mid-epoch recovery")
        restored_progress = _validated_epoch_progress(
            initial_epoch_progress,
            epoch=(
                int(checkpoint_epoch_index)
                if checkpoint_epoch_index is not None
                else epoch_index
            ),
            next_batch_index=start_batch_index,
            num_classes=int(config["model"]["num_classes"]),
            total_batches=total_batches,
            accumulation_steps=accumulation_steps,
        )
        totals.update(restored_progress["loss_totals"])
        batches = int(restored_progress["batches"])
        metrics.confusion[:] = restored_progress["confusion_matrix"]
        restored_native = restored_progress["native_confusion_matrix"]
        if native_metrics is None and restored_native is not None:
            raise RuntimeError(
                "recovery checkpoint contains native metrics for a non-native model"
            )
        if native_metrics is not None:
            if restored_native is None:
                raise RuntimeError(
                    "recovery checkpoint is missing native metrics"
                )
            native_metrics.confusion[:] = restored_native
    if training:
        optimizer.zero_grad(set_to_none=True)
    group_size = accumulation_steps
    progress_interval = int(
        config["train"].get("progress_log_interval_batches", 0)
    )
    epoch_started = time.perf_counter()
    loss_config = config
    contrast_start = config["train"].get("family_contrast_start_epoch")
    if contrast_start is not None and epoch_index is not None:
        start = int(contrast_start)
        ramp_epochs = max(
            int(config["train"].get("family_contrast_ramp_epochs", 0)), 0
        )
        if epoch_index < start:
            multiplier = 0.0
        elif ramp_epochs > 0:
            multiplier = min(
                float(epoch_index - start + 1) / float(ramp_epochs), 1.0
            )
        else:
            multiplier = 1.0
        loss_config = copy.deepcopy(config)
        loss_config["loss"]["family_contrast_weight"] = (
            float(config["loss"].get("family_contrast_weight", 0.0))
            * multiplier
        )

    # Constructing a DataLoader iterator consumes a torch RNG value for worker
    # base seeds.  For a mid-epoch resume the original iterator was already
    # constructed, so restore the saved model RNG *after* creating this new
    # iterator.  Dataset samples themselves are deterministic by index.
    loader_iterator = iter(loader)
    if resume_rng_state is not None:
        _restore_rng_state(resume_rng_state)
    for batch_index, batch in enumerate(
        loader_iterator, start=start_batch_index
    ):
        batch = _move_batch(batch, device)
        optimizer_updated = False
        safe_optimizer_boundary = False
        gradient_norm = float("nan")
        loss_sar_valid = batch.get("sar_valid")
        if loss_sar_valid is None:
            loss_sar_valid = torch.zeros(
                (
                    batch["ms"].shape[0],
                    max(int(batch["ms"].shape[-2]) // 4, 1),
                    max(int(batch["ms"].shape[-1]) // 4, 1),
                ),
                dtype=torch.bool,
                device=device,
            )
        if training and batch_index % accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
            group_size = min(
                accumulation_steps,
                total_batches - batch_index,
            )
        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device.type,
                enabled=bool(
                    teacher is not None
                    and teacher_amp_enabled
                    and device.type == "cuda"
                ),
                dtype=autocast_dtype,
            ):
                teacher_logits = (
                    _forward_model(teacher, batch)["logits"]
                    if teacher is not None
                    else None
                )
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=autocast_dtype,
            ):
                outputs = _forward_model(model, batch)
                loss_microbatch_size = loss_config["train"].get(
                    "loss_microbatch_size"
                )
                if training and loss_microbatch_size is not None:
                    loss, components = grouped_segmentation_loss(
                        outputs,
                        batch["target"],
                        loss_sar_valid,
                        loss_config,
                        group_size=int(loss_microbatch_size),
                        teacher_logits=teacher_logits,
                        phase=phase,
                        sar_aux_eligible=batch.get("sar_aux_eligible"),
                        ms75_truth_probability=batch.get(
                            "ms75_truth_probability"
                        ),
                        local512_truth_probability=batch.get(
                            "local512_truth_probability"
                        ),
                        fusion_truth_probability_valid=batch.get(
                            "fusion_truth_probability_valid"
                        ),
                    )
                else:
                    loss, components = segmentation_loss(
                        outputs,
                        batch["target"],
                        loss_sar_valid,
                        loss_config,
                        teacher_logits=teacher_logits,
                        phase=phase,
                        sar_aux_eligible=batch.get("sar_aux_eligible"),
                        ms75_truth_probability=batch.get(
                            "ms75_truth_probability"
                        ),
                        local512_truth_probability=batch.get(
                            "local512_truth_probability"
                        ),
                        fusion_truth_probability_valid=batch.get(
                            "fusion_truth_probability_valid"
                        ),
                    )
            component_scalars: dict[str, float] = {}
            if strict_step_finiteness and not bool(
                torch.isfinite(outputs["logits"].detach()).all().item()
            ):
                if training:
                    optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "non-finite logits before loss backward: "
                    f"epoch={epoch_index} batch={batch_index}"
                )
            if strict_step_finiteness and not bool(
                torch.isfinite(loss.detach()).item()
            ):
                if training:
                    optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "non-finite loss before backward: "
                    f"epoch={epoch_index} batch={batch_index}"
                )
            for key, value in components.items():
                scalar = float(value.detach())
                if strict_step_finiteness and not math.isfinite(scalar):
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "non-finite loss component before backward: "
                        f"component={key} epoch={epoch_index} "
                        f"batch={batch_index}"
                    )
                component_scalars[key] = scalar
            if training:
                scaler.scale(loss / group_size).backward()
                should_step = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == total_batches
                )
                if should_step:
                    safe_optimizer_boundary = True
                    scaler.unscale_(optimizer)
                    gradient_norm = float(
                        clip_grad_norm_(
                            model.parameters(),
                            float(config["train"]["gradient_clip_norm"]),
                            error_if_nonfinite=(
                                strict_step_finiteness
                                and not scaler.is_enabled()
                            ),
                        )
                    )
                    if (
                        strict_step_finiteness
                        and not scaler.is_enabled()
                        and not math.isfinite(gradient_norm)
                    ):
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            "non-finite gradient norm before optimizer step: "
                            f"epoch={epoch_index} batch={batch_index}"
                        )
                    optimizer_updated = _scaled_optimizer_step(
                        scaler,
                        optimizer,
                        minimum_scale=float(
                            config["train"].get("amp_min_scale", 0.0)
                        ),
                    )
                    if optimizer_updated:
                        _project_model_constraints(model)
                        if bool(
                            config["train"].get(
                                "validate_step_finiteness", False
                            )
                        ):
                            _validate_module_finiteness(
                                model, label="training model"
                            )
                    if optimizer_updated and ema_model is not None:
                        ema_model.update_parameters(model)
                        if bool(
                            config["train"].get(
                                "validate_step_finiteness", False
                            )
                        ):
                            _validate_module_finiteness(
                                ema_model.module, label="EMA model"
                            )
                    if optimizer_updated and (
                        scheduler is not None
                        and str(config["train"].get("scheduler_unit", "epoch"))
                        == "step"
                    ):
                        scheduler.step()

        prediction = outputs["logits"].argmax(dim=1)
        metrics.update(prediction, batch["target"])
        if native_metrics is not None:
            _, native_target, _ = native_sar_targets(
                batch["target"],
                batch["sar_valid"],
                int(config["model"]["num_classes"]),
                size=tuple(outputs["sar_native_logits"].shape[-2:]),
                min_purity=float(
                    config["loss"].get("sar_native_min_purity", 0.6)
                ),
                dtype=outputs["sar_native_logits"].dtype,
            )
            native_metrics.update(
                outputs["sar_native_logits"].argmax(dim=1),
                native_target,
            )
        for key, scalar in component_scalars.items():
            totals[key] += scalar
        batches += 1
        if training and safe_optimizer_boundary and recovery_boundary_callback:
            recovery_boundary_callback(
                {
                    "epoch": (
                        int(epoch_index) if epoch_index is not None else -1
                    ),
                    "next_batch_index": int(batch_index + 1),
                    "batches": int(batches),
                    "total_batches": int(total_batches),
                    "loss_totals": dict(totals),
                    "confusion_matrix": metrics.confusion.copy(),
                    "native_confusion_matrix": (
                        native_metrics.confusion.copy()
                        if native_metrics is not None
                        else None
                    ),
                    "loss_finite": bool(torch.isfinite(loss.detach()).item()),
                    # A normal FP16 overflow may intentionally skip an update.
                    # Keep that diagnostic out of finite-gated recovery payloads.
                    "gradient_norm": (
                        gradient_norm
                        if math.isfinite(gradient_norm)
                        else None
                    ),
                    "gradient_norm_finite": math.isfinite(gradient_norm),
                },
                bool(optimizer_updated),
            )
        if (
            progress_label is not None
            and progress_interval > 0
            and (batches % progress_interval == 0 or batches == total_batches)
        ):
            elapsed = time.perf_counter() - epoch_started
            rate = batches / max(elapsed, 1e-9)
            progress = {
                "event": f"{progress_label}_progress",
                "epoch": epoch_index,
                "batch": batches,
                "batches_total": total_batches,
                "elapsed_seconds": elapsed,
                "batches_per_second": rate,
                "eta_seconds": (total_batches - batches) / max(rate, 1e-9),
                "mean_total_loss": totals.get("total", 0.0) / batches,
            }
            if device.type == "cuda":
                progress["cuda_allocated_mib"] = (
                    torch.cuda.memory_allocated(device) / 1024**2
                )
                progress["cuda_reserved_mib"] = (
                    torch.cuda.memory_reserved(device) / 1024**2
                )
            print(json.dumps(progress), flush=True)

    averaged = {key: value / max(batches, 1) for key, value in totals.items()}
    computed = metrics.compute()
    if native_metrics is not None:
        computed.update(
            {
                f"native_{key}": value
                for key, value in native_metrics.compute().items()
            }
        )
    return averaged, computed


def _scheduler(
    optimizer: torch.optim.Optimizer,
    total_units: int,
    warmup_units: int,
    minimum_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    minimum_lr_ratio = float(minimum_lr_ratio)
    if not 0.0 <= minimum_lr_ratio <= 1.0:
        raise ValueError("minimum_lr_ratio must be in [0, 1]")

    def multiplier(unit: int) -> float:
        if warmup_units and unit < warmup_units:
            return float(unit + 1) / float(warmup_units)
        progress = (unit - warmup_units) / max(total_units - warmup_units, 1)
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _checkpoint_data_provenance(config: dict) -> dict | None:
    """Record the exact FBP scenes that may have influenced a checkpoint.

    The MS75 confusion curriculum accepts only genuinely out-of-fold
    predictions.  A sidecar fold declaration is not sufficient evidence: the
    checkpoint itself must state which scenes were optimized and which were
    held out.  Keeping this small, tensor-free payload in every checkpoint
    lets the index builder verify that contract without loading model weights
    into memory.
    """

    data = config["data"]
    if str(data.get("dataset_format", "ms_sar_manifest")) != "fbp_ms":
        return None
    root = Path(data["root"])
    train_scene_ids = read_fbp_split(root, str(data["train_split"]))
    validation_split = data.get("val_split")
    heldout_scene_ids: list[str] = []
    if validation_split:
        validation_scene_ids = read_fbp_split(root, str(validation_split))
        overlap = set(validation_scene_ids) & set(train_scene_ids)
        # A legacy tile-level validation loader can use the same scene list as
        # training.  Such a checkpoint is intentionally not representable as
        # an OOF fold and therefore records no held-out scenes.
        if not overlap:
            heldout_scene_ids = list(validation_scene_ids)
        elif set(validation_scene_ids) != set(train_scene_ids):
            raise ValueError(
                "FBP checkpoint provenance rejects partially overlapping "
                "train/validation scene lists"
            )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "FBP",
        "source_split": "official_train",
        "train_scene_ids": list(train_scene_ids),
        "heldout_scene_ids": heldout_scene_ids,
    }
    if str(data.get("protocol_stage", "")) == (
        "local_detail_cnxb256_fusionaware_a1_full120_10k"
    ):
        histogram_identity = data.get(
            "fusionaware_a1_class_histogram_identity"
        )
        if not isinstance(histogram_identity, Mapping):
            raise RuntimeError(
                "FusionAware A1 checkpoint lacks class-histogram identity"
            )
        provenance["fusionaware_a1"] = {
            "schema_version": 1,
            "official30_read": False,
            "ms75_truth_probability_manifest": str(
                data["ms75_truth_probability_manifest"]
            ),
            "ms75_truth_probability_manifest_sha256": str(
                data["ms75_truth_probability_manifest_sha256"]
            ),
            "local512_truth_probability_manifest": str(
                data["local512_truth_probability_manifest"]
            ),
            "local512_truth_probability_manifest_sha256": str(
                data["local512_truth_probability_manifest_sha256"]
            ),
            "class_histogram": dict(histogram_identity),
            "initialization_checkpoint": str(
                config["train"][
                    "fusionaware_a1_initialization_checkpoint"
                ]
            ),
            "initialization_checkpoint_sha256": str(
                config["train"][
                    "fusionaware_a1_initialization_checkpoint_sha256"
                ]
            ),
        }
    return provenance


def _training_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    ema_model: AveragedModel | None,
    config: dict,
    epoch: int,
    best_miou: float,
    best_epoch: int,
    global_optimizer_step: int,
    data_provenance: dict | None,
    epoch_complete: bool,
    epoch_progress: Mapping[str, Any] | None = None,
    paired_continuation: Mapping[str, Any] | None = None,
    successful_update_budget: int | None = None,
    budget_complete: bool = False,
    sampling_plan_audits: list[Mapping[str, Any]] | None = None,
    optimizer_update_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one self-contained epoch or optimizer-boundary checkpoint."""

    if (
        paired_continuation is not None
        and successful_update_budget is not None
    ):
        raise ValueError(
            "paired and ordinary successful-update budgets are mutually exclusive"
        )
    if (
        budget_complete
        and paired_continuation is None
        and successful_update_budget is None
    ):
        raise ValueError(
            "budget_complete requires an explicit successful-update budget"
        )
    if budget_complete and epoch_complete:
        raise ValueError(
            "successful-budget completion must not masquerade as epoch_complete"
        )
    if not epoch_complete:
        if epoch_progress is None:
            raise ValueError("mid-epoch checkpoint requires epoch_progress")
        if int(epoch_progress.get("epoch", -1)) != int(epoch):
            raise ValueError(
                "mid-epoch checkpoint progress must use its resumable local epoch"
            )
        next_batch_index = int(epoch_progress["next_batch_index"])
        if next_batch_index <= 0:
            raise ValueError(
                "mid-epoch checkpoint must advance at least one batch"
            )
    else:
        if epoch_progress is not None:
            raise ValueError(
                "completed epoch checkpoint cannot carry partial progress"
            )
        next_batch_index = 0
    validation_model = ema_model.module if ema_model is not None else model
    state: dict[str, Any] = {
        "epoch": int(epoch),
        "epoch_complete": bool(epoch_complete),
        "next_batch_index": next_batch_index,
        "global_optimizer_step": int(global_optimizer_step),
        "recovery_checkpoint_version": RECOVERY_CHECKPOINT_VERSION,
        # `model` is always evaluation-ready.  The raw trainable state remains
        # separate when EMA is active so resume does not optimize EMA weights.
        "model": validation_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_miou": float(best_miou),
        "best_epoch": int(best_epoch),
        "config_fingerprint": config_fingerprint(config),
        "rng_state": _capture_rng_state(),
    }
    model_checkpoint_role = getattr(model, "checkpoint_role", None)
    if model_checkpoint_role is not None:
        state["checkpoint_role"] = str(model_checkpoint_role)
        state["oof_proxy_role"] = str(getattr(model, "oof_proxy_role"))
        state["target_successful_optimizer_updates"] = int(
            getattr(model, "target_successful_optimizer_updates")
        )
        encoder = getattr(model, "encoder")
        state["encoder_frozen"] = not any(
            parameter.requires_grad for parameter in encoder.parameters()
        )
        from tcmmsline3.ms75_aligned_proxy import (
            canonical_module_tensor_sha256,
        )

        encoder_digest = canonical_module_tensor_sha256(encoder)
        initial_encoder_digest = str(model.encoder_initial_tensor_sha256)
        if encoder_digest != initial_encoder_digest:
            raise RuntimeError(
                "aligned OOF frozen encoder changed after initialization"
            )
        state["initialization_provenance"] = {
            **dict(model.encoder_initialization_provenance),
            "encoder_checkpoint_tensor_sha256": encoder_digest,
        }
        execution_provenance = getattr(
            model, "aligned_execution_provenance", None
        )
        if not isinstance(execution_provenance, Mapping):
            raise RuntimeError("aligned OOF lacks execution provenance")
        state["execution_provenance"] = dict(execution_provenance)
    oof_source_binding = getattr(model, "oof_source_protocol_binding", None)
    if oof_source_binding is not None:
        state["oof_source_protocol_binding"] = dict(oof_source_binding)
    neutral_initialization = getattr(
        model, "neutral_oof_initialization_provenance", None
    )
    if neutral_initialization is not None:
        from tcmmsline3.ms75_neutral_oof import canonical_module_tensor_sha256

        state["checkpoint_role"] = "ms75_clean_neutral_oof_fold"
        state["oof_proxy_role"] = "ms75_clean_neutral_oof"
        state["initialization_provenance"] = {
            **dict(neutral_initialization),
            "backbone_checkpoint_tensor_sha256": (
                canonical_module_tensor_sha256(getattr(model, "backbone"))
            ),
        }
        neutral_execution = getattr(
            model, "neutral_oof_execution_provenance", None
        )
        if not isinstance(neutral_execution, Mapping):
            raise RuntimeError("neutral OOF lacks execution provenance")
        state["execution_provenance"] = dict(neutral_execution)
    if paired_continuation is not None:
        state["checkpoint_role"] = PAIRED_CONTINUATION_CHECKPOINT_ROLE
        state["paired_continuation"] = dict(paired_continuation)
        state["budget_complete"] = bool(budget_complete)
        if optimizer_update_audit is None:
            # Unit callers and legacy no-skip fixtures get the exact canonical
            # audit for an all-successful prefix.  Production always supplies
            # its live attempt/skip history explicitly below.
            optimizer_update_audit = _paired_optimizer_update_audit(
                successful_updates=global_optimizer_step
            )
        state["optimizer_update_audit"] = (
            _validate_paired_optimizer_update_audit(
                optimizer_update_audit,
                successful_updates=global_optimizer_step,
            )
        )
        if budget_complete:
            target = int(
                paired_continuation["max_successful_optimizer_updates"]
            )
            if int(global_optimizer_step) != target:
                raise ValueError(
                    "budget_complete checkpoint must contain exactly the target "
                    "successful optimizer updates"
                )
            state["terminal_successful_updates"] = target
        if sampling_plan_audits is None or not sampling_plan_audits:
            raise ValueError(
                "paired continuation checkpoint requires sampling plan audits"
            )
        state["sampling_plan_audits"] = [
            dict(value) for value in sampling_plan_audits
        ]
        common_confusion = paired_continuation.get(
            "common_clean_confusion_index"
        )
        if common_confusion is not None:
            if not isinstance(common_confusion, Mapping):
                raise ValueError("paired clean confusion contract is invalid")
            for audit in state["sampling_plan_audits"]:
                if audit.get("confusion_index_file_sha256") != (
                    common_confusion.get("index_file_sha256")
                ):
                    raise ValueError(
                        "paired sampling audit used another confusion index"
                    )
                if audit.get("class_index_file_sha256") != (
                    paired_continuation.get("common_class_index_sha256")
                ):
                    raise ValueError(
                        "paired sampling audit used another class index"
                    )
    elif successful_update_budget is not None:
        target = int(successful_update_budget)
        if target <= 0:
            raise ValueError(
                "successful optimizer-update budget must be positive"
            )
        state.setdefault(
            "checkpoint_role", SUCCESSFUL_UPDATE_BUDGET_CHECKPOINT_ROLE
        )
        state["optimizer_budget_checkpoint_role"] = (
            SUCCESSFUL_UPDATE_BUDGET_CHECKPOINT_ROLE
        )
        state["successful_update_budget"] = target
        milestones = [
            int(value)
            for value in config["train"].get(
                "successful_update_milestones", []
            )
        ]
        if milestones:
            state["successful_update_milestones"] = milestones
        primary_update = config["train"].get(
            "primary_successful_optimizer_update"
        )
        if primary_update is not None:
            state["primary_successful_optimizer_update"] = int(
                primary_update
            )
        if str(config.get("data", {}).get("protocol_stage", "")) == (
            "local_detail_cnxb256_fusionaware_a1_full120_10k"
        ):
            initialization_provenance = getattr(
                model, "fusionaware_a1_initialization_provenance", None
            )
            if not isinstance(initialization_provenance, Mapping):
                raise RuntimeError(
                    "FusionAware A1 checkpoint lacks initialization provenance"
                )
            state["fusionaware_a1_initialization"] = dict(
                initialization_provenance
            )
            state["initial_ema_n_averaged"] = 1
            state["ema_seed_is_evaluated_initialization"] = True
        state["budget_complete"] = bool(budget_complete)
        state["selection_metric"] = str(
            config.get("train", {}).get("selection_metric", "mean_iou")
        )
        if optimizer_update_audit is None:
            optimizer_update_audit = _successful_optimizer_update_audit(
                successful_updates=global_optimizer_step
            )
        validated_update_audit = _validate_successful_optimizer_update_audit(
            optimizer_update_audit,
            successful_updates=global_optimizer_step,
        )
        state["optimizer_update_audit"] = validated_update_audit
        state["optimizer_attempt_count"] = validated_update_audit[
            "optimizer_attempt_count"
        ]
        state["skipped_optimizer_attempt_count"] = validated_update_audit[
            "skipped_optimizer_attempt_count"
        ]
        state["skipped_optimizer_attempt_indices"] = validated_update_audit[
            "skipped_optimizer_attempt_indices"
        ]
        if budget_complete:
            if int(global_optimizer_step) != target:
                raise ValueError(
                    "budget_complete checkpoint must contain exactly the target "
                    "successful optimizer updates"
                )
            state["terminal_successful_updates"] = target
    elif optimizer_update_audit is not None:
        raise ValueError(
            "optimizer update audit requires a successful-update budget"
        )
    if data_provenance is not None:
        state["data_provenance"] = data_provenance
    if epoch_progress is not None:
        progress_payload = dict(epoch_progress)
        # Avoid raw NumPy arrays so strict metadata readers can still open a
        # recovery checkpoint with torch.load(weights_only=True).
        for key in ("confusion_matrix", "native_confusion_matrix"):
            value = progress_payload.get(key)
            if value is not None:
                progress_payload[key] = (
                    torch.as_tensor(value, dtype=torch.int64).cpu().clone()
                )
        state["epoch_progress"] = progress_payload
    if ema_model is not None:
        state["training_model"] = model.state_dict()
        state["ema_n_averaged"] = int(ema_model.n_averaged.item())
    if bool(config["train"].get("validate_checkpoint_finiteness", False)):
        _validate_finite_checkpoint_payload(state)
    return state


def _publish_paired_budget_completion(
    state: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    output: Path,
    elapsed_seconds: float,
) -> None:
    """Atomically publish terminal state before its completion marker."""

    if not bool(state.get("budget_complete", False)):
        raise ValueError(
            "paired terminal publication requires budget_complete"
        )
    paired = state.get("paired_continuation")
    if not isinstance(paired, Mapping):
        raise ValueError("paired terminal state has no provenance")
    successful_updates = int(state.get("global_optimizer_step", -1))
    target = int(paired["max_successful_optimizer_updates"])
    if successful_updates != target:
        raise ValueError(
            "paired terminal state does not match its update budget"
        )
    optimizer_update_audit = _validate_paired_optimizer_update_audit(
        state.get("optimizer_update_audit"),
        successful_updates=successful_updates,
    )
    if (
        int(paired.get("initial_ema_n_averaged", -1)) != 1
        or paired.get("ema_seed_is_evaluated_base") is not True
        or int(state.get("ema_n_averaged", -1)) != target + 1
    ):
        raise ValueError(
            "paired terminal EMA must retain the evaluated Base seed and every "
            "successful continuation update"
        )
    sampling_plan_audits = state.get("sampling_plan_audits")
    if not isinstance(sampling_plan_audits, list) or not sampling_plan_audits:
        raise ValueError("paired terminal state has no sampling plan audits")
    # ``last.pt`` is written first by the caller at the optimizer boundary.
    # If the process dies while publishing ``final.pt``, auto-resume can read
    # that terminal last.pt and idempotently finish this transaction.
    final_checkpoint = checkpoint_dir / "final.pt"
    atomic_torch_save(dict(state), final_checkpoint)
    final_checkpoint_sha256 = sha256_file(final_checkpoint)
    last_checkpoint = checkpoint_dir / "last.pt"
    if not last_checkpoint.is_file():
        raise RuntimeError(
            "paired completion requires terminal last.pt before final.pt"
        )
    last_checkpoint_sha256 = sha256_file(last_checkpoint)
    # torch.save is not guaranteed byte-deterministic across two independent
    # serializations, so identity is asserted from terminal metadata/model
    # state and both file hashes are recorded rather than requiring equal file
    # bytes.
    atomic_json_dump(
        {
            "status": "completed",
            "checkpoint_role": PAIRED_CONTINUATION_CHECKPOINT_ROLE,
            "completion_kind": "successful_optimizer_update_budget",
            "budget_complete": True,
            "terminal_successful_updates": successful_updates,
            "arm": paired["arm"],
            "base_checkpoint_sha256": paired["base_checkpoint_sha256"],
            "base_model_tensor_sha256": paired["base_model_tensor_sha256"],
            "paired_protocol_manifest_sha256": paired[
                "paired_protocol_manifest_sha256"
            ],
            "common_class_index_path": paired["common_class_index_path"],
            "common_class_index_sha256": paired["common_class_index_sha256"],
            "common_clean_confusion_index": paired[
                "common_clean_confusion_index"
            ],
            "sampling_plan_audits": sampling_plan_audits,
            "optimizer_update_audit": optimizer_update_audit,
            "config_fingerprint": state["config_fingerprint"],
            "last_checkpoint": str(last_checkpoint),
            "last_checkpoint_sha256": last_checkpoint_sha256,
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": final_checkpoint_sha256,
            "elapsed_seconds": float(elapsed_seconds),
        },
        output / "completed.json",
    )


def _successful_update_evaluation_payload(
    training_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract a compact evaluation-only terminal budget checkpoint."""

    target = int(training_checkpoint.get("successful_update_budget", -1))
    audit = _validate_successful_update_budget_resume(
        training_checkpoint, target=target
    )
    if not bool(training_checkpoint.get("budget_complete", False)):
        raise ValueError(
            "terminal evaluation checkpoint requires budget_complete"
        )
    model_state = training_checkpoint.get("model")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("terminal evaluation checkpoint has no model state")
    ema_count = training_checkpoint.get("ema_n_averaged")
    uses_ema = ema_count is not None
    ema_seed_offset = _successful_update_ema_seed_offset(training_checkpoint)
    if uses_ema and int(ema_count) != target + ema_seed_offset:
        raise ValueError(
            "terminal EMA count must equal successful updates plus its seed"
        )
    payload: dict[str, Any] = {
        "checkpoint_schema_version": 1,
        "checkpoint_role": (
            SUCCESSFUL_UPDATE_EMA_CHECKPOINT_ROLE
            if uses_ema
            else SUCCESSFUL_UPDATE_MODEL_CHECKPOINT_ROLE
        ),
        "completion_kind": "successful_optimizer_update_budget",
        "budget_complete": True,
        "target_successful_optimizer_updates": target,
        "global_optimizer_step": target,
        "optimizer_update_audit": audit,
        "model": model_state,
        "config_fingerprint": training_checkpoint["config_fingerprint"],
        "source_epoch": int(training_checkpoint.get("epoch", -1)),
        "source_epoch_complete": bool(
            training_checkpoint.get("epoch_complete", False)
        ),
        "source_next_batch_index": int(
            training_checkpoint.get("next_batch_index", 0)
        ),
    }
    if uses_ema:
        payload["ema_n_averaged"] = int(ema_count)
    if ema_seed_offset:
        payload["initial_ema_n_averaged"] = 1
        payload["ema_seed_is_evaluated_initialization"] = True
        payload["fusionaware_a1_initialization"] = dict(
            training_checkpoint["fusionaware_a1_initialization"]
        )
    provenance = training_checkpoint.get("data_provenance")
    if provenance is not None:
        payload["data_provenance"] = provenance
    return payload


def _successful_update_milestone_payload(
    training_checkpoint: Mapping[str, Any], *, milestone: int
) -> dict[str, Any]:
    """Extract an immutable evaluation-only EMA at a precommitted update."""

    target = int(training_checkpoint.get("successful_update_budget", -1))
    _validate_successful_update_budget_resume(training_checkpoint, target=target)
    milestone = int(milestone)
    if not 0 < milestone < target:
        raise ValueError(
            "non-terminal successful-update milestone must lie inside the budget"
        )
    if int(training_checkpoint.get("global_optimizer_step", -1)) != milestone:
        raise ValueError("milestone checkpoint successful-update count mismatch")
    if bool(training_checkpoint.get("budget_complete", False)):
        raise ValueError("non-terminal milestone cannot be budget_complete")
    model_state = training_checkpoint.get("model")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("milestone checkpoint has no evaluation model state")
    ema_count = int(training_checkpoint.get("ema_n_averaged", -1))
    ema_seed_offset = _successful_update_ema_seed_offset(training_checkpoint)
    if ema_count != milestone + ema_seed_offset:
        raise ValueError(
            "milestone EMA count must equal successful updates plus its seed"
        )
    payload: dict[str, Any] = {
        "checkpoint_schema_version": 1,
        "checkpoint_role": "ema_successful_optimizer_update_milestone",
        "completion_kind": "precommitted_successful_optimizer_update_milestone",
        "budget_complete": False,
        "successful_update_budget": target,
        "target_successful_optimizer_updates": milestone,
        "global_optimizer_step": milestone,
        "ema_n_averaged": ema_count,
        "model": model_state,
        "config_fingerprint": training_checkpoint["config_fingerprint"],
        "source_epoch": int(training_checkpoint.get("epoch", -1)),
        "source_epoch_complete": bool(
            training_checkpoint.get("epoch_complete", False)
        ),
        "source_next_batch_index": int(
            training_checkpoint.get("next_batch_index", 0)
        ),
        "primary_precommitted": _is_primary_successful_update(
            training_checkpoint, milestone
        ),
    }
    if ema_seed_offset:
        payload["initial_ema_n_averaged"] = 1
        payload["ema_seed_is_evaluated_initialization"] = True
        payload["fusionaware_a1_initialization"] = dict(
            training_checkpoint["fusionaware_a1_initialization"]
        )
    provenance = training_checkpoint.get("data_provenance")
    if provenance is not None:
        payload["data_provenance"] = provenance
    return payload


def _publish_successful_update_milestone(
    state: Mapping[str, Any], *, checkpoint_dir: Path, milestone: int
) -> Path:
    path = checkpoint_dir / f"ema_successful_updates_{int(milestone):07d}.pt"
    expected = _successful_update_milestone_payload(
        state, milestone=milestone
    )
    if path.exists():
        try:
            observed = torch.load(
                str(path), map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:  # pragma: no cover - older supported PyTorch
            observed = torch.load(
                str(path), map_location="cpu", weights_only=False
            )
        if not isinstance(observed, Mapping):
            raise RuntimeError(
                f"successful-update milestone is not a mapping: {path}"
            )
        identity_keys = (
            "checkpoint_schema_version",
            "checkpoint_role",
            "completion_kind",
            "budget_complete",
            "successful_update_budget",
            "target_successful_optimizer_updates",
            "global_optimizer_step",
            "ema_n_averaged",
            "config_fingerprint",
            "primary_precommitted",
            "initial_ema_n_averaged",
            "ema_seed_is_evaluated_initialization",
            "fusionaware_a1_initialization",
            "data_provenance",
        )
        changed = {
            key: {"expected": expected.get(key), "observed": observed.get(key)}
            for key in identity_keys
            if observed.get(key) != expected.get(key)
        }
        expected_model = expected.get("model")
        observed_model = observed.get("model")
        if (
            not isinstance(expected_model, Mapping)
            or not isinstance(observed_model, Mapping)
            or _model_tensor_digest(observed_model)
            != _model_tensor_digest(expected_model)
        ):
            changed["model"] = {
                "expected": "identical tensor digest",
                "observed": "changed",
            }
        if changed:
            raise RuntimeError(
                f"successful-update milestone identity changed at {path}: {changed}"
            )
        return path
    atomic_torch_save(expected, path)
    return path


def _publish_successful_update_budget_completion(
    state: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    output: Path,
    elapsed_seconds: float,
) -> None:
    """Publish ordinary strict-budget artifacts before completed.json."""

    target = int(state.get("successful_update_budget", -1))
    audit = _validate_successful_update_budget_resume(state, target=target)
    if not bool(state.get("budget_complete", False)):
        raise ValueError("successful-update terminal state is not complete")
    last_checkpoint = checkpoint_dir / "last.pt"
    if not last_checkpoint.is_file():
        raise RuntimeError(
            "successful-update completion requires terminal last.pt first"
        )
    last_checkpoint_sha256 = sha256_file(last_checkpoint)

    best_checkpoint = checkpoint_dir / "best.pt"
    terminalize_best = str(state.get("selection_metric", "")) == "last_epoch"
    if terminalize_best:
        atomic_torch_save(dict(state), best_checkpoint)
        best_checkpoint_role = "terminal_successful_optimizer_update_budget"
    else:
        if not best_checkpoint.is_file():
            raise RuntimeError(
                "validated selection has no best.pt at budget completion"
            )
        best_checkpoint_role = "validated_selection"
    best_checkpoint_sha256 = sha256_file(best_checkpoint)

    uses_ema = state.get("ema_n_averaged") is not None
    evaluation_name = (
        f"ema_successful_updates_{target:07d}.pt"
        if uses_ema
        else f"model_successful_updates_{target:07d}.pt"
    )
    evaluation_checkpoint = checkpoint_dir / evaluation_name
    # Re-publishing from the durable terminal last.pt is intentionally
    # idempotent: an interrupted transaction simply replaces this compact
    # derivative before completed.json is written last.
    atomic_torch_save(
        _successful_update_evaluation_payload(state), evaluation_checkpoint
    )
    evaluation_checkpoint_sha256 = sha256_file(evaluation_checkpoint)

    milestone_values = [
        int(value) for value in state.get("successful_update_milestones", [])
    ]
    primary_value = state.get("primary_successful_optimizer_update")
    primary_update = int(primary_value) if primary_value is not None else None
    milestone_records: list[dict[str, Any]] = []
    primary_checkpoint: Path | None = None
    primary_checkpoint_sha256: str | None = None
    primary_payload: Mapping[str, Any] | None = None
    for milestone in milestone_values:
        path = (
            evaluation_checkpoint
            if milestone == target
            else checkpoint_dir / f"ema_successful_updates_{milestone:07d}.pt"
        )
        if not path.is_file():
            raise RuntimeError(
                "precommitted successful-update milestone is missing at "
                f"completion: {path}"
            )
        try:
            payload = torch.load(
                str(path), map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:  # pragma: no cover - older supported PyTorch
            payload = torch.load(
                str(path), map_location="cpu", weights_only=False
            )
        if (
            not isinstance(payload, Mapping)
            or int(payload.get("global_optimizer_step", -1)) != milestone
            or payload.get("config_fingerprint") != state["config_fingerprint"]
        ):
            raise RuntimeError(
                f"precommitted milestone identity mismatch: {path}"
            )
        digest = sha256_file(path)
        record = {
            "successful_optimizer_updates": milestone,
            "checkpoint": str(path),
            "checkpoint_sha256": digest,
            "checkpoint_role": payload.get("checkpoint_role"),
            "precommitted_role": (
                "primary" if milestone == primary_update else "diagnostic"
            ),
        }
        milestone_records.append(record)
        if milestone == primary_update:
            if payload.get("primary_precommitted") is not True:
                raise RuntimeError(
                    "primary successful-update milestone lacks its precommitment"
                )
            primary_checkpoint = path
            primary_checkpoint_sha256 = digest
            primary_payload = payload

    official_evaluation_checkpoint = evaluation_checkpoint
    official_evaluation_checkpoint_sha256 = evaluation_checkpoint_sha256
    official_evaluation_checkpoint_role = (
        SUCCESSFUL_UPDATE_EMA_CHECKPOINT_ROLE
        if uses_ema
        else SUCCESSFUL_UPDATE_MODEL_CHECKPOINT_ROLE
    )
    if primary_update is not None:
        if primary_update not in milestone_values or primary_update >= target:
            raise RuntimeError(
                "precommitted primary must be a non-terminal milestone"
            )
        assert primary_checkpoint is not None
        assert primary_checkpoint_sha256 is not None
        assert primary_payload is not None
        # ``best.pt`` must never silently point at the diagnostic 12k tail.
        # Publish the compact 10k EMA as the conventional best artifact too.
        atomic_torch_save(dict(primary_payload), best_checkpoint)
        best_checkpoint_role = "precommitted_primary_successful_update"
        best_checkpoint_sha256 = sha256_file(best_checkpoint)
        official_evaluation_checkpoint = primary_checkpoint
        official_evaluation_checkpoint_sha256 = primary_checkpoint_sha256
        official_evaluation_checkpoint_role = str(
            primary_payload.get("checkpoint_role")
        )

    progress = state.get("epoch_progress")
    effective_epoch = None
    if isinstance(progress, Mapping):
        effective_epoch = progress.get("effective_epoch")
    completion: dict[str, Any] = {
        "status": "completed",
        "checkpoint_role": SUCCESSFUL_UPDATE_BUDGET_CHECKPOINT_ROLE,
        "completion_kind": "successful_optimizer_update_budget",
        "budget_complete": True,
        "target_successful_optimizer_updates": target,
        "terminal_successful_updates": target,
        "optimizer_attempt_count": audit["optimizer_attempt_count"],
        "skipped_optimizer_attempt_count": audit[
            "skipped_optimizer_attempt_count"
        ],
        "skipped_optimizer_attempt_indices": audit[
            "skipped_optimizer_attempt_indices"
        ],
        "optimizer_update_audit": audit,
        "terminal_epoch": int(state.get("epoch", -1)),
        "terminal_effective_epoch": (
            int(effective_epoch) if effective_epoch is not None else None
        ),
        "terminal_epoch_complete": bool(state.get("epoch_complete", False)),
        "terminal_next_batch_index": int(state.get("next_batch_index", 0)),
        "ema_n_averaged": (
            int(state["ema_n_averaged"])
            if state.get("ema_n_averaged") is not None
            else None
        ),
        "selection_metric": state.get("selection_metric"),
        "best_checkpoint_role": best_checkpoint_role,
        "config_fingerprint": state["config_fingerprint"],
        "last_checkpoint": str(last_checkpoint),
        "last_checkpoint_sha256": last_checkpoint_sha256,
        "best_checkpoint": str(best_checkpoint),
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "evaluation_checkpoint": str(official_evaluation_checkpoint),
        "evaluation_checkpoint_sha256": official_evaluation_checkpoint_sha256,
        "evaluation_checkpoint_role": official_evaluation_checkpoint_role,
        "diagnostic_terminal_checkpoint": str(evaluation_checkpoint),
        "diagnostic_terminal_checkpoint_sha256": evaluation_checkpoint_sha256,
        "diagnostic_terminal_checkpoint_role": (
            SUCCESSFUL_UPDATE_EMA_CHECKPOINT_ROLE
            if uses_ema
            else SUCCESSFUL_UPDATE_MODEL_CHECKPOINT_ROLE
        ),
        "successful_update_checkpoints": milestone_records,
        "elapsed_seconds": float(elapsed_seconds),
    }
    if primary_update is not None:
        completion["primary_successful_optimizer_update"] = primary_update
        completion["primary_checkpoint"] = str(primary_checkpoint)
        completion["primary_checkpoint_sha256"] = primary_checkpoint_sha256
        completion["terminal_is_diagnostic_only"] = True
    if uses_ema:
        completion["ema_checkpoint"] = str(official_evaluation_checkpoint)
        completion["ema_checkpoint_sha256"] = official_evaluation_checkpoint_sha256
    provenance = state.get("data_provenance")
    if provenance is not None:
        completion["data_provenance"] = provenance
    atomic_json_dump(completion, output / "completed.json")


def _ema_milestone_payload(
    training_checkpoint: Mapping[str, Any],
    *,
    completed_epochs: int,
) -> dict[str, Any]:
    """Extract an evaluation-only EMA checkpoint from a full train state.

    The payload intentionally omits raw training weights, Adam moments,
    scheduler, scaler, and RNG state.  It is therefore compact and directly
    consumable by ``evaluate.py``, but it can never be mistaken for a resume
    checkpoint.
    """

    completed_epochs = int(completed_epochs)
    if completed_epochs <= 0:
        raise ValueError("EMA milestone must complete at least one epoch")
    epoch = int(training_checkpoint.get("epoch", -1))
    if not bool(training_checkpoint.get("epoch_complete", False)):
        raise ValueError("EMA milestone requires an epoch-complete checkpoint")
    if epoch + 1 != completed_epochs:
        raise ValueError("EMA milestone epoch does not match its source state")
    model_state = training_checkpoint.get("model")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("EMA milestone source has no evaluation model state")
    fingerprint = training_checkpoint.get("config_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("EMA milestone source has no config fingerprint")
    ema_n_averaged = int(training_checkpoint.get("ema_n_averaged", 0))
    if ema_n_averaged <= 0:
        raise ValueError("EMA milestone source has no averaged updates")
    payload: dict[str, Any] = {
        "checkpoint_schema_version": 1,
        "checkpoint_role": "ema_milestone",
        "epoch": epoch,
        "completed_epochs": completed_epochs,
        "global_optimizer_step": int(
            training_checkpoint.get("global_optimizer_step", 0)
        ),
        "model": model_state,
        "config_fingerprint": fingerprint,
        "ema_n_averaged": ema_n_averaged,
    }
    provenance = training_checkpoint.get("data_provenance")
    if provenance is not None:
        payload["data_provenance"] = provenance
    return payload


def _load_ema_milestone_metadata(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        try:
            candidate = torch.load(
                str(path),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:  # pragma: no cover - older supported PyTorch
            candidate = torch.load(
                str(path), map_location="cpu", weights_only=False
            )
    except Exception:
        return None
    return candidate if isinstance(candidate, Mapping) else None


def _ema_milestone_matches(
    candidate: Mapping[str, Any] | None,
    *,
    completed_epochs: int,
    expected_config_fingerprint: str,
) -> bool:
    return bool(
        candidate is not None
        and int(candidate.get("checkpoint_schema_version", -1)) == 1
        and candidate.get("checkpoint_role") == "ema_milestone"
        and int(candidate.get("completed_epochs", -1)) == int(completed_epochs)
        and int(candidate.get("epoch", -1)) == int(completed_epochs) - 1
        and candidate.get("config_fingerprint") == expected_config_fingerprint
        and int(candidate.get("global_optimizer_step", -1)) >= 0
        and int(candidate.get("ema_n_averaged", 0)) > 0
        and isinstance(candidate.get("model"), Mapping)
        and "optimizer" not in candidate
        and "training_model" not in candidate
        and "scheduler" not in candidate
        and "scaler" not in candidate
        and "rng_state" not in candidate
    )


def _publish_ema_milestone(
    training_checkpoint: Mapping[str, Any],
    path: Path,
    *,
    completed_epochs: int,
) -> bool:
    """Atomically publish one immutable EMA milestone.

    Returns ``True`` when a file was written and ``False`` when an already
    valid file made the operation idempotent.  A conflicting artifact fails
    closed instead of silently overwriting a scientific result.
    """

    fingerprint = str(training_checkpoint.get("config_fingerprint", ""))
    existing = _load_ema_milestone_metadata(path)
    if existing is not None:
        if _ema_milestone_matches(
            existing,
            completed_epochs=completed_epochs,
            expected_config_fingerprint=fingerprint,
        ):
            return False
        raise RuntimeError(f"EMA milestone is stale or malformed: {path}")
    if path.exists():
        raise RuntimeError(f"EMA milestone cannot be read safely: {path}")
    payload = _ema_milestone_payload(
        training_checkpoint, completed_epochs=completed_epochs
    )
    atomic_torch_save(payload, path)
    return True


def _reconcile_ema_milestones(
    resume_checkpoint: Mapping[str, Any],
    checkpoint_dir: Path,
    *,
    milestone_epochs: list[int],
    expected_config_fingerprint: str,
) -> None:
    """Validate earlier milestones and repair the last-to-milestone window."""

    if not milestone_epochs:
        return
    checkpoint_epoch = int(resume_checkpoint.get("epoch", -1))
    epoch_complete = bool(resume_checkpoint.get("epoch_complete", True))
    completed_epochs = checkpoint_epoch + (1 if epoch_complete else 0)
    for milestone in milestone_epochs:
        if milestone > completed_epochs:
            continue
        path = checkpoint_dir / f"ema_epoch_{milestone:04d}.pt"
        existing = _load_ema_milestone_metadata(path)
        if _ema_milestone_matches(
            existing,
            completed_epochs=milestone,
            expected_config_fingerprint=expected_config_fingerprint,
        ):
            continue
        # Only the just-completed epoch's EMA weights still exist verbatim in
        # last.pt.  This is the crash window we can repair exactly.
        if epoch_complete and milestone == checkpoint_epoch + 1:
            if path.exists():
                raise RuntimeError(
                    f"EMA milestone is stale or malformed: {path}"
                )
            _publish_ema_milestone(
                resume_checkpoint,
                path,
                completed_epochs=milestone,
            )
            print(
                json.dumps(
                    {
                        "event": "ema_milestone_reconciled",
                        "completed_epochs": milestone,
                        "source": "resume_checkpoint",
                        "destination": str(path),
                    }
                ),
                flush=True,
            )
            continue
        raise RuntimeError(
            "an earlier EMA milestone is missing/stale and cannot be "
            f"reconstructed from the current resume checkpoint: {path}"
        )


def _reconcile_best_checkpoint(
    resume_checkpoint: Mapping[str, Any],
    best_path: Path,
    *,
    expected_config_fingerprint: str,
) -> None:
    """Repair the crash window between end-of-epoch last.pt and best.pt.

    New checkpoints declare exactly which epoch owns ``best_miou``. If the
    resumed last.pt is itself that completed best epoch, it is a complete
    canonical source from which a missing/stale best.pt can be republished.
    A missing older best cannot be reconstructed from a newer non-best last,
    so fail loudly rather than evaluating the wrong model.
    """

    if "best_epoch" not in resume_checkpoint:
        # Backward compatibility for legacy runs created before the explicit
        # reconciliation contract. Their historical best.pt remains untouched.
        return
    best_epoch = int(resume_checkpoint["best_epoch"])
    if best_epoch < 0:
        return

    def metadata_matches(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            candidate = torch.load(
                str(path),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:  # pragma: no cover - older supported PyTorch
            candidate = torch.load(
                str(path), map_location="cpu", weights_only=False
            )
        except Exception:
            return False
        return bool(
            isinstance(candidate, Mapping)
            and candidate.get("config_fingerprint")
            == expected_config_fingerprint
            and int(candidate.get("epoch", -1)) == best_epoch
            and int(candidate.get("best_epoch", -1)) == best_epoch
            and float(candidate.get("best_miou", float("nan")))
            == float(resume_checkpoint["best_miou"])
        )

    if metadata_matches(best_path):
        return
    resume_epoch = int(resume_checkpoint.get("epoch", -1))
    if bool(resume_checkpoint.get("epoch_complete", True)) and (
        resume_epoch == best_epoch
    ):
        atomic_torch_save(dict(resume_checkpoint), best_path)
        print(
            json.dumps(
                {
                    "event": "best_checkpoint_reconciled",
                    "best_epoch": best_epoch,
                    "source": "resume_checkpoint",
                    "destination": str(best_path),
                }
            ),
            flush=True,
        )
        return
    raise RuntimeError(
        "best.pt is missing/stale and the resumed checkpoint does not contain "
        f"the declared best epoch {best_epoch}"
    )


def _build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    *,
    staged_parameters: list[torch.nn.Parameter],
) -> tuple[
    torch.optim.AdamW,
    torch.optim.lr_scheduler.LambdaLR,
    int,
    int,
]:
    """Build the single production optimizer/scheduler used by train and probe."""

    staged_ids = {id(parameter) for parameter in staged_parameters}
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in staged_ids
    ]
    if not regular_parameters and not staged_parameters:
        raise ValueError("no trainable model parameters remain")
    train = config["train"]
    data = config["data"]
    base_learning_rate = float(train["learning_rate"])
    parameter_groups: list[dict[str, Any]] = []
    if regular_parameters:
        backbone_multiplier = float(train.get("backbone_lr_multiplier", 1.0))
        regular_ids = {id(parameter) for parameter in regular_parameters}
        backbone_prefixes = tuple(
            str(value)
            for value in train.get("backbone_prefixes", ["backbone."])
        )
        backbone_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if id(parameter) in regular_ids
            and any(name.startswith(prefix) for prefix in backbone_prefixes)
        ]
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter
            for parameter in regular_parameters
            if id(parameter) not in backbone_ids
        ]
        if head_parameters:
            parameter_groups.append(
                {
                    "params": head_parameters,
                    "lr": base_learning_rate,
                    "group_name": "heads",
                }
            )
        if backbone_parameters:
            parameter_groups.append(
                {
                    "params": backbone_parameters,
                    "lr": base_learning_rate * backbone_multiplier,
                    "group_name": "backbone",
                }
            )
    if staged_parameters:
        parameter_groups.append(
            {
                "params": staged_parameters,
                "lr": base_learning_rate
                * float(train.get("staged_lr_multiplier", 0.1)),
                "group_name": "staged_unfreeze",
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=base_learning_rate,
        weight_decay=float(train["weight_decay"]),
    )
    scheduler_unit = str(train.get("scheduler_unit", "epoch"))
    samples_per_epoch = int(
        data.get("debug_samples_per_epoch", data["samples_per_epoch"])
    )
    batches_per_epoch = samples_per_epoch // int(train["batch_size"])
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / int(train.get("gradient_accumulation_steps", 1))
    )
    if scheduler_unit == "step":
        successful_update_budget = train.get(
            "max_successful_optimizer_updates"
        )
        scheduler_successful_update_horizon = train.get(
            "scheduler_total_successful_optimizer_updates"
        )
        total_scheduler_units = (
            int(scheduler_successful_update_horizon)
            if scheduler_successful_update_horizon is not None
            else int(successful_update_budget)
            if successful_update_budget is not None
            else optimizer_steps_per_epoch * int(train["epochs"])
        )
        warmup_scheduler_units = int(train.get("warmup_steps", 0))
    else:
        total_scheduler_units = int(train["epochs"])
        warmup_scheduler_units = int(train["warmup_epochs"])
    scheduler = _scheduler(
        optimizer,
        total_scheduler_units,
        warmup_scheduler_units,
        minimum_lr_ratio=float(train.get("minimum_lr_ratio", 0.0)),
    )
    return (
        optimizer,
        scheduler,
        optimizer_steps_per_epoch,
        total_scheduler_units,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = load_config(args.config)
    configured_output = Path(config["experiment"]["output_dir"]).resolve()
    existing_paired_completion = _validate_existing_paired_completion(
        config, configured_output
    )
    if existing_paired_completion is not None:
        print(
            json.dumps(
                {
                    "event": "paired_continuation_already_complete",
                    "arm": existing_paired_completion["arm"],
                    "terminal_successful_updates": existing_paired_completion[
                        "terminal_successful_updates"
                    ],
                    "final_checkpoint_sha256": existing_paired_completion[
                        "final_checkpoint_sha256"
                    ],
                }
            ),
            flush=True,
        )
        return
    existing_budget_completion = (
        _validate_existing_successful_budget_completion(
            config, configured_output
        )
    )
    if existing_budget_completion is not None:
        print(
            json.dumps(
                {
                    "event": "successful_update_budget_already_complete",
                    "terminal_successful_updates": (
                        existing_budget_completion[
                            "terminal_successful_updates"
                        ]
                    ),
                    "evaluation_checkpoint_sha256": (
                        existing_budget_completion[
                            "evaluation_checkpoint_sha256"
                        ]
                    ),
                }
            ),
            flush=True,
        )
        return
    oof_source_protocol: dict[str, Any] | None = None
    oof_source_binding: dict[str, Any] | None = None
    if str(config["data"].get("protocol_stage", "")) in {
        "oof_neutral",
        "oof_neutral_proxy",
        "oof_aligned_proxy",
    }:
        from tcmmsline3.ms75_oof_source_protocol import (
            load_oof_source_protocol_binding,
        )

        oof_source_protocol, oof_source_binding = (
            load_oof_source_protocol_binding(config, project_root=project_root)
        )
    seed_everything(int(config["experiment"]["seed"]))
    device = resolve_device(str(config["train"]["device"]))
    deterministic_algorithms = bool(
        config["train"].get("deterministic_algorithms", False)
    )
    if deterministic_algorithms:
        # Paired runs fail closed if CUDA reaches an operation without a
        # deterministic implementation.  Warning-only mode would silently
        # reintroduce run noise into the small expected treatment effect.
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = bool(
        config["train"].get("cudnn_benchmark", False) and device.type == "cuda"
    )
    output = prepare_output(config, project_root)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ema_milestone_epochs = [
        int(value) for value in config["train"].get("ema_milestone_epochs", [])
    ]
    resume_path = _resolve_resume_checkpoint(config, checkpoint_dir, output)

    model = build_model(config).to(device)
    if oof_source_binding is not None:
        model.oof_source_protocol_binding = dict(oof_source_binding)
    if str(config["model"].get("architecture")) == "ms75_oof_aligned_swin":
        if oof_source_protocol is None:
            raise ValueError(
                "aligned OOF lacks source protocol startup binding"
            )
        expected_encoder_digest = str(
            oof_source_protocol.get("generic_initial_tensor_sha256", "")
        )
        if model.encoder_initial_tensor_sha256 != expected_encoder_digest:
            raise ValueError(
                "aligned OOF generic pretrained tensor differs from protocol"
            )
        if model.encoder_initialization_provenance.get("pretrained_cfg") != (
            oof_source_protocol.get("encoder_pretrained_cfg")
        ):
            raise ValueError(
                "aligned OOF timm pretrained source differs from protocol"
            )
        from tcmmsline3.ms75_aligned_proxy import (
            build_aligned_execution_provenance,
        )

        model.aligned_execution_provenance = (
            build_aligned_execution_provenance(device)
        )
        if model.aligned_execution_provenance["software_dependencies"] != (
            oof_source_protocol.get("software_dependencies")
        ):
            raise ValueError(
                "aligned OOF installed dependencies differ from protocol"
            )
    if str(config["data"].get("protocol_stage", "")) in {
        "oof_neutral",
        "oof_neutral_proxy",
    }:
        from tcmmsline3.ms75_neutral_oof import (
            build_neutral_execution_provenance,
            canonical_module_tensor_sha256,
            validate_neutral_proxy_config,
        )

        validate_neutral_proxy_config(config)
        backbone = getattr(model, "backbone", None)
        if not isinstance(backbone, torch.nn.Module):
            raise ValueError(
                "clean neutral OOF model has no auditable backbone"
            )
        model.neutral_oof_initialization_provenance = {
            "schema_version": 1,
            "scope": "generic_pretraining_only",
            "provider": "torchvision",
            "initialization": "convnext_tiny_imagenet1k_v1",
            "pretrained": True,
            "backbone_initial_tensor_sha256": canonical_module_tensor_sha256(
                backbone
            ),
            "fbp_checkpoint_opened": False,
            "full120_checkpoint_opened": False,
            "teacher_checkpoint_opened": False,
        }
        if oof_source_binding is None:
            raise ValueError(
                "neutral OOF lacks source protocol startup binding"
            )
        if oof_source_protocol is None:
            raise ValueError("neutral OOF lacks loaded source protocol")
        if oof_source_protocol.get("generic_initial_tensor_sha256") != (
            model.neutral_oof_initialization_provenance[
                "backbone_initial_tensor_sha256"
            ]
        ):
            raise ValueError(
                "neutral OOF generic pretrained tensor differs from protocol"
            )
        model.neutral_oof_execution_provenance = (
            build_neutral_execution_provenance(device)
        )
        if model.neutral_oof_execution_provenance[
            "software_dependencies"
        ] != oof_source_protocol.get("software_dependencies"):
            raise ValueError(
                "neutral OOF installed dependencies differ from protocol"
            )
    paired_continuation: dict[str, Any] | None = None
    if resume_path is None:
        paired_continuation = _initialize_model_from_config(
            model, config, output
        )

    trainable_module_names = [
        str(name) for name in config["train"].get("trainable_modules", [])
    ]
    if trainable_module_names:
        _configure_trainable_modules(model, trainable_module_names)
        print(
            json.dumps({"trainable_modules": trainable_module_names}),
            flush=True,
        )

    frozen_modules = [
        str(name) for name in config["train"].get("freeze_modules", [])
    ]
    for module_name in frozen_modules:
        module = _resolve_module(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    staged_module_names = [
        str(name)
        for name in config["train"].get("staged_unfreeze_modules", [])
    ]
    staged_modules = [
        _resolve_module(model, name) for name in staged_module_names
    ]
    permanently_frozen = {
        id(parameter)
        for module_name in frozen_modules
        for parameter in _resolve_module(model, module_name).parameters()
    }
    staged_parameters = {
        id(parameter): parameter
        for module in staged_modules
        for parameter in module.parameters()
        if id(parameter) not in permanently_frozen
    }

    teacher_path = config["train"].get("teacher_checkpoint")
    fixed_anchor_teacher_enabled = bool(
        config["train"].get("fixed_anchor_teacher_enabled", False)
    )
    if teacher_path and fixed_anchor_teacher_enabled:
        raise ValueError(
            "legacy teacher_checkpoint and fixed MS75 anchor teacher are "
            "mutually exclusive"
        )
    if fixed_anchor_teacher_enabled:
        teacher = _build_ms75_fixed_anchor_teacher(config, device)
    else:
        teacher = (
            _build_teacher(config, teacher_path, device)
            if teacher_path
            else None
        )
    teacher_amp_enabled = bool(
        fixed_anchor_teacher_enabled
        and config["train"].get("fixed_anchor_teacher_amp", False)
    )
    planned_staged_parameters = list(staged_parameters.values())
    scheduler_unit = str(config["train"].get("scheduler_unit", "epoch"))
    optimizer, scheduler, optimizer_steps_per_epoch, _ = (
        _build_optimizer_and_scheduler(
            model,
            config,
            staged_parameters=planned_staged_parameters,
        )
    )
    use_ema = bool(config["train"].get("use_ema", False))
    ema_model = (
        AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(
                float(config["train"].get("ema_decay", 0.999))
            ),
            use_buffers=True,
        ).to(device)
        if use_ema
        else None
    )
    if paired_continuation is not None:
        if ema_model is None:
            raise RuntimeError("paired continuation requires EMA")
        _seed_paired_ema_from_evaluated_base(ema_model, model)
        paired_continuation["initial_ema_n_averaged"] = 1
        paired_continuation["ema_seed_is_evaluated_base"] = True
        paired_continuation["initial_ema_model_tensor_sha256"] = (
            paired_continuation["base_model_tensor_sha256"]
        )
        atomic_json_dump(
            paired_continuation,
            output / "paired_continuation_initialization.json",
        )
    elif resume_path is None and bool(
        config["train"].get("fusionaware_a1_initialization_checkpoint")
    ):
        if ema_model is None:
            raise RuntimeError("FusionAware A1 requires EMA")
        initialization_provenance = getattr(
            model, "fusionaware_a1_initialization_provenance", None
        )
        if not isinstance(initialization_provenance, Mapping):
            raise RuntimeError(
                "FusionAware A1 lacks strict warm-start provenance"
            )
        _seed_paired_ema_from_evaluated_base(ema_model, model)
        seed_digest = _model_tensor_digest(model.state_dict())
        seeded_provenance = {
            **dict(initialization_provenance),
            "initial_ema_n_averaged": 1,
            "ema_seed_is_evaluated_initialization": True,
            "initial_ema_model_tensor_sha256": seed_digest,
        }
        model.fusionaware_a1_initialization_provenance = dict(
            seeded_provenance
        )
        atomic_json_dump(
            seeded_provenance,
            output / "fusionaware_a1_initialization.json",
        )
    scaler = _build_grad_scaler(config, device)

    validation_enabled = bool(config["train"].get("validation_enabled", True))
    start_epoch = 0
    resume_next_batch_index = 0
    resume_epoch_progress: Mapping[str, Any] | None = None
    resume_rng_state: Mapping[str, Any] | None = None
    global_optimizer_step = 0
    paired_optimizer_attempts = 0
    paired_skipped_optimizer_attempts: list[int] = []
    budget_optimizer_attempts = 0
    budget_skipped_optimizer_attempts: list[int] = []
    ordinary_successful_update_budget = _ordinary_successful_update_budget(
        config
    )
    best_miou = -1.0
    best_epoch = -1
    resumed_completed_sampling_plan_audits: list[dict[str, Any]] = []
    if resume_path:
        def load_recovery_candidate(path: str | Path) -> Mapping[str, Any]:
            candidate = torch.load(
                str(path), map_location="cpu", weights_only=False
            )
            return _validate_recovery_candidate_for_config(candidate, config)

        try:
            checkpoint = load_recovery_candidate(resume_path)
        except Exception as last_error:
            previous_path = checkpoint_dir / "previous.pt"
            automatic_last = checkpoint_dir / "last.pt"
            may_fallback = (
                config["train"].get("resume_checkpoint") is None
                and bool(
                    config["train"].get(
                        "retain_previous_recovery_checkpoint", False
                    )
                )
                and Path(resume_path).resolve() == automatic_last.resolve()
                and previous_path.is_file()
            )
            if not may_fallback:
                raise
            try:
                checkpoint = load_recovery_candidate(previous_path)
            except Exception as previous_error:
                raise RuntimeError(
                    "both rolling recovery checkpoints are invalid"
                ) from previous_error
            print(
                json.dumps(
                    {
                        "event": "rolling_recovery_fallback",
                        "rejected_checkpoint": str(automatic_last),
                        "selected_checkpoint": str(previous_path),
                        "rejected_error": repr(last_error),
                    }
                ),
                flush=True,
            )
            resume_path = str(previous_path)
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("resume checkpoint must be a mapping")
        _validate_resume_checkpoint_role(checkpoint)
        if oof_source_binding is not None:
            from tcmmsline3.ms75_oof_source_protocol import (
                validate_checkpoint_oof_source_protocol_binding,
            )

            validate_checkpoint_oof_source_protocol_binding(
                checkpoint, oof_source_binding
            )
        if str(config["model"].get("architecture")) == (
            "ms75_oof_aligned_swin"
        ):
            from tcmmsline3.ms75_aligned_proxy import (
                validate_aligned_checkpoint_initialization,
            )

            validate_aligned_checkpoint_initialization(
                checkpoint,
                expected_initial_tensor_sha256=(
                    oof_source_protocol["generic_initial_tensor_sha256"]
                ),
                expected_pretrained_cfg=oof_source_protocol[
                    "encoder_pretrained_cfg"
                ],
            )
            from tcmmsline3.ms75_aligned_proxy import (
                validate_aligned_execution_provenance,
            )

            checkpoint_execution = checkpoint.get("execution_provenance")
            validate_aligned_execution_provenance(
                checkpoint_execution,
                expected_software_dependencies=(
                    oof_source_protocol["software_dependencies"]
                ),
            )
            if checkpoint_execution != model.aligned_execution_provenance:
                raise RuntimeError(
                    "aligned OOF resume execution environment changed"
                )
        if str(config["data"].get("protocol_stage", "")) == (
            "oof_neutral_proxy"
        ):
            from tcmmsline3.ms75_neutral_oof import (
                validate_neutral_checkpoint_initialization,
                validate_neutral_execution_provenance,
            )

            validate_neutral_checkpoint_initialization(
                checkpoint,
                expected_initial_tensor_sha256=(
                    model.neutral_oof_initialization_provenance[
                        "backbone_initial_tensor_sha256"
                    ]
                ),
            )
            checkpoint_execution = checkpoint.get("execution_provenance")
            validate_neutral_execution_provenance(
                checkpoint_execution,
                expected_software_dependencies=(
                    oof_source_protocol["software_dependencies"]
                ),
            )
            if checkpoint_execution != model.neutral_oof_execution_provenance:
                raise RuntimeError(
                    "neutral OOF resume execution environment changed"
                )
        if checkpoint.get("config_fingerprint") != config_fingerprint(config):
            raise RuntimeError("resume checkpoint config fingerprint mismatch")
        if bool(config["train"].get("validate_checkpoint_finiteness", False)):
            _validate_finite_checkpoint_payload(checkpoint)
        expected_paired = _paired_continuation_metadata_from_config(config)
        if expected_paired is not None:
            paired_continuation = _validate_paired_continuation_resume(
                checkpoint, expected_paired
            )
            if bool(checkpoint.get("budget_complete", False)):
                # Reconcile the narrow crash window after terminal last.pt but
                # before final.pt/completed.json, without taking another step.
                _atomic_save_rolling_recovery(
                    checkpoint, checkpoint_dir=checkpoint_dir, config=config
                )
                _publish_paired_budget_completion(
                    checkpoint,
                    checkpoint_dir=checkpoint_dir,
                    output=output,
                    elapsed_seconds=0.0,
                )
                print(
                    json.dumps(
                        {
                            "event": "paired_budget_completion_reconciled",
                            "global_optimizer_step": int(
                                checkpoint["global_optimizer_step"]
                            ),
                        }
                    ),
                    flush=True,
                )
                return
        elif ordinary_successful_update_budget is not None:
            _validate_successful_update_budget_resume(
                checkpoint, target=ordinary_successful_update_budget
            )
            if bool(checkpoint.get("budget_complete", False)):
                # The durable terminal last.pt is the transaction journal. A
                # restart in the best/compact/completed publication window
                # finishes that transaction without taking another update.
                _atomic_save_rolling_recovery(
                    checkpoint, checkpoint_dir=checkpoint_dir, config=config
                )
                _publish_successful_update_budget_completion(
                    checkpoint,
                    checkpoint_dir=checkpoint_dir,
                    output=output,
                    elapsed_seconds=0.0,
                )
                print(
                    json.dumps(
                        {
                            "event": (
                                "successful_update_budget_completion_reconciled"
                            ),
                            "global_optimizer_step": int(
                                checkpoint["global_optimizer_step"]
                            ),
                        }
                    ),
                    flush=True,
                )
                return
        elif checkpoint.get("optimizer_budget_checkpoint_role") == (
            SUCCESSFUL_UPDATE_BUDGET_CHECKPOINT_ROLE
        ):
            raise RuntimeError(
                "successful-update budget checkpoint cannot resume without its "
                "configured target"
            )
        elif checkpoint.get("checkpoint_role") == (
            PAIRED_CONTINUATION_CHECKPOINT_ROLE
        ):
            raise RuntimeError(
                "paired continuation checkpoint cannot resume under another role"
            )
        model.load_state_dict(
            checkpoint.get(
                "training_model", checkpoint.get("model", checkpoint)
            ),
            strict=True,
        )
        _project_model_constraints(model)
        _restore_fusionaware_a1_resume_provenance(
            model, checkpoint, config, output
        )
        if ema_model is not None:
            ema_model.module.load_state_dict(
                checkpoint.get("model", checkpoint), strict=True
            )
            _project_model_constraints(ema_model.module)
            ema_model.n_averaged.copy_(
                torch.as_tensor(
                    checkpoint.get("ema_n_averaged", 1),
                    dtype=ema_model.n_averaged.dtype,
                    device=ema_model.n_averaged.device,
                )
            )
        print(
            json.dumps({"resume_checkpoint": str(resume_path)}),
            flush=True,
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        checkpoint_epoch = int(checkpoint["epoch"])
        epoch_complete = bool(checkpoint.get("epoch_complete", True))
        if epoch_complete:
            start_epoch = checkpoint_epoch + 1
        else:
            if int(checkpoint.get("recovery_checkpoint_version", -1)) != (
                RECOVERY_CHECKPOINT_VERSION
            ):
                raise RuntimeError("unsupported mid-epoch recovery checkpoint")
            start_epoch = checkpoint_epoch
            resume_next_batch_index = int(
                checkpoint.get("next_batch_index", -1)
            )
            if resume_next_batch_index <= 0:
                raise RuntimeError("invalid mid-epoch recovery batch position")
            resume_epoch_progress = checkpoint.get("epoch_progress")
        rng_value = checkpoint.get("rng_state")
        if rng_value is not None:
            if not isinstance(rng_value, Mapping):
                raise RuntimeError("resume checkpoint RNG state is malformed")
            resume_rng_state = rng_value
        elif not epoch_complete:
            raise RuntimeError(
                "mid-epoch recovery checkpoint has no RNG state"
            )
        global_optimizer_step = int(
            checkpoint.get(
                "global_optimizer_step",
                (
                    max(int(scheduler.last_epoch), 0)
                    if scheduler_unit == "step"
                    else max(start_epoch, 0) * optimizer_steps_per_epoch
                ),
            )
        )
        if global_optimizer_step < 0:
            raise RuntimeError(
                "resume checkpoint has a negative optimizer step"
            )
        if paired_continuation is not None and global_optimizer_step >= int(
            paired_continuation["max_successful_optimizer_updates"]
        ):
            raise RuntimeError(
                "non-terminal paired checkpoint has exhausted its update budget"
            )
        if paired_continuation is not None:
            update_audit = _validate_paired_optimizer_update_audit(
                checkpoint.get("optimizer_update_audit"),
                successful_updates=global_optimizer_step,
            )
            paired_optimizer_attempts = int(
                update_audit["optimizer_attempt_count"]
            )
            paired_skipped_optimizer_attempts = [
                int(value)
                for value in update_audit["skipped_optimizer_attempt_indices"]
            ]
            if int(scheduler.last_epoch) != global_optimizer_step:
                raise RuntimeError(
                    "paired continuation scheduler.last_epoch does not equal "
                    "global successful optimizer updates"
                )
            if ema_model is None or int(ema_model.n_averaged.item()) != (
                global_optimizer_step + 1
            ):
                raise RuntimeError(
                    "paired continuation EMA count does not equal the evaluated "
                    "Base seed plus global successful optimizer updates"
                )
            audits = checkpoint.get("sampling_plan_audits")
            if (
                not isinstance(audits, list)
                or not audits
                or any(not isinstance(value, Mapping) for value in audits)
            ):
                raise RuntimeError(
                    "paired continuation resume has no valid sampling plan audits"
                )
            restored_audits = [dict(value) for value in audits]
            if epoch_complete:
                resumed_completed_sampling_plan_audits = restored_audits
            else:
                current_audit = restored_audits[-1]
                expected_effective = checkpoint_epoch + int(
                    paired_continuation["continuation_epoch_offset"]
                )
                if int(current_audit.get("epoch", -1)) != expected_effective:
                    raise RuntimeError(
                        "paired resume sampling audit has wrong effective epoch"
                    )
                if int(current_audit.get("class_schedule_epoch", -1)) != (
                    checkpoint_epoch
                ) or int(current_audit.get("consumed_samples", -1)) != (
                    resume_next_batch_index
                    * int(config["train"]["batch_size"])
                ):
                    raise RuntimeError(
                        "paired resume sampling audit disagrees with its local "
                        "epoch/batch position"
                    )
                resumed_completed_sampling_plan_audits = restored_audits[:-1]
        elif ordinary_successful_update_budget is not None:
            update_audit = _validate_successful_update_budget_resume(
                checkpoint, target=ordinary_successful_update_budget
            )
            budget_optimizer_attempts = int(
                update_audit["optimizer_attempt_count"]
            )
            budget_skipped_optimizer_attempts = [
                int(value)
                for value in update_audit["skipped_optimizer_attempt_indices"]
            ]
            if global_optimizer_step >= ordinary_successful_update_budget:
                raise RuntimeError(
                    "non-terminal successful-update checkpoint exhausted its budget"
                )
            if int(scheduler.last_epoch) != global_optimizer_step:
                raise RuntimeError(
                    "successful-update scheduler.last_epoch does not equal "
                    "global successful optimizer updates"
                )
            ema_seed_offset = _successful_update_ema_seed_offset(checkpoint)
            if ema_model is not None and int(ema_model.n_averaged.item()) != (
                global_optimizer_step + ema_seed_offset
            ):
                raise RuntimeError(
                    "ordinary successful-update EMA count does not equal global "
                    "successful optimizer updates plus its initialization seed"
                )
            if global_optimizer_step in {
                int(value)
                for value in checkpoint.get(
                    "successful_update_milestones", []
                )
            } and global_optimizer_step < ordinary_successful_update_budget:
                _publish_successful_update_milestone(
                    checkpoint,
                    checkpoint_dir=checkpoint_dir,
                    milestone=global_optimizer_step,
                )
        best_miou = float(checkpoint["best_miou"])
        best_epoch = int(checkpoint.get("best_epoch", -1))
        _reconcile_best_checkpoint(
            checkpoint,
            checkpoint_dir / "best.pt",
            expected_config_fingerprint=config_fingerprint(config),
        )
        _reconcile_ema_milestones(
            checkpoint,
            checkpoint_dir,
            milestone_epochs=ema_milestone_epochs,
            expected_config_fingerprint=config_fingerprint(config),
        )

    train_loader = build_dataloader(
        config,
        "train",
        start_batch_index=resume_next_batch_index,
    )
    val_loader = (
        build_dataloader(config, "val") if validation_enabled else None
    )
    # For an epoch-complete checkpoint the next uninterrupted action is loader
    # iterator construction, so restore before entering the loop.  Mid-epoch
    # RNG is restored inside _run_epoch after its replacement iterator exists.
    if resume_rng_state is not None and resume_next_batch_index == 0:
        _restore_rng_state(resume_rng_state)

    staged_unfreeze_epoch = config["train"].get("staged_unfreeze_epoch")
    continuation_epoch_offset = int(
        config["train"].get("continuation_epoch_offset", 0)
    )

    def apply_staged_freeze(epoch: int) -> bool:
        if not staged_modules:
            return False
        should_unfreeze = staged_unfreeze_epoch is not None and epoch >= int(
            staged_unfreeze_epoch
        )
        for parameter in planned_staged_parameters:
            parameter.requires_grad_(should_unfreeze)
        return should_unfreeze

    apply_staged_freeze(start_epoch + continuation_epoch_offset)

    metrics_path = output / "metrics.jsonl"
    started = time.time()
    objective = str(config["train"].get("objective", "joint"))
    selection_metric = str(config["train"].get("selection_metric", "mean_iou"))
    data_provenance = _checkpoint_data_provenance(config)
    recovery_interval_steps = int(
        config["train"].get("recovery_checkpoint_interval_steps", 0)
    )
    if recovery_interval_steps < 0:
        raise ValueError(
            "train.recovery_checkpoint_interval_steps must be non-negative"
        )
    successful_update_budget = (
        int(paired_continuation["max_successful_optimizer_updates"])
        if paired_continuation is not None
        else ordinary_successful_update_budget
    )
    successful_update_milestones = {
        int(value)
        for value in config["train"].get(
            "successful_update_milestones", []
        )
    }
    configured_budget = config["train"].get("max_successful_optimizer_updates")
    if successful_update_budget is not None and (
        configured_budget is None
        or int(configured_budget) != successful_update_budget
    ):
        raise RuntimeError(
            "runtime successful-update budget differs from the resolved config"
        )
    termination_signal: list[int | None] = [None]

    def request_graceful_checkpoint(signum: int, _frame: object) -> None:
        # Signal handlers must not serialize a multi-gigabyte checkpoint while
        # autograd/Adam is mutating it.  Record the request and publish at the
        # next accumulation/optimizer boundary instead.
        if termination_signal[0] is None:
            termination_signal[0] = int(signum)
            print(
                json.dumps(
                    {
                        "event": "termination_signal_received",
                        "signal": int(signum),
                        "action": "checkpoint_at_next_optimizer_boundary",
                    }
                ),
                flush=True,
            )

    for signal_name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is not None:
            signal.signal(signal_value, request_graceful_checkpoint)

    last_periodic_recovery_step = global_optimizer_step
    completed_sampling_plan_audits = resumed_completed_sampling_plan_audits
    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        effective_epoch = epoch + continuation_epoch_offset
        if epoch != start_epoch and resume_next_batch_index:
            # The offset applies only to the interrupted epoch.  Subsequent
            # epochs must again traverse the complete deterministic dataset.
            train_loader = build_dataloader(config, "train")
            resume_next_batch_index = 0
        staged_unfrozen = apply_staged_freeze(effective_epoch)
        sar_pretrain_epochs = int(
            config["train"].get("sar_pretrain_epochs", 0)
        )
        router_warmup_epochs = int(
            config["train"].get("router_warmup_epochs", 0)
        )
        if objective == "sar_native":
            train_phase = "sar_native"
        elif effective_epoch < sar_pretrain_epochs:
            train_phase = "sar_pretrain"
        elif (
            isinstance(model, SymmSafeFusion)
            and effective_epoch < router_warmup_epochs
        ):
            train_phase = "expert_warmup"
        else:
            train_phase = "joint"
        if isinstance(model, SymmSafeFusion):
            configured_routing_mode = str(
                config["model"].get("routing_mode", "learned")
            )
            model.set_routing_mode(
                "uniform"
                if train_phase == "expert_warmup"
                else configured_routing_mode
            )
        current_learning_rate = optimizer.param_groups[0]["lr"]
        if paired_continuation is not None:
            train_loader.dataset.set_epoch(
                effective_epoch, class_schedule_epoch=epoch
            )
        else:
            train_loader.dataset.set_epoch(effective_epoch)

        def current_sampling_audits(
            *, consumed_samples: int
        ) -> list[dict[str, Any]] | None:
            if paired_continuation is None:
                return None
            audit_method = getattr(
                train_loader.dataset, "sampling_plan_audit", None
            )
            if not callable(audit_method):
                raise RuntimeError(
                    "paired continuation dataset has no sampling plan audit"
                )
            current = dict(
                audit_method(consumed_samples=int(consumed_samples))
            )
            if (
                int(current.get("epoch", -1)) != effective_epoch
                or int(current.get("class_schedule_epoch", -1)) != epoch
            ):
                raise RuntimeError(
                    "paired continuation sampling audit epoch identity mismatch"
                )
            return [*completed_sampling_plan_audits, current]

        def recovery_boundary(
            epoch_progress: dict[str, Any], optimizer_updated: bool
        ) -> None:
            nonlocal global_optimizer_step, last_periodic_recovery_step
            nonlocal paired_optimizer_attempts
            nonlocal budget_optimizer_attempts
            nonlocal best_miou, best_epoch
            # _run_epoch receives the effective phase/data epoch (25+local),
            # while the outer checkpoint/loop resumes by local epoch. Persist
            # both explicitly and keep the legacy `epoch` key local so the
            # existing offset sampler validator can resume without ambiguity.
            epoch_progress = _paired_checkpoint_epoch_progress(
                epoch_progress,
                local_epoch=epoch,
                effective_epoch=effective_epoch,
            )
            if paired_continuation is not None:
                paired_optimizer_attempts += 1
                if not optimizer_updated:
                    paired_skipped_optimizer_attempts.append(
                        paired_optimizer_attempts
                    )
            elif ordinary_successful_update_budget is not None:
                budget_optimizer_attempts += 1
                if not optimizer_updated:
                    budget_skipped_optimizer_attempts.append(
                        budget_optimizer_attempts
                    )
            if optimizer_updated:
                global_optimizer_step += 1
            if (
                successful_update_budget is not None
                and global_optimizer_step > successful_update_budget
            ):
                raise RuntimeError(
                    "training advanced beyond its successful-update budget"
                )
            budget_reached = (
                optimizer_updated
                and successful_update_budget is not None
                and global_optimizer_step == successful_update_budget
            )
            milestone_reached = (
                optimizer_updated
                and ordinary_successful_update_budget is not None
                and global_optimizer_step in successful_update_milestones
                and global_optimizer_step < ordinary_successful_update_budget
            )
            periodic = (
                optimizer_updated
                and recovery_interval_steps > 0
                and global_optimizer_step % recovery_interval_steps == 0
                and global_optimizer_step != last_periodic_recovery_step
                and int(epoch_progress["next_batch_index"])
                < int(epoch_progress["total_batches"])
            )
            interrupted = termination_signal[0] is not None
            if (
                not periodic
                and not interrupted
                and not budget_reached
                and not milestone_reached
            ):
                return
            if (
                budget_reached
                and paired_continuation is None
                and selection_metric == "last_epoch"
            ):
                # For a strict budget, the budget boundary is the precommitted
                # terminal selection even when it lies inside the attempt-
                # capacity epoch. Keep best.pt metadata honest about its source.
                best_miou = float(epoch + 1)
                best_epoch = epoch
            state = _training_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema_model=ema_model,
                config=config,
                epoch=epoch,
                best_miou=best_miou,
                best_epoch=best_epoch,
                global_optimizer_step=global_optimizer_step,
                data_provenance=data_provenance,
                epoch_complete=False,
                epoch_progress=epoch_progress,
                paired_continuation=paired_continuation,
                successful_update_budget=(
                    ordinary_successful_update_budget
                    if paired_continuation is None
                    else None
                ),
                budget_complete=budget_reached,
                sampling_plan_audits=current_sampling_audits(
                    consumed_samples=int(epoch_progress["next_batch_index"])
                    * int(config["train"]["batch_size"])
                ),
                optimizer_update_audit=(
                    _paired_optimizer_update_audit(
                        successful_updates=global_optimizer_step,
                        optimizer_attempts=paired_optimizer_attempts,
                        skipped_attempt_indices=(
                            paired_skipped_optimizer_attempts
                        ),
                    )
                    if paired_continuation is not None
                    else (
                        _successful_optimizer_update_audit(
                            successful_updates=global_optimizer_step,
                            optimizer_attempts=budget_optimizer_attempts,
                            skipped_attempt_indices=(
                                budget_skipped_optimizer_attempts
                            ),
                        )
                        if ordinary_successful_update_budget is not None
                        else None
                    )
                ),
            )
            _atomic_save_rolling_recovery(
                state, checkpoint_dir=checkpoint_dir, config=config
            )
            last_periodic_recovery_step = global_optimizer_step
            if milestone_reached:
                milestone_path = _publish_successful_update_milestone(
                    state,
                    checkpoint_dir=checkpoint_dir,
                    milestone=global_optimizer_step,
                )
                print(
                    json.dumps(
                        {
                            "event": "successful_update_milestone_published",
                            "global_optimizer_step": global_optimizer_step,
                            "checkpoint": str(milestone_path),
                            "checkpoint_sha256": sha256_file(milestone_path),
                            "primary_precommitted": (
                                _is_primary_successful_update(
                                    config["train"], global_optimizer_step
                                )
                            ),
                        }
                    ),
                    flush=True,
                )
            print(
                json.dumps(
                    {
                        "event": "recovery_checkpoint_saved",
                        "epoch": epoch,
                        "next_batch_index": int(
                            epoch_progress["next_batch_index"]
                        ),
                        "global_optimizer_step": global_optimizer_step,
                        "optimizer_attempt_count": (
                            paired_optimizer_attempts
                            if paired_continuation is not None
                            else (
                                budget_optimizer_attempts
                                if ordinary_successful_update_budget
                                is not None
                                else None
                            )
                        ),
                        "reason": (
                            "successful_update_budget"
                            if budget_reached
                            else "signal"
                            if interrupted
                            else "successful_update_milestone"
                            if milestone_reached
                            else "periodic"
                        ),
                    }
                ),
                flush=True,
            )
            if budget_reached:
                if paired_continuation is not None:
                    _publish_paired_budget_completion(
                        state,
                        checkpoint_dir=checkpoint_dir,
                        output=output,
                        elapsed_seconds=time.time() - started,
                    )
                    completion_event = (
                        "paired_successful_update_budget_complete"
                    )
                else:
                    _publish_successful_update_budget_completion(
                        state,
                        checkpoint_dir=checkpoint_dir,
                        output=output,
                        elapsed_seconds=time.time() - started,
                    )
                    completion_event = "successful_update_budget_complete"
                print(
                    json.dumps(
                        {
                            "event": completion_event,
                            "arm": (
                                paired_continuation["arm"]
                                if paired_continuation is not None
                                else None
                            ),
                            "global_optimizer_step": global_optimizer_step,
                            "optimizer_attempt_count": (
                                paired_optimizer_attempts
                                if paired_continuation is not None
                                else budget_optimizer_attempts
                            ),
                            "skipped_optimizer_attempt_count": (
                                len(paired_skipped_optimizer_attempts)
                                if paired_continuation is not None
                                else len(budget_skipped_optimizer_attempts)
                            ),
                            "epoch": epoch,
                            "effective_epoch": effective_epoch,
                            "next_batch_index": int(
                                epoch_progress["next_batch_index"]
                            ),
                        }
                    ),
                    flush=True,
                )
                raise _SuccessfulUpdateBudgetComplete
            if interrupted:
                raise SystemExit(128 + int(termination_signal[0]))

        try:
            train_loss, train_metrics = _run_epoch(
                model=model,
                loader=train_loader,
                config=config,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                teacher=teacher,
                teacher_amp_enabled=teacher_amp_enabled,
                ema_model=ema_model,
                phase=train_phase,
                epoch_index=effective_epoch,
                checkpoint_epoch_index=epoch,
                progress_label="train",
                start_batch_index=(
                    resume_next_batch_index if epoch == start_epoch else 0
                ),
                initial_epoch_progress=(
                    resume_epoch_progress if epoch == start_epoch else None
                ),
                resume_rng_state=(
                    resume_rng_state
                    if epoch == start_epoch and resume_next_batch_index
                    else None
                ),
                recovery_boundary_callback=recovery_boundary,
            )
        except _SuccessfulUpdateBudgetComplete:
            return
        resume_epoch_progress = None
        resume_rng_state = None
        completed_sampling_plan_audits = (
            current_sampling_audits(consumed_samples=len(train_loader.dataset))
            or []
        )
        validation_model = ema_model.module if ema_model is not None else model
        if val_loader is not None:
            val_loss, val_metrics = _run_epoch(
                model=validation_model,
                loader=val_loader,
                config=config,
                device=device,
                optimizer=None,
                scheduler=None,
                scaler=scaler,
                teacher=teacher,
                teacher_amp_enabled=teacher_amp_enabled,
                phase=(
                    "sar_native"
                    if objective == "sar_native"
                    else (
                        "expert_warmup"
                        if train_phase == "expert_warmup"
                        else "joint"
                    )
                ),
                epoch_index=effective_epoch,
                progress_label="validation",
            )
        else:
            val_loss, val_metrics = {}, {}
        if scheduler_unit == "epoch":
            scheduler.step()
        record = {
            "epoch": epoch,
            "effective_epoch": effective_epoch,
            "train_phase": train_phase,
            "routing_mode": (
                model.routing_mode
                if isinstance(model, SymmSafeFusion)
                else None
            ),
            "staged_modules_unfrozen": staged_unfrozen,
            "learning_rate": current_learning_rate,
            "learning_rates": [
                group["lr"] for group in optimizer.param_groups
            ],
            "next_learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
            "train": {**train_loss, **train_metrics},
            "val": {**val_loss, **val_metrics},
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        if selection_metric == "last_epoch":
            validation_score = float(epoch + 1)
        else:
            if selection_metric not in val_metrics:
                raise RuntimeError(
                    f"selection metric {selection_metric!r} was not produced"
                )
            validation_score = float(val_metrics[selection_metric])
        checkpoint_eligible = paired_continuation is None and (
            objective == "sar_native" or train_phase == "joint"
        )
        is_new_best = checkpoint_eligible and validation_score > best_miou
        if is_new_best:
            best_miou = validation_score
            best_epoch = epoch
        state = _training_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema_model=ema_model,
            config=config,
            epoch=epoch,
            best_miou=best_miou,
            best_epoch=best_epoch,
            global_optimizer_step=global_optimizer_step,
            data_provenance=data_provenance,
            epoch_complete=True,
            paired_continuation=paired_continuation,
            successful_update_budget=(
                ordinary_successful_update_budget
                if paired_continuation is None
                else None
            ),
            sampling_plan_audits=(
                completed_sampling_plan_audits
                if paired_continuation is not None
                else None
            ),
            optimizer_update_audit=(
                _paired_optimizer_update_audit(
                    successful_updates=global_optimizer_step,
                    optimizer_attempts=paired_optimizer_attempts,
                    skipped_attempt_indices=paired_skipped_optimizer_attempts,
                )
                if paired_continuation is not None
                else (
                    _successful_optimizer_update_audit(
                        successful_updates=global_optimizer_step,
                        optimizer_attempts=budget_optimizer_attempts,
                        skipped_attempt_indices=(
                            budget_skipped_optimizer_attempts
                        ),
                    )
                    if ordinary_successful_update_budget is not None
                    else None
                )
            ),
        )
        _atomic_save_rolling_recovery(
            state, checkpoint_dir=checkpoint_dir, config=config
        )
        if epoch + 1 in ema_milestone_epochs:
            milestone_path = checkpoint_dir / f"ema_epoch_{epoch + 1:04d}.pt"
            written = _publish_ema_milestone(
                state,
                milestone_path,
                completed_epochs=epoch + 1,
            )
            print(
                json.dumps(
                    {
                        "event": "ema_milestone_saved",
                        "completed_epochs": epoch + 1,
                        "path": str(milestone_path),
                        "written": written,
                    }
                ),
                flush=True,
            )
        if (epoch + 1) % int(config["train"]["checkpoint_every"]) == 0:
            atomic_torch_save(
                state, checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
            )
        if is_new_best:
            atomic_torch_save(state, checkpoint_dir / "best.pt")
        if termination_signal[0] is not None:
            print(
                json.dumps(
                    {
                        "event": "termination_checkpoint_saved",
                        "epoch": epoch,
                        "epoch_complete": True,
                        "global_optimizer_step": global_optimizer_step,
                    }
                ),
                flush=True,
            )
            raise SystemExit(128 + int(termination_signal[0]))

    if successful_update_budget is not None:
        run_kind = (
            "paired continuation"
            if paired_continuation is not None
            else "ordinary strict-budget training"
        )
        raise RuntimeError(
            f"{run_kind} exhausted its attempt-capacity epochs before reaching "
            f"exactly {successful_update_budget} successful optimizer updates; "
            f"last successful count={global_optimizer_step}"
        )

    atomic_json_dump(
        {
            "status": "completed",
            "selection_metric": selection_metric,
            "best_val_score": best_miou,
            "best_epoch": best_epoch,
            "best_val_mean_iou": (
                best_miou if selection_metric == "mean_iou" else None
            ),
            "epochs": int(config["train"]["epochs"]),
            "ema_milestone_epochs": ema_milestone_epochs,
            "ema_milestone_checkpoints": [
                str(checkpoint_dir / f"ema_epoch_{epoch:04d}.pt")
                for epoch in ema_milestone_epochs
            ],
            "elapsed_seconds": time.time() - started,
        },
        output / "completed.json",
    )


if __name__ == "__main__":
    main()
