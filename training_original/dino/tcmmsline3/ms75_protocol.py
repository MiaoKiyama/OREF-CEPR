"""Immutable protocol checks for the complete 70.1564 CASTR anchor.

These checks are intentionally narrower than the general Line3 configuration
validator.  They describe the one authenticated fbponlyms artifact used to
initialize MS75 and fail closed when a seemingly harmless inference or data
change would make its published whole-scene result irreproducible.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .ms75_data import OfficialFBPProtocol


MS75_ANCHOR_SHA256 = (
    "0060b69245e30d2753be72fa60ae1c99ff4d24bf8f018fee8ed7a6520c8d6cf6"
)
MS75_ANCHOR_TENSOR_COUNT = 596
MS75_SOURCE_GIT_REF = "baseline-ms-70.1564"
MS75_SOURCE_GIT_COMMIT = "5f0075bfea3c4296472a863dd8d679ce777e2856"
MS75_SOURCE_MEAN_NRGB = (0.506, 0.371, 0.390, 0.363)
MS75_SOURCE_STD_NRGB = (0.254, 0.244, 0.231, 0.231)
MS75_LINE3_MEAN_BGRN = (0.363, 0.390, 0.371, 0.506)
MS75_LINE3_STD_BGRN = (0.231, 0.231, 0.244, 0.254)

MS75_FORMAL_FROZEN_MODULES = (
    "local_encoder",
    "context_encoder",
    "context_projection",
    "global_head",
    "structure_input",
    "structure_blocks",
)
MS75_FORMAL_STAGED_MODULES = (
    "decoder",
    "local_head",
    "context_fusion",
    "structure_fusion",
    "context_refine",
    "structure_refine",
    "context_head",
    "structure_head",
    "router",
)
MS75_FORMAL_EMA_MILESTONES = (5, 10, 15, 20, 25)
MS75_FORMAL_POST_WARMUP_TRAINABLE_PARAMETERS = 28_893_927

# The evaluation-ready EMA state published as the historical Line3 71.322573
# no-TTA baseline.  A paired continuation is intentionally pinned to this
# exact file rather than to a mutable "best.pt" pathname.
MS75_PAIRED_CONTINUATION_BASE_SHA256 = (
    "dc52e3c98a9624cb5bfd9c40ef58402ef7e08cff0556a68e5641d782f77a6bf5"
)
MS75_PAIRED_CONTINUATION_UPDATES = 20_000
MS75_PAIRED_CONTINUATION_EPOCH_OFFSET = 25
MS75_PAIRED_CONTINUATION_WARMUP_STEPS = 500

MS75_ANCHOR_PREFIX_COUNTS = {
    "context_encoder": 120,
    "context_fusion": 27,
    "context_head": 2,
    "context_projection": 3,
    "context_refine": 3,
    "decoder": 36,
    "global_head": 2,
    "local_encoder": 325,
    "local_head": 2,
    "router": 5,
    "structure_blocks": 36,
    "structure_fusion": 27,
    "structure_head": 2,
    "structure_input": 3,
    "structure_refine": 3,
}

# FBP Table 4, in the canonical order distributed with the official code.
MS75_OFFICIAL_TEST_SCENE_IDS = (
    "GF2_PMS1__L1A0001064454-MSS1",
    "GF2_PMS1__L1A0001118839-MSS1",
    "GF2_PMS1__L1A0001344822-MSS1",
    "GF2_PMS1__L1A0001348919-MSS1",
    "GF2_PMS1__L1A0001366278-MSS1",
    "GF2_PMS1__L1A0001366284-MSS1",
    "GF2_PMS1__L1A0001395956-MSS1",
    "GF2_PMS1__L1A0001432972-MSS1",
    "GF2_PMS1__L1A0001670888-MSS1",
    "GF2_PMS1__L1A0001680857-MSS1",
    "GF2_PMS1__L1A0001680858-MSS1",
    "GF2_PMS1__L1A0001757429-MSS1",
    "GF2_PMS1__L1A0001765574-MSS1",
    "GF2_PMS2__L1A0000607677-MSS2",
    "GF2_PMS2__L1A0000607681-MSS2",
    "GF2_PMS2__L1A0000718813-MSS2",
    "GF2_PMS2__L1A0001038935-MSS2",
    "GF2_PMS2__L1A0001038936-MSS2",
    "GF2_PMS2__L1A0001119060-MSS2",
    "GF2_PMS2__L1A0001367840-MSS2",
    "GF2_PMS2__L1A0001378491-MSS2",
    "GF2_PMS2__L1A0001378501-MSS2",
    "GF2_PMS2__L1A0001396036-MSS2",
    "GF2_PMS2__L1A0001396037-MSS2",
    "GF2_PMS2__L1A0001416129-MSS2",
    "GF2_PMS2__L1A0001471436-MSS2",
    "GF2_PMS2__L1A0001517494-MSS2",
    "GF2_PMS2__L1A0001591676-MSS2",
    "GF2_PMS2__L1A0001787564-MSS2",
    "GF2_PMS2__L1A0001821754-MSS2",
)


def _require_equal(name: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ValueError(f"{name} must be {expected!r}, found {observed!r}")


def _require_float_sequence(
    name: str,
    observed: object,
    expected: Sequence[float],
) -> None:
    try:
        values = tuple(float(value) for value in observed)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric sequence") from error
    if values != tuple(expected):
        raise ValueError(
            f"{name} must be {tuple(expected)!r}, found {values!r}"
        )


def validate_official_fbp_table4_protocol(
    root: str | Path,
) -> OfficialFBPProtocol:
    """Require the exact ordered 120/30 protocol and official Table 4 test."""

    protocol = OfficialFBPProtocol.from_root(root)
    if protocol.test_scene_ids != MS75_OFFICIAL_TEST_SCENE_IDS:
        raise ValueError(
            "FBP test split is not the exact ordered official Table 4 list"
        )
    if len(protocol.all_scene_ids) != 150:
        raise ValueError("FBP all split must contain exactly 150 scenes")
    return protocol


def validate_ms75_formal_protocol(
    config: Mapping[str, Any],
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the exact data/model/inference boundary of formal MS75."""

    data = config.get("data")
    model = config.get("model")
    loss = config.get("loss")
    train = config.get("train")
    inference = config.get("inference")
    if not isinstance(data, Mapping) or not isinstance(model, Mapping):
        raise ValueError("MS75 protocol requires data and model mappings")
    if not isinstance(loss, Mapping) or not isinstance(train, Mapping):
        raise ValueError("MS75 protocol requires loss and train mappings")
    if not isinstance(inference, Mapping):
        raise ValueError("MS75 protocol requires an inference mapping")

    required_data = {
        "dataset_format": "fbp_ms",
        "dataset_variant": "ms75_multiscale",
        "protocol_stage": "final",
        "official_120_30_protocol": True,
        "expected_scene_count": 150,
        "expected_hw": [6908, 7300],
        "train_split": "train",
        "val_split": None,
        "test_split": "test",
        "load_sar": False,
        "patch_size": 512,
        "global_input_size": 512,
    }
    for key, expected in required_data.items():
        _require_equal(f"data.{key}", data.get(key), expected)

    normalization = data.get("normalization")
    if not isinstance(normalization, Mapping) or not isinstance(
        normalization.get("ms"), Mapping
    ):
        raise ValueError("data.normalization.ms is required")
    ms_normalization = normalization["ms"]
    _require_equal(
        "data.normalization.ms.divisor", ms_normalization.get("divisor"), 255.0
    )
    _require_float_sequence(
        "data.normalization.ms.mean",
        ms_normalization.get("mean"),
        MS75_LINE3_MEAN_BGRN,
    )
    _require_float_sequence(
        "data.normalization.ms.std",
        ms_normalization.get("std"),
        MS75_LINE3_STD_BGRN,
    )

    _require_equal(
        "model.architecture", model.get("architecture"), "ms75_multiscale"
    )
    _require_equal("model.fusion_mode", model.get("fusion_mode"), "ms_only")
    _require_equal("model.num_classes", model.get("num_classes"), 24)
    ms75 = model.get("ms75")
    if not isinstance(ms75, Mapping):
        raise ValueError("model.ms75 is required")
    _require_equal("model.ms75.num_classes", ms75.get("num_classes"), 24)
    _require_equal(
        "model.ms75.freeze_context_batch_norm_stats",
        ms75.get("freeze_context_batch_norm_stats"),
        True,
    )
    local_encoder = ms75.get("local_encoder")
    context_encoder = ms75.get("context_encoder", ms75.get("global_encoder"))
    if not isinstance(local_encoder, Mapping) or not isinstance(
        context_encoder, Mapping
    ):
        raise ValueError(
            "MS75 local and complete-anchor context encoders are required"
        )
    _require_equal(
        "model.ms75.local_encoder.band_order",
        local_encoder.get("band_order"),
        ["NIR", "R", "G", "B"],
    )
    _require_equal(
        "model.ms75.context_encoder.band_order",
        context_encoder.get("band_order"),
        ["NIR", "R", "G", "B"],
    )
    required_local_encoder = {
        "type": "timm",
        "name": "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
        "in_channels": 4,
        "pretrained": False,
        "img_size": 512,
        "out_indices": [0, 1, 2, 3],
    }
    required_context_encoder = {
        "type": "timm",
        "name": "resnet18.a1_in1k",
        "in_channels": 4,
        "pretrained": False,
        "out_indices": [1, 2, 3, 4],
    }
    for key, expected in required_local_encoder.items():
        _require_equal(
            f"model.ms75.local_encoder.{key}", local_encoder.get(key), expected
        )
    for key, expected in required_context_encoder.items():
        _require_equal(
            f"model.ms75.context_encoder.{key}",
            context_encoder.get(key),
            expected,
        )
    for key, expected in {
        "decoder_channels": 256,
        "pool_scales": [1, 2, 3, 6],
        "evidence_channels": 128,
        "attention_channels": 128,
        "attention_heads": 4,
        "max_context_grid": 16,
        "structure_groups": [
            "water",
            "vegetation",
            "built",
            "transport",
            "snow_bare",
        ],
        "structure_layers": 2,
        "structure_dilations": [1, 2, 4, 8],
        "router_signal_mode": "local_entropy",
        "router_detach_signals": False,
    }.items():
        _require_equal(f"model.ms75.{key}", ms75.get(key), expected)

    required_training = {
        "epochs": 25,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "amp": True,
        "use_ema": True,
        "ema_decay": 0.9999,
        "ema_milestone_epochs": list(MS75_FORMAL_EMA_MILESTONES),
        "ms75_anchor_source_sha256": MS75_ANCHOR_SHA256,
        "ms75_anchor_destination_sha256": MS75_ANCHOR_SHA256,
        "freeze_modules": list(MS75_FORMAL_FROZEN_MODULES),
        "staged_unfreeze_modules": list(MS75_FORMAL_STAGED_MODULES),
        "staged_unfreeze_epoch": 5,
        "staged_lr_multiplier": 0.05,
        "fixed_anchor_teacher_enabled": True,
        "fixed_anchor_teacher_amp": True,
        "validation_enabled": False,
        "selection_metric": "last_epoch",
    }
    for key, expected in required_training.items():
        _require_equal(f"train.{key}", train.get(key), expected)
    if not train.get("ms75_anchor_checkpoint"):
        raise ValueError("train.ms75_anchor_checkpoint is required")

    required_no_harm = {
        "fixed_anchor_no_harm_weight": 0.25,
        "fixed_anchor_no_harm_confidence": 0.70,
        "fixed_anchor_no_harm_temperature": 2.0,
        "teacher_distillation_weight": 0.0,
    }
    for key, expected in required_no_harm.items():
        _require_equal(f"loss.{key}", loss.get(key), expected)

    required_inference = {
        "batch_size": 2,
        "max_windows_per_scene": None,
        "tile_stride": 256,
        "tta_transforms": ["none"],
        "blend_window": "hann",
        "hann_floor": 0.05,
        "amp": True,
        "routing_mode": "learned",
        "protocol_role": "complete_anchor_formal",
    }
    for key, expected in required_inference.items():
        _require_equal(f"inference.{key}", inference.get(key), expected)

    protocol = None
    if data_root is not None:
        protocol = validate_official_fbp_table4_protocol(data_root)
    return {
        "official_120_30": True,
        "train_scene_count": (
            len(protocol.train_scene_ids) if protocol is not None else 120
        ),
        "test_scene_count": (
            len(protocol.test_scene_ids) if protocol is not None else 30
        ),
        "test_scene_ids": list(MS75_OFFICIAL_TEST_SCENE_IDS),
        "channel_contract": {
            "line3": "B,G,R,NIR",
            "complete_anchor": "NIR,R,G,B",
            "permutation": [3, 2, 1, 0],
        },
        "whole_scene_context_size": 512,
        "normalized_boxes": "[y0/H,x0/W,y1/H,x1/W]",
        "training_protection": {
            "fixed_teacher": True,
            "teacher_amp": True,
            "permanently_frozen_modules": list(MS75_FORMAL_FROZEN_MODULES),
            "staged_modules": list(MS75_FORMAL_STAGED_MODULES),
            "staged_unfreeze_epoch": 5,
            "staged_lr_multiplier": 0.05,
            "no_harm_weight": 0.25,
            "no_harm_confidence": 0.70,
            "no_harm_temperature": 2.0,
            "ema_milestone_epochs": list(MS75_FORMAL_EMA_MILESTONES),
        },
        "inference": dict(required_inference),
    }


def validate_ms75_paired_continuation_protocol(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed around the paired 71-model fixed-budget continuation.

    This role deliberately does not reuse :func:`validate_ms75_formal_protocol`:
    the historical 25-epoch fit remains immutable, while this experiment has
    a fresh optimizer and terminates by successful GradScaler updates rather
    than by an epoch-selection checkpoint.
    """

    data = config.get("data")
    model = config.get("model")
    loss = config.get("loss")
    train = config.get("train")
    inference = config.get("inference")
    if not all(
        isinstance(value, Mapping)
        for value in (data, model, loss, train, inference)
    ):
        raise ValueError(
            "paired MS75 continuation requires data/model/loss/train/inference mappings"
        )
    assert isinstance(data, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(loss, Mapping)
    assert isinstance(train, Mapping)
    assert isinstance(inference, Mapping)

    for key, expected in {
        "dataset_format": "fbp_ms",
        "dataset_variant": "ms75_multiscale",
        "protocol_stage": "final",
        "official_120_30_protocol": True,
        "expected_scene_count": 150,
        "train_split": "train",
        "val_split": None,
        "test_split": "test",
        "load_sar": False,
    }.items():
        _require_equal(f"data.{key}", data.get(key), expected)
    _require_equal(
        "model.architecture", model.get("architecture"), "ms75_multiscale"
    )
    _require_equal("model.num_classes", model.get("num_classes"), 24)

    dynamic = data.get("dynamic_sampling")
    if not isinstance(dynamic, Mapping):
        raise ValueError("paired continuation requires dynamic_sampling")
    for key, expected in {
        "scene_fraction": 0.60,
        "class_fraction": 0.20,
        "confusion_fraction": 0.20,
        "center_jitter": 128,
        "max_scene_repeat": 64,
    }.items():
        _require_equal(
            f"data.dynamic_sampling.{key}", dynamic.get(key), expected
        )
    for key in (
        "class_index",
        "class_index_sha256",
        "confusion_index",
        "confusion_index_sha256",
        "class_probability_manifest",
        "class_probability_manifest_sha256",
    ):
        if not str(dynamic.get(key, "")).strip():
            raise ValueError(f"data.dynamic_sampling.{key} is required")

    arm = str(train.get("paired_continuation_arm", ""))
    if arm not in {"A0", "A1"}:
        raise ValueError("train.paired_continuation_arm must be A0 or A1")
    base_checkpoint = train.get("continuation_initialization_checkpoint")
    if not str(base_checkpoint or "").strip():
        raise ValueError(
            "train.continuation_initialization_checkpoint is required"
        )
    _require_equal(
        "train.continuation_initialization_expected_sha256",
        str(
            train.get("continuation_initialization_expected_sha256", "")
        ).lower(),
        MS75_PAIRED_CONTINUATION_BASE_SHA256,
    )
    pair_manifest = train.get("paired_protocol_manifest")
    pair_manifest_sha = str(
        train.get("paired_protocol_manifest_expected_sha256", "")
    ).lower()
    if not str(pair_manifest or "").strip():
        raise ValueError("train.paired_protocol_manifest is required")
    if len(pair_manifest_sha) != 64 or any(
        character not in "0123456789abcdef" for character in pair_manifest_sha
    ):
        raise ValueError(
            "train.paired_protocol_manifest_expected_sha256 must be lowercase SHA256"
        )

    required_training = {
        "device": "cuda",
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.000015,
        "backbone_lr_multiplier": 0.30,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "scheduler_unit": "step",
        "warmup_steps": MS75_PAIRED_CONTINUATION_WARMUP_STEPS,
        "max_successful_optimizer_updates": MS75_PAIRED_CONTINUATION_UPDATES,
        "continuation_epoch_offset": MS75_PAIRED_CONTINUATION_EPOCH_OFFSET,
        "amp": True,
        "use_ema": True,
        "ema_decay": 0.9999,
        "ema_milestone_epochs": [],
        "staged_unfreeze_epoch": 5,
        "staged_lr_multiplier": 0.05,
        "freeze_modules": list(MS75_FORMAL_FROZEN_MODULES),
        "staged_unfreeze_modules": list(MS75_FORMAL_STAGED_MODULES),
        "family_contrast_start_epoch": 18,
        "family_contrast_ramp_epochs": 3,
        "fixed_anchor_teacher_enabled": True,
        "fixed_anchor_teacher_amp": True,
        "validation_enabled": False,
        "selection_metric": "last_epoch",
        "auto_resume": True,
        "resume_checkpoint": None,
        "initialization_checkpoint": None,
        "ms_initialization_checkpoint": None,
        "sar_initialization_checkpoint": None,
        "cudnn_benchmark": False,
        "deterministic_algorithms": True,
    }
    for key, expected in required_training.items():
        _require_equal(f"train.{key}", train.get(key), expected)
    if train.get("initialization_checkpoint"):
        raise ValueError(
            "paired continuation forbids legacy initialization_checkpoint"
        )
    if train.get("initialization_exclude_prefixes"):
        raise ValueError(
            "paired continuation requires a strict complete model load"
        )
    if train.get("teacher_checkpoint"):
        raise ValueError(
            "paired continuation uses only the independent fixed anchor teacher"
        )
    if not train.get("ms75_anchor_checkpoint"):
        raise ValueError(
            "paired continuation requires the authenticated fixed teacher anchor"
        )
    _require_equal(
        "train.ms75_anchor_source_sha256",
        train.get("ms75_anchor_source_sha256"),
        MS75_ANCHOR_SHA256,
    )
    for key, expected in {
        "family_contrast_weight": 0.05,
        "fixed_anchor_no_harm_weight": 0.25,
        "fixed_anchor_no_harm_confidence": 0.70,
        "fixed_anchor_no_harm_temperature": 2.0,
        "teacher_distillation_weight": 0.0,
    }.items():
        _require_equal(f"loss.{key}", loss.get(key), expected)
    _require_equal(
        "train.ms75_anchor_destination_sha256",
        train.get("ms75_anchor_destination_sha256"),
        MS75_ANCHOR_SHA256,
    )
    _require_equal(
        "inference.protocol_role",
        inference.get("protocol_role"),
        "paired_continuation_train",
    )
    _require_equal(
        "inference.evaluate_test_after_training",
        inference.get("evaluate_test_after_training"),
        False,
    )
    for key, expected in {
        "tile_stride": 256,
        "tta_transforms": ["none"],
        "blend_window": "hann",
        "hann_floor": 0.05,
        "amp": True,
        "routing_mode": "learned",
    }.items():
        _require_equal(f"inference.{key}", inference.get(key), expected)

    samples_per_epoch = int(
        data.get("debug_samples_per_epoch", data.get("samples_per_epoch", 0))
    )
    planned_updates = (
        int(train.get("epochs", 0))
        * samples_per_epoch
        // int(train.get("batch_size", 1))
    )
    if planned_updates <= MS75_PAIRED_CONTINUATION_UPDATES:
        raise ValueError(
            "paired continuation must plan more batches than its successful "
            "update budget so AMP skips cannot end the run early"
        )

    return {
        "protocol_role": "paired_continuation_train",
        "arm": arm,
        "base_checkpoint_sha256": MS75_PAIRED_CONTINUATION_BASE_SHA256,
        "pair_manifest_sha256": pair_manifest_sha,
        "successful_optimizer_updates": MS75_PAIRED_CONTINUATION_UPDATES,
        "continuation_epoch_offset": MS75_PAIRED_CONTINUATION_EPOCH_OFFSET,
        "warmup_successful_updates": MS75_PAIRED_CONTINUATION_WARMUP_STEPS,
        "official_test_during_training": False,
    }


def validate_ms75_paired_evaluation_protocol(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the two immutable historical A/B evaluation overlays.

    These roles retain the formal model/data contract but intentionally do not
    call :func:`validate_ms75_formal_protocol`, whose role field is itself part
    of the immutable publication identity.  Instead, first substitute that
    one role locally and validate every formal field, then enforce the exact
    no-TTA or Flip4 endpoint plus the historical CUDA/cuDNN execution choice.
    """

    inference = config.get("inference")
    train = config.get("train")
    if not isinstance(inference, Mapping) or not isinstance(train, Mapping):
        raise ValueError(
            "paired MS75 evaluation requires train/inference mappings"
        )
    role = str(inference.get("protocol_role", ""))
    if role not in {
        "paired_continuation_no_tta_evaluation",
        "paired_continuation_flip4_evaluation",
    }:
        raise ValueError("invalid paired MS75 evaluation role")
    _require_equal("train.device", train.get("device"), "cuda")
    _require_equal(
        "train.cudnn_benchmark", train.get("cudnn_benchmark"), True
    )

    formal_config = dict(config)
    formal_inference = dict(inference)
    formal_inference["protocol_role"] = "complete_anchor_formal"
    formal_inference["tta_transforms"] = ["none"]
    formal_inference["batch_size"] = 2
    formal_config["inference"] = formal_inference
    validate_ms75_formal_protocol(formal_config)

    expected_transforms = (
        ["none"]
        if role == "paired_continuation_no_tta_evaluation"
        else ["none", "horizontal", "vertical", "both"]
    )
    expected_batch_size = (
        2 if role == "paired_continuation_no_tta_evaluation" else 1
    )
    _require_equal(
        "inference.tta_transforms",
        list(inference.get("tta_transforms", [])),
        expected_transforms,
    )
    _require_equal(
        "inference.batch_size", inference.get("batch_size"), expected_batch_size
    )
    for key, expected in {
        "max_windows_per_scene": None,
        "tile_stride": 256,
        "blend_window": "hann",
        "hann_floor": 0.05,
        "amp": True,
        "routing_mode": "learned",
    }.items():
        _require_equal(f"inference.{key}", inference.get(key), expected)
    return {
        "protocol_role": role,
        "official_120_30": True,
        "num_classes": 24,
        "tta_transforms": expected_transforms,
        "historical_execution": {
            "device": "cuda",
            "cudnn_benchmark": True,
        },
    }


def validate_ms75_source_checkpoint(
    checkpoint: object,
) -> dict[str, Any]:
    """Validate the authenticated checkpoint's complete 596-tensor graph."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("MS75 source checkpoint must be a mapping")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("MS75 source checkpoint must contain model state")
    non_tensor = sorted(
        str(key)
        for key, value in state.items()
        if not isinstance(value, torch.Tensor)
    )
    if non_tensor:
        raise ValueError(
            f"MS75 model state contains non-tensors: {non_tensor[:5]}"
        )
    if any(str(key).startswith(("module.", "_orig_mod.")) for key in state):
        raise ValueError(
            "authenticated MS75 state must use canonical unwrapped keys"
        )
    counts = Counter(str(key).split(".", 1)[0] for key in state)
    observed_counts = dict(sorted(counts.items()))
    if len(state) != MS75_ANCHOR_TENSOR_COUNT:
        raise ValueError(
            f"complete MS75 anchor requires {MS75_ANCHOR_TENSOR_COUNT} tensors, "
            f"found {len(state)}"
        )
    if observed_counts != MS75_ANCHOR_PREFIX_COUNTS:
        raise ValueError(
            "complete MS75 anchor prefix counts differ from the authenticated "
            f"596-tensor graph: {observed_counts!r}"
        )

    source_config = checkpoint.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("MS75 source checkpoint has no embedded config")
    data = source_config.get("data")
    model = source_config.get("model")
    inference = source_config.get("inference")
    if not all(
        isinstance(value, Mapping) for value in (data, model, inference)
    ):
        raise ValueError(
            "MS75 embedded data/model/inference configs are required"
        )
    assert isinstance(data, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(inference, Mapping)
    for key, expected in {
        "train_split": "train",
        "expected_hw": [6908, 7300],
        "crop_size": 512,
        "context_size": 512,
    }.items():
        _require_equal(f"source.config.data.{key}", data.get(key), expected)
    _require_float_sequence(
        "source.config.data.mean", data.get("mean"), MS75_SOURCE_MEAN_NRGB
    )
    _require_float_sequence(
        "source.config.data.std", data.get("std"), MS75_SOURCE_STD_NRGB
    )
    for key, expected in {
        "num_classes": 24,
        "enable_context": True,
        "enable_structure": True,
        "decoder_channels": 256,
        "evidence_channels": 128,
        "attention_channels": 128,
        "attention_heads": 4,
        "max_context_grid": 16,
        "structure_layers": 2,
    }.items():
        _require_equal(f"source.config.model.{key}", model.get(key), expected)
    for encoder_name in ("local_encoder", "context_encoder"):
        encoder = model.get(encoder_name)
        if not isinstance(encoder, Mapping):
            raise ValueError(f"source.config.model.{encoder_name} is required")
        _require_equal(
            f"source.config.model.{encoder_name}.band_order",
            encoder.get("band_order"),
            ["NIR", "R", "G", "B"],
        )
    for key, expected in {"stride": 256, "batch_size": 2, "amp": True}.items():
        _require_equal(
            f"source.config.inference.{key}", inference.get(key), expected
        )
    _require_equal("source.checkpoint.stage", checkpoint.get("stage"), "joint")
    _require_equal(
        "source.checkpoint.inference_routing_mode",
        checkpoint.get("inference_routing_mode"),
        "learned",
    )
    return {
        "complete_anchor": True,
        "expected_anchor_tensor_count": MS75_ANCHOR_TENSOR_COUNT,
        "loaded_anchor_tensor_count": len(state),
        "ignored_source_tensor_count": 0,
        "source_prefix_counts": observed_counts,
        "stage": "joint",
        "routing_mode": "learned",
        "source_inference": {"stride": 256, "batch_size": 2, "amp": True},
    }


def compare_logit_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    """Return exact and decision-level equivalence diagnostics."""

    same_shape = tuple(reference.shape) == tuple(candidate.shape)
    report: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "same_shape": same_shape,
        "elementwise_exact": False,
        "max_abs_error": None,
        "max_relative_error": None,
        "different_element_count": None,
        "argmax_mismatch_pixels": None,
    }
    if not same_shape:
        return report
    left = reference.detach().cpu()
    right = candidate.detach().cpu()
    report["elementwise_exact"] = bool(torch.equal(left, right))
    left_float = left.float()
    right_float = right.float()
    difference = (left_float - right_float).abs()
    report["max_abs_error"] = (
        float(difference.max().item()) if difference.numel() else 0.0
    )
    denominator = left_float.abs().clamp_min(torch.finfo(torch.float32).tiny)
    relative = difference / denominator
    report["max_relative_error"] = (
        float(relative.max().item()) if relative.numel() else 0.0
    )
    report["different_element_count"] = int(
        torch.count_nonzero(left != right).item()
    )
    report["argmax_mismatch_pixels"] = int(
        torch.count_nonzero(
            left_float.argmax(dim=1) != right_float.argmax(dim=1)
        ).item()
    )
    return report


def require_exact_logit_equivalence(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    report = compare_logit_tensors(reference, candidate)
    if not report["elementwise_exact"]:
        raise RuntimeError(f"{name} is not elementwise exact: {report}")
    return report


__all__ = [
    "MS75_ANCHOR_PREFIX_COUNTS",
    "MS75_ANCHOR_SHA256",
    "MS75_ANCHOR_TENSOR_COUNT",
    "MS75_LINE3_MEAN_BGRN",
    "MS75_LINE3_STD_BGRN",
    "MS75_OFFICIAL_TEST_SCENE_IDS",
    "MS75_SOURCE_GIT_COMMIT",
    "MS75_SOURCE_GIT_REF",
    "MS75_SOURCE_MEAN_NRGB",
    "MS75_SOURCE_STD_NRGB",
    "compare_logit_tensors",
    "require_exact_logit_equivalence",
    "validate_ms75_formal_protocol",
    "validate_ms75_source_checkpoint",
    "validate_official_fbp_table4_protocol",
]
