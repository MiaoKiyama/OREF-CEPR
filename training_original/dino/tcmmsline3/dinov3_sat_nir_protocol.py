"""Fail-closed configuration contract for the DINOv3-SAT/NIR development expert."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from .dinov3_sat_nir_model import (
    DINOV3_SAT_RGB_MEAN,
    DINOV3_SAT_RGB_STD,
    DINOV3_SAT_VITL16_SHA256,
)


DINOV3_SAT_NIR_ARCHITECTURE = "dinov3_sat_nir_dpt"
DINOV3_SAT_NIR_FOLD0_SPLIT = "ms75_oof_seed1337_fold_0_train"
DINOV3_SAT_NIR_FOLD0_HELD_SPLIT = "ms75_oof_seed1337_fold_0_heldout"
DINOV3_SAT_NIR_CLASS_INDEX = (
    "metadata/ms75_oof_seed1337_fold_0_train_points.npz"
)
DINOV3_SAT_NIR_CLASS_INDEX_SHA256 = (
    "c14f3ed42ea1a3bf5e372ffc46d0294c30a7f18724738f9c23fb1d2c5253ac84"
)
DINOV3_SAT_NIR_RARE_CLASS_IDS = (7, 9, 15, 18, 19, 22, 23)
DINOV3_SAT_NIR_HARD_CLASS_IDS = (0, 4, 5, 6, 8, 10, 11, 12, 13, 14, 17, 20, 21)
# Fail closed if the contaminated teacher-derived archive is reintroduced.
DINOV3_SAT_NIR_FORBIDDEN_CONFUSION_INDEX_BASENAME = (
    "ms75_oof_confusions_train.npz"
)
DINOV3_SAT_NIR_MILESTONES = (9_600, 14_400, 19_200, 24_000, 28_800, 33_600)
DINOV3_SAT_NIR_PERMANENT_FREEZE_MODULES = (
    "backbone.patch_embed.proj.rgb_projection",
    *(f"backbone.blocks.{index}" for index in range(12)),
)
DINOV3_SAT_NIR_STAGED_MODULES = (
    *(f"backbone.blocks.{index}" for index in range(12, 24)),
    "backbone.norm",
)
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        raise ValueError("cyclic DINOv3SATNIR config inheritance")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("DINOv3SATNIR config root must be a mapping")
    base = payload.pop("_base_", None)
    if base is None:
        return payload
    parent = _load_yaml((resolved.parent / str(base)).resolve(), (*stack, resolved))
    return _deep_merge(parent, payload)


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        replacement = os.environ.get(name, default)
        if replacement is None:
            raise ValueError(f"environment variable {name!r} is required")
        return replacement

    return _ENV_PATTERN.sub(replace, value)


def _require_equal(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise ValueError(
            f"DINOv3SATNIR protocol requires {name}={expected!r}, observed={observed!r}"
        )


def _zero_loss_contract(loss: Mapping[str, Any]) -> None:
    forbidden = (
        "ms_aux_weight",
        "sar_aux_weight",
        "sar_native_aux_weight",
        "aux_weight",
        "spectral_aux_weight",
        "context_aux_weight",
        "short_aux_weight",
        "global_aux_weight",
        "boundary_weight",
        "prototype_alignment_weight",
        "private_orthogonality_weight",
        "negative_transfer_weight",
        "utility_routing_weight",
        "alignment_offset_weight",
        "teacher_distillation_weight",
        "fixed_anchor_no_harm_weight",
        "fusion_mixture_nll_weight",
        "family_ce_weight",
        "hierarchy_consistency_weight",
        "family_region_aux_weight",
        "family_child_weight",
        "family_contrast_weight",
        "family_region_confusion_weight",
        "family_region_no_harm_weight",
        "region_prototype_weight",
        "confusion_margin_weight",
    )
    changed = {key: loss.get(key) for key in forbidden if float(loss.get(key, 0.0))}
    if changed:
        raise ValueError(f"DINOv3SATNIR forbids unrelated auxiliary losses: {changed}")


def validate_dinov3_sat_nir_config(
    config: Mapping[str, Any], *, require_pretrained_file: bool = False
) -> None:
    for section in ("experiment", "data", "model", "loss", "train", "inference"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"DINOv3SATNIR config lacks section {section!r}")
    experiment = config["experiment"]
    data = config["data"]
    model = config["model"]
    loss = config["loss"]
    train = config["train"]
    inference = config["inference"]
    role = str(data.get("protocol_stage", ""))
    is_smoke = role == "dinov3_sat_nir_fold0_smoke"
    is_probe = role == "dinov3_sat_nir_fold0_probe"
    if role not in {
        "dinov3_sat_nir_fold0_train96",
        "dinov3_sat_nir_fold0_smoke",
        "dinov3_sat_nir_fold0_probe",
    }:
        raise ValueError("unsupported DINOv3SATNIR protocol_stage")

    _require_equal(int(experiment.get("seed", -1)), 1337, "experiment.seed")
    _require_equal(data.get("dataset_format"), "fbp_ms", "data.dataset_format")
    _require_equal(data.get("dataset_variant"), "legacy", "data.dataset_variant")
    _require_equal(data.get("official_120_30_protocol"), False, "official protocol")
    _require_equal(data.get("load_sar"), False, "data.load_sar")
    _require_equal(int(data.get("patch_size", -1)), 512, "data.patch_size")
    _require_equal(int(data.get("sar_size", -1)), 128, "data.sar_size")
    _require_equal(int(data.get("context_size", -1)), 512, "data.context_size")
    _require_equal(
        int(data.get("context_input_size", -1)), 512, "data.context_input_size"
    )
    _require_equal(data.get("train_split"), DINOV3_SAT_NIR_FOLD0_SPLIT, "data.train_split")
    for split_name in ("val_split", "test_split"):
        _require_equal(
            data.get(split_name),
            DINOV3_SAT_NIR_FOLD0_HELD_SPLIT,
            f"data.{split_name}",
        )
    _require_equal(int(data.get("samples_per_epoch", -1)), 30_720, "samples_per_epoch")
    _require_equal(int(data.get("scene_block_size", -1)), 16, "scene block size")
    _require_equal(int(data.get("max_scene_repeat", -1)), 16, "max scene repeat")
    _require_equal(data.get("normalization", {}).get("ms", {}).get("mean"), [0.363, 0.390, 0.371, 0.506], "MS mean")
    _require_equal(data.get("normalization", {}).get("ms", {}).get("std"), [0.231, 0.231, 0.244, 0.254], "MS std")
    dynamic = data.get("dynamic_sampling", {})
    _require_equal(
        data.get("rare_class_ids"),
        list(DINOV3_SAT_NIR_RARE_CLASS_IDS),
        "sampling rare classes",
    )
    _require_equal(
        data.get("hard_class_ids"),
        list(DINOV3_SAT_NIR_HARD_CLASS_IDS),
        "sampling known-hard classes",
    )
    if set(DINOV3_SAT_NIR_RARE_CLASS_IDS) & set(DINOV3_SAT_NIR_HARD_CLASS_IDS):
        raise AssertionError("DINOv3SATNIR rare/hard class contracts overlap")
    _require_equal(
        (
            float(dynamic.get("scene_fraction", -1.0)),
            float(dynamic.get("rare_fraction", -1.0)),
            float(dynamic.get("hard_fraction", -1.0)),
        ),
        (0.50, 0.30, 0.20),
        "unified sampling fractions",
    )
    forbidden_dynamic_keys = {
        "class_fraction",
        "confusion_fraction",
        "confusion_index",
        "confusion_index_sha256",
    }
    present_forbidden = sorted(forbidden_dynamic_keys & set(dynamic))
    if present_forbidden:
        raise ValueError(
            "DINOv3SATNIR rejects legacy teacher/confusion sampling keys: "
            f"{present_forbidden}"
        )
    allowed_dynamic_keys = {
        "scene_fraction",
        "rare_fraction",
        "hard_fraction",
        "class_index",
        "class_index_sha256",
        "center_jitter",
        "max_scene_repeat",
    }
    unexpected_dynamic = sorted(set(dynamic) - allowed_dynamic_keys)
    if unexpected_dynamic:
        raise ValueError(
            f"unexpected DINOv3SATNIR dynamic-sampling keys: {unexpected_dynamic}"
        )
    if any(
        DINOV3_SAT_NIR_FORBIDDEN_CONFUSION_INDEX_BASENAME in str(value)
        for value in dynamic.values()
    ):
        raise ValueError("DINOv3SATNIR rejects the legacy teacher/confusion archive")
    _require_equal(int(dynamic.get("center_jitter", -1)), 128, "center jitter")
    _require_equal(int(dynamic.get("max_scene_repeat", -1)), 16, "dynamic block")
    _require_equal(dynamic.get("class_index"), DINOV3_SAT_NIR_CLASS_INDEX, "class index")
    _require_equal(
        dynamic.get("class_index_sha256"),
        DINOV3_SAT_NIR_CLASS_INDEX_SHA256,
        "class index SHA256",
    )
    _require_equal(model.get("architecture"), DINOV3_SAT_NIR_ARCHITECTURE, "architecture")
    _require_equal(model.get("fusion_mode"), "ms_only", "fusion_mode")
    _require_equal(int(model.get("num_classes", -1)), 24, "num_classes")
    _require_equal(
        model.get("backbone_model"),
        "vit_large_patch16_dinov3.sat493m",
        "backbone",
    )
    _require_equal(model.get("input_band_order"), ["B", "G", "R", "NIR"], "band order")
    _require_equal(int(model.get("input_size", -1)), 512, "model.input_size")
    _require_equal(int(model.get("padded_size", -1)), 512, "model.padded_size")
    _require_equal(int(model.get("patch_size", -1)), 16, "model.patch_size")
    _require_equal(model.get("feature_indices"), [5, 11, 17, 23], "feature indices")
    _require_equal(
        model.get("reassemble_channels"),
        [96, 192, 384, 768],
        "DPT reassembly channels",
    )
    _require_equal(int(model.get("decoder_channels", -1)), 256, "decoder channels")
    _require_equal(float(model.get("initial_nir_gate", -1.0)), 0.1, "NIR gate")
    _require_equal(model.get("nir_initialization"), "rgb_patch_weight_mean", "NIR initialization")
    _require_equal(model.get("freeze_prefix_tokens"), True, "prefix-token freeze")
    _require_equal(model.get("gradient_checkpointing"), True, "checkpointing")
    _require_equal(
        model.get("rgb_normalization", {}).get("mean"),
        list(DINOV3_SAT_RGB_MEAN),
        "DINOv3 SAT RGB mean",
    )
    _require_equal(
        model.get("rgb_normalization", {}).get("std"),
        list(DINOV3_SAT_RGB_STD),
        "DINOv3 SAT RGB std",
    )
    _require_equal(model.get("pretrained"), True, "pretrained")
    _require_equal(
        model.get("pretrained_file_sha256"),
        DINOV3_SAT_VITL16_SHA256,
        "pretrained SHA256",
    )
    if require_pretrained_file and not Path(str(model.get("pretrained_file", ""))).is_file():
        raise FileNotFoundError(
            f"missing pinned DINOv3-SAT pretrained file: {model.get('pretrained_file')}"
        )

    _require_equal(loss.get("class_weights"), None, "loss.class_weights")
    _require_equal(float(loss.get("cross_entropy_weight", -1.0)), 1.0, "CE weight")
    _require_equal(float(loss.get("dice_weight", -1.0)), 0.5, "Dice weight")
    _require_equal(float(loss.get("lovasz_weight", -1.0)), 0.25, "Lovasz weight")
    _zero_loss_contract(loss)

    _require_equal(train.get("device"), "cuda", "train.device")
    physical_batch = int(train.get("batch_size", -1))
    if physical_batch not in {1, 2, 4, 8, 16}:
        raise ValueError("DINOv3SATNIR physical batch must be one of 16/8/4/2/1")
    _require_equal(
        physical_batch * int(train.get("gradient_accumulation_steps", -1)),
        64,
        "effective crop batch",
    )
    _require_equal(
        train.get("loss_microbatch_size"),
        1,
        "per-image loss microbatch",
    )
    _require_equal(float(train.get("learning_rate", -1.0)), 2.0e-4, "head LR")
    _require_equal(
        float(train.get("backbone_lr_multiplier", -1.0)),
        0.05,
        "backbone LR multiplier",
    )
    _require_equal(float(train.get("layer_decay", -1.0)), 0.9, "layer decay")
    _require_equal(float(train.get("weight_decay", -1.0)), 0.05, "weight decay")
    _require_equal(train.get("scheduler_unit"), "step", "scheduler unit")
    _require_equal(float(train.get("gradient_clip_norm", -1.0)), 1.0, "gradient clip")
    _require_equal(train.get("amp"), True, "AMP")
    _require_equal(float(train.get("amp_initial_scale", -1.0)), 1024.0, "AMP initial scale")
    _require_equal(train.get("use_ema"), True, "EMA")
    _require_equal(float(train.get("ema_decay", -1.0)), 0.999, "EMA decay")
    _require_equal(train.get("validation_enabled"), False, "validation")
    _require_equal(train.get("selection_metric"), "last_epoch", "selection metric")
    _require_equal(train.get("teacher_checkpoint"), None, "teacher checkpoint")
    _require_equal(train.get("initialization_checkpoint"), None, "initializer")
    _require_equal(train.get("trainable_modules"), [], "trainable modules")
    _require_equal(
        train.get("freeze_modules"),
        list(DINOV3_SAT_NIR_PERMANENT_FREEZE_MODULES),
        "permanently frozen modules",
    )
    _require_equal(
        train.get("staged_unfreeze_modules"),
        list(DINOV3_SAT_NIR_STAGED_MODULES),
        "staged backbone modules",
    )
    _require_equal(
        int(train.get("staged_unfreeze_epoch", -1)),
        0 if (is_smoke or is_probe) else 5,
        "staged unfreeze epoch",
    )
    _require_equal(inference.get("evaluate_test_after_training"), False, "automatic test")
    _require_equal(inference.get("tta_transforms"), ["none"], "TTA")

    if is_smoke or is_probe:
        if "smoke" not in str(experiment.get("name", "")).lower():
            if not is_probe or "probe" not in str(experiment.get("name", "")).lower():
                raise ValueError("DINOv3SATNIR short-run experiment name is invalid")
        updates = 1 if is_probe else 10
        label = "probe" if is_probe else "smoke"
        _require_equal(
            int(data.get("debug_samples_per_epoch", -1)),
            updates * 64,
            f"{label} samples",
        )
        _require_equal(
            int(train.get("max_successful_optimizer_updates", -1)),
            updates,
            f"{label} budget",
        )
        _require_equal(
            int(train.get("scheduler_total_successful_optimizer_updates", -1)),
            updates,
            f"{label} scheduler",
        )
        _require_equal(train.get("successful_update_milestones"), [], f"{label} milestones")
    else:
        if data.get("debug_samples_per_epoch") is not None:
            raise ValueError("formal DINOv3SATNIR training forbids debug sample limits")
        _require_equal(int(train.get("epochs", -1)), 71, "attempt capacity epochs")
        _require_equal(int(train.get("warmup_steps", -1)), 2_000, "warmup steps")
        _require_equal(int(train.get("max_successful_optimizer_updates", -1)), 33_600, "formal budget")
        _require_equal(int(train.get("scheduler_total_successful_optimizer_updates", -1)), 33_600, "scheduler horizon")
        _require_equal(
            train.get("successful_update_milestones"),
            list(DINOV3_SAT_NIR_MILESTONES),
            "successful-update milestones",
        )
        _require_equal(train.get("primary_successful_optimizer_update"), None, "unselected primary")


def load_dinov3_sat_nir_config(
    path: str | Path, *, require_pretrained_file: bool = False
) -> dict[str, Any]:
    config = _expand(_load_yaml(Path(path)))
    # Environment expansion is textual; normalize the two probe-selected
    # integers before validation and before the trainer fingerprints config.
    config["train"]["batch_size"] = int(config["train"]["batch_size"])
    config["train"]["gradient_accumulation_steps"] = int(
        config["train"]["gradient_accumulation_steps"]
    )
    validate_dinov3_sat_nir_config(
        config, require_pretrained_file=require_pretrained_file
    )
    return config


__all__ = [
    "DINOV3_SAT_NIR_ARCHITECTURE",
    "DINOV3_SAT_NIR_CLASS_INDEX",
    "DINOV3_SAT_NIR_CLASS_INDEX_SHA256",
    "DINOV3_SAT_NIR_FORBIDDEN_CONFUSION_INDEX_BASENAME",
    "DINOV3_SAT_NIR_FOLD0_HELD_SPLIT",
    "DINOV3_SAT_NIR_FOLD0_SPLIT",
    "DINOV3_SAT_NIR_HARD_CLASS_IDS",
    "DINOV3_SAT_NIR_MILESTONES",
    "DINOV3_SAT_NIR_PERMANENT_FREEZE_MODULES",
    "DINOV3_SAT_NIR_RARE_CLASS_IDS",
    "DINOV3_SAT_NIR_STAGED_MODULES",
    "load_dinov3_sat_nir_config",
    "validate_dinov3_sat_nir_config",
]
