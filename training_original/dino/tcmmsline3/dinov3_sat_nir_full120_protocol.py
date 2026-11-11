"""Isolated full120 refit contract for the held24-selected DINOv3-SAT/NIR expert.

No train96 validator is weakened: these are new full120 stages and artifacts.
The selected checkpoint fixes only the duration; initialization remains pinned
official DINOv3-SAT pretraining. Official30 rasters are never opened here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .dinov3_sat_nir_model import (
    DINOV3_SAT_RGB_MEAN, DINOV3_SAT_RGB_STD, DINOV3_SAT_VITL16_SHA256,
)
from .dinov3_sat_nir_protocol import (
    DINOV3_SAT_NIR_ARCHITECTURE, DINOV3_SAT_NIR_HARD_CLASS_IDS,
    DINOV3_SAT_NIR_RARE_CLASS_IDS, DINOV3_SAT_NIR_PERMANENT_FREEZE_MODULES,
    DINOV3_SAT_NIR_STAGED_MODULES, _expand, _load_yaml, _zero_loss_contract,
)

DINOV3_SAT_NIR_FULL120_SPLIT = "train"
DINOV3_SAT_NIR_FULL120_CLASS_INDEX = (
    "metadata/local_detail_points_train120_seed1337_stride16_cap64.npz"
)
DINOV3_SAT_NIR_FULL120_CLASS_INDEX_SHA256 = (
    "61d66cad70beceab3193bff70fe3033156db16a65a4894c41b30de4d2087943f"
)
DINOV3_SAT_NIR_FULL120_SAMPLES_PER_EPOCH = 38_400
DINOV3_SAT_NIR_FULL120_EPOCH = 20
DINOV3_SAT_NIR_FULL120_UPDATES = 12_000
DINOV3_SAT_NIR_FULL120_SCHEDULER_UPDATES = 42_000
DINOV3_SAT_NIR_FULL120_WARMUP_UPDATES = 2_500
DINOV3_SAT_NIR_FULL120_MILESTONES = (12_000,)
DINOV3_SAT_NIR_FULL120_STAGES = (
    "dinov3_sat_nir_full120_train",
    "dinov3_sat_nir_full120_probe",
    "dinov3_sat_nir_full120_smoke",
)
DINOV3_SAT_NIR_FULL120_SELECTION_PROVENANCE = {
    "source_summary_path": "/archive/hot0/yeqi/experiments/TCMMSline3_dinov3_sat_nir/held24_early_epoch_selection_gpu1_20260904_v1/selection_summary.json",
    "source_summary_file_sha256": "ad75c4e4bc803734e9f5464e3f5a1cc4fe49a59a94ef91f0d617b78b43537acf",
    "source_summary_receipt_sha256": "3dd5cdc57b7275b36532c1b612842a674c2ec495962693158754786b779cb64c",
    "source_final_audit_path": "/archive/hot5/yeqi/runtime_fixes/dinov3_gpu1_skip_eval_20260906_v2/final_audit.json",
    "source_final_audit_file_sha256": "bb8fc96bd38b77a345c55242279b437d25ca5a4f0c740939b6a8fe3289fcc68f",
    "source_final_audit_sidecar_sha256": "31aaad8b64f4f0855d9d5691a6f23434a0a9f395168e5b62bbf8f29525238c74",
    "selected_epoch": 20,
    "selected_train96_successful_updates": 9_600,
    "selected_full120_successful_updates": 12_000,
    "selected_candidate_path": "/archive/hot0/yeqi/experiments/TCMMSline3_dinov3_sat_nir/train96_fold0_gpu1_early_20260904_v1/checkpoints/ema_successful_updates_0009600.pt",
    "selected_candidate_sha256": "309002bc69b5dbb66e99d9101742d96a2d17ad5263bd12ec4787c08ec3df1d85",
    "selected_held24_global_miou": 0.6784711587649954,
    "original_epoch20_control_is_not_a_selection_candidate": True,
    "original_epoch20_control_held24_global_miou": 0.6927302780767143,
    "selection_split": "ms75_oof_seed1337_fold_0_heldout",
    "selection_rule": "maximum_held24_global_confusion_miou_tie_earlier",
    "selection_scope": "one_global_24x24_confusion_per_epoch",
    "scene_scale_numerator": 120,
    "scene_scale_denominator": 96,
    "train96_checkpoint_loaded": False,
    "official30_candidate_observed_before_commitment": False,
}


def _equal(value: Any, expected: Any, name: str) -> None:
    # Bool must never silently satisfy an integer/float protocol counter.
    if isinstance(expected, bool):
        valid = type(value) is bool and value is expected
    elif isinstance(expected, int):
        valid = type(value) is int and value == expected
    elif isinstance(expected, float):
        valid = type(value) in (int, float) and math.isfinite(value) and value == expected
    elif isinstance(expected, Mapping):
        if not isinstance(value, Mapping) or set(value) != set(expected):
            valid = False
        else:
            for key, item in expected.items():
                _equal(value[key], item, f"{name}.{key}")
            return
    elif isinstance(expected, (list, tuple)):
        if type(value) is not type(expected) or len(value) != len(expected):
            valid = False
        else:
            for index, item in enumerate(expected):
                _equal(value[index], item, f"{name}[{index}]")
            return
    else:
        valid = value == expected
    if not valid:
        raise ValueError(
            f"DINOv3-SAT full120 requires {name}={expected!r}, observed={value!r}"
        )


def _anchors(section: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, value in expected.items():
        _equal(section.get(key), value, f"{label}.{key}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def verify_dinov3_full120_selection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read only the pinned held24 JSON; never load its checkpoint as weights."""
    provenance = config.get("selection_provenance")
    _equal(provenance, DINOV3_SAT_NIR_FULL120_SELECTION_PROVENANCE, "selection_provenance")
    path = Path(provenance["source_summary_path"])
    _equal(_sha256(path), provenance["source_summary_file_sha256"], "selection summary SHA256")
    summary = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in summary.items() if key != "receipt_sha256"}
    _equal(summary.get("receipt_sha256"), provenance["source_summary_receipt_sha256"], "selection receipt")
    _equal(_canonical(body), summary["receipt_sha256"], "selection receipt self-hash")
    _anchors(summary, {
        "schema": "dinov3_sat_nir_early_held24_selection_v1",
        "status": "completed",
        "selection_scene_count": 24,
        "selection_split": provenance["selection_split"],
        "selection_scope": provenance["selection_scope"],
        "selection_rule": provenance["selection_rule"],
        "selected_epoch": 20,
        "selected_global_optimizer_step": 9_600,
        "selected_candidate": provenance["selected_candidate_path"],
        "selected_candidate_sha256": provenance["selected_candidate_sha256"],
        "selected_mean_iou": provenance["selected_held24_global_miou"],
        "selected_full120_stop_updates": 12_000,
        "global_confusion_not_scene_miou_average": True,
        "official30_raster_opened": False,
        "selection_candidates_are_early_replay_only": True,
        "original_epoch20_control_is_not_a_selection_candidate": True,
    }, "held24 summary")
    candidates = summary.get("candidate_results")
    if not isinstance(candidates, list) or [row.get("epoch") for row in candidates] != [5, 8, 10, 12, 15, 18, 20]:
        raise ValueError("held24 candidate grid changed")
    for row in candidates:
        _equal(row.get("global_optimizer_step"), row["epoch"] * 480, "held24 update mapping")
        if type(row.get("mean_iou")) not in (float, int) or not math.isfinite(row["mean_iou"]):
            raise ValueError("held24 candidate has invalid mIoU")
    winner = min(candidates, key=lambda row: (-row["mean_iou"], row["epoch"]))
    _equal(winner["epoch"], 20, "recomputed selected epoch")
    _equal(winner["mean_iou"], provenance["selected_held24_global_miou"], "recomputed selected mIoU")
    _equal(
        summary["run_identity"]["candidate_sha256"]["20"],
        provenance["selected_candidate_sha256"], "selected checkpoint identity",
    )
    _equal(summary["run_identity"]["original_epoch20_control"]["selection_candidate"],
           False, "original epoch20 control selection eligibility")
    _equal(summary["original_epoch20_control_result"]["mean_iou"],
           provenance["original_epoch20_control_held24_global_miou"], "original control mIoU")
    audit_path = Path(provenance["source_final_audit_path"])
    _equal(_sha256(audit_path), provenance["source_final_audit_file_sha256"], "selection final audit SHA256")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _equal(audit.get("sidecar_sha256"), provenance["source_final_audit_sidecar_sha256"], "selection final audit sidecar")
    _equal(_canonical({key: value for key, value in audit.items() if key != "sidecar_sha256"}),
           audit["sidecar_sha256"], "selection final audit self-hash")
    _anchors(audit, {
        "schema": "dinov3_skip_eval_final_audit_v1", "selected_epoch": 20,
        "selected_mean_iou": provenance["selected_held24_global_miou"],
    }, "held24 final audit")
    _anchors(audit["selection_summary"], {
        "path": provenance["source_summary_path"],
        "sha256": provenance["source_summary_file_sha256"],
    }, "held24 final audit summary binding")
    _anchors(audit["native_terminal_validation"], {
        "status": "validated", "scene_receipt_count": 24,
        "selected_epoch": 20, "selected_mean_iou": provenance["selected_held24_global_miou"],
        "selection_summary_sha256": provenance["source_summary_receipt_sha256"],
        "run_identity_sha256": summary["run_identity_sha256"],
        "official30_raster_opened": False,
    }, "held24 native terminal validation")
    return summary


def validate_dinov3_sat_nir_config(
    config: Mapping[str, Any], *, require_pretrained_file: bool = False,
) -> None:
    for key in ("experiment", "data", "model", "loss", "train", "inference", "selection_provenance"):
        if not isinstance(config.get(key), Mapping):
            raise ValueError(f"DINOv3-SAT full120 config lacks mapping {key!r}")
    experiment, data, model = config["experiment"], config["data"], config["model"]
    loss, train, inference = config["loss"], config["train"], config["inference"]
    role = data.get("protocol_stage")
    if role not in DINOV3_SAT_NIR_FULL120_STAGES:
        raise ValueError("unsupported DINOv3-SAT full120 protocol_stage")
    _equal(experiment.get("seed"), 1337, "experiment.seed")
    if "full120" not in str(experiment.get("name", "")):
        raise ValueError("experiment.name must identify full120")
    if not str(experiment.get("output_dir", "")):
        raise ValueError("full120 output_dir is required")
    _equal(config["selection_provenance"], DINOV3_SAT_NIR_FULL120_SELECTION_PROVENANCE, "selection_provenance")
    _anchors(data, {
        "dataset_format": "fbp_ms", "dataset_variant": "legacy",
        "official_120_30_protocol": True, "expected_scene_count": 150,
        "expected_hw": [6908, 7300], "train_split": "train",
        "val_split": "train", "test_split": "train",
        "load_sar": False, "sar_pairing": "zero",
        "fixed_40k_tile_protocol": False, "fixed_tile_index": None,
        "fixed_tile_subsets": None, "validation_exact_fixed_tiles": False,
        "validation_holdout_fraction": 0.0,
        "patch_size": 512, "sar_size": 128, "sar_native_size": 128,
        "context_size": 512, "context_input_size": 512,
        "samples_per_epoch": 38_400, "eval_patches_per_scene": 1,
        "min_labeled_fraction": 0.5, "sampling_attempts": 16,
        "scene_block_size": 16, "max_scene_repeat": 16,
        "rare_class_ids": list(DINOV3_SAT_NIR_RARE_CLASS_IDS),
        "hard_class_ids": list(DINOV3_SAT_NIR_HARD_CLASS_IDS),
        "normalization": {"ms": {
            "divisor": 255.0, "mean": [0.363, 0.390, 0.371, 0.506],
            "std": [0.231, 0.231, 0.244, 0.254],
        }},
        "spectral_augmentation": {
            "enabled": True, "probability": 0.8,
            "global_gain_range": [0.90, 1.10], "global_offset_range": [-0.02, 0.02],
            "band_gain_range": [0.97, 1.03], "band_offset_range": [-0.008, 0.008],
        },
        "dynamic_sampling": {
            "scene_fraction": 0.50, "rare_fraction": 0.30, "hard_fraction": 0.20,
            "class_index": DINOV3_SAT_NIR_FULL120_CLASS_INDEX,
            "class_index_sha256": DINOV3_SAT_NIR_FULL120_CLASS_INDEX_SHA256,
            "center_jitter": 128, "max_scene_repeat": 16,
        },
    }, "data")
    for key in ("load_pan", "teacher_sampling", "teacher_enabled"):
        if data.get(key, False) is not False:
            raise ValueError(f"full120 forbids data.{key}")
    for key in ("train_scene_ids", "val_scene_ids", "test_scene_ids"):
        if key in data:
            raise ValueError(f"full120 forbids alternate explicit data.{key}")
    _anchors(model, {
        "architecture": DINOV3_SAT_NIR_ARCHITECTURE, "fusion_mode": "ms_only",
        "num_classes": 24, "backbone_model": "vit_large_patch16_dinov3.sat493m",
        "input_band_order": ["B", "G", "R", "NIR"],
        "input_size": 512, "padded_size": 512, "patch_size": 16,
        "feature_indices": [5, 11, 17, 23],
        "reassemble_channels": [96, 192, 384, 768], "decoder_channels": 256,
        "initial_nir_gate": 0.1, "nir_initialization": "rgb_patch_weight_mean",
        "rgb_normalization": {"mean": list(DINOV3_SAT_RGB_MEAN), "std": list(DINOV3_SAT_RGB_STD)},
        "pretrained": True, "pretrained_file_sha256": DINOV3_SAT_VITL16_SHA256,
        "gradient_checkpointing": True, "freeze_prefix_tokens": True,
    }, "model")
    _anchors(loss, {
        "class_weights": None, "cross_entropy_weight": 1.0,
        "dice_weight": 0.5, "lovasz_weight": 0.25,
    }, "loss")
    _zero_loss_contract(loss)
    physical = train.get("batch_size")
    if type(physical) is not int or physical != 8:
        raise ValueError("physical batch must preserve the selected DINOv3-SAT batch8")
    accumulation = train.get("gradient_accumulation_steps")
    if type(accumulation) is not int:
        raise ValueError("gradient_accumulation_steps must be an integer")
    _equal(physical * accumulation, 64, "effective crop batch")
    _anchors(train, {
        "device": "cuda", "loss_microbatch_size": 1,
        "learning_rate": 2e-4, "backbone_lr_multiplier": 0.05,
        "layer_decay": 0.9, "weight_decay": 0.05, "scheduler_unit": "step",
        "warmup_steps": 2_500, "warmup_epochs": 0,
        "scheduler_total_successful_optimizer_updates": 42_000,
        "minimum_lr_ratio": 0.0, "gradient_clip_norm": 1.0,
        "amp": True, "amp_initial_scale": 1024.0,
        "amp_growth_interval": 1_000_000,
        "amp_growth_factor": 2.0, "amp_backoff_factor": 0.5,
        "use_ema": True, "ema_decay": 0.999, "ema_milestone_epochs": [],
        "validation_enabled": False,
        "selection_metric": "last_epoch",
        "freeze_modules": list(DINOV3_SAT_NIR_PERMANENT_FREEZE_MODULES),
        "staged_unfreeze_modules": list(DINOV3_SAT_NIR_STAGED_MODULES),
        "staged_unfreeze_epoch": 5 if role.endswith("_train") else 0,
        "trainable_modules": [], "initialization_checkpoint": None,
        "initialization_exclude_prefixes": [], "ms_initialization_checkpoint": None,
        "sar_initialization_checkpoint": None, "teacher_checkpoint": None,
        "continuation_initialization_checkpoint": None, "ms75_anchor_checkpoint": None,
        "fixed_anchor_teacher_enabled": False, "resume_checkpoint": None,
        "auto_resume": True, "primary_successful_optimizer_update": None,
        "checkpoint_every": 1000,
    }, "train")
    # Preserve the native FP16 loss path; DINOv2 BF16/forced-FP32 fixes are not this expert.
    _equal(train.get("amp_dtype", "float16"), "float16", "train.amp_dtype")
    _equal(loss.get("force_main_losses_fp32", False), False, "loss.force_main_losses_fp32")
    _equal(train.get("cudnn_benchmark"), True, "train.cudnn_benchmark")
    _anchors(inference, {
        "protocol_role": role, "batch_size": 1, "max_windows_per_scene": None,
        "tile_stride": 256, "tta_transforms": ["none"],
        "blend_window": "hann", "hann_floor": 0.05, "amp": True,
        "evaluate_test_after_training": False,
    }, "inference")
    if role == "dinov3_sat_nir_full120_train":
        _anchors(train, {
            "epochs": 21, "max_successful_optimizer_updates": 12_000,
            "successful_update_milestones": [12_000],
        }, "train")
        _equal(data.get("debug_samples_per_epoch"), None, "formal debug_samples_per_epoch")
    else:
        is_probe = role.endswith("_probe")
        budget = 1 if is_probe else 10
        _anchors(train, {
            "epochs": 2, "max_successful_optimizer_updates": budget,
            "successful_update_milestones": [],
        }, "train")
        _equal(data.get("debug_samples_per_epoch"), budget * 64, "short-run sample budget")
        if ("probe" if is_probe else "smoke") not in experiment["name"]:
            raise ValueError("short-run experiment name does not match its role")
    if require_pretrained_file:
        pretrained = Path(str(model.get("pretrained_file", "")))
        if not pretrained.is_file():
            raise FileNotFoundError(f"missing pinned DINOv3-SAT pretraining: {pretrained}")
        _equal(_sha256(pretrained), DINOV3_SAT_VITL16_SHA256, "pretrained file SHA256")
        verify_dinov3_full120_selection(config)


def load_dinov3_sat_nir_config(
    path: str | Path, *, require_pretrained_file: bool = False,
) -> dict[str, Any]:
    config = _expand(_load_yaml(Path(path)))
    # Textual environment-selected memory batch is normalized before hashing.
    for key in ("batch_size", "gradient_accumulation_steps"):
        value = config["train"][key]
        if isinstance(value, bool):
            raise ValueError(f"train.{key} must not be boolean")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"train.{key} must be an exact integer")
        config["train"][key] = int(value)
    validate_dinov3_sat_nir_config(config, require_pretrained_file=require_pretrained_file)
    return config

