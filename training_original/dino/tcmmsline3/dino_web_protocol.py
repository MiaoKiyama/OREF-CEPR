"""Fail-closed contracts for the three independent WEB-pretrained DINO runs.

Configuration identity is separate from source identity: a held24-selected
full120 refit may change its duration and split, never its implementation or
pretraining. No development checkpoint is an initialization source.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .dinov3_sat_nir_protocol import _expand, _load_yaml, _zero_loss_contract
from .dino_web_model import DINO_WEB_VARIANTS
from .runtime import config_fingerprint

ARCHITECTURE = "dino_web_nir_dpt"
TRAIN96_SPLIT = "ms75_oof_seed1337_fold_0_train"
HELD24_SPLIT = "ms75_oof_seed1337_fold_0_heldout"
TRAIN96_INDEX = "metadata/ms75_oof_seed1337_fold_0_train_points.npz"
TRAIN96_INDEX_SHA256 = (
    "c14f3ed42ea1a3bf5e372ffc46d0294c30a7f18724738f9c23fb1d2c5253ac84"
)
FULL120_INDEX = "metadata/local_detail_points_train120_seed1337_stride16_cap64.npz"
FULL120_INDEX_SHA256 = (
    "61d66cad70beceab3193bff70fe3033156db16a65a4894c41b30de4d2087943f"
)
RARE_CLASS_IDS = (7, 9, 15, 18, 19, 22, 23)
HARD_CLASS_IDS = (0, 4, 5, 6, 8, 10, 11, 12, 13, 14, 17, 20, 21)
CANDIDATE_EPOCHS = (4, 8, 12, 16, 18, 20, 24, 30, 40, 50, 60, 70)
SELECTION_RULE = "maximum_held24_global_confusion_miou_tie_earlier"
RUNTIME_BINDING_KEY = "dino_web_runtime_binding"
FULL120_SWEEP_EPOCHS = tuple(range(20, 31))
FULL120_SWEEP_SOURCE_PATHS = (
    "tcmmsline3/dino_web_protocol.py",
    "tools/dino_web_full120_sweep_server_ops.py",
    "tools/run_dino_web_full120_sweep.py",
)
V2L_FULL120_SWEEP_EPOCHS = tuple(range(16, 23))
V2L_FULL120_SWEEP_SOURCE_PATHS = (
    "tcmmsline3/dino_web_protocol.py",
    "tools/run_dino_web_v2l_full120_sweep.py",
    "tools/dino_web_v2l_full120_server_ops.py",
)
V3L_MULTIGPU_SOURCE_PATHS = (*FULL120_SWEEP_SOURCE_PATHS,
    "tools/train_dino_web_multigpu.py",
    "tools/run_dino_web_v3l_multigpu.py",
    "tools/dino_web_v3l_multigpu_server_ops.py",
)
V3L_MULTIGPU_CONTRACT = {
    "role": "dino_web_v3l_four_gpu_synchronous_forward_v1",
    "device_ids": [0, 1, 2, 3], "global_physical_batch": 16,
    "per_device_batch": 4, "gradient_accumulation_steps": 4,
    "effective_batch": 64, "loss_microbatch_size": 1,
    "optimizer_ema_and_checkpoint_owner": "single_master_raw_model",
    "sampler": "unchanged_ordered_single_loader",
    "autocast_dtype": "bfloat16", "autocast_cache_enabled": False,
}
V3L_SINGLEGPU_SOURCE_PATHS = (*FULL120_SWEEP_SOURCE_PATHS,
    "tools/run_dino_web_v3l_single.py",
    "tools/dino_web_v3l_single_server_ops.py",
)
V3L_SINGLEGPU_CONTRACT = {
    "role": "dino_web_v3l_single_gpu_legacy_training_v1",
    # CUDA-local numbering; the launcher independently pins the physical GPU.
    "device_ids": [0], "global_physical_batch": 8,
    "per_device_batch": 8, "gradient_accumulation_steps": 8,
    "effective_batch": 64, "loss_microbatch_size": 1,
    "optimizer_ema_and_checkpoint_owner": "single_raw_model",
    "sampler": "unchanged_ordered_single_loader",
    "autocast_dtype": "bfloat16", "autocast_cache_enabled": False,
}
VARIANTS = {
    "v3b": ("vit_base_patch16_dinov3.lvd1689m", 12, 16, 512, [2, 5, 8, 11]),
    "v3l": ("vit_large_patch16_dinov3.lvd1689m", 24, 16, 512, [5, 11, 17, 23]),
    "v2l": ("vit_large_patch14_dinov2.lvd142m", 24, 14, 518, [5, 11, 17, 23]),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _equal(value: Any, expected: Any, name: str) -> None:
    # In particular, True is not a valid optimizer-step count.
    valid = value == expected
    if isinstance(expected, bool) or isinstance(expected, int):
        valid = valid and type(value) is type(expected)
    if isinstance(expected, float):
        valid = type(value) in (int, float) and math.isfinite(value) and valid
    if not valid:
        raise ValueError(f"DINO WEB requires {name}={expected!r}, observed={value!r}")


def _anchors(
    section: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    for key, value in expected.items():
        _equal(section.get(key), value, f"{label}.{key}")


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[0-9a-f]{64}", value)
        or len(set(value)) == 1
    ):
        raise ValueError(f"DINO WEB requires an authenticated SHA256 for {label}")
    return value


def stage_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    stage = config["data"].get("protocol_stage")
    if stage not in ("dino_web_train96", "dino_web_full120"):
        raise ValueError("DINO WEB requires an explicit train96 or full120 stage")
    full = stage == "dino_web_full120"
    return {
        "stage": "full120" if full else "train96",
        "scene_count": 120 if full else 96,
        "samples": 38_400 if full else 30_720,
        "updates_per_epoch": 600 if full else 480,
        "horizon": 42_000 if full else 33_600,
        "warmup": 2_500 if full else 2_000,
        "train_split": "train" if full else TRAIN96_SPLIT,
        "held_split": "train" if full else HELD24_SPLIT,
        "class_index": FULL120_INDEX if full else TRAIN96_INDEX,
        "class_index_sha256": FULL120_INDEX_SHA256 if full else TRAIN96_INDEX_SHA256,
        "index_source_split": "official_train" if full else "official_train_oof_fold",
    }


def freeze_contract(variant: str) -> tuple[list[str], list[str]]:
    if variant == "v2l":
        return [], []
    depth = VARIANTS[variant][1]
    permanent = [
        "backbone.patch_embed.proj.rgb_projection",
        *[f"backbone.blocks.{i}" for i in range(depth // 2)],
    ]
    staged = [
        *[f"backbone.blocks.{i}" for i in range(depth // 2, depth)],
        "backbone.norm",
    ]
    return permanent, staged


def validate_selection_provenance(
    config: Mapping[str, Any], *, verify_file: bool
) -> dict[str, Any]:
    provenance = config.get("selection_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("full120 requires a committed held24 selection_provenance")
    epoch = provenance.get("selected_epoch")
    if type(epoch) is not int or epoch not in CANDIDATE_EPOCHS:
        raise ValueError("full120 selected_epoch is not a precommitted candidate")
    _anchors(
        provenance,
        {
            "selected_train96_successful_updates": epoch * 480,
            "selected_full120_successful_updates": epoch * 600,
            "selection_split": HELD24_SPLIT,
            "selection_rule": SELECTION_RULE,
            "train96_checkpoint_loaded": False,
        },
        "selection_provenance",
    )
    for key in ("source_summary_sha256", "selected_candidate_sha256"):
        _digest(provenance.get(key), f"selection_provenance.{key}")
    if not str(provenance.get("source_summary_path", "")):
        raise ValueError("full120 selection summary path is absent")
    if not verify_file:
        return dict(provenance)
    path = Path(str(provenance["source_summary_path"]))
    _equal(
        sha256_file(path),
        provenance["source_summary_sha256"],
        "selection summary SHA256",
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    _anchors(
        summary,
        {
            "role": "dino_web_held24_selection",
            "status": "completed",
            "variant": config["model"]["variant"],
            "pretrained_file_sha256": config["model"]["pretrained_file_sha256"],
            "candidate_epochs": list(CANDIDATE_EPOCHS),
            "selected_epoch": epoch,
            "selected_candidate_sha256": provenance["selected_candidate_sha256"],
            "selection_split": HELD24_SPLIT,
            "selection_rule": SELECTION_RULE,
            "official30_used": False,
        },
        "held24 summary",
    )
    binding = summary.get("train96_runtime_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("held24 summary lacks train96 source/config/split binding")
    _anchors(
        binding,
        {
            "role": "dino_web_checkpoint_runtime_binding",
            "stage": "train96",
            "mode": "formal",
            "variant": config["model"]["variant"],
            "pretrained_file_sha256": config["model"]["pretrained_file_sha256"],
        },
        "selected train96 binding",
    )
    _digest(binding.get("source_manifest_sha256"), "selected source manifest")
    _digest(binding.get("config_fingerprint"), "selected config fingerprint")
    return dict(summary)


def validate_full120_epoch_sweep(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate the explicitly authorized variant-specific exploration.

    The held24-selected reference remains e24 for V3B/V3L and e18 for V2L.
    Additional epochs are exploration candidates, not a new held24 selection.
    Short-run modes retain their existing one/ten-update smoke budget below.
    """
    sweep = config.get("full120_epoch_sweep")
    if sweep is None:
        return None
    if not isinstance(sweep, Mapping):
        raise ValueError("full120_epoch_sweep must be an explicit mapping")
    if stage_parameters(config)["stage"] != "full120":
        raise ValueError("full120_epoch_sweep is forbidden for train96")
    variant = config["model"].get("variant")
    candidate_epochs, reference_epoch, stop_epoch = FULL120_SWEEP_EPOCHS, 24, 30
    if variant == "v3l":
        allowed_source_paths = _v3l_execution_source_paths(config)
    elif variant == "v2l":
        candidate_epochs, reference_epoch, stop_epoch = V2L_FULL120_SWEEP_EPOCHS, 18, 22
        allowed_source_paths = V2L_FULL120_SWEEP_SOURCE_PATHS
        _equal(config.get("execution_contract"), None, "V2L unchanged single-GPU execution")
        physical = config["train"].get("batch_size")
        if type(physical) is not int or physical not in (4, 8):
            raise ValueError("V2L full120 sweep physical batch must be 8 or 4")
    else:
        _equal(variant, "v3b", "epoch sweep variant")
        allowed_source_paths = FULL120_SWEEP_SOURCE_PATHS
    _equal(
        set(sweep),
        {"schema_version", "role", "purpose", "candidate_epochs", "reference_epoch",
         "stop_epoch", "official30_started", "source_compatibility"},
        "full120_epoch_sweep keys",
    )
    _anchors(
        sweep,
        {
            "schema_version": 1,
            "role": "dino_web_full120_epoch_sweep",
            "purpose": "user_authorized_checkpoint_exploration",
            "candidate_epochs": list(candidate_epochs),
            "reference_epoch": reference_epoch,
            "stop_epoch": stop_epoch,
            "official30_started": False,
        },
        "full120_epoch_sweep",
    )
    if any(type(epoch) is not int for epoch in sweep["candidate_epochs"]):
        raise ValueError("epoch sweep candidates must contain integer epochs")
    _equal(
        config.get("selection_provenance", {}).get("selected_epoch"),
        sweep["reference_epoch"],
        "epoch sweep held24 reference epoch",
    )
    compatibility = sweep.get("source_compatibility")
    if not isinstance(compatibility, Mapping):
        raise ValueError("epoch sweep requires explicit source_compatibility")
    _equal(
        set(compatibility),
        {"role", "source_manifest_path", "source_manifest_sha256",
         "target_source_manifest_sha256", "allowed_changed_paths"},
        "source_compatibility keys",
    )
    _anchors(
        compatibility,
        {
            "role": "dino_web_epoch_sweep_source_compatibility",
            "allowed_changed_paths": list(allowed_source_paths),
        },
        "source_compatibility",
    )
    for key in ("source_manifest_sha256", "target_source_manifest_sha256"):
        _digest(compatibility.get(key), f"source_compatibility.{key}")
    if not isinstance(compatibility.get("source_manifest_path"), str) or not compatibility["source_manifest_path"]:
        raise ValueError("epoch sweep source_manifest_path is absent")
    return dict(sweep)


def _v3l_execution_source_paths(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Bind each explicit execution mode to its own narrow source allowlist."""
    execution = config.get("execution_contract")
    # Canonical equality rejects bool/int substitutions inside nested dicts.
    for contract, source_paths in (
        (V3L_SINGLEGPU_CONTRACT, V3L_SINGLEGPU_SOURCE_PATHS),
        (V3L_MULTIGPU_CONTRACT, V3L_MULTIGPU_SOURCE_PATHS),
    ):
        if execution == contract and canonical_sha256(execution) == canonical_sha256(contract):
            return source_paths
    raise ValueError("DINO WEB requires an exact V3L single or four GPU execution contract")


def full120_training_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the held24 reference and the actual stopping budget distinct."""
    if stage_parameters(config)["stage"] != "full120":
        raise ValueError("full120_training_plan requires the full120 stage")
    validate_selection_provenance(config, verify_file=False)
    sweep = validate_full120_epoch_sweep(config)
    reference = config["selection_provenance"]["selected_epoch"]
    candidates = list(sweep["candidate_epochs"]) if sweep else [reference]
    stop = sweep["stop_epoch"] if sweep else reference
    return {
        "reference_epoch": reference,
        "reference_successful_updates": reference * 600,
        "stop_epoch": stop,
        "terminal_successful_updates": stop * 600,
        "candidate_epochs": candidates,
        "successful_update_milestones": [epoch * 600 for epoch in candidates],
        "primary_successful_optimizer_update": reference * 600 if sweep else None,
        "exploratory_epoch_sweep": sweep is not None,
    }


def _validated_source_manifest_files(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Verify canonical manifest digest and reject ambiguous file inventories."""
    _anchors(manifest, {"schema_version": 1, "role": "dino_web_runtime_source_manifest"}, "source manifest")
    _equal(set(manifest), {"schema_version", "role", "files", "source_manifest_sha256"}, "source manifest keys")
    body = {key: value for key, value in manifest.items() if key != "source_manifest_sha256"}
    _equal(canonical_sha256(body), manifest["source_manifest_sha256"], "source manifest canonical SHA256")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source manifest requires a nonempty file inventory")
    result: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError("source manifest contains an invalid file entry")
        name = entry["path"]
        if (
            not isinstance(name, str) or not name or "\\" in name
            or name.startswith("/") or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or name in result
        ):
            raise ValueError("source manifest contains an ambiguous file path")
        result[name] = _digest(entry["sha256"], f"source manifest file {name}")
    return result


def validate_full120_sweep_source_compatibility(
    config: Mapping[str, Any], summary: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate a narrow orchestration-only extension of the selected run.

    This never permits model/data/loss/optimizer/trainer changes or a blanket
    source mismatch bypass. Both the old and current manifests are pinned.
    """
    sweep = validate_full120_epoch_sweep(config)
    if sweep is None:
        raise ValueError("source compatibility requires an explicit epoch sweep")
    compatibility = sweep["source_compatibility"]
    previous = json.loads(Path(compatibility["source_manifest_path"]).read_text(encoding="utf-8"))
    if not isinstance(previous, Mapping):
        raise ValueError("selected source manifest is not a mapping")
    old_files = _validated_source_manifest_files(previous)
    new_files = _validated_source_manifest_files(manifest)
    expected_old = summary["train96_runtime_binding"]["source_manifest_sha256"]
    _equal(previous["source_manifest_sha256"], expected_old, "held24 selected source manifest")
    _equal(previous["source_manifest_sha256"], compatibility["source_manifest_sha256"], "epoch sweep source manifest")
    _equal(manifest["source_manifest_sha256"], compatibility["target_source_manifest_sha256"], "epoch sweep target source manifest")
    removed = sorted(set(old_files) - set(new_files))
    if removed:
        raise ValueError(f"epoch sweep cannot remove selected source files: {removed}")
    changed = sorted(name for name in new_files if old_files.get(name) != new_files[name])
    variant = config["model"]["variant"]
    if variant == "v3l":
        allowed = _v3l_execution_source_paths(config)
    elif variant == "v2l":
        allowed = V2L_FULL120_SWEEP_SOURCE_PATHS
    else:
        allowed = FULL120_SWEEP_SOURCE_PATHS
    forbidden = sorted(set(changed) - set(allowed))
    if forbidden:
        raise ValueError(f"epoch sweep changed protected training source: {forbidden}")
    return {
        "role": "dino_web_epoch_sweep_source_compatibility_audit",
        "status": "passed",
        "source_manifest_sha256": previous["source_manifest_sha256"],
        "target_source_manifest_sha256": manifest["source_manifest_sha256"],
        "changed_paths": changed,
        "protected_training_source_unchanged": True,
    }


def validate_dino_web_config(
    config: Mapping[str, Any], *, require_pretrained_file: bool = True
) -> None:
    for key in ("experiment", "data", "model", "loss", "train", "inference"):
        if not isinstance(config.get(key), Mapping):
            raise ValueError(f"DINO WEB configuration lacks {key}")
    experiment, data, model, loss, train, inference = [
        config[key]
        for key in ("experiment", "data", "model", "loss", "train", "inference")
    ]
    variant = str(model.get("variant", ""))
    if variant not in VARIANTS:
        raise ValueError("DINO WEB variant must be v3b, v3l or v2l")
    stage = stage_parameters(config)
    full = stage["stage"] == "full120"
    mode = experiment.get("mode", "formal")
    if mode not in ("formal", "probe", "smoke", "fault_recovery"):
        raise ValueError("unsupported DINO WEB experiment.mode")
    name = str(experiment.get("name", ""))
    if (
        variant not in name
        or stage["stage"] not in name
        or (mode != "formal" and mode not in name)
    ):
        raise ValueError(
            "experiment name must identify its variant, stage and short-run mode"
        )
    _equal(experiment.get("seed"), 1337, "seed")
    _anchors(
        data,
        {
            "dataset_format": "fbp_ms",
            "dataset_variant": "legacy",
            "official_120_30_protocol": full,
            "load_sar": False,
            "patch_size": 512,
            "sar_size": 128,
            "context_size": 512,
            "context_input_size": 512,
            "train_split": stage["train_split"],
            "val_split": stage["held_split"],
            "test_split": stage["held_split"],
            "samples_per_epoch": stage["samples"],
            "scene_block_size": 16,
            "max_scene_repeat": 16,
            "rare_class_ids": list(RARE_CLASS_IDS),
            "hard_class_ids": list(HARD_CLASS_IDS),
        },
        "data",
    )
    _equal(data.get("load_pan", False), False, "load_pan")
    for key in ("debug_scene_limit", "fixed_tile_index", "fixed_tile_subsets"):
        _equal(data.get(key), None, f"data.{key}")
    for key in ("fixed_40k_tile_protocol", "validation_exact_fixed_tiles"):
        _equal(data.get(key, False), False, f"data.{key}")
    dynamic = data.get("dynamic_sampling", {})
    expected_dynamic = {
        "scene_fraction": 0.5,
        "rare_fraction": 0.3,
        "hard_fraction": 0.2,
        "class_index": stage["class_index"],
        "class_index_sha256": stage["class_index_sha256"],
        "center_jitter": 128,
        "max_scene_repeat": 16,
    }
    _equal(set(dynamic), set(expected_dynamic), "dynamic sampling keys (label-only)")
    _anchors(dynamic, expected_dynamic, "dynamic_sampling")
    _anchors(
        data.get("normalization", {}).get("ms", {}),
        {
            "divisor": 255.0,
            "mean": [0.363, 0.390, 0.371, 0.506],
            "std": [0.231, 0.231, 0.244, 0.254],
        },
        "MS normalization",
    )
    backbone, _, patch, padded, indices = VARIANTS[variant]
    if variant == "v2l" and model.get("backbone_model") == "vit_large_patch14_dinov2":
        backbone = "vit_large_patch14_dinov2"
    _anchors(
        model,
        {
            "architecture": ARCHITECTURE,
            "fusion_mode": "ms_only",
            "num_classes": 24,
            "backbone_model": backbone,
            "input_band_order": ["B", "G", "R", "NIR"],
            "input_size": 512,
            "padded_size": padded,
            "patch_size": patch,
            "feature_indices": indices,
            "reassemble_channels": [96, 192, 384, 768],
            "decoder_channels": 256,
            "initial_nir_gate": 0.1,
            "nir_initialization": "rgb_patch_weight_mean",
            "gradient_checkpointing": True,
            "freeze_prefix_tokens": variant != "v2l",
            "pretrained": True,
            "autocast_cache_enabled": False,
        },
        "model",
    )
    _anchors(
        model.get("rgb_normalization", {}),
        {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "RGB normalization",
    )
    pinned_sha = _digest(model.get("pretrained_file_sha256"), "pretrained_file_sha256")
    _equal(
        pinned_sha,
        DINO_WEB_VARIANTS[variant].sha256,
        "official variant pretrained SHA256",
    )
    if not str(model.get("pretrained_file", "")):
        raise ValueError("DINO WEB requires an explicit pretrained_file")
    if require_pretrained_file:
        _equal(
            sha256_file(str(model["pretrained_file"])),
            pinned_sha,
            "pretrained file SHA256",
        )
    _anchors(
        loss,
        {
            "class_weights": None,
            "force_main_losses_fp32": True,
            "cross_entropy_weight": 1.0,
            "dice_weight": 0.5,
            "lovasz_weight": 0.25,
        },
        "loss",
    )
    _zero_loss_contract(loss)
    physical = train.get("batch_size")
    if type(physical) is not int or physical not in (1, 2, 4, 8, 16):
        raise ValueError("DINO WEB physical batch must be 1/2/4/8/16")
    accumulation = train.get("gradient_accumulation_steps")
    if type(accumulation) is not int or physical * accumulation != 64:
        raise ValueError("DINO WEB effective crop batch must remain 64")
    if config.get("execution_contract") is not None:
        _v3l_execution_source_paths(config)
        execution = config["execution_contract"]
        _equal(variant, "v3l", "explicit GPU execution variant")
        _equal(full, True, "explicit GPU execution full120 stage")
        _equal(physical, execution["global_physical_batch"], "explicit GPU execution global physical batch")
        _equal(accumulation, execution["gradient_accumulation_steps"], "explicit GPU execution accumulation")
        if config.get("full120_epoch_sweep") is None:
            raise ValueError("explicit GPU execution requires its explicit epoch sweep")
    _anchors(
        train,
        {
            "device": "cuda",
            "loss_microbatch_size": 1,
            "learning_rate": 2e-4,
            "backbone_lr_multiplier": 0.1 if variant == "v2l" else 0.05,
            "layer_decay": 0.9,
            "weight_decay": 0.05,
            "scheduler_unit": "step",
            "warmup_steps": stage["warmup"],
            "warmup_epochs": 0,
            "scheduler_total_successful_optimizer_updates": stage["horizon"],
            "gradient_clip_norm": 1.0,
            "amp": True,
            "amp_dtype": "bfloat16",
            "use_ema": True,
            "ema_decay": 0.999,
            "validation_enabled": False,
            "selection_metric": "last_epoch",
            "trainable_modules": [],
        },
        "train",
    )
    for key in (
        "initialization_checkpoint",
        "ms_initialization_checkpoint",
        "sar_initialization_checkpoint",
        "teacher_checkpoint",
        "continuation_initialization_checkpoint",
        "ms75_anchor_checkpoint",
    ):
        _equal(
            train.get(key), None, f"train.{key} (fresh pretrained initialization only)"
        )
    _equal(train.get("fixed_anchor_teacher_enabled", False), False, "fixed teacher")
    permanent, staged = freeze_contract(variant)
    _equal(train.get("freeze_modules"), permanent, "permanent freeze")
    _equal(train.get("staged_unfreeze_modules"), staged, "staged freeze")
    _equal(
        train.get("staged_unfreeze_epoch"),
        (5 if mode == "formal" else 0) if staged else None,
        "staged unfreeze epoch",
    )
    _anchors(
        inference,
        {"evaluate_test_after_training": False, "tta_transforms": ["none"]},
        "inference",
    )
    selected = (
        validate_selection_provenance(config, verify_file=require_pretrained_file)
        if full
        else None
    )
    if not full and config.get("selection_provenance") is not None:
        raise ValueError("train96 cannot inherit another model's selection")
    validate_full120_epoch_sweep(config)
    if mode == "formal":
        _equal(data.get("debug_samples_per_epoch"), None, "formal debug crop limit")
        full_plan = full120_training_plan(config) if full else None
        terminal = full_plan["terminal_successful_updates"] if full_plan else 33_600
        milestones = full_plan["successful_update_milestones"] if full_plan else [epoch * 480 for epoch in CANDIDATE_EPOCHS]
        _anchors(
            train,
            {
                "max_successful_optimizer_updates": terminal,
                "successful_update_milestones": milestones,
                "primary_successful_optimizer_update": full_plan["primary_successful_optimizer_update"] if full_plan else None,
            },
            "formal train",
        )
        _equal(
            train.get("epochs"),
            terminal // stage["updates_per_epoch"] + 1,
            "attempt capacity epochs",
        )
    else:
        budget = 1 if mode == "probe" else 10
        _anchors(
            train,
            {
                "max_successful_optimizer_updates": budget,
                "successful_update_milestones": [],
                "primary_successful_optimizer_update": None,
            },
            "short run",
        )
        if type(train.get("epochs")) is not int or train["epochs"] not in (1, 2):
            raise ValueError("short-run attempt capacity must be one or two epochs")
        debug = data.get("debug_samples_per_epoch")
        if debug is not None:
            _equal(debug, budget * 64, "explicit short-run debug crop limit")
    resume = train.get("resume_checkpoint")
    if resume:
        output = Path(str(experiment["output_dir"])).resolve()
        if Path(str(resume)).resolve().parent != output / "checkpoints":
            raise ValueError(
                "resume must be local to this independent run's checkpoints directory"
            )


def load_dino_web_config(
    path: str | Path, *, require_pretrained_file: bool = True
) -> dict[str, Any]:
    config = _expand(_load_yaml(Path(path)))
    # Only environment-substituted batch counters need textual normalization.
    for key in ("batch_size", "gradient_accumulation_steps"):
        value = config["train"][key]
        if type(value) is str:
            config["train"][key] = int(value)
    validate_dino_web_config(config, require_pretrained_file=require_pretrained_file)
    return config


def build_dino_web_source_manifest(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    # Configs are intentionally bound separately: selected full120 config is
    # generated after held24, whereas implementation identity must stay equal.
    paths = {root / "train.py", root / "tools" / "train_dino_web.py"}
    paths.update((root / "tcmmsline3").rglob("*.py"))
    paths.update((root / "tools").glob("*dino_web*.py"))
    files = []
    for path in sorted(paths):
        relative = path.resolve().relative_to(root).as_posix()
        value = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        files.append({"path": relative, "sha256": hashlib.sha256(value).hexdigest()})
    body = {
        "schema_version": 1,
        "role": "dino_web_runtime_source_manifest",
        "files": files,
    }
    return {**body, "source_manifest_sha256": canonical_sha256(body)}


def resolve_dino_web_runtime_binding(
    config: Mapping[str, Any],
    *,
    project_root: str | Path,
    data_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = build_dino_web_source_manifest(project_root)
    if data_identity is None:
        from .dino_web_data import audit_dino_web_data_identity

        audit = audit_dino_web_data_identity(config)
    else:
        audit = dict(data_identity)
    stage = stage_parameters(config)["stage"]
    selection = config.get("selection_provenance")
    sweep = validate_full120_epoch_sweep(config)
    compatibility_audit = None
    if stage == "full120":
        summary = validate_selection_provenance(config, verify_file=True)
        if sweep is not None:
            compatibility_audit = validate_full120_sweep_source_compatibility(config, summary, manifest)
        else:
            _equal(
                summary["train96_runtime_binding"]["source_manifest_sha256"],
                manifest["source_manifest_sha256"],
                "selected train96 source manifest",
            )
    binding = {
        "schema_version": 1,
        "role": "dino_web_checkpoint_runtime_binding",
        "variant": config["model"]["variant"],
        "stage": stage,
        "mode": config["experiment"].get("mode", "formal"),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "source_manifest_file_count": len(manifest["files"]),
        "config_fingerprint": config_fingerprint(dict(config)),
        "pretrained_file_sha256": config["model"]["pretrained_file_sha256"],
        "data_identity": audit,
        "selection_provenance": dict(selection) if selection else None,
        "fresh_initialization": "pinned_official_web_pretraining_only",
    }
    if sweep is not None:
        binding["full120_epoch_sweep"] = dict(sweep)
        binding["full120_training_plan"] = full120_training_plan(config)
        binding["source_compatibility_audit"] = compatibility_audit
    return binding


def validate_dino_web_checkpoint_runtime_binding(
    checkpoint: Mapping[str, Any], current_binding: Mapping[str, Any]
) -> None:
    provenance = checkpoint.get("data_provenance", {})
    stored = (
        provenance.get(RUNTIME_BINDING_KEY) if isinstance(provenance, Mapping) else None
    )
    if stored != dict(current_binding):
        raise RuntimeError(
            "DINO WEB checkpoint source/config/pretraining/split/mode binding mismatch before state load"
        )
