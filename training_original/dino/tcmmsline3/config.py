from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.environ.get(name, default)
        if resolved is None:
            raise ValueError(f"environment variable {name!r} is required")
        return resolved

    return _ENV_PATTERN.sub(replace, value)


def _deep_merge(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_fusionaware_a1_class_histogram(data: dict[str, Any]) -> dict[str, Any]:
    """Authenticate and resolve the frozen official-train120 class histogram."""

    path_value = str(
        data.get("fusionaware_a1_class_histogram_manifest", "")
    ).strip()
    if not path_value:
        raise ValueError(
            "FusionAware A1 requires its class-histogram manifest"
        )
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"missing FusionAware A1 class-histogram manifest: {path}"
        )
    expected_file_sha256 = str(
        data.get("fusionaware_a1_class_histogram_manifest_sha256", "")
    )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256):
        raise ValueError(
            "data.fusionaware_a1_class_histogram_manifest_sha256 must be "
            "lowercase SHA256"
        )
    observed_file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_file_sha256 != expected_file_sha256:
        raise ValueError(
            "FusionAware A1 class-histogram manifest SHA256 mismatch"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid FusionAware A1 class-histogram manifest: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("FusionAware A1 class-histogram root must be a mapping")
    exact_contract = {
        "schema_version": 1,
        "artifact_role": "official120_training_label_class_histogram",
        "dataset": "fbp_ms",
        "split": "official_train120",
        "official30_read": False,
        "class_count": 24,
        "count_unit": "canonical_target_pixels",
        "valid_rule": "0 <= canonical_target < 24",
        "ignore_rule": "canonical_target < 0 or canonical_target >= 24",
        "ignore_label": -1,
        "valid_label_range": [0, 23],
    }
    changed = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in exact_contract.items()
        if payload.get(key) != expected
    }
    if changed:
        raise ValueError(
            "FusionAware A1 class-histogram contract changed: "
            f"{changed}"
        )

    scene_ids = payload.get("ordered_scene_ids")
    if (
        not isinstance(scene_ids, list)
        or len(scene_ids) != 120
        or any(not isinstance(value, str) or not value for value in scene_ids)
        or len(set(scene_ids)) != 120
    ):
        raise ValueError(
            "FusionAware A1 histogram requires 120 unique ordered scene IDs"
        )
    if payload.get("ordered_scene_ids_sha256") != _canonical_json_sha256(
        scene_ids
    ):
        raise ValueError(
            "FusionAware A1 ordered scene-ID SHA256 mismatch"
        )

    counts = payload.get("pixel_counts")
    if (
        not isinstance(counts, list)
        or len(counts) != 24
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in counts
        )
    ):
        raise ValueError(
            "FusionAware A1 pixel_counts must be 24 positive integers"
        )
    if payload.get("pixel_counts_sha256") != _canonical_json_sha256(counts):
        raise ValueError("FusionAware A1 pixel-count SHA256 mismatch")
    if payload.get("total_valid_pixels") != sum(counts):
        raise ValueError("FusionAware A1 total_valid_pixels mismatch")
    source_train_list_sha256 = str(
        payload.get("source_train_list_sha256", "")
    )
    if not re.fullmatch(r"[0-9a-f]{64}", source_train_list_sha256):
        raise ValueError(
            "FusionAware A1 source_train_list_sha256 must be lowercase SHA256"
        )
    split_path = (
        Path(str(data.get("root", "")))
        / "splits"
        / f"{str(data.get('train_split', ''))}.txt"
    ).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(
            f"missing FusionAware A1 source train list: {split_path}"
        )
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != (
        source_train_list_sha256
    ):
        raise ValueError(
            "FusionAware A1 source train-list file SHA256 mismatch"
        )
    actual_scene_ids = [
        value.strip()
        for value in split_path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if actual_scene_ids != scene_ids:
        raise ValueError(
            "FusionAware A1 histogram scene order differs from the actual train list"
        )
    records = payload.get("label_records")
    if not isinstance(records, list) or len(records) != 120:
        raise ValueError("FusionAware A1 requires 120 label_records")
    for scene_id, record in zip(scene_ids, records):
        if not isinstance(record, dict) or record.get("scene_id") != scene_id:
            raise ValueError(
                "FusionAware A1 label_records must follow ordered_scene_ids"
            )
        if record.get("shape") != [6908, 7300]:
            raise ValueError(
                f"FusionAware A1 label shape changed for {scene_id}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
            raise ValueError(
                f"FusionAware A1 label SHA256 is invalid for {scene_id}"
            )

    sorted_counts = sorted(counts)
    median = float(sorted_counts[(len(sorted_counts) - 1) // 2])
    raw_weights = [
        min(max(math.sqrt(median / float(count)), 0.5), 3.0)
        for count in counts
    ]
    expectation = sum(
        float(count) * weight for count, weight in zip(counts, raw_weights)
    ) / float(sum(counts))
    weights = [weight / expectation for weight in raw_weights]
    return {
        "path": str(path),
        "file_sha256": expected_file_sha256,
        "ordered_scene_ids": list(scene_ids),
        "ordered_scene_ids_sha256": payload["ordered_scene_ids_sha256"],
        "pixel_counts": list(counts),
        "pixel_counts_sha256": payload["pixel_counts_sha256"],
        "total_valid_pixels": int(payload["total_valid_pixels"]),
        "source_train_list_sha256": source_train_list_sha256,
        "label_records": [dict(record) for record in records],
        "label_records_sha256": _canonical_json_sha256(records),
        "class_weights": weights,
    }


def _load_yaml(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"cyclic config inheritance: {chain}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"config root must be a mapping: {resolved}")
    base_ref = payload.pop("_base_", None)
    if base_ref is None:
        return payload
    base_path = (resolved.parent / str(base_ref)).resolve()
    return _deep_merge(_load_yaml(base_path, (*stack, resolved)), payload)


def _reject_pan(value: Any, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            if "pan" in str(key).lower():
                raise ValueError(
                    f"PAN is outside TCMMSline3 scope; remove config key {key_path!r}"
                )
            _reject_pan(item, key_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_pan(item, f"{path}[{index}]")
    elif isinstance(value, str) and value.strip().lower() == "pan":
        raise ValueError(f"PAN is outside TCMMSline3 scope at {path!r}")


def validate_config(config: dict[str, Any]) -> None:
    _reject_pan(config)
    required = ("experiment", "data", "model", "loss", "train")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing config sections: {missing}")

    patch_size = int(config["data"]["patch_size"])
    sar_size = int(config["data"]["sar_size"])
    sar_native_size = int(config["data"].get("sar_native_size", sar_size))
    if patch_size <= 0 or sar_size <= 0 or sar_native_size <= 0:
        raise ValueError(
            "patch_size, sar_size, and sar_native_size must be positive"
        )
    if patch_size % 4:
        raise ValueError("patch_size must be divisible by 4")
    if sar_size != patch_size // 4:
        raise ValueError(
            "sar_size must equal patch_size / 4 so SAR stays on the coarse grid"
        )
    if int(config["model"]["num_classes"]) != 24:
        raise ValueError(
            "the retained GID contract requires exactly 24 classes"
        )
    dataset_format = str(
        config["data"].get("dataset_format", "ms_sar_manifest")
    )
    dataset_variant = str(config["data"].get("dataset_variant", "legacy"))
    if dataset_format not in {"ms_sar_manifest", "fbp_ms"}:
        raise ValueError(
            "data.dataset_format must be ms_sar_manifest or fbp_ms"
        )
    if dataset_format == "fbp_ms":
        if int(config["data"].get("expected_scene_count", 0)) != 150:
            raise ValueError("FBP data requires expected_scene_count=150")
        if bool(config["data"].get("load_sar", False)):
            raise ValueError("FBP optical data cannot load SAR")
        protocol_stage = str(config["data"].get("protocol_stage", "final"))
        required_split_keys = (
            ("train_split", "test_split")
            if dataset_variant == "ms75_multiscale"
            and protocol_stage == "final"
            else ("train_split", "val_split", "test_split")
        )
        for key in required_split_keys:
            if not str(config["data"].get(key, "")).strip():
                raise ValueError(f"FBP data requires data.{key}")
        if config["data"].get("sampling", {}).get("strategy") == "directed":
            if not str(config["data"].get("sampling_index", "")).strip():
                raise ValueError(
                    "directed FBP sampling requires sampling_index"
                )
        if config["data"].get("sampling_index") and config["data"].get(
            "fixed_tile_index"
        ):
            raise ValueError(
                "FBP sampling_index and fixed_tile_index are mutually exclusive"
            )
        debug_scene_limit = config["data"].get("debug_scene_limit")
        if debug_scene_limit is not None:
            if int(debug_scene_limit) <= 0:
                raise ValueError("data.debug_scene_limit must be positive")
            if (
                "smoke"
                not in str(config["experiment"].get("name", "")).lower()
            ):
                raise ValueError(
                    "data.debug_scene_limit is restricted to smoke experiments"
                )
        debug_fixed_tile_limit = config["data"].get("debug_fixed_tile_limit")
        if debug_fixed_tile_limit is not None:
            if int(debug_fixed_tile_limit) <= 0:
                raise ValueError(
                    "data.debug_fixed_tile_limit must be positive"
                )
            if (
                "smoke"
                not in str(config["experiment"].get("name", "")).lower()
            ):
                raise ValueError(
                    "data.debug_fixed_tile_limit is restricted to smoke experiments"
                )
        debug_samples = config["data"].get("debug_samples_per_epoch")
        if debug_samples is not None:
            if int(debug_samples) <= 0:
                raise ValueError(
                    "data.debug_samples_per_epoch must be positive"
                )
            if (
                "smoke"
                not in str(config["experiment"].get("name", "")).lower()
            ):
                raise ValueError(
                    "data.debug_samples_per_epoch is restricted to smoke experiments"
                )
        if bool(config["data"].get("official_120_30_protocol", False)):
            configured_splits = {
                key: (
                    ""
                    if config["data"].get(key) is None
                    else str(config["data"].get(key, ""))
                )
                for key in ("train_split", "val_split", "test_split")
            }
            expected_splits = (
                {
                    "train_split": "train_dev",
                    "val_split": "val",
                    "test_split": "test",
                }
                if dataset_variant == "ms75_multiscale"
                and protocol_stage == "development"
                else {
                    "train_split": "train",
                    "val_split": (
                        "" if dataset_variant == "ms75_multiscale" else "train"
                    ),
                    "test_split": "test",
                }
            )
            if configured_splits != expected_splits:
                if dataset_variant == "ms75_multiscale":
                    raise ValueError(
                        "MS75 development requires train_dev/val/test; final "
                        "120/30 fitting requires train/test with no val_split"
                    )
                raise ValueError(
                    "official FBP 120/30 protocol requires train_split=train, "
                    "val_split=train and test_split=test"
                )
            if dataset_variant == "ms75_multiscale":
                if protocol_stage not in {"development", "final"}:
                    raise ValueError(
                        "MS75 data.protocol_stage must be development or final"
                    )
                if config["data"].get("fixed_tile_index"):
                    raise ValueError(
                        "MS75 dynamic sampling cannot use fixed_tile_index"
                    )
                fixed_protocol = False
            else:
                fixed_protocol = bool(
                    config["data"].get("fixed_40k_tile_protocol", False)
                )
            if fixed_protocol:
                if not str(config["data"].get("fixed_tile_index", "")).strip():
                    raise ValueError(
                        "fixed FBP 40k protocol requires data.fixed_tile_index"
                    )
                if config["data"].get("fixed_tile_subsets") != {
                    "train": "train",
                    "val": "val",
                }:
                    raise ValueError(
                        "fixed FBP 40k protocol requires train/val tile subsets"
                    )
                if not bool(
                    config["data"].get("validation_exact_fixed_tiles", False)
                ):
                    raise ValueError(
                        "fixed FBP 40k protocol requires exact validation tiles"
                    )
                if patch_size != 512:
                    raise ValueError(
                        "fixed FBP 40k protocol requires data.patch_size=512"
                    )
                if int(config["data"].get("samples_per_epoch", 0)) != 32000:
                    raise ValueError(
                        "fixed FBP 40k protocol requires 32000 training samples"
                    )
                if (
                    float(
                        config["data"].get("validation_holdout_fraction", 0.0)
                    )
                    != 0.0
                ):
                    raise ValueError(
                        "fixed FBP tiles cannot be combined with hash validation"
                    )
                rare_probability = float(
                    config["data"].get("fixed_tile_rare_probability", 0.0)
                )
                if not 0.0 <= rare_probability <= 1.0:
                    raise ValueError(
                        "data.fixed_tile_rare_probability must be in [0, 1]"
                    )
            elif dataset_variant != "ms75_multiscale":
                holdout = float(
                    config["data"].get("validation_holdout_fraction", 0.0)
                )
                configured_architecture = str(
                    config.get("model", {}).get("architecture", "")
                )
                local_full120_train_only = (
                    protocol_stage in {
                        "local_detail_cnxb_full120_train120",
                        "local_detail_cnxb256_full120_train120",
                        "local_detail_cnxb256_full120_epoch40_selected_train120",
                        "local_detail_cnxb512_full120_epoch30_selected_train120",
                        "local_detail_cnxb256_fusionaware_a0_full120_epoch40",
                        "local_detail_cnxb256_fusionaware_a1_full120_10k",
                    }
                    and configured_architecture == "local_detail_cnxb"
                )
                bclean_full120_train_only = (
                    protocol_stage
                    == "hrnet_w32_ocr_bclean10k_full120_train120"
                    and configured_architecture == "hrnet_w32_ocr"
                )
                full120_train_only = (
                    local_full120_train_only or bclean_full120_train_only
                ) and (
                    not bool(
                        config.get("train", {}).get("validation_enabled", True)
                    )
                    and str(
                        config.get("train", {}).get("selection_metric", "")
                    )
                    == "last_epoch"
                )
                if full120_train_only and holdout != 0.0:
                    protocol_name = (
                        "LocalDetail" if local_full120_train_only else "B-clean"
                    )
                    raise ValueError(
                        f"train-only {protocol_name} full120 fitting must not hide a "
                        "spatial validation partition"
                    )
                if not full120_train_only and not 0.0 < holdout < 1.0:
                    raise ValueError(
                        "official FBP in-training validation requires either "
                        "fixed_40k_tile_protocol or "
                        "validation_holdout_fraction in (0, 1)"
                    )
    architecture = str(config["model"].get("architecture", "ms_sar_fusion"))
    if architecture not in {
        "ms_sar_fusion",
        "sar_native",
        "symm_safe_fusion",
        "strong_ms_context",
        "ms70_context",
        "local_detail_cnxb",
        "hrnet_w32_ocr",
        "ms75_multiscale",
        "ms75_oof_aligned_swin",
    }:
        raise ValueError(
            "model.architecture must be ms_sar_fusion, sar_native, "
            "symm_safe_fusion, strong_ms_context, ms70_context, "
            "local_detail_cnxb, hrnet_w32_ocr, ms75_multiscale, or "
            "ms75_oof_aligned_swin"
        )
    if architecture == "ms75_oof_aligned_swin":
        from .ms75_aligned_protocol import validate_aligned_proxy_config

        validate_aligned_proxy_config(config)
    if architecture == "sar_native" and sar_native_size < sar_size:
        raise ValueError(
            "sar_native_size must not be coarser than the fusion SAR grid"
        )
    fusion_mode = str(config["model"]["fusion_mode"])
    if fusion_mode not in {
        "ms_only",
        "sar_only",
        "concat",
        "region",
        "local_residual",
        "symm_safe",
    }:
        raise ValueError("unsupported fusion_mode")
    if architecture == "symm_safe_fusion":
        if fusion_mode not in {"symm_safe", "ms_only"}:
            raise ValueError(
                "symm_safe_fusion requires fusion_mode=symm_safe or ms_only"
            )
        router_grid = int(config["model"].get("router_grid", 8))
        if router_grid <= 0 or router_grid > sar_size:
            raise ValueError("model.router_grid must be in [1, data.sar_size]")
        channels = int(config["model"]["width"]) * 4
        adapter_bottleneck = int(config["model"].get("adapter_bottleneck", 16))
        if not 0 < adapter_bottleneck < channels:
            raise ValueError(
                "model.adapter_bottleneck must be in (0, width * 4)"
            )
        if int(config["model"].get("expert_hidden", 64)) <= 0:
            raise ValueError("model.expert_hidden must be positive")
        if int(config["model"].get("router_hidden", 64)) <= 0:
            raise ValueError("model.router_hidden must be positive")
        routing_mode = str(config["model"].get("routing_mode", "learned"))
        if routing_mode not in {
            "learned",
            "uniform",
            "null",
            "sar",
            "collaborative",
            "both",
        }:
            raise ValueError("unsupported model.routing_mode")
        include_both = bool(config["model"].get("router_include_both", False))
        if routing_mode == "both" and not include_both:
            raise ValueError(
                "model.routing_mode=both requires model.router_include_both=true"
            )
        initial_both_probability = float(
            config["model"].get("router_initial_both_probability", 0.0)
        )
        if not 0 <= initial_both_probability < 1:
            raise ValueError(
                "model.router_initial_both_probability must be in [0, 1)"
            )
        if initial_both_probability > 0 and not include_both:
            raise ValueError(
                "router_initial_both_probability requires router_include_both"
            )
        max_scale = float(config["model"]["max_sar_residual_scale"])
        initial_scale = float(
            config["model"].get("initial_sar_residual_scale", 0.1)
        )
        if not 0 < initial_scale < max_scale:
            raise ValueError(
                "initial_sar_residual_scale must be in (0, max_sar_residual_scale)"
            )
        boundary_attenuation = float(
            config["model"].get("boundary_attenuation", 0.0)
        )
        if not 0 <= boundary_attenuation <= 1:
            raise ValueError("model.boundary_attenuation must be in [0, 1]")
        if float(config["model"].get("max_alignment_offset_pixels", 1.0)) < 0:
            raise ValueError(
                "model.max_alignment_offset_pixels must be non-negative"
            )
    if (
        not bool(config["data"].get("load_sar", True))
        and fusion_mode != "ms_only"
    ):
        raise ValueError("data.load_sar=false requires fusion_mode=ms_only")
    if architecture == "local_detail_cnxb":
        if fusion_mode != "ms_only":
            raise ValueError("local_detail_cnxb requires fusion_mode=ms_only")
        if dataset_format != "fbp_ms":
            raise ValueError("local_detail_cnxb requires data.dataset_format=fbp_ms")
        if patch_size not in {256, 512}:
            raise ValueError(
                "local_detail_cnxb requires data.patch_size to be 256 or 512"
            )
        context_size = int(config["data"].get("context_size", patch_size))
        context_input_size = int(
            config["data"].get("context_input_size", patch_size)
        )
        if context_size != patch_size or context_input_size != patch_size:
            raise ValueError(
                "local_detail_cnxb forbids external context; context_size and "
                "context_input_size must equal patch_size"
            )
        if str(config["model"].get("backbone_variant", "base")) != "base":
            raise ValueError("local_detail_cnxb requires backbone_variant=base")
        for key, default in (
            ("decoder_channels", 192),
            ("detail_channels", 48),
            ("detail_fusion_channels", 64),
        ):
            if int(config["model"].get(key, default)) <= 0:
                raise ValueError(f"model.{key} must be positive")
        pool_scales = config["model"].get("pool_scales", [1, 2, 3, 6])
        if (
            not isinstance(pool_scales, (list, tuple))
            or not pool_scales
            or any(int(value) <= 0 for value in pool_scales)
            or len({int(value) for value in pool_scales}) != len(pool_scales)
        ):
            raise ValueError(
                "model.pool_scales must contain unique positive integers"
            )
        maximum_detail_scale = float(
            config["model"].get("max_detail_logit_scale", 0.5)
        )
        initial_detail_scale = float(
            config["model"].get("initial_detail_logit_scale", 0.1)
        )
        if not (
            math.isfinite(maximum_detail_scale)
            and math.isfinite(initial_detail_scale)
            and 0.0 < initial_detail_scale < maximum_detail_scale
        ):
            raise ValueError(
                "initial_detail_logit_scale must be finite and in "
                "(0, max_detail_logit_scale)"
            )
    if architecture == "hrnet_w32_ocr":
        if fusion_mode != "ms_only":
            raise ValueError("hrnet_w32_ocr requires fusion_mode=ms_only")
        if dataset_format != "fbp_ms":
            raise ValueError("hrnet_w32_ocr requires data.dataset_format=fbp_ms")
        if patch_size != 384 or sar_size != 96:
            raise ValueError(
                "hrnet_w32_ocr B-clean requires patch_size=384 and sar_size=96"
            )
        context_size = int(config["data"].get("context_size", patch_size))
        context_input_size = int(
            config["data"].get("context_input_size", patch_size)
        )
        if context_size != patch_size or context_input_size != patch_size:
            raise ValueError(
                "hrnet_w32_ocr forbids external context; context sizes must "
                "equal patch_size"
            )
        if not bool(config["model"].get("pretrained", False)):
            raise ValueError("hrnet_w32_ocr requires pinned ImageNet pretraining")
        if str(config["model"].get("pretrained_model", "")) != (
            "hrnet_w32.ms_in1k"
        ):
            raise ValueError(
                "hrnet_w32_ocr pretrained_model must be hrnet_w32.ms_in1k"
            )
        pretrained_file = str(
            config["model"].get("pretrained_file", "")
        ).strip()
        pretrained_sha256 = str(
            config["model"].get("pretrained_file_sha256", "")
        ).strip()
        if not pretrained_file or not pretrained_file.lower().endswith(
            ".safetensors"
        ):
            raise ValueError(
                "hrnet_w32_ocr requires an explicit .safetensors pretrained_file"
            )
        if re.fullmatch(r"[0-9A-Fa-f]{64}", pretrained_sha256) is None:
            raise ValueError(
                "hrnet_w32_ocr requires a 64-hex pretrained_file_sha256"
            )
        if int(config["model"].get("stem_in_channels", 0)) != 4:
            raise ValueError("hrnet_w32_ocr requires stem_in_channels=4")
        if list(config["model"].get("input_band_order", [])) != [
            "B",
            "G",
            "R",
            "NIR",
        ]:
            raise ValueError(
                "hrnet_w32_ocr input_band_order must be B/G/R/NIR"
            )
        if str(config["model"].get("nir_init", "")) != "rgb_mean":
            raise ValueError("hrnet_w32_ocr requires nir_init=rgb_mean")
        ocr_channels = int(config["model"].get("ocr_channels", 0))
        key_channels = int(config["model"].get("ocr_key_channels", 0))
        boundary_channels = int(config["model"].get("boundary_channels", 0))
        if (
            ocr_channels <= 0
            or key_channels <= 0
            or key_channels > ocr_channels
            or boundary_channels <= 0
        ):
            raise ValueError(
                "hrnet_w32_ocr channels must be positive and key_channels "
                "must not exceed ocr_channels"
            )
        ocr_dropout = float(config["model"].get("ocr_dropout", -1.0))
        if not math.isfinite(ocr_dropout) or not 0.0 <= ocr_dropout < 1.0:
            raise ValueError("model.ocr_dropout must be finite and in [0, 1)")
        if str(config["model"].get("norm_layer", "")) != "group_norm":
            raise ValueError("hrnet_w32_ocr heads require norm_layer=group_norm")
        if int(config["model"].get("group_norm_groups", 0)) <= 0:
            raise ValueError("model.group_norm_groups must be positive")
        if not bool(
            config["model"].get("freeze_backbone_batch_norm_stats", False)
        ):
            raise ValueError(
                "hrnet_w32_ocr requires frozen backbone BatchNorm running stats"
            )
        forbidden_local_keys = {
            "backbone_variant",
            "decoder_channels",
            "detail_channels",
            "detail_fusion_channels",
            "max_detail_logit_scale",
            "initial_detail_logit_scale",
            "pool_scales",
        }
        retained_local_keys = sorted(forbidden_local_keys & set(config["model"]))
        if retained_local_keys:
            raise ValueError(
                "hrnet_w32_ocr config retains LocalDetail-only model keys: "
                f"{retained_local_keys}"
            )
        aux_weight = float(config["loss"].get("aux_weight", 0.0))
        if not math.isfinite(aux_weight) or aux_weight < 0.0:
            raise ValueError("loss.aux_weight must be finite and non-negative")
    if architecture in {"strong_ms_context", "ms70_context"}:
        if fusion_mode != "ms_only":
            raise ValueError(f"{architecture} requires fusion_mode=ms_only")
        context_size = int(config["data"].get("context_size", patch_size))
        context_input_size = int(
            config["data"].get("context_input_size", patch_size)
        )
        if context_size < patch_size:
            raise ValueError("data.context_size must be at least patch_size")
        if (context_size - patch_size) % 2:
            raise ValueError(
                "data.context_size - patch_size must be even for centering"
            )
        if patch_size % 32 or context_input_size % 32:
            raise ValueError(
                "strong_ms_context patch/context input sizes must be divisible by 32"
            )
        if int(config["model"].get("decoder_channels", 192)) <= 0:
            raise ValueError("model.decoder_channels must be positive")
        if int(config["model"].get("embedding_dim", 96)) <= 0:
            raise ValueError("model.embedding_dim must be positive")
        if int(config["model"].get("prototypes_per_class", 3)) <= 0:
            raise ValueError("model.prototypes_per_class must be positive")
        if float(config["model"].get("prototype_temperature", 10.0)) <= 0:
            raise ValueError("model.prototype_temperature must be positive")
        if float(config["model"].get("max_prototype_logit_scale", 0.25)) < 0:
            raise ValueError(
                "model.max_prototype_logit_scale must be non-negative"
            )
        if architecture == "ms70_context":
            if str(config["model"].get("backbone_variant", "base")) not in {
                "tiny",
                "base",
            }:
                raise ValueError("model.backbone_variant must be tiny or base")
            if int(config["model"].get("context_encoder_width", 64)) <= 0:
                raise ValueError(
                    "model.context_encoder_width must be positive"
                )
            if int(config["model"].get("context_token_grid", 4)) <= 0:
                raise ValueError("model.context_token_grid must be positive")
        if bool(config["model"].get("enable_spectral_refinement", False)):
            if int(config["model"].get("spectral_width", 64)) <= 0:
                raise ValueError("model.spectral_width must be positive")
            max_spectral_scale = float(
                config["model"].get("max_spectral_logit_scale", 0.5)
            )
            initial_spectral_scale = float(
                config["model"].get("initial_spectral_logit_scale", 0.1)
            )
            if not 0 < initial_spectral_scale < max_spectral_scale:
                raise ValueError(
                    "initial_spectral_logit_scale must be in "
                    "(0, max_spectral_logit_scale)"
                )
        if bool(config["model"].get("enable_family_region_refinement", False)):
            if int(config["model"].get("family_region_width", 64)) <= 0:
                raise ValueError("model.family_region_width must be positive")
            grids = config["model"].get("family_region_grids", [1, 2, 4])
            if (
                not isinstance(grids, list)
                or not grids
                or any(int(value) <= 0 for value in grids)
                or len({int(value) for value in grids}) != len(grids)
            ):
                raise ValueError(
                    "model.family_region_grids must contain unique positive integers"
                )
            max_family_region_scale = float(
                config["model"].get("max_family_region_logit_scale", 0.5)
            )
            initial_family_region_scale = float(
                config["model"].get("initial_family_region_logit_scale", 0.1)
            )
            if not (0 < initial_family_region_scale < max_family_region_scale):
                raise ValueError(
                    "initial_family_region_logit_scale must be in "
                    "(0, max_family_region_logit_scale)"
                )
            scale_override = config["model"].get(
                "family_region_scale_override"
            )
            if scale_override is not None and not (
                0 <= float(scale_override) <= max_family_region_scale
            ):
                raise ValueError(
                    "model.family_region_scale_override must be in "
                    "[0, max_family_region_logit_scale]"
                )
            uncertainty_margin = config["model"].get(
                "family_region_uncertainty_margin_threshold"
            )
            if uncertainty_margin is not None and not (
                0 < float(uncertainty_margin) <= 1
            ):
                raise ValueError(
                    "model.family_region_uncertainty_margin_threshold must be in (0, 1]"
                )
            if bool(
                config["model"].get(
                    "family_region_condition_on_spectral_logits", False
                )
            ) and not bool(
                config["model"].get("enable_spectral_refinement", False)
            ):
                raise ValueError(
                    "spectral-conditioned family refinement requires "
                    "model.enable_spectral_refinement"
                )
            if (
                bool(
                    config["model"].get(
                        "family_region_fine_uncertainty_gate", False
                    )
                )
                and uncertainty_margin is None
            ):
                raise ValueError(
                    "model.family_region_fine_uncertainty_gate requires "
                    "family_region_uncertainty_margin_threshold"
                )
    if architecture == "ms75_multiscale":
        if fusion_mode != "ms_only":
            raise ValueError("ms75_multiscale requires fusion_mode=ms_only")
        if dataset_format != "fbp_ms" or dataset_variant != "ms75_multiscale":
            raise ValueError(
                "ms75_multiscale requires FBP data with "
                "data.dataset_variant=ms75_multiscale"
            )
        if bool(config["data"].get("load_sar", False)):
            raise ValueError(
                "MS75 is a pure-MS experiment and cannot load SAR"
            )
        ms75_protocol_stage = str(
            config["data"].get("protocol_stage", "final")
        )
        if (
            not bool(config["data"].get("official_120_30_protocol", False))
            and ms75_protocol_stage != "ms75_clean_fold0_train96"
        ):
            raise ValueError(
                "MS75 requires either the strict official 120/30 protocol or "
                "the explicit clean fold0 train96 protocol"
            )
        short_context_size = int(
            config["data"].get("short_context_size", patch_size * 3)
        )
        short_input_size = int(
            config["data"].get("short_input_size", patch_size)
        )
        global_input_size = int(
            config["data"].get("global_input_size", patch_size)
        )
        if min(short_context_size, short_input_size, global_input_size) <= 0:
            raise ValueError("MS75 context input sizes must be positive")
        if short_context_size < patch_size:
            raise ValueError(
                "MS75 short_context_size must be at least patch_size"
            )
        if (short_context_size - patch_size) % 2:
            raise ValueError(
                "MS75 short context and local patch must have the same center"
            )
        if any(
            value % 32
            for value in (patch_size, short_input_size, global_input_size)
        ):
            raise ValueError(
                "MS75 local, short-input and global-input sizes must be divisible by 32"
            )
        dynamic = config["data"].get("dynamic_sampling", {})
        fractions = [
            float(dynamic.get("scene_fraction", 0.60)),
            float(dynamic.get("class_fraction", 0.20)),
            float(dynamic.get("confusion_fraction", 0.20)),
        ]
        if (
            not all(math.isfinite(value) for value in fractions)
            or any(value < 0 for value in fractions)
            or abs(sum(fractions) - 1.0) > 1e-8
        ):
            raise ValueError(
                "MS75 dynamic sampling fractions must be non-negative and sum to one"
            )
        if (
            fractions[1] > 0
            and not str(dynamic.get("class_index", "")).strip()
        ):
            raise ValueError(
                "MS75 class-balanced sampling requires class_index"
            )
        if (
            fractions[2] > 0
            and not str(dynamic.get("confusion_index", "")).strip()
        ):
            raise ValueError(
                "MS75 confusion-balanced sampling requires confusion_index"
            )
        for kind, fraction in zip(("scene", "class", "confusion"), fractions):
            if kind == "scene" or fraction <= 0:
                continue
            digest = str(dynamic.get(f"{kind}_index_sha256", ""))
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(
                    f"MS75 {kind} sampling requires a pinned lowercase "
                    f"{kind}_index_sha256"
                )
        if int(dynamic.get("center_jitter", 0)) < 0:
            raise ValueError("MS75 center_jitter must be non-negative")
        if int(dynamic.get("max_scene_repeat", 8)) <= 0:
            raise ValueError("MS75 max_scene_repeat must be positive")
        if int(config["data"].get("global_cache_scenes", 120)) <= 0:
            raise ValueError("MS75 global_cache_scenes must be positive")
        for key in (
            "horizontal_flip_probability",
            "vertical_flip_probability",
        ):
            probability = float(config["data"].get(key, 0.0))
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"MS75 data.{key} must be in [0,1]")
        ms75_model = config["model"].get("ms75", {})
        if int(ms75_model.get("num_classes", 24)) != 24:
            raise ValueError("MS75 nested model must use exactly 24 classes")
        encoder_configs = {
            "local_encoder": ms75_model.get("local_encoder", {}),
            "short_encoder": ms75_model.get("short_encoder", {}),
            "context_encoder": ms75_model.get(
                "context_encoder", ms75_model.get("global_encoder", {})
            ),
        }
        for encoder_name, encoder in encoder_configs.items():
            encoder_type = str(encoder.get("type", "tiny"))
            if encoder_type not in {"tiny", "timm"}:
                raise ValueError(
                    f"model.ms75.{encoder_name}.type must be tiny or timm"
                )
            if int(encoder.get("in_channels", 4)) != 4:
                raise ValueError(
                    f"model.ms75.{encoder_name} must consume four bands"
                )
            if encoder_type == "tiny":
                channels = encoder.get("channels")
                if (
                    not isinstance(channels, list)
                    or len(channels) < 2
                    or any(int(value) <= 0 for value in channels)
                ):
                    raise ValueError(
                        f"model.ms75.{encoder_name}.channels must be a "
                        "positive multi-level list"
                    )
            elif not str(encoder.get("name", "")).strip():
                raise ValueError(
                    f"model.ms75.{encoder_name}.name is required for timm"
                )
        spectral_channels = ms75_model.get("spectral_channels", [])
        if (
            not isinstance(spectral_channels, list)
            or len(spectral_channels) < 2
            or any(int(value) <= 0 for value in spectral_channels)
        ):
            raise ValueError(
                "model.ms75.spectral_channels must be a positive multi-level list"
            )
        if int(ms75_model.get("decoder_channels", 256)) <= 0:
            raise ValueError("model.ms75.decoder_channels must be positive")
        if int(ms75_model.get("fusion_channels", 64)) <= 0:
            raise ValueError("model.ms75.fusion_channels must be positive")
        if int(ms75_model.get("contrast_channels", 64)) <= 0:
            raise ValueError("model.ms75.contrast_channels must be positive")
        if not isinstance(
            ms75_model.get("freeze_context_batch_norm_stats", False), bool
        ):
            raise ValueError(
                "model.ms75.freeze_context_batch_norm_stats must be boolean"
            )
        initial_context = float(
            ms75_model.get("initial_context_residual_scale", 0.0)
        )
        max_context = float(ms75_model.get("max_context_residual_scale", 1.0))
        if not 0 <= initial_context <= max_context:
            raise ValueError("invalid MS75 context residual scales")
        initial_hierarchy = float(ms75_model.get("initial_hierarchy_mix", 0.0))
        max_hierarchy = float(ms75_model.get("max_hierarchy_mix", 0.75))
        if not 0 <= initial_hierarchy <= max_hierarchy <= 1:
            raise ValueError("invalid MS75 hierarchy mixture")
    spectral_augmentation = config["data"].get("spectral_augmentation", {})
    if bool(spectral_augmentation.get("enabled", False)):
        probability = float(spectral_augmentation.get("probability", 0.0))
        if not 0 <= probability <= 1:
            raise ValueError(
                "data.spectral_augmentation.probability must be in [0, 1]"
            )
        for key in (
            "global_gain_range",
            "global_offset_range",
            "band_gain_range",
            "band_offset_range",
        ):
            values = spectral_augmentation.get(key)
            if (
                not isinstance(values, list)
                or len(values) != 2
                or float(values[0]) > float(values[1])
            ):
                raise ValueError(
                    f"data.spectral_augmentation.{key} must be [min, max]"
                )
    if (
        dataset_format == "ms_sar_manifest"
        and float(config["data"]["max_time_gap_days"]) > 20
    ):
        raise ValueError(
            "TCMMSline3 must not reintroduce rejected >20-day pairs"
        )
    pairing = str(config["data"].get("sar_pairing", "aligned"))
    if pairing not in {
        "aligned",
        "same_scene_spatial",
        "cross_scene",
        "zero",
    }:
        raise ValueError(
            "data.sar_pairing must be aligned, same_scene_spatial, cross_scene, or zero"
        )
    if pairing == "cross_scene":
        for split_name in ("train", "val", "test"):
            if len(config["data"].get(f"{split_name}_scene_ids", [])) < 2:
                raise ValueError(
                    "cross_scene SAR pairing requires at least two scenes "
                    f"in the {split_name} split"
                )

    objective = str(config["train"].get("objective", "joint"))
    physical_batch_size = int(config["train"].get("batch_size", 0))
    if physical_batch_size <= 0:
        raise ValueError("train.batch_size must be positive")
    if int(config["train"].get("gradient_accumulation_steps", 1)) <= 0:
        raise ValueError("train.gradient_accumulation_steps must be positive")
    loss_microbatch_size = config["train"].get("loss_microbatch_size")
    if loss_microbatch_size is not None:
        if (
            isinstance(loss_microbatch_size, bool)
            or not isinstance(loss_microbatch_size, int)
            or loss_microbatch_size <= 0
            or physical_batch_size % loss_microbatch_size
        ):
            raise ValueError(
                "train.loss_microbatch_size must be a positive integer that "
                "divides train.batch_size"
            )
    if int(config["train"].get("recovery_checkpoint_interval_steps", 0)) < 0:
        raise ValueError(
            "train.recovery_checkpoint_interval_steps must be non-negative"
        )
    scheduler_unit = str(config["train"].get("scheduler_unit", "epoch"))
    if scheduler_unit not in {"epoch", "step"}:
        raise ValueError("train.scheduler_unit must be epoch or step")
    minimum_lr_ratio = float(config["train"].get("minimum_lr_ratio", 0.0))
    if not math.isfinite(minimum_lr_ratio) or not 0.0 <= minimum_lr_ratio <= 1.0:
        raise ValueError("train.minimum_lr_ratio must be finite and in [0, 1]")
    if int(config["train"].get("warmup_steps", 0)) < 0:
        raise ValueError("train.warmup_steps must be non-negative")
    if (
        scheduler_unit == "step"
        and int(config["train"].get("warmup_steps", 0)) <= 0
    ):
        raise ValueError("step scheduler requires positive train.warmup_steps")
    if bool(config["train"].get("use_ema", False)):
        ema_decay = float(config["train"].get("ema_decay", 0.999))
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("train.ema_decay must be in (0, 1)")
    if "deterministic_algorithms" in config["train"] and not isinstance(
        config["train"]["deterministic_algorithms"], bool
    ):
        raise ValueError("train.deterministic_algorithms must be boolean")
    amp_initial_scale = float(
        config["train"].get("amp_initial_scale", 65536.0)
    )
    if not math.isfinite(amp_initial_scale) or amp_initial_scale <= 0.0:
        raise ValueError("train.amp_initial_scale must be finite and positive")
    amp_growth_interval = config["train"].get("amp_growth_interval", 2000)
    if (
        isinstance(amp_growth_interval, bool)
        or not isinstance(amp_growth_interval, int)
        or amp_growth_interval <= 0
    ):
        raise ValueError("train.amp_growth_interval must be a positive integer")
    amp_growth_factor = float(config["train"].get("amp_growth_factor", 2.0))
    if not math.isfinite(amp_growth_factor) or amp_growth_factor <= 1.0:
        raise ValueError(
            "train.amp_growth_factor must be finite and greater than one"
        )
    amp_backoff_factor = float(
        config["train"].get("amp_backoff_factor", 0.5)
    )
    if (
        not math.isfinite(amp_backoff_factor)
        or not 0.0 < amp_backoff_factor < 1.0
    ):
        raise ValueError(
            "train.amp_backoff_factor must be finite and in (0, 1)"
        )
    if objective not in {"joint", "sar_native"}:
        raise ValueError("train.objective must be joint or sar_native")
    if architecture == "sar_native" and objective != "sar_native":
        raise ValueError(
            "sar_native architecture requires train.objective=sar_native"
        )
    if architecture != "sar_native" and objective == "sar_native":
        raise ValueError(
            "sar_native objective requires model.architecture=sar_native"
        )
    selection_metric = str(config["train"].get("selection_metric", "mean_iou"))
    if selection_metric not in {"mean_iou", "native_mean_iou", "last_epoch"}:
        raise ValueError(
            "train.selection_metric must be mean_iou, native_mean_iou, or last_epoch"
        )
    if objective == "sar_native" and selection_metric != "native_mean_iou":
        raise ValueError(
            "sar_native training must select checkpoints by native_mean_iou"
        )
    sar_initialization = config["train"].get("sar_initialization_checkpoint")
    if sar_initialization and architecture not in {
        "ms_sar_fusion",
        "symm_safe_fusion",
    }:
        raise ValueError(
            "train.sar_initialization_checkpoint is only valid for MS-SAR fusion models"
        )
    ms_initialization = config["train"].get("ms_initialization_checkpoint")
    if ms_initialization and architecture != "symm_safe_fusion":
        raise ValueError(
            "train.ms_initialization_checkpoint is only valid for "
            "symm_safe_fusion models"
        )
    initialization = config["train"].get("initialization_checkpoint")
    continuation_initialization = config["train"].get(
        "continuation_initialization_checkpoint"
    )
    continuation_initialization_hash = config["train"].get(
        "continuation_initialization_expected_sha256"
    )
    if continuation_initialization and architecture != "ms75_multiscale":
        raise ValueError(
            "train.continuation_initialization_checkpoint is only valid for "
            "ms75_multiscale"
        )
    if bool(continuation_initialization) != bool(
        continuation_initialization_hash
    ):
        raise ValueError(
            "continuation initialization requires both checkpoint and expected SHA256"
        )
    if (
        continuation_initialization_hash is not None
        and re.fullmatch(
            r"[0-9A-Fa-f]{64}", str(continuation_initialization_hash)
        )
        is None
    ):
        raise ValueError(
            "train.continuation_initialization_expected_sha256 must contain "
            "exactly 64 hexadecimal characters"
        )
    if continuation_initialization and initialization:
        raise ValueError(
            "paired continuation initialization cannot be combined with "
            "train.initialization_checkpoint"
        )
    continuation_offset = int(
        config["train"].get("continuation_epoch_offset", 0)
    )
    if continuation_offset < 0:
        raise ValueError(
            "train.continuation_epoch_offset must be non-negative"
        )
    successful_budget = config["train"].get("max_successful_optimizer_updates")
    if successful_budget is not None:
        if (
            isinstance(successful_budget, bool)
            or not isinstance(successful_budget, int)
            or successful_budget <= 0
        ):
            raise ValueError(
                "train.max_successful_optimizer_updates must be a positive integer"
            )
        if scheduler_unit != "step":
            raise ValueError(
                "a successful optimizer-update budget requires "
                "train.scheduler_unit=step"
            )
        if int(config["train"].get("warmup_steps", 0)) >= int(
            successful_budget
        ):
            raise ValueError(
                "train.warmup_steps must be smaller than the successful "
                "optimizer-update budget"
            )
    successful_milestones = config["train"].get(
        "successful_update_milestones", []
    )
    if not isinstance(successful_milestones, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in successful_milestones
    ):
        raise ValueError(
            "train.successful_update_milestones must be positive integers"
        )
    if successful_milestones != sorted(set(successful_milestones)):
        raise ValueError(
            "train.successful_update_milestones must be strictly increasing and unique"
        )
    if successful_milestones and successful_budget is None:
        raise ValueError(
            "train.successful_update_milestones require a successful-update budget"
        )
    if successful_budget is not None and any(
        value > int(successful_budget) for value in successful_milestones
    ):
        raise ValueError(
            "train.successful_update_milestones cannot exceed the update budget"
        )
    primary_successful_update = config["train"].get(
        "primary_successful_optimizer_update"
    )
    if primary_successful_update is not None:
        if (
            isinstance(primary_successful_update, bool)
            or not isinstance(primary_successful_update, int)
            or primary_successful_update not in successful_milestones
        ):
            raise ValueError(
                "train.primary_successful_optimizer_update must be a listed milestone"
            )
        if successful_budget is None or primary_successful_update >= int(
            successful_budget
        ):
            raise ValueError(
                "train.primary_successful_optimizer_update must be non-terminal"
            )
    scheduler_successful_update_horizon = config["train"].get(
        "scheduler_total_successful_optimizer_updates"
    )
    if scheduler_successful_update_horizon is not None:
        if (
            isinstance(scheduler_successful_update_horizon, bool)
            or not isinstance(scheduler_successful_update_horizon, int)
            or scheduler_successful_update_horizon <= 0
        ):
            raise ValueError(
                "train.scheduler_total_successful_optimizer_updates must be "
                "a positive integer"
            )
        if scheduler_unit != "step":
            raise ValueError(
                "train.scheduler_total_successful_optimizer_updates requires "
                "train.scheduler_unit=step"
            )
        if int(config["train"].get("warmup_steps", 0)) >= int(
            scheduler_successful_update_horizon
        ):
            raise ValueError(
                "train.warmup_steps must be smaller than the successful-update "
                "scheduler horizon"
            )
        if successful_budget is not None and int(
            scheduler_successful_update_horizon
        ) < int(successful_budget):
            raise ValueError(
                "train.scheduler_total_successful_optimizer_updates must be "
                "greater than or equal to train.max_successful_optimizer_updates"
            )
    ms75_anchor = config["train"].get("ms75_anchor_checkpoint")
    if ms75_anchor and architecture != "ms75_multiscale":
        raise ValueError(
            "train.ms75_anchor_checkpoint is only valid for ms75_multiscale"
        )
    if ms75_anchor and initialization:
        raise ValueError(
            "MS75 anchor initialization cannot be combined with a full checkpoint"
        )
    if (
        architecture == "local_detail_cnxb"
        and str(config["data"].get("protocol_stage", "final"))
        == "local_detail_cnxb256_clean_fold0_train96"
    ):
        clean_split = "ms75_oof_seed1337_fold_0_train"
        if any(
            str(config["data"].get(key, "")) != clean_split
            for key in ("train_split", "val_split", "test_split")
        ):
            raise ValueError(
                "clean LocalDetail-256 fold0 requires every loader alias to "
                "use the precommitted train96 split"
            )
        forbidden_clean_initializers = {
            "train.initialization_checkpoint": initialization,
            "train.ms_initialization_checkpoint": ms_initialization,
            "train.sar_initialization_checkpoint": sar_initialization,
            "train.teacher_checkpoint": config["train"].get("teacher_checkpoint"),
            "train.continuation_initialization_checkpoint": (
                continuation_initialization
            ),
            "train.ms75_anchor_checkpoint": ms75_anchor,
        }
        enabled = [
            key for key, value in forbidden_clean_initializers.items() if value
        ]
        if enabled:
            raise ValueError(
                "clean LocalDetail-256 fold0 forbids checkpoint initializers: "
                f"{enabled}"
            )
        if bool(
            config["train"].get("fixed_anchor_teacher_enabled", False)
        ) or float(config["loss"].get("fixed_anchor_no_harm_weight", 0.0)) != 0.0:
            raise ValueError(
                "clean LocalDetail-256 fold0 forbids fixed/full120 teachers"
            )
        clean_local256_contract = {
            "epochs": 61,
            "batch_size": 32,
            "gradient_accumulation_steps": 2,
            "loss_microbatch_size": 4,
            "scheduler_unit": "step",
            "warmup_steps": 3_840,
            "max_successful_optimizer_updates": 28_800,
            "amp_initial_scale": 8_192.0,
            "amp_growth_interval": 1_000_000,
            "amp_growth_factor": 2.0,
            "amp_backoff_factor": 0.5,
            "use_ema": True,
            "ema_decay": 0.999,
            "ema_milestone_epochs": [],
            "checkpoint_every": 100,
            "recovery_checkpoint_interval_steps": 500,
            "validation_enabled": False,
            "selection_metric": "last_epoch",
            "auto_resume": True,
        }
        changed_local256_contract = {
            key: {"expected": expected, "observed": config["train"].get(key)}
            for key, expected in clean_local256_contract.items()
            if config["train"].get(key) != expected
        }
        if changed_local256_contract:
            raise ValueError(
                "clean LocalDetail-256 fold0 optimizer/AMP contract changed: "
                f"{changed_local256_contract}"
            )
        clean_local256_data = {
            "patch_size": 256,
            "sar_size": 64,
            "context_size": 256,
            "context_input_size": 256,
            "samples_per_epoch": 30_720,
            "min_labeled_fraction": 0.5,
            "sampling_attempts": 16,
        }
        changed_local256_data = {
            key: {"expected": expected, "observed": config["data"].get(key)}
            for key, expected in clean_local256_data.items()
            if config["data"].get(key) != expected
        }
        if changed_local256_data:
            raise ValueError(
                "clean LocalDetail-256 fold0 data contract changed: "
                f"{changed_local256_data}"
            )
        if config["model"].get("pretrained_file_sha256") != (
            "6075fbad31c688f9ed013922a59daf0fc2f4dc12e6e43bc01fa840c7bd1ca2cd"
        ):
            raise ValueError(
                "clean LocalDetail-256 fold0 pretrained checkpoint hash changed"
            )
        clean_local256_normalization = {
            "divisor": 255.0,
            "mean": [0.363, 0.390, 0.371, 0.506],
            "std": [0.231, 0.231, 0.244, 0.254],
        }
        observed_ms_normalization = (
            config["data"].get("normalization", {}).get("ms", {})
        )
        if observed_ms_normalization != clean_local256_normalization:
            raise ValueError(
                "clean LocalDetail-256 fold0 MS normalization changed"
            )
        clean_local256_loss_weights = {
            "cross_entropy_weight": 1.0,
            "dice_weight": 0.5,
            "lovasz_weight": 0.25,
            "spectral_aux_weight": 0.1,
            "boundary_weight": 0.05,
            "family_ce_weight": 0.0,
            "hierarchy_consistency_weight": 0.0,
            "family_region_aux_weight": 0.0,
            "region_prototype_weight": 0.0,
            "confusion_margin_weight": 0.0,
            "ms_aux_weight": 0.0,
            "sar_aux_weight": 0.0,
            "sar_native_aux_weight": 0.0,
            "prototype_alignment_weight": 0.0,
            "private_orthogonality_weight": 0.0,
            "negative_transfer_weight": 0.0,
            "utility_routing_weight": 0.0,
            "alignment_offset_weight": 0.0,
            "teacher_distillation_weight": 0.0,
            "context_aux_weight": 0.0,
            "short_aux_weight": 0.0,
            "global_aux_weight": 0.0,
            "family_child_weight": 0.0,
            "family_contrast_weight": 0.0,
            "family_region_confusion_weight": 0.0,
            "family_region_no_harm_weight": 0.0,
            "fixed_anchor_no_harm_weight": 0.0,
        }
        observed_local256_loss_weights = {
            key: value
            for key, value in config["loss"].items()
            if key.endswith("_weight")
        }
        if observed_local256_loss_weights != clean_local256_loss_weights:
            raise ValueError(
                "clean LocalDetail-256 fold0 loss weights changed: "
                f"{observed_local256_loss_weights}"
            )
        inference = config.get("inference", {})
        if (
            str(inference.get("protocol_role", ""))
            != "local_detail_cnxb256_clean_fold0_train96"
            or list(inference.get("tta_transforms", [])) != ["none"]
            or bool(inference.get("evaluate_test_after_training", False))
            or int(inference.get("batch_size", 0)) != 16
            or int(inference.get("tile_stride", 0)) != 128
        ):
            raise ValueError(
                "clean LocalDetail-256 fold0 inference contract changed"
            )
    if (
        architecture == "ms75_multiscale"
        and str(config["data"].get("protocol_stage", "final")) == "development"
        and ms75_anchor
    ):
        raise ValueError(
            "development MS75 forbids an anchor trained on the full official "
            "120 scenes; use no anchor until a verifiable 108-scene anchor "
            "protocol is provided"
        )
    legacy_anchor_hash = config["train"].get("ms75_anchor_expected_sha256")
    if legacy_anchor_hash is not None:
        raise ValueError(
            "ms75_anchor_expected_sha256 is ambiguous; use both "
            "ms75_anchor_source_sha256 and ms75_anchor_destination_sha256"
        )
    anchor_source_hash = config["train"].get("ms75_anchor_source_sha256")
    anchor_destination_hash = config["train"].get(
        "ms75_anchor_destination_sha256"
    )
    anchor_hashes = {
        "ms75_anchor_source_sha256": anchor_source_hash,
        "ms75_anchor_destination_sha256": anchor_destination_hash,
    }
    for field, value in anchor_hashes.items():
        if value is not None and (
            not ms75_anchor
            or re.fullmatch(r"[0-9A-Fa-f]{64}", str(value)) is None
        ):
            raise ValueError(
                f"{field} requires an anchor and exactly 64 hex characters"
            )
    if ms75_anchor and (
        anchor_source_hash is None or anchor_destination_hash is None
    ):
        raise ValueError(
            "MS75 anchor requires separate source and destination SHA256 pins"
        )
    initialization_excludes = [
        str(value)
        for value in config["train"].get("initialization_exclude_prefixes", [])
    ]
    if initialization_excludes and not initialization:
        raise ValueError(
            "train.initialization_exclude_prefixes requires "
            "train.initialization_checkpoint"
        )
    if any(not value for value in initialization_excludes):
        raise ValueError("initialization exclude prefixes must be non-empty")
    if (
        architecture == "symm_safe_fusion"
        and initialization
        and (ms_initialization or sar_initialization)
    ):
        raise ValueError(
            "a full SymmSafe initialization cannot be combined with MS/SAR "
            "partial initialization checkpoints"
        )
    trainable_modules = [
        str(value) for value in config["train"].get("trainable_modules", [])
    ]
    if any(not value for value in trainable_modules):
        raise ValueError("train.trainable_modules entries must be non-empty")
    if trainable_modules and config["train"].get("freeze_modules"):
        raise ValueError(
            "train.trainable_modules cannot be combined with freeze_modules"
        )
    if bool(config["train"].get("verify_frozen_anchor", False)):
        if not initialization:
            raise ValueError(
                "train.verify_frozen_anchor requires initialization_checkpoint"
            )
        if not trainable_modules:
            raise ValueError(
                "train.verify_frozen_anchor requires an explicit "
                "trainable_modules allowlist"
            )
        if config["train"].get("staged_unfreeze_modules"):
            raise ValueError(
                "train.verify_frozen_anchor forbids staged_unfreeze_modules"
            )
        expected_mutable_prefixes = {
            f"{module_name}." for module_name in trainable_modules
        }
        if set(initialization_excludes) != expected_mutable_prefixes:
            raise ValueError(
                "verified frozen-anchor runs require initialization exclusions "
                "to exactly match trainable_modules"
            )
    router_warmup_epochs = int(config["train"].get("router_warmup_epochs", 0))
    if router_warmup_epochs < 0:
        raise ValueError("train.router_warmup_epochs must be non-negative")
    staged_unfreeze_epoch = config["train"].get("staged_unfreeze_epoch")
    if staged_unfreeze_epoch is not None and int(staged_unfreeze_epoch) < 0:
        raise ValueError("train.staged_unfreeze_epoch must be non-negative")
    if float(config["train"].get("staged_lr_multiplier", 0.1)) <= 0:
        raise ValueError("train.staged_lr_multiplier must be positive")
    if float(config["train"].get("backbone_lr_multiplier", 1.0)) <= 0:
        raise ValueError("train.backbone_lr_multiplier must be positive")
    backbone_prefixes = [
        str(value)
        for value in config["train"].get("backbone_prefixes", ["backbone."])
    ]
    if not backbone_prefixes or any(not value for value in backbone_prefixes):
        raise ValueError(
            "train.backbone_prefixes must contain non-empty prefixes"
        )
    contrast_start = config["train"].get("family_contrast_start_epoch")
    if contrast_start is not None and int(contrast_start) < 0:
        raise ValueError(
            "train.family_contrast_start_epoch must be non-negative"
        )
    if int(config["train"].get("family_contrast_ramp_epochs", 0)) < 0:
        raise ValueError(
            "train.family_contrast_ramp_epochs must be non-negative"
        )
    fixed_teacher_enabled = config["train"].get(
        "fixed_anchor_teacher_enabled", False
    )
    fixed_teacher_amp = config["train"].get("fixed_anchor_teacher_amp", False)
    if not isinstance(fixed_teacher_enabled, bool):
        raise ValueError("train.fixed_anchor_teacher_enabled must be boolean")
    if not isinstance(fixed_teacher_amp, bool):
        raise ValueError("train.fixed_anchor_teacher_amp must be boolean")
    if fixed_teacher_amp and not fixed_teacher_enabled:
        raise ValueError(
            "train.fixed_anchor_teacher_amp requires the fixed anchor teacher"
        )
    if fixed_teacher_enabled and config["train"].get("teacher_checkpoint"):
        raise ValueError(
            "fixed anchor teacher cannot be combined with teacher_checkpoint"
        )
    milestone_value = config["train"].get("ema_milestone_epochs", [])
    if not isinstance(milestone_value, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in milestone_value
    ):
        raise ValueError(
            "train.ema_milestone_epochs must be a list of integers"
        )
    ema_milestone_epochs = [int(value) for value in milestone_value]
    if ema_milestone_epochs != sorted(set(ema_milestone_epochs)):
        raise ValueError(
            "train.ema_milestone_epochs must be strictly increasing and unique"
        )
    if any(
        value < 1 or value > int(config["train"]["epochs"])
        for value in ema_milestone_epochs
    ):
        raise ValueError(
            "train.ema_milestone_epochs must lie within completed epochs"
        )
    if ema_milestone_epochs and not bool(
        config["train"].get("use_ema", False)
    ):
        raise ValueError(
            "train.ema_milestone_epochs requires train.use_ema=true"
        )
    validation_enabled = bool(config["train"].get("validation_enabled", True))
    if architecture == "ms75_multiscale":
        protocol_stage = str(config["data"].get("protocol_stage", "final"))
        if protocol_stage == "final" and (
            validation_enabled or selection_metric != "last_epoch"
        ):
            raise ValueError(
                "final MS75 fitting uses all 120 scenes and requires "
                "validation_enabled=false with selection_metric=last_epoch"
            )
        if protocol_stage == "development" and not validation_enabled:
            raise ValueError("development MS75 training requires validation")
        if protocol_stage == "development" and bool(
            config.get("inference", {}).get(
                "evaluate_test_after_training", False
            )
        ):
            raise ValueError(
                "development MS75 must not evaluate the official test scenes"
            )

    for key in (
        "sar_dropout_probability",
        "sar_noise_probability",
        "sar_shift_probability",
    ):
        probability = float(config["train"].get(key, 0.0))
        if not 0 <= probability <= 1:
            raise ValueError(f"train.{key} must be in [0, 1]")
    if float(config["train"].get("sar_noise_std", 0.0)) < 0:
        raise ValueError("train.sar_noise_std must be non-negative")
    if int(config["train"].get("sar_shift_max_pixels", 0)) < 0:
        raise ValueError("train.sar_shift_max_pixels must be non-negative")

    route_target_mode = str(config["loss"].get("route_target_mode", "hard"))
    if route_target_mode not in {"hard", "soft"}:
        raise ValueError("loss.route_target_mode must be hard or soft")
    if float(config["loss"].get("route_utility_temperature", 1.0)) <= 0:
        raise ValueError("loss.route_utility_temperature must be positive")
    if int(config["loss"].get("route_min_class_region_pixels", 1)) < 1:
        raise ValueError("loss.route_min_class_region_pixels must be positive")
    if float(config["loss"].get("route_min_utility_gap", 0.0)) < 0:
        raise ValueError("loss.route_min_utility_gap must be non-negative")
    negative_transfer_reduction = str(
        config["loss"].get("negative_transfer_reduction", "pixel")
    )
    if negative_transfer_reduction not in {"pixel", "class_region_best"}:
        raise ValueError(
            "loss.negative_transfer_reduction must be pixel or class_region_best"
        )
    if int(config["loss"].get("negative_transfer_min_pixels", 1)) < 1:
        raise ValueError("loss.negative_transfer_min_pixels must be positive")
    for key in (
        "lovasz_weight",
        "family_ce_weight",
        "hierarchy_consistency_weight",
        "boundary_weight",
        "spectral_aux_weight",
        "context_aux_weight",
        "short_aux_weight",
        "global_aux_weight",
        "family_region_aux_weight",
        "family_region_confusion_weight",
        "family_region_confusion_margin",
        "family_region_no_harm_weight",
        "fixed_anchor_no_harm_weight",
        "family_child_weight",
        "family_contrast_weight",
        "region_prototype_weight",
        "confusion_margin_weight",
        "confusion_margin",
        "fusion_mixture_nll_weight",
        "fusionaware_a1_dual_fusion_weight",
        "fusionaware_a1_repair_weight",
        "fusionaware_a1_class_balanced_weight",
        "fusionaware_a1_noharm_weight",
    ):
        value = float(config["loss"].get(key, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"loss.{key} must be non-negative")
    fusion_mixture_weight = float(
        config["loss"].get("fusion_mixture_nll_weight", 0.0)
    )
    fusion_anchor_probability = float(
        config["loss"].get("fusion_anchor_probability", 0.5)
    )
    if not math.isfinite(fusion_anchor_probability) or not (
        0.0 < fusion_anchor_probability < 1.0
    ):
        raise ValueError("loss.fusion_anchor_probability must be in (0, 1)")
    truth_probability_manifest = str(
        config["data"].get("ms75_truth_probability_manifest", "")
    ).strip()
    fusionaware_a1_enabled = bool(
        config["loss"].get("fusionaware_a1_enabled", False)
    )
    if fusion_mixture_weight > 0.0:
        if str(config["data"].get("protocol_stage", "")) != (
            "local_detail_cnxb256_fusionaware_a0_full120_epoch40"
        ):
            raise ValueError(
                "fusion mixture NLL is currently authorized only for the "
                "explicit FusionAware A0 full120 protocol stage"
            )
        if architecture != "local_detail_cnxb" or dataset_format != "fbp_ms":
            raise ValueError(
                "fusion mixture NLL is restricted to the FBP LocalDetail expert"
            )
        if not truth_probability_manifest:
            raise ValueError(
                "fusion mixture NLL requires data.ms75_truth_probability_manifest"
            )
        truth_manifest_digest = str(
            config["data"].get(
                "ms75_truth_probability_manifest_sha256", ""
            )
        )
        if not re.fullmatch(r"[0-9a-f]{64}", truth_manifest_digest):
            raise ValueError(
                "data.ms75_truth_probability_manifest_sha256 must be lowercase SHA256"
            )
        if not str(
            config["data"].get(
                "ms75_truth_probability_expected_provenance", ""
            )
        ).strip():
            raise ValueError(
                "fusion mixture NLL requires an expected MS75 producer provenance"
            )
        for key in (
            "ms75_truth_probability_expected_checkpoint_sha256",
            "ms75_truth_probability_expected_config_sha256",
        ):
            expected_producer_digest = str(config["data"].get(key, ""))
            if not re.fullmatch(r"[0-9a-f]{64}", expected_producer_digest):
                raise ValueError(f"data.{key} must be lowercase SHA256")
        verify_probability_files = config["data"].get(
            "ms75_truth_probability_verify_file_sha256"
        )
        if verify_probability_files is not True:
            raise ValueError(
                "fusion mixture NLL requires full truth-probability file SHA256 "
                "verification"
            )
        if int(
            config["data"].get("ms75_truth_probability_cache_scenes", 2)
        ) <= 0:
            raise ValueError(
                "data.ms75_truth_probability_cache_scenes must be positive"
            )
        if validation_enabled:
            raise ValueError(
                "FusionAware fitting must not select on an in-training validation loss"
            )
    elif truth_probability_manifest and not fusionaware_a1_enabled:
        raise ValueError(
            "MS75 truth-probability manifest requires a positive fusion mixture weight"
        )

    if str(config["data"].get("protocol_stage", "")) == (
        "local_detail_cnxb256_fusionaware_a0_full120_epoch40"
    ):
        a0_contract: dict[str, tuple[object, object]] = {
            "experiment.name": (
                config["experiment"].get("name"),
                "tcmmsline3_local256_fusionaware_a0_full120_epoch40",
            ),
            "data.dataset_format": (dataset_format, "fbp_ms"),
            "data.dataset_variant": (dataset_variant, "legacy"),
            "data.official_120_30_protocol": (
                config["data"].get("official_120_30_protocol"),
                True,
            ),
            "data.train_split": (config["data"].get("train_split"), "train"),
            "data.val_split": (config["data"].get("val_split"), "train"),
            "data.test_split": (config["data"].get("test_split"), "test"),
            "data.validation_holdout_fraction": (
                config["data"].get("validation_holdout_fraction"),
                0.0,
            ),
            "data.sampling_index": (
                config["data"].get("sampling_index"),
                "metadata/local_detail_points_train120_seed1337_stride16_cap64.npz",
            ),
            "data.patch_size": (config["data"].get("patch_size"), 256),
            "data.sar_size": (config["data"].get("sar_size"), 64),
            "data.context_size": (config["data"].get("context_size"), 256),
            "data.context_input_size": (
                config["data"].get("context_input_size"),
                256,
            ),
            "data.samples_per_epoch": (
                config["data"].get("samples_per_epoch"),
                38_400,
            ),
            "data.expected_hw": (
                config["data"].get("expected_hw"),
                [6908, 7300],
            ),
            "data.load_sar": (config["data"].get("load_sar"), False),
            "data.num_workers": (config["data"].get("num_workers"), 8),
            "data.sampling.strategy": (
                config["data"].get("sampling", {}).get("strategy"),
                "directed",
            ),
            "data.sampling.rare_fraction": (
                config["data"].get("sampling", {}).get("rare_fraction"),
                0.30,
            ),
            "data.sampling.hard_fraction": (
                config["data"].get("sampling", {}).get("hard_fraction"),
                0.20,
            ),
            "data.sampling.center_jitter": (
                config["data"].get("sampling", {}).get("center_jitter"),
                64,
            ),
            "data.spectral_augmentation.enabled": (
                config["data"].get("spectral_augmentation", {}).get("enabled"),
                True,
            ),
            "data.spectral_augmentation.probability": (
                config["data"].get("spectral_augmentation", {}).get(
                    "probability"
                ),
                0.8,
            ),
            "data.ms75_truth_probability_expected_provenance": (
                config["data"].get(
                    "ms75_truth_probability_expected_provenance"
                ),
                "in_sample_full120_ms75_no_tta",
            ),
            "data.ms75_truth_probability_expected_checkpoint_sha256": (
                config["data"].get(
                    "ms75_truth_probability_expected_checkpoint_sha256"
                ),
                "dc52e3c98a9624cb5bfd9c40ef58402ef7e08cff0556a68e5641d782f77a6bf5",
            ),
            "data.ms75_truth_probability_expected_config_sha256": (
                config["data"].get(
                    "ms75_truth_probability_expected_config_sha256"
                ),
                "6e419b8374e711eee92c9d12e422de45f7ea36b9ba367f4e8dae5e6f2ad22228",
            ),
            "data.ms75_truth_probability_verify_file_sha256": (
                config["data"].get(
                    "ms75_truth_probability_verify_file_sha256"
                ),
                True,
            ),
            "model.architecture": (architecture, "local_detail_cnxb"),
            "model.pretrained": (config["model"].get("pretrained"), True),
            "model.pretrained_file_sha256": (
                config["model"].get("pretrained_file_sha256"),
                "6075fbad31c688f9ed013922a59daf0fc2f4dc12e6e43bc01fa840c7bd1ca2cd",
            ),
            "model.backbone_variant": (
                config["model"].get("backbone_variant"),
                "base",
            ),
            "model.decoder_channels": (
                config["model"].get("decoder_channels"),
                192,
            ),
            "model.detail_channels": (
                config["model"].get("detail_channels"),
                48,
            ),
            "model.detail_fusion_channels": (
                config["model"].get("detail_fusion_channels"),
                64,
            ),
            "loss.class_weights": (config["loss"].get("class_weights"), None),
            "loss.cross_entropy_weight": (
                config["loss"].get("cross_entropy_weight"),
                1.0,
            ),
            "loss.dice_weight": (config["loss"].get("dice_weight"), 0.5),
            "loss.lovasz_weight": (config["loss"].get("lovasz_weight"), 0.25),
            "loss.spectral_aux_weight": (
                config["loss"].get("spectral_aux_weight"),
                0.10,
            ),
            "loss.boundary_weight": (
                config["loss"].get("boundary_weight"),
                0.05,
            ),
            "loss.fusion_mixture_nll_weight": (
                config["loss"].get("fusion_mixture_nll_weight"),
                0.5,
            ),
            "loss.fusion_anchor_probability": (
                config["loss"].get("fusion_anchor_probability"),
                0.5,
            ),
            **{
                f"loss.{key}": (config["loss"].get(key, 0.0), 0.0)
                for key in (
                    "family_ce_weight",
                    "hierarchy_consistency_weight",
                    "context_aux_weight",
                    "short_aux_weight",
                    "global_aux_weight",
                    "family_region_aux_weight",
                    "region_prototype_weight",
                    "confusion_margin_weight",
                    "family_child_weight",
                    "family_contrast_weight",
                    "family_region_confusion_weight",
                    "family_region_no_harm_weight",
                    "fixed_anchor_no_harm_weight",
                    "ms_aux_weight",
                    "sar_aux_weight",
                    "sar_native_aux_weight",
                    "prototype_alignment_weight",
                    "private_orthogonality_weight",
                    "negative_transfer_weight",
                    "utility_routing_weight",
                    "alignment_offset_weight",
                    "teacher_distillation_weight",
                )
            },
            "train.epochs": (config["train"].get("epochs"), 41),
            "train.batch_size": (config["train"].get("batch_size"), 32),
            "train.gradient_accumulation_steps": (
                config["train"].get("gradient_accumulation_steps"),
                2,
            ),
            "train.loss_microbatch_size": (
                config["train"].get("loss_microbatch_size"),
                4,
            ),
            "train.learning_rate": (
                config["train"].get("learning_rate"),
                0.0003,
            ),
            "train.backbone_lr_multiplier": (
                config["train"].get("backbone_lr_multiplier"),
                0.1,
            ),
            "train.weight_decay": (
                config["train"].get("weight_decay"),
                0.01,
            ),
            "train.scheduler_unit": (
                config["train"].get("scheduler_unit"),
                "step",
            ),
            "train.warmup_steps": (
                config["train"].get("warmup_steps"),
                4_800,
            ),
            "train.max_successful_optimizer_updates": (
                config["train"].get("max_successful_optimizer_updates"),
                23_999,
            ),
            "train.scheduler_total_successful_optimizer_updates": (
                config["train"].get(
                    "scheduler_total_successful_optimizer_updates"
                ),
                36_000,
            ),
            "train.amp": (config["train"].get("amp"), True),
            "train.amp_initial_scale": (
                config["train"].get("amp_initial_scale"),
                8192.0,
            ),
            "train.amp_growth_interval": (
                config["train"].get("amp_growth_interval"),
                1_000_000,
            ),
            "train.gradient_clip_norm": (
                config["train"].get("gradient_clip_norm"),
                1.0,
            ),
            "train.use_ema": (config["train"].get("use_ema"), True),
            "train.ema_decay": (config["train"].get("ema_decay"), 0.999),
            "train.ema_milestone_epochs": (
                config["train"].get("ema_milestone_epochs"),
                [],
            ),
            "train.validation_enabled": (
                config["train"].get("validation_enabled"),
                False,
            ),
            "train.selection_metric": (
                config["train"].get("selection_metric"),
                "last_epoch",
            ),
            "train.initialization_checkpoint": (
                config["train"].get("initialization_checkpoint"),
                None,
            ),
            "train.teacher_checkpoint": (
                config["train"].get("teacher_checkpoint"),
                None,
            ),
            "train.ms75_anchor_checkpoint": (
                config["train"].get("ms75_anchor_checkpoint"),
                None,
            ),
            "train.auto_resume": (config["train"].get("auto_resume"), True),
            "inference.protocol_role": (
                config.get("inference", {}).get("protocol_role"),
                "local256_fusionaware_a0_full120_epoch40",
            ),
            "inference.batch_size": (
                config.get("inference", {}).get("batch_size"),
                16,
            ),
            "inference.tile_stride": (
                config.get("inference", {}).get("tile_stride"),
                128,
            ),
            "inference.tta_transforms": (
                config.get("inference", {}).get("tta_transforms"),
                ["none"],
            ),
            "inference.blend_window": (
                config.get("inference", {}).get("blend_window"),
                "hann",
            ),
            "inference.hann_floor": (
                config.get("inference", {}).get("hann_floor"),
                0.05,
            ),
            "inference.amp": (
                config.get("inference", {}).get("amp"),
                True,
            ),
            "inference.evaluate_test_after_training": (
                config.get("inference", {}).get(
                    "evaluate_test_after_training"
                ),
                False,
            ),
        }
        for field, (observed, required_value) in a0_contract.items():
            bool_type_mismatch = isinstance(required_value, bool) and not isinstance(
                observed, bool
            )
            if bool_type_mismatch or observed != required_value:
                raise ValueError(
                    "FusionAware A0 contract requires "
                    f"{field}={required_value!r}, observed={observed!r}"
                )

    if fusionaware_a1_enabled:
        if str(config["data"].get("protocol_stage", "")) != (
            "local_detail_cnxb256_fusionaware_a1_full120_10k"
        ):
            raise ValueError(
                "FusionAware A1 is authorized only for its explicit full120 stage"
            )
        if architecture != "local_detail_cnxb" or dataset_format != "fbp_ms":
            raise ValueError(
                "FusionAware A1 is restricted to the FBP LocalDetail expert"
            )
        if fusion_mixture_weight != 0.0:
            raise ValueError(
                "FusionAware A1 requires legacy fusion_mixture_nll_weight=0"
            )
        local512_manifest = str(
            config["data"].get("local512_truth_probability_manifest", "")
        ).strip()
        if not truth_probability_manifest or not local512_manifest:
            raise ValueError(
                "FusionAware A1 requires both MS75 and Local512 manifests"
            )
        for prefix in ("ms75", "local512"):
            manifest_digest = str(
                config["data"].get(
                    f"{prefix}_truth_probability_manifest_sha256", ""
                )
            )
            if not re.fullmatch(r"[0-9a-f]{64}", manifest_digest):
                raise ValueError(
                    f"data.{prefix}_truth_probability_manifest_sha256 must be "
                    "lowercase SHA256"
                )
            for suffix in (
                "expected_checkpoint_sha256",
                "expected_config_sha256",
            ):
                value = str(
                    config["data"].get(
                        f"{prefix}_truth_probability_{suffix}", ""
                    )
                )
                if not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError(
                        f"data.{prefix}_truth_probability_{suffix} must be "
                        "lowercase SHA256"
                    )
            if config["data"].get(
                f"{prefix}_truth_probability_verify_file_sha256"
            ) is not True:
                raise ValueError(
                    f"FusionAware A1 requires full {prefix} map SHA256 verification"
                )
            if int(
                config["data"].get(
                    f"{prefix}_truth_probability_cache_scenes", 0
                )
            ) != 2:
                raise ValueError(
                    f"FusionAware A1 requires {prefix} probability cache_scenes=2"
                )

        histogram = _load_fusionaware_a1_class_histogram(config["data"])
        config["data"]["fusionaware_a1_class_histogram_identity"] = {
            key: value
            for key, value in histogram.items()
            if key != "class_weights"
        }
        config["loss"]["fusionaware_a1_class_weights"] = list(
            histogram["class_weights"]
        )

        initialization_path = str(
            config["train"].get(
                "fusionaware_a1_initialization_checkpoint", ""
            )
        ).strip()
        initialization_sha256 = str(
            config["train"].get(
                "fusionaware_a1_initialization_checkpoint_sha256", ""
            )
        )
        if not initialization_path:
            raise ValueError(
                "FusionAware A1 requires its A0 EMA initialization checkpoint"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", initialization_sha256):
            raise ValueError(
                "train.fusionaware_a1_initialization_checkpoint_sha256 must be "
                "lowercase SHA256"
            )

        a1_contract: dict[str, tuple[object, object]] = {
            "experiment.name": (
                config["experiment"].get("name"),
                "tcmmsline3_local256_fusionaware_a1_full120_10k",
            ),
            "data.dataset_variant": (dataset_variant, "legacy"),
            "data.official_120_30_protocol": (
                config["data"].get("official_120_30_protocol"), True
            ),
            "data.train_split": (config["data"].get("train_split"), "train"),
            "data.val_split": (config["data"].get("val_split"), "train"),
            "data.test_split": (config["data"].get("test_split"), "test"),
            "data.validation_holdout_fraction": (
                config["data"].get("validation_holdout_fraction"), 0.0
            ),
            "data.patch_size": (config["data"].get("patch_size"), 256),
            "data.sar_size": (config["data"].get("sar_size"), 64),
            "data.context_size": (config["data"].get("context_size"), 256),
            "data.context_input_size": (
                config["data"].get("context_input_size"), 256
            ),
            "data.samples_per_epoch": (
                config["data"].get("samples_per_epoch"), 38_400
            ),
            "data.expected_hw": (config["data"].get("expected_hw"), [6908, 7300]),
            "data.load_sar": (config["data"].get("load_sar"), False),
            "data.num_workers": (config["data"].get("num_workers"), 8),
            "data.spectral_augmentation.enabled": (
                config["data"].get("spectral_augmentation", {}).get("enabled"),
                False,
            ),
            "data.spectral_augmentation.probability": (
                config["data"].get("spectral_augmentation", {}).get("probability"),
                0.0,
            ),
            "data.ms75_truth_probability_expected_provenance": (
                config["data"].get(
                    "ms75_truth_probability_expected_provenance"
                ),
                "in_sample_full120_ms75_no_tta",
            ),
            "data.local512_truth_probability_expected_provenance": (
                config["data"].get(
                    "local512_truth_probability_expected_provenance"
                ),
                "in_sample_full120_local512e30_no_tta",
            ),
            "data.ms75_truth_probability_expected_checkpoint_sha256": (
                config["data"].get(
                    "ms75_truth_probability_expected_checkpoint_sha256"
                ),
                "dc52e3c98a9624cb5bfd9c40ef58402ef7e08cff0556a68e5641d782f77a6bf5",
            ),
            "data.ms75_truth_probability_expected_config_sha256": (
                config["data"].get(
                    "ms75_truth_probability_expected_config_sha256"
                ),
                "6e419b8374e711eee92c9d12e422de45f7ea36b9ba367f4e8dae5e6f2ad22228",
            ),
            "data.local512_truth_probability_expected_checkpoint_sha256": (
                config["data"].get(
                    "local512_truth_probability_expected_checkpoint_sha256"
                ),
                "d34024ac0ad419c6b2f08e1dee0c28030fec6aeb8050e31bc6d3becf05b331eb",
            ),
            "data.local512_truth_probability_expected_config_sha256": (
                config["data"].get(
                    "local512_truth_probability_expected_config_sha256"
                ),
                "f967b7b644222b5788fa6e87252351b5e96a5b6487cb3cc304cbe63f18fba4c0",
            ),
            "model.architecture": (architecture, "local_detail_cnxb"),
            "model.pretrained": (config["model"].get("pretrained"), False),
            "loss.class_weights": (config["loss"].get("class_weights"), None),
            "loss.fusion_mixture_nll_weight": (fusion_mixture_weight, 0.0),
            "loss.fusionaware_a1_dual_fusion_weight": (
                config["loss"].get("fusionaware_a1_dual_fusion_weight"), 1.0
            ),
            "loss.fusionaware_a1_repair_weight": (
                config["loss"].get("fusionaware_a1_repair_weight"), 0.20
            ),
            "loss.fusionaware_a1_class_balanced_weight": (
                config["loss"].get("fusionaware_a1_class_balanced_weight"), 0.10
            ),
            "loss.fusionaware_a1_noharm_weight": (
                config["loss"].get("fusionaware_a1_noharm_weight"), 0.25
            ),
            "train.epochs": (config["train"].get("epochs"), 21),
            "train.batch_size": (config["train"].get("batch_size"), 32),
            "train.gradient_accumulation_steps": (
                config["train"].get("gradient_accumulation_steps"), 2
            ),
            "train.loss_microbatch_size": (
                config["train"].get("loss_microbatch_size"), 4
            ),
            "train.learning_rate": (
                config["train"].get("learning_rate"), 3.0e-5
            ),
            "train.backbone_lr_multiplier": (
                config["train"].get("backbone_lr_multiplier"), 0.1
            ),
            "train.weight_decay": (config["train"].get("weight_decay"), 0.01),
            "train.scheduler_unit": (
                config["train"].get("scheduler_unit"), "step"
            ),
            "train.warmup_steps": (config["train"].get("warmup_steps"), 500),
            "train.max_successful_optimizer_updates": (
                config["train"].get("max_successful_optimizer_updates"), 12_000
            ),
            "train.scheduler_total_successful_optimizer_updates": (
                config["train"].get(
                    "scheduler_total_successful_optimizer_updates"
                ),
                12_000,
            ),
            "train.minimum_lr_ratio": (
                config["train"].get("minimum_lr_ratio"), 0.1
            ),
            "train.gradient_clip_norm": (
                config["train"].get("gradient_clip_norm"), 1.0
            ),
            "train.use_ema": (config["train"].get("use_ema"), True),
            "train.ema_decay": (config["train"].get("ema_decay"), 0.999),
            "train.successful_update_milestones": (
                config["train"].get("successful_update_milestones"),
                [8_000, 10_000, 12_000],
            ),
            "train.primary_successful_optimizer_update": (
                config["train"].get("primary_successful_optimizer_update"),
                10_000,
            ),
            "train.validation_enabled": (
                config["train"].get("validation_enabled"), False
            ),
            "train.selection_metric": (
                config["train"].get("selection_metric"), "last_epoch"
            ),
            "train.initialization_checkpoint": (
                config["train"].get("initialization_checkpoint"), None
            ),
            "train.continuation_initialization_checkpoint": (
                config["train"].get("continuation_initialization_checkpoint"),
                None,
            ),
            "train.auto_resume": (config["train"].get("auto_resume"), True),
            "inference.protocol_role": (
                config.get("inference", {}).get("protocol_role"),
                "local256_fusionaware_a1_full120_primary_10k",
            ),
            "inference.evaluate_test_after_training": (
                config.get("inference", {}).get(
                    "evaluate_test_after_training"
                ),
                False,
            ),
        }
        for field, (observed, required_value) in a1_contract.items():
            bool_type_mismatch = isinstance(required_value, bool) and not isinstance(
                observed, bool
            )
            if bool_type_mismatch or observed != required_value:
                raise ValueError(
                    "FusionAware A1 contract requires "
                    f"{field}={required_value!r}, observed={observed!r}"
                )
    elif str(
        config["data"].get("local512_truth_probability_manifest", "")
    ).strip():
        raise ValueError(
            "Local512 truth-probability manifest requires FusionAware A1"
        )
    if float(config["loss"].get("family_contrast_temperature", 0.08)) <= 0:
        raise ValueError("loss.family_contrast_temperature must be positive")
    if (
        int(config["loss"].get("family_contrast_max_samples_per_class", 64))
        < 2
    ):
        raise ValueError(
            "loss.family_contrast_max_samples_per_class must be at least two"
        )
    no_harm_confidence = float(
        config["loss"].get("family_region_no_harm_confidence", 0.7)
    )
    if not 0 <= no_harm_confidence <= 1:
        raise ValueError(
            "loss.family_region_no_harm_confidence must be in [0, 1]"
        )
    if (
        float(config["loss"].get("family_region_no_harm_temperature", 2.0))
        <= 0
    ):
        raise ValueError(
            "loss.family_region_no_harm_temperature must be positive"
        )
    fixed_anchor_no_harm_weight = float(
        config["loss"].get("fixed_anchor_no_harm_weight", 0.0)
    )
    if not math.isfinite(fixed_anchor_no_harm_weight):
        raise ValueError("loss.fixed_anchor_no_harm_weight must be finite")
    fixed_anchor_confidence = float(
        config["loss"].get("fixed_anchor_no_harm_confidence", 0.7)
    )
    if not math.isfinite(fixed_anchor_confidence) or not (
        0 <= fixed_anchor_confidence <= 1
    ):
        raise ValueError(
            "loss.fixed_anchor_no_harm_confidence must be in [0, 1]"
        )
    fixed_anchor_temperature = float(
        config["loss"].get("fixed_anchor_no_harm_temperature", 2.0)
    )
    if not math.isfinite(fixed_anchor_temperature) or (
        fixed_anchor_temperature <= 0
    ):
        raise ValueError(
            "loss.fixed_anchor_no_harm_temperature must be positive"
        )
    if fixed_teacher_enabled != (fixed_anchor_no_harm_weight > 0.0):
        raise ValueError(
            "fixed_anchor_teacher_enabled and a positive "
            "fixed_anchor_no_harm_weight must be enabled together"
        )
    if fixed_teacher_enabled:
        if architecture != "ms75_multiscale":
            raise ValueError(
                "the fixed anchor teacher is only valid for ms75_multiscale"
            )
        if str(config["data"].get("protocol_stage", "final")) != "final":
            raise ValueError(
                "the fixed anchor teacher is restricted to final MS75 fitting"
            )
        if (
            not ms75_anchor
            or not anchor_source_hash
            or not anchor_destination_hash
        ):
            raise ValueError(
                "the fixed anchor teacher requires the authenticated MS75 anchor"
            )
    if (
        architecture == "ms75_multiscale"
        and str(config["data"].get("protocol_stage", "final"))
        == "ms75_clean_fold0_train96"
    ):
        clean_split = "ms75_oof_seed1337_fold_0_train"
        if any(
            str(config["data"].get(key, "")) != clean_split
            for key in ("train_split", "val_split", "test_split")
        ):
            raise ValueError(
                "clean MS75 fold0 requires every loader alias to use the "
                "precommitted train96 split"
            )
        forbidden_clean_initializers = {
            "train.initialization_checkpoint": initialization,
            "train.ms_initialization_checkpoint": ms_initialization,
            "train.sar_initialization_checkpoint": sar_initialization,
            "train.teacher_checkpoint": config["train"].get(
                "teacher_checkpoint"
            ),
            "train.continuation_initialization_checkpoint": (
                continuation_initialization
            ),
            "train.ms75_anchor_checkpoint": ms75_anchor,
        }
        enabled = [
            key for key, value in forbidden_clean_initializers.items() if value
        ]
        if enabled:
            raise ValueError(
                "clean MS75 fold0 forbids FBP/checkpoint initializers: "
                f"{enabled}"
            )
        if fixed_teacher_enabled or fixed_anchor_no_harm_weight != 0.0:
            raise ValueError(
                "clean MS75 fold0 forbids the full120 anchor teacher"
            )
        clean_optimizer_contract = {
            "epochs": 26,
            "scheduler_unit": "step",
            "max_successful_optimizer_updates": 192_000,
            "amp_initial_scale": 2048.0,
            "amp_growth_interval": 1_000_000,
            "amp_growth_factor": 2.0,
            "amp_backoff_factor": 0.5,
            "ema_milestone_epochs": [],
            "checkpoint_every": 100,
        }
        changed_optimizer_contract = {
            key: {"expected": expected, "observed": config["train"].get(key)}
            for key, expected in clean_optimizer_contract.items()
            if config["train"].get(key) != expected
        }
        if changed_optimizer_contract:
            raise ValueError(
                "clean MS75 fold0 optimizer/AMP contract changed: "
                f"{changed_optimizer_contract}"
            )
        if validation_enabled or selection_metric != "last_epoch":
            raise ValueError(
                "clean MS75 fold0 requires validation_enabled=false and "
                "selection_metric=last_epoch"
            )
        if bool(
            config.get("inference", {}).get(
                "evaluate_test_after_training", False
            )
        ):
            raise ValueError(
                "clean MS75 fold0 forbids automatic held/test evaluation"
            )
        if str(config.get("inference", {}).get("protocol_role", "")) != (
            "ms75_clean_fold0_train96"
        ):
            raise ValueError(
                "clean MS75 fold0 requires its dedicated inference role"
            )
    fine_to_family = {
        0: 0,
        11: 0,
        17: 0,
        18: 0,
        19: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 2,
        6: 2,
        7: 2,
        8: 3,
        9: 3,
        10: 4,
        12: 4,
        13: 4,
        14: 4,
        15: 5,
        16: 5,
        20: 6,
        21: 6,
        22: 6,
        23: 6,
    }
    for group in config["loss"].get("family_region_confusion_groups", []):
        class_ids = [int(value) for value in group]
        if len(class_ids) < 2 or len(set(class_ids)) != len(class_ids):
            raise ValueError(
                "loss.family_region_confusion_groups entries need at least "
                "two unique classes"
            )
        if any(value not in fine_to_family for value in class_ids):
            raise ValueError(
                "loss.family_region_confusion_groups must use IDs in [0, 23]"
            )
        if len({fine_to_family[value] for value in class_ids}) != 1:
            raise ValueError(
                "loss.family_region_confusion_groups cannot cross families"
            )
    if int(config["loss"].get("region_min_pixels", 1)) < 1:
        raise ValueError("loss.region_min_pixels must be positive")
    if float(config["loss"].get("region_temperature", 1.0)) <= 0:
        raise ValueError("loss.region_temperature must be positive")
    for group in config["loss"].get("confusion_groups", []):
        class_ids = [int(value) for value in group]
        if len(class_ids) < 2 or len(set(class_ids)) != len(class_ids):
            raise ValueError(
                "loss.confusion_groups entries need at least two unique classes"
            )
        if any(value < 0 or value >= 24 for value in class_ids):
            raise ValueError(
                "loss.confusion_groups must use zero-based IDs in [0, 23]"
            )

    inference = config.get("inference", {})
    if int(inference.get("batch_size", 0)) <= 0:
        raise ValueError("inference.batch_size must be positive")
    tile_stride = int(inference.get("tile_stride", patch_size))
    if tile_stride <= 0 or tile_stride > patch_size:
        raise ValueError(
            "inference.tile_stride must be in (0, data.patch_size]"
        )
    tta_transforms = [
        str(value) for value in inference.get("tta_transforms", ["none"])
    ]
    supported_tta = {"none", "horizontal", "vertical", "both"}
    if (
        not tta_transforms
        or len(tta_transforms) != len(set(tta_transforms))
        or any(value not in supported_tta for value in tta_transforms)
    ):
        raise ValueError(
            "inference.tta_transforms must contain unique supported transforms"
        )
    if str(inference.get("blend_window", "uniform")) not in {
        "uniform",
        "hann",
    }:
        raise ValueError("inference.blend_window must be uniform or hann")
    if "hann_floor" in inference:
        hann_floor = inference["hann_floor"]
        if isinstance(hann_floor, bool):
            raise ValueError(
                "inference.hann_floor must be a finite number in (0, 1]"
            )
        hann_floor = float(hann_floor)
        if not math.isfinite(hann_floor) or not 0.0 < hann_floor <= 1.0:
            raise ValueError(
                "inference.hann_floor must be a finite number in (0, 1]"
            )
    if "amp" in inference and not isinstance(inference["amp"], bool):
        raise ValueError("inference.amp must be boolean")
    if "routing_mode" in inference and str(inference["routing_mode"]) not in {
        "learned",
    }:
        raise ValueError("inference.routing_mode must be learned")
    if (
        tile_stride != patch_size or tta_transforms != ["none"]
    ) and inference.get("max_windows_per_scene") is not None:
        raise ValueError(
            "overlap/TTA inference requires max_windows_per_scene=null"
        )
    if (
        architecture == "ms75_multiscale"
        and str(config["data"].get("protocol_stage", "final")) == "final"
    ):
        protocol_role = str(inference.get("protocol_role", ""))
        if protocol_role == "complete_anchor_formal":
            # Local import keeps the general configuration module lightweight
            # while making the formal 70.1564 protocol fail closed even when
            # callers bypass tools/preflight.py.
            from .ms75_protocol import validate_ms75_formal_protocol

            validate_ms75_formal_protocol(config)
        elif protocol_role == "derived_flip4_tta":
            if (
                int(inference.get("tile_stride", 0)) != 256
                or tta_transforms != ["none", "horizontal", "vertical", "both"]
                or str(inference.get("blend_window")) != "hann"
                or float(inference.get("hann_floor", -1.0)) != 0.05
                or inference.get("amp") is not True
                or str(inference.get("routing_mode")) != "learned"
            ):
                raise ValueError(
                    "derived MS75 flip4 TTA must retain stride=256, Hann "
                    "floor=0.05, CUDA AMP and learned routing"
                )
        elif protocol_role == "paired_continuation_train":
            # This is a fixed-budget training protocol, not a relaxation of
            # the immutable 71-model publication role.  Keep its checks in a
            # separate validator so the old formal role remains byte-for-byte
            # reproducible.
            from .ms75_protocol import (
                validate_ms75_paired_continuation_protocol,
            )

            validate_ms75_paired_continuation_protocol(config)
        elif protocol_role in {
            "paired_continuation_no_tta_evaluation",
            "paired_continuation_flip4_evaluation",
        }:
            from .ms75_protocol import (
                validate_ms75_paired_evaluation_protocol,
            )

            validate_ms75_paired_evaluation_protocol(config)
        else:
            raise ValueError(
                "final MS75 requires inference.protocol_role to be "
                "complete_anchor_formal, derived_flip4_tta, or "
                "a paired-continuation train/evaluation role"
            )

    analysis = config.get("analysis", {})
    boundary_tolerance = int(
        analysis.get("boundary_f1_tolerance_ms_pixels", 3)
    )
    if boundary_tolerance < 0:
        raise ValueError(
            "analysis.boundary_f1_tolerance_ms_pixels must be non-negative"
        )
    component_scale = analysis.get("component_scale", {})
    if component_scale.get("enabled", False):
        class_ids = [
            int(value) for value in component_scale.get("class_ids", [])
        ]
        if not class_ids or any(
            value < 0 or value >= 24 for value in class_ids
        ):
            raise ValueError(
                "analysis.component_scale.class_ids must contain zero-based "
                "class IDs in [0, 23]"
            )
        thresholds = [
            float(value)
            for value in component_scale.get("area_thresholds_m2", [])
        ]
        if (
            len(thresholds) != 2
            or thresholds[0] <= 0
            or thresholds[1] <= thresholds[0]
        ):
            raise ValueError(
                "analysis.component_scale.area_thresholds_m2 must contain "
                "two increasing positive values"
            )

    sampling = config["data"].get("sampling", {})
    strategy = str(sampling.get("strategy", "uniform"))
    if strategy not in {"uniform", "directed", "neutral_scene_class_uniform"}:
        raise ValueError(
            "data.sampling.strategy must be uniform, directed, or "
            "neutral_scene_class_uniform"
        )
    if strategy == "directed":
        rare_fraction = float(sampling.get("rare_fraction", 0.0))
        hard_fraction = float(sampling.get("hard_fraction", 0.0))
        if rare_fraction < 0 or hard_fraction < 0:
            raise ValueError(
                "directed sampling fractions must be non-negative"
            )
        if rare_fraction + hard_fraction >= 1:
            raise ValueError(
                "directed sampling must retain a positive uniform fraction"
            )
        for key in ("rare_class_ids", "hard_class_ids"):
            class_ids = [int(value) for value in sampling.get(key, [])]
            if not class_ids:
                raise ValueError(f"directed sampling requires {key}")
            if any(value < 0 or value >= 24 for value in class_ids):
                raise ValueError(
                    f"{key} must contain zero-based class IDs in [0, 23]"
                )
    elif strategy == "neutral_scene_class_uniform":
        from .ms75_neutral_oof import (
            validate_neutral_proxy_config,
            validate_neutral_sampling_config,
        )

        validate_neutral_sampling_config(sampling)
        if not str(config["data"].get("sampling_index", "")).strip():
            raise ValueError(
                "neutral_scene_class_uniform requires a fold-train sampling_index"
            )
        if str(config["data"].get("protocol_stage", "")) == (
            "oof_neutral_proxy"
        ):
            validate_neutral_proxy_config(config)

    if dataset_format == "ms_sar_manifest":
        splits = [
            set(map(str, config["data"].get(f"{name}_scene_ids", [])))
            for name in ("train", "val", "test")
        ]
        if any(
            left & right
            for i, left in enumerate(splits)
            for right in splits[i + 1 :]
        ):
            raise ValueError("train/val/test scene lists must be disjoint")


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config = _expand_env(_load_yaml(Path(path), ()))
    validate_config(config)
    return config
