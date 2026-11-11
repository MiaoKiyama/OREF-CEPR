from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import re
from collections.abc import Mapping
from typing import Any


ALIGNED_OOF_ARCHITECTURE = "ms75_oof_aligned_swin"
ALIGNED_OOF_ROLE = "ms75_clean_aligned_swin_oof"
ALIGNED_OOF_CHECKPOINT_ROLE = f"{ALIGNED_OOF_ROLE}_training"
ALIGNED_OOF_ENCODER = (
    "swin_large_patch4_window12_384.ms_in22k_ft_in1k"
)
ALIGNED_OOF_PATCH_SIZE = 512
ALIGNED_OOF_SUCCESSFUL_UPDATES = 20_000
ALIGNED_OOF_CLASS_IDS = list(range(24))
ALIGNED_OOF_SAMPLER_POLICY = "scene_uniform_0.5_class_uniform_0.5"
ALIGNED_OOF_ENCODER_INITIALIZATION = "timm_official_imagenet"
ALIGNED_OOF_CONFIG_PATH = "configs/line3_ms75_oof_aligned_swin.yaml"
ALIGNED_OOF_IMPLEMENTATION_PATHS = (
    ALIGNED_OOF_CONFIG_PATH,
    "configs/base.yaml",
    "evaluate.py",
    "train.py",
    "tcmmsline3/config.py",
    "tcmmsline3/data.py",
    "tcmmsline3/model.py",
    "tcmmsline3/ms75_aligned_protocol.py",
    "tcmmsline3/ms75_aligned_proxy.py",
    "tcmmsline3/ms75_data.py",
    "tcmmsline3/ms75_model.py",
    "tcmmsline3/ms75_oof_source_protocol.py",
    "tcmmsline3/runtime.py",
    "scripts/run_line3_ms75_oof_aligned_swin.sh",
    "scripts/run_line3_ms75_clean_paired_training.sh",
    "scripts/run_line3_resilient.sh",
    "tools/finalize_ms75_oof_provenance.py",
    "tools/finalize_ms75_aligned_oof_completion.py",
    "tools/build_ms75_oof_confusion_index.py",
    "tools/predict_ms75_oof_fold.py",
    "tools/prepare_ms75_oof_folds.py",
    "tools/write_ms75_aligned_source_protocol.py",
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def source_protocol_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("source_protocol_sha256", None)
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def current_software_dependencies() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in ("torch", "timm", "torchvision"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"aligned OOF requires installed dependency {distribution!r}"
            ) from error
    return versions


def build_aligned_source_protocol(
    *,
    config_sha256: str,
    source_revision: str,
    implementation_sha256: Mapping[str, str],
    encoder_pretrained_cfg: Mapping[str, Any],
    generic_initial_tensor_sha256: str,
    software_dependencies: Mapping[str, str],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": 1,
        "role": ALIGNED_OOF_ROLE,
        "source_split": "official_train",
        "official_test_opened": False,
        "full120_checkpoint_opened": False,
        "manual_class_ids": False,
        "class_ids": list(ALIGNED_OOF_CLASS_IDS),
        "sampler_policy": ALIGNED_OOF_SAMPLER_POLICY,
        "initialization_scope": "generic_pretraining_only",
        "config_path": ALIGNED_OOF_CONFIG_PATH,
        "config_sha256": str(config_sha256).lower(),
        "source_revision": str(source_revision).lower(),
        "implementation_sha256": {
            str(path): str(digest).lower()
            for path, digest in sorted(implementation_sha256.items())
        },
        "patch_size": ALIGNED_OOF_PATCH_SIZE,
        "encoder": ALIGNED_OOF_ENCODER,
        "encoder_initialization": ALIGNED_OOF_ENCODER_INITIALIZATION,
        "encoder_pretrained_cfg": copy.deepcopy(dict(encoder_pretrained_cfg)),
        "generic_initial_tensor_sha256": str(
            generic_initial_tensor_sha256
        ).lower(),
        "software_dependencies": {
            str(name): str(version)
            for name, version in sorted(software_dependencies.items())
        },
        "encoder_frozen": True,
        "successful_updates_per_fold": ALIGNED_OOF_SUCCESSFUL_UPDATES,
    }
    protocol = {**body, "source_protocol_sha256": source_protocol_sha256(body)}
    validate_aligned_source_protocol(protocol)
    return protocol


def validate_aligned_source_protocol(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("aligned OOF source_protocol must be a mapping")
    expected_keys = {
        "schema_version",
        "role",
        "source_split",
        "official_test_opened",
        "full120_checkpoint_opened",
        "manual_class_ids",
        "class_ids",
        "sampler_policy",
        "initialization_scope",
        "config_path",
        "config_sha256",
        "source_revision",
        "implementation_sha256",
        "patch_size",
        "encoder",
        "encoder_initialization",
        "encoder_pretrained_cfg",
        "generic_initial_tensor_sha256",
        "software_dependencies",
        "encoder_frozen",
        "successful_updates_per_fold",
        "source_protocol_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("aligned OOF source_protocol has unexpected fields")
    expected = {
        "schema_version": 1,
        "role": ALIGNED_OOF_ROLE,
        "source_split": "official_train",
        "official_test_opened": False,
        "full120_checkpoint_opened": False,
        "manual_class_ids": False,
        "class_ids": list(ALIGNED_OOF_CLASS_IDS),
        "sampler_policy": ALIGNED_OOF_SAMPLER_POLICY,
        "initialization_scope": "generic_pretraining_only",
        "config_path": ALIGNED_OOF_CONFIG_PATH,
        "patch_size": ALIGNED_OOF_PATCH_SIZE,
        "encoder": ALIGNED_OOF_ENCODER,
        "encoder_initialization": ALIGNED_OOF_ENCODER_INITIALIZATION,
        "encoder_frozen": True,
        "successful_updates_per_fold": ALIGNED_OOF_SUCCESSFUL_UPDATES,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(
                f"aligned OOF source_protocol {key} must be {expected_value!r}"
            )
    for field, length in (("config_sha256", 64), ("source_revision", 40)):
        observed = str(value.get(field, ""))
        if re.fullmatch(rf"[0-9a-f]{{{length}}}", observed) is None:
            raise ValueError(
                f"aligned OOF source_protocol {field} must be {length} lowercase hex"
            )
    generic_digest = str(value.get("generic_initial_tensor_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", generic_digest) is None:
        raise ValueError("aligned OOF generic initialization digest is invalid")
    pretrained_cfg = value.get("encoder_pretrained_cfg")
    if not isinstance(pretrained_cfg, Mapping) or not pretrained_cfg:
        raise ValueError("aligned OOF timm pretrained_cfg identity is missing")
    if "swin_large_patch4_window12_384" not in str(
        pretrained_cfg.get("architecture", "")
    ):
        raise ValueError("aligned OOF timm pretrained_cfg architecture is wrong")
    if not any(pretrained_cfg.get(key) for key in ("url", "hf_hub_id")):
        raise ValueError("aligned OOF timm pretrained_cfg has no official source")
    if pretrained_cfg.get("file"):
        raise ValueError("aligned OOF timm pretrained_cfg cannot use a local file")
    software_dependencies = value.get("software_dependencies")
    if (
        not isinstance(software_dependencies, Mapping)
        or set(software_dependencies) != {"python", "torch", "timm", "torchvision"}
        or any(
            not isinstance(version, str) or not version.strip()
            for version in software_dependencies.values()
        )
    ):
        raise ValueError("aligned OOF software dependency identity is invalid")
    implementation = value.get("implementation_sha256")
    if not isinstance(implementation, Mapping):
        raise ValueError("aligned OOF source_protocol implementation SHA map is missing")
    if set(implementation) != set(ALIGNED_OOF_IMPLEMENTATION_PATHS):
        raise ValueError(
            "aligned OOF source_protocol implementation map is not exact"
        )
    for path, digest in implementation.items():
        if (
            not isinstance(path, str)
            or not path
            or path.startswith(("/", "\\"))
            or "\\" in path
            or ".." in path.split("/")
            or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
        ):
            raise ValueError(
                "aligned OOF implementation SHA map must bind relative POSIX paths"
            )
    expected_sha = source_protocol_sha256(value)
    if value.get("source_protocol_sha256") != expected_sha:
        raise ValueError("aligned OOF source_protocol self-SHA256 mismatch")
    return copy.deepcopy(dict(value))


def validate_aligned_proxy_config(config: Mapping[str, Any]) -> None:
    data = config["data"]
    model = config["model"]
    train = config["train"]
    inference = config.get("inference", {})

    if str(model.get("architecture")) != ALIGNED_OOF_ARCHITECTURE:
        raise ValueError("aligned OOF validator received another architecture")
    if str(model.get("fusion_mode")) != "ms_only":
        raise ValueError("aligned OOF proxy requires fusion_mode=ms_only")
    if int(model.get("num_classes", -1)) != len(ALIGNED_OOF_CLASS_IDS):
        raise ValueError("aligned OOF proxy requires exactly 24 classes")
    if (
        str(data.get("dataset_format")) != "fbp_ms"
        or str(data.get("dataset_variant")) != "legacy"
        or str(data.get("protocol_stage")) != "oof_aligned_proxy"
    ):
        raise ValueError("aligned OOF proxy requires its isolated FBP protocol stage")
    if bool(data.get("official_120_30_protocol", False)):
        raise ValueError("aligned OOF training cannot expose the official test protocol")
    if str(data.get("train_split")) == str(data.get("val_split")):
        raise ValueError("aligned OOF train and heldout splits must differ")
    if str(data.get("test_split")) != str(data.get("val_split")):
        raise ValueError("aligned OOF test_split must alias heldout val_split")
    if bool(data.get("load_sar", False)):
        raise ValueError("aligned OOF proxy is optical-only")
    if int(data.get("patch_size", -1)) != ALIGNED_OOF_PATCH_SIZE:
        raise ValueError("aligned OOF proxy requires 512-pixel crops")
    sampling = data.get("sampling", {})
    if not isinstance(sampling, Mapping) or sampling.get("strategy") != (
        "neutral_scene_class_uniform"
    ):
        raise ValueError("aligned OOF proxy requires the clean 50/50 sampler")
    if (
        float(sampling.get("scene_fraction", -1.0)) != 0.5
        or float(sampling.get("class_fraction", -1.0)) != 0.5
        or int(sampling.get("center_jitter", -1)) != 0
    ):
        raise ValueError(
            "aligned OOF proxy requires 50% scene + 50% all-24 class-uniform crops"
        )
    forbidden_sampling = {
        "class_ids",
        "rare_class_ids",
        "hard_class_ids",
        "target_class_ids",
    }
    if forbidden_sampling & set(sampling):
        raise ValueError("aligned OOF proxy forbids manually selected class IDs")
    if not str(data.get("sampling_index", "")).strip():
        raise ValueError("aligned OOF proxy requires its fold-train-only class index")
    if data.get("fixed_tile_index"):
        raise ValueError("aligned OOF proxy forbids fixed tile indices")

    aligned = model.get("aligned_proxy")
    if not isinstance(aligned, Mapping):
        raise ValueError("model.aligned_proxy must be a mapping")
    encoder = aligned.get("encoder")
    if not isinstance(encoder, Mapping):
        raise ValueError("aligned OOF proxy requires an encoder mapping")
    expected_encoder = {
        "type": "timm",
        "name": ALIGNED_OOF_ENCODER,
        "in_channels": 4,
        "band_order": ["B", "G", "R", "NIR"],
        "pretrained": True,
        "prefer_official_url": True,
        "img_size": ALIGNED_OOF_PATCH_SIZE,
        "out_indices": [0, 1, 2, 3],
        "gradient_checkpointing": False,
    }
    for key, expected_value in expected_encoder.items():
        if encoder.get(key) != expected_value:
            raise ValueError(
                f"aligned OOF encoder {key} must be {expected_value!r}"
            )
    if encoder.get("pretrained_file"):
        raise ValueError("aligned OOF proxy forbids a local pretrained checkpoint")
    if int(aligned.get("decoder_channels", 0)) <= 0:
        raise ValueError("aligned OOF decoder_channels must be positive")

    forbidden_checkpoints = (
        "initialization_checkpoint",
        "ms_initialization_checkpoint",
        "sar_initialization_checkpoint",
        "continuation_initialization_checkpoint",
        "ms75_anchor_checkpoint",
        "teacher_checkpoint",
    )
    if any(train.get(key) for key in forbidden_checkpoints):
        raise ValueError(
            "aligned OOF proxy forbids FBP/full120/teacher checkpoint initialization"
        )
    if bool(train.get("fixed_anchor_teacher_enabled", False)):
        raise ValueError("aligned OOF proxy forbids a fixed anchor teacher")
    from .ms75_oof_source_protocol import validate_oof_source_binding_config

    validate_oof_source_binding_config(train)
    if list(train.get("trainable_modules", [])) != ["decoder", "classifier"]:
        raise ValueError(
            "aligned OOF proxy must freeze its encoder and train decoder/classifier only"
        )
    if train.get("freeze_modules") or train.get("staged_unfreeze_modules"):
        raise ValueError("aligned OOF proxy forbids staged encoder unfreezing")
    if bool(train.get("amp", False)):
        raise ValueError(
            "aligned OOF exact-update accounting requires AMP disabled (no skipped steps)"
        )
    batch_size = int(train.get("batch_size", 0))
    samples = int(data.get("samples_per_epoch", 0))
    accumulation = int(train.get("gradient_accumulation_steps", 1))
    epochs = int(train.get("epochs", 0))
    if batch_size <= 0 or samples <= 0 or samples % batch_size:
        raise ValueError("aligned OOF samples_per_epoch must divide by batch_size")
    updates = epochs * math.ceil((samples // batch_size) / accumulation)
    if updates != ALIGNED_OOF_SUCCESSFUL_UPDATES:
        raise ValueError(
            "aligned OOF proxy must schedule exactly 20,000 optimizer updates"
        )
    if str(train.get("selection_metric")) != "last_epoch":
        raise ValueError("aligned OOF proxy must publish the final epoch checkpoint")
    if bool(train.get("validation_enabled", True)):
        raise ValueError("aligned OOF training must not optimize/select on heldout scenes")
    if bool(train.get("use_ema", False)):
        raise ValueError("aligned OOF proxy uses the raw final decoder, not EMA")
    if str(inference.get("protocol_role")) != ALIGNED_OOF_ROLE:
        raise ValueError("aligned OOF inference protocol role is missing")
    if bool(inference.get("evaluate_test_after_training", True)):
        raise ValueError("aligned OOF training cannot invoke generic test evaluation")
    if list(inference.get("tta_transforms", [])) != ["none"]:
        raise ValueError("aligned OOF predictions require no TTA")
