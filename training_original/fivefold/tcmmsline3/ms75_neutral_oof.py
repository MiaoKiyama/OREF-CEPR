"""Deterministic, class-neutral sampling for clean MS75 OOF miners."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import random
import re
import sys
from typing import Any

import torch


NUM_CLASSES = 24
NEUTRAL_SAMPLER_STRATEGY = "neutral_scene_class_uniform"
NEUTRAL_SAMPLER_POLICY = "scene_uniform_0.5_class_uniform_0.5"
NEUTRAL_CLASS_IDS = tuple(range(NUM_CLASSES))
NEUTRAL_OOF_ROLE = "ms75_clean_neutral_oof"
NEUTRAL_OOF_CHECKPOINT_ROLE = "ms75_clean_neutral_oof_fold"
NEUTRAL_OOF_CONFIG_PATH = "configs/line3_ms75_oof_neutral.yaml"
NEUTRAL_OOF_SUCCESSFUL_UPDATES = 1_920
NEUTRAL_OOF_IMPLEMENTATION_PATHS = (
    NEUTRAL_OOF_CONFIG_PATH,
    "configs/base.yaml",
    "evaluate.py",
    "train.py",
    "tcmmsline3/config.py",
    "tcmmsline3/data.py",
    "tcmmsline3/model.py",
    "tcmmsline3/ms75_neutral_oof.py",
    "tcmmsline3/ms75_data.py",
    "tcmmsline3/ms75_oof_source_protocol.py",
    "tcmmsline3/runtime.py",
    "scripts/run_line3_ms75_oof_clean_neutral.sh",
    "scripts/run_line3_resilient.sh",
    "tools/finalize_ms75_oof_provenance.py",
    "tools/finalize_ms75_neutral_oof_completion.py",
    "tools/build_ms75_oof_confusion_index.py",
    "tools/predict_ms75_oof_fold.py",
    "tools/prepare_ms75_oof_folds.py",
    "tools/write_ms75_neutral_source_protocol.py",
)


def current_neutral_software_dependencies() -> dict[str, str]:
    """Return the dependency identity content-addressed by the protocol."""

    import numpy
    import torchvision

    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": str(numpy.__version__),
    }


def build_neutral_execution_provenance(
    device: torch.device,
) -> dict[str, Any]:
    """Capture the software and accelerator identity used for one fold."""

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "software_dependencies": current_neutral_software_dependencies(),
        "torch_cuda_version": (
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": device.type,
        "gpu_name": None,
        "gpu_compute_capability": None,
        "gpu_total_memory_bytes": None,
        "gpu_uuid": None,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_uuid = getattr(properties, "uuid", None)
        if gpu_uuid is None:
            raw_uuid_loader = getattr(torch.cuda, "_raw_device_uuid_nvml", None)
            nvml_index_loader = getattr(torch.cuda, "_get_nvml_device_index", None)
            if callable(raw_uuid_loader) and callable(nvml_index_loader):
                uuids = raw_uuid_loader()
                if uuids is not None:
                    gpu_uuid = uuids[int(nvml_index_loader(device))]
        if gpu_uuid is None:
            raise RuntimeError("neutral OOF cannot authenticate the CUDA GPU UUID")
        provenance.update(
            {
                "gpu_name": str(properties.name),
                "gpu_compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_uuid": str(gpu_uuid),
            }
        )
    return provenance


def validate_neutral_execution_provenance(
    value: Mapping[str, Any],
    *,
    expected_software_dependencies: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "software_dependencies",
        "torch_cuda_version",
        "cudnn_version",
        "device_type",
        "gpu_name",
        "gpu_compute_capability",
        "gpu_total_memory_bytes",
        "gpu_uuid",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("neutral OOF execution provenance is incomplete")
    software = value.get("software_dependencies")
    if (
        not isinstance(software, Mapping)
        or set(software) != {"python", "torch", "torchvision", "numpy"}
        or any(
            not isinstance(version, str) or not version
            for version in software.values()
        )
    ):
        raise ValueError("neutral OOF execution dependency identity is invalid")
    if expected_software_dependencies is not None and dict(software) != dict(
        expected_software_dependencies
    ):
        raise ValueError("neutral OOF execution dependencies differ from protocol")
    device_type = value.get("device_type")
    if device_type not in {"cpu", "cuda"}:
        raise ValueError("neutral OOF execution device type is invalid")
    if device_type == "cuda":
        capability = value.get("gpu_compute_capability")
        if (
            not isinstance(value.get("torch_cuda_version"), str)
            or not value["torch_cuda_version"]
            or not isinstance(value.get("cudnn_version"), int)
            or not isinstance(value.get("gpu_name"), str)
            or not value["gpu_name"]
            or not isinstance(capability, list)
            or len(capability) != 2
            or not all(isinstance(item, int) and item >= 0 for item in capability)
            or not isinstance(value.get("gpu_total_memory_bytes"), int)
            or int(value["gpu_total_memory_bytes"]) <= 0
            or not isinstance(value.get("gpu_uuid"), str)
            or not value["gpu_uuid"]
        ):
            raise ValueError("neutral OOF CUDA execution identity is invalid")
    elif any(
        value.get(key) is not None
        for key in (
            "gpu_name",
            "gpu_compute_capability",
            "gpu_total_memory_bytes",
            "gpu_uuid",
        )
    ):
        raise ValueError("neutral OOF CPU execution cannot claim a GPU identity")
    return copy.deepcopy(dict(value))


def validate_neutral_sampling_config(value: Mapping[str, Any]) -> None:
    """Validate the exact clean-OOF sampler contract.

    Manual class subsets are forbidden even when their configured fractions
    would be zero: retaining such fields would make the source ambiguous.
    """

    if value.get("strategy") != NEUTRAL_SAMPLER_STRATEGY:
        raise ValueError("clean OOF requires neutral_scene_class_uniform")
    forbidden = {
        "rare_fraction",
        "hard_fraction",
        "rare_class_ids",
        "hard_class_ids",
    }
    leaked = forbidden & set(value)
    if leaked:
        raise ValueError(
            "clean OOF neutral sampling forbids manual class fields: "
            + ", ".join(sorted(leaked))
        )
    if float(value.get("scene_fraction", -1.0)) != 0.5:
        raise ValueError("clean OOF scene_fraction must equal 0.5")
    if float(value.get("class_fraction", -1.0)) != 0.5:
        raise ValueError("clean OOF class_fraction must equal 0.5")
    if int(value.get("center_jitter", 0)) < 0:
        raise ValueError("clean OOF center_jitter must be non-negative")


def validate_neutral_proxy_config(config: Mapping[str, Any]) -> None:
    data = config.get("data", {})
    model = config.get("model", {})
    train = config.get("train", {})
    inference = config.get("inference", {})
    if (
        data.get("dataset_format") != "fbp_ms"
        or data.get("dataset_variant") != "legacy"
        or data.get("protocol_stage") != "oof_neutral_proxy"
        or data.get("official_120_30_protocol") is not False
        or data.get("test_split") != data.get("val_split")
        or data.get("train_split") == data.get("val_split")
    ):
        raise ValueError("neutral OOF requires its isolated heldout-alias protocol")
    validate_neutral_sampling_config(data.get("sampling", {}))
    if not str(data.get("sampling_index", "")).strip():
        raise ValueError("neutral OOF requires its fold-train point index")
    if (
        model.get("architecture") != "strong_ms_context"
        or model.get("fusion_mode") != "ms_only"
        or model.get("pretrained") is not True
        or int(model.get("num_classes", -1)) != NUM_CLASSES
    ):
        raise ValueError("neutral OOF model must be pretrained pure-MS ConvNeXt")
    forbidden = (
        "initialization_checkpoint",
        "ms_initialization_checkpoint",
        "sar_initialization_checkpoint",
        "teacher_checkpoint",
        "continuation_initialization_checkpoint",
        "ms75_anchor_checkpoint",
    )
    if any(train.get(key) for key in forbidden) or bool(
        train.get("fixed_anchor_teacher_enabled", False)
    ):
        raise ValueError("neutral OOF forbids every FBP/full120/teacher initializer")
    if (
        train.get("amp") is not False
        or train.get("use_ema") is not False
        or train.get("validation_enabled") is not False
        or train.get("selection_metric") != "last_epoch"
    ):
        raise ValueError(
            "neutral OOF requires raw non-EMA exact-budget last-epoch training"
        )
    batch = int(train.get("batch_size", 0))
    samples = int(data.get("samples_per_epoch", 0))
    accumulation = int(train.get("gradient_accumulation_steps", 1))
    epochs = int(train.get("epochs", 0))
    if batch <= 0 or samples <= 0 or samples % batch:
        raise ValueError("neutral OOF samples must divide exactly by batch size")
    updates = epochs * (samples // batch) // accumulation
    if updates != NEUTRAL_OOF_SUCCESSFUL_UPDATES:
        raise ValueError(
            f"neutral OOF must schedule {NEUTRAL_OOF_SUCCESSFUL_UPDATES} updates"
        )
    if list(inference.get("tta_transforms", [])) != ["none"] or bool(
        inference.get("evaluate_test_after_training", True)
    ):
        raise ValueError("neutral OOF prediction must be no-TTA and train-only")


def neutral_assignment(
    index: int,
    *,
    epoch: int,
    samples_per_epoch: int,
    seed: int,
) -> tuple[str, int | None]:
    """Return the exact arm and (for class arm) globally scheduled class.

    Arms alternate, giving every prefix an error of at most one half sample.
    The class arm cycles through one seed-fixed permutation of all 24 classes,
    so its prefix error is below one sample for every class.  Epochs continue
    the global schedule instead of restarting it.
    """

    if int(index) < 0 or int(epoch) < 0 or int(samples_per_epoch) <= 0:
        raise ValueError("neutral assignment indices must be non-negative")
    global_slot = int(epoch) * int(samples_per_epoch) + int(index)
    class_on_even = bool(int(seed) & 1)
    is_class = (global_slot % 2 == 0) == class_on_even
    if not is_class:
        return "scene", None
    class_ordinal = global_slot // 2
    order = list(NEUTRAL_CLASS_IDS)
    random.Random(int(seed) + 7_919).shuffle(order)
    return "class", int(order[class_ordinal % NUM_CLASSES])


def neutral_schedule_audit(
    count: int,
    *,
    epoch: int,
    samples_per_epoch: int,
    seed: int,
) -> dict[str, object]:
    """Describe and verify one deterministic sampler prefix."""

    arm_counts = {"scene": 0, "class": 0}
    class_counts = [0] * NUM_CLASSES
    max_arm_prefix_error = 0.0
    max_class_prefix_error = 0.0
    for index in range(int(count)):
        arm, class_id = neutral_assignment(
            index,
            epoch=epoch,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )
        arm_counts[arm] += 1
        prefix = index + 1
        max_arm_prefix_error = max(
            max_arm_prefix_error,
            abs(arm_counts["class"] - prefix * 0.5),
        )
        if class_id is not None:
            class_counts[class_id] += 1
            class_prefix = arm_counts["class"]
            max_class_prefix_error = max(
                max_class_prefix_error,
                max(
                    abs(value - class_prefix / NUM_CLASSES)
                    for value in class_counts
                ),
            )
    return {
        "schema_version": 1,
        "sampler_policy": NEUTRAL_SAMPLER_POLICY,
        "manual_class_ids": False,
        "class_ids": list(NEUTRAL_CLASS_IDS),
        "epoch": int(epoch),
        "sample_count": int(count),
        "arm_counts": arm_counts,
        "class_counts": class_counts,
        "max_arm_prefix_error": max_arm_prefix_error,
        "max_class_prefix_error": max_class_prefix_error,
    }


def canonical_module_tensor_sha256(module: torch.nn.Module) -> str:
    """Hash module tensors independently of checkpoint serialization."""

    return canonical_tensor_mapping_sha256(module.state_dict())


def canonical_tensor_mapping_sha256(
    value: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(value.items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("neutral tensor mapping contains a non-tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_neutral_checkpoint_initialization(
    checkpoint: Mapping[str, Any],
    *,
    expected_initial_tensor_sha256: str,
) -> dict[str, Any]:
    initialization = checkpoint.get("initialization_provenance")
    model = checkpoint.get("model")
    if not isinstance(initialization, Mapping) or not isinstance(model, Mapping):
        raise ValueError("neutral OOF checkpoint lacks model/init provenance")
    selected = {
        str(name)[len("backbone.") :]: tensor
        for name, tensor in model.items()
        if str(name).startswith("backbone.")
    }
    if not selected:
        raise ValueError("neutral OOF checkpoint has no backbone tensors")
    observed_final = canonical_tensor_mapping_sha256(selected)
    expected = {
        "schema_version": 1,
        "scope": "generic_pretraining_only",
        "provider": "torchvision",
        "initialization": "convnext_tiny_imagenet1k_v1",
        "pretrained": True,
        "fbp_checkpoint_opened": False,
        "full120_checkpoint_opened": False,
        "teacher_checkpoint_opened": False,
        "backbone_initial_tensor_sha256": expected_initial_tensor_sha256,
        "backbone_checkpoint_tensor_sha256": observed_final,
    }
    if any(initialization.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("neutral OOF checkpoint initialization identity mismatch")
    return dict(initialization)


def source_protocol_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("source_protocol_sha256", None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_neutral_source_protocol(
    *,
    config_sha256: str,
    source_revision: str,
    implementation_sha256: Mapping[str, str],
    generic_initial_tensor_sha256: str,
    software_dependencies: Mapping[str, str],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": 1,
        "role": NEUTRAL_OOF_ROLE,
        "source_split": "official_train",
        "official_test_opened": False,
        "full120_checkpoint_opened": False,
        "manual_class_ids": False,
        "class_ids": list(NEUTRAL_CLASS_IDS),
        "sampler_policy": NEUTRAL_SAMPLER_POLICY,
        "initialization_scope": "generic_pretraining_only",
        "config_path": NEUTRAL_OOF_CONFIG_PATH,
        "config_sha256": str(config_sha256),
        "source_revision": str(source_revision),
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "generic_initial_tensor_sha256": str(generic_initial_tensor_sha256),
        "software_dependencies": dict(sorted(software_dependencies.items())),
    }
    result = {**body, "source_protocol_sha256": source_protocol_sha256(body)}
    return validate_neutral_source_protocol(result)


def validate_neutral_source_protocol(
    value: Mapping[str, Any],
) -> dict[str, Any]:
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
        "generic_initial_tensor_sha256",
        "software_dependencies",
        "source_protocol_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("neutral OOF source_protocol has unexpected fields")
    expected = {
        "schema_version": 1,
        "role": NEUTRAL_OOF_ROLE,
        "source_split": "official_train",
        "official_test_opened": False,
        "full120_checkpoint_opened": False,
        "manual_class_ids": False,
        "class_ids": list(NEUTRAL_CLASS_IDS),
        "sampler_policy": NEUTRAL_SAMPLER_POLICY,
        "initialization_scope": "generic_pretraining_only",
        "config_path": NEUTRAL_OOF_CONFIG_PATH,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(
                f"neutral OOF source_protocol {key} must be {expected_value!r}"
            )
    for key, length in (("config_sha256", 64), ("source_revision", 40)):
        if re.fullmatch(rf"[0-9a-f]{{{length}}}", str(value.get(key, ""))) is None:
            raise ValueError(f"neutral OOF {key} must be lowercase hex")
    if re.fullmatch(
        r"[0-9a-f]{64}", str(value.get("generic_initial_tensor_sha256", ""))
    ) is None:
        raise ValueError("neutral OOF generic initial tensor SHA256 is invalid")
    software = value.get("software_dependencies")
    if not isinstance(software, Mapping) or set(software) != {
        "python",
        "torch",
        "torchvision",
        "numpy",
    }:
        raise ValueError("neutral OOF software dependency identity is incomplete")
    if any(not isinstance(item, str) or not item for item in software.values()):
        raise ValueError("neutral OOF software dependency version is invalid")
    implementation = value.get("implementation_sha256")
    if not isinstance(implementation, Mapping):
        raise ValueError("neutral OOF implementation SHA map is missing")
    if set(implementation) != set(NEUTRAL_OOF_IMPLEMENTATION_PATHS):
        raise ValueError("neutral OOF implementation SHA map is not exact")
    for path, digest in implementation.items():
        if (
            not isinstance(path, str)
            or path.startswith(("/", "\\"))
            or "\\" in path
            or ".." in path.split("/")
            or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
        ):
            raise ValueError("neutral OOF implementation SHA entry is invalid")
    if value.get("source_protocol_sha256") != source_protocol_sha256(value):
        raise ValueError("neutral OOF source_protocol self-SHA256 mismatch")
    return copy.deepcopy(dict(value))


__all__ = [
    "NEUTRAL_CLASS_IDS",
    "NEUTRAL_OOF_CHECKPOINT_ROLE",
    "NEUTRAL_OOF_CONFIG_PATH",
    "NEUTRAL_OOF_IMPLEMENTATION_PATHS",
    "NEUTRAL_OOF_ROLE",
    "NEUTRAL_OOF_SUCCESSFUL_UPDATES",
    "NEUTRAL_SAMPLER_POLICY",
    "NEUTRAL_SAMPLER_STRATEGY",
    "canonical_module_tensor_sha256",
    "canonical_tensor_mapping_sha256",
    "build_neutral_execution_provenance",
    "build_neutral_source_protocol",
    "current_neutral_software_dependencies",
    "neutral_assignment",
    "neutral_schedule_audit",
    "validate_neutral_sampling_config",
    "validate_neutral_proxy_config",
    "validate_neutral_checkpoint_initialization",
    "validate_neutral_execution_provenance",
    "validate_neutral_source_protocol",
]
