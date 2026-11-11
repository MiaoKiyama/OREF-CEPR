from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .ms75_aligned_protocol import (
    ALIGNED_OOF_CHECKPOINT_ROLE,
    ALIGNED_OOF_ENCODER,
    ALIGNED_OOF_ENCODER_INITIALIZATION,
    ALIGNED_OOF_ROLE,
    ALIGNED_OOF_SUCCESSFUL_UPDATES,
    current_software_dependencies,
)
from .ms75_model import UPerDecoder, build_pyramid_encoder


def canonical_tensor_mapping_sha256(value: Mapping[str, torch.Tensor]) -> str:
    """Hash state keys, tensor metadata, and exact CPU bytes deterministically."""

    digest = hashlib.sha256()
    for name, source in sorted(value.items()):
        if not isinstance(source, torch.Tensor):
            raise ValueError("canonical tensor mapping contains a non-tensor")
        tensor = source.detach().cpu().contiguous()
        metadata = (
            f"{name}\0{tensor.dtype}\0{','.join(map(str, tensor.shape))}\0"
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def canonical_module_tensor_sha256(module: nn.Module) -> str:
    return canonical_tensor_mapping_sha256(module.state_dict())


def canonical_checkpoint_submodule_sha256(
    state: Mapping[str, Any], *, prefix: str
) -> str:
    selected = {
        str(name)[len(prefix) :]: tensor
        for name, tensor in state.items()
        if str(name).startswith(prefix)
    }
    if not selected:
        raise ValueError(f"checkpoint state has no {prefix!r} tensors")
    return canonical_tensor_mapping_sha256(selected)


def timm_pretrained_cfg_identity(encoder: nn.Module) -> dict[str, Any]:
    timm_model = getattr(encoder, "model", None)
    pretrained_cfg = getattr(timm_model, "pretrained_cfg", {})
    if not isinstance(pretrained_cfg, Mapping):
        pretrained_cfg = {}

    def portable(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [portable(item) for item in value]
        return str(value)

    selected_cfg_keys = (
        "architecture",
        "url",
        "hf_hub_id",
        "file",
        "input_size",
        "first_conv",
        "classifier",
    )
    return {
        key: portable(pretrained_cfg[key])
        for key in selected_cfg_keys
        if key in pretrained_cfg
    }


def build_aligned_execution_provenance(
    device: torch.device,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "software_dependencies": current_software_dependencies(),
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
            raise RuntimeError("aligned OOF cannot authenticate the CUDA GPU UUID")
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


def validate_aligned_execution_provenance(
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
        raise ValueError("aligned OOF execution provenance is incomplete")
    software = value.get("software_dependencies")
    if (
        not isinstance(software, Mapping)
        or set(software) != {"python", "torch", "timm", "torchvision"}
        or any(not isinstance(version, str) or not version for version in software.values())
    ):
        raise ValueError("aligned OOF execution dependency identity is invalid")
    if expected_software_dependencies is not None and dict(software) != dict(
        expected_software_dependencies
    ):
        raise ValueError("aligned OOF execution dependencies differ from protocol")
    device_type = value.get("device_type")
    if device_type not in {"cpu", "cuda"}:
        raise ValueError("aligned OOF execution device type is invalid")
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
            raise ValueError("aligned OOF CUDA execution identity is invalid")
    elif any(
        value.get(key) is not None
        for key in (
            "gpu_name",
            "gpu_compute_capability",
            "gpu_total_memory_bytes",
            "gpu_uuid",
        )
    ):
        raise ValueError("aligned OOF CPU execution cannot claim a GPU identity")
    return dict(value)


def validate_aligned_checkpoint_initialization(
    checkpoint: Mapping[str, Any],
    *,
    expected_initial_tensor_sha256: str | None = None,
    expected_pretrained_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    initialization = checkpoint.get("initialization_provenance")
    if not isinstance(initialization, Mapping):
        raise ValueError("aligned OOF checkpoint has no initialization provenance")
    expected = {
        "schema_version": 1,
        "scope": "generic_pretraining_only",
        "provider": "timm",
        "initialization": ALIGNED_OOF_ENCODER_INITIALIZATION,
        "configured_encoder": ALIGNED_OOF_ENCODER,
        "pretrained": True,
    }
    for key, expected_value in expected.items():
        if initialization.get(key) != expected_value:
            raise ValueError(
                f"aligned OOF initialization {key} must be {expected_value!r}"
            )
    initial_digest = str(
        initialization.get("encoder_initial_tensor_sha256", "")
    )
    checkpoint_digest = str(
        initialization.get("encoder_checkpoint_tensor_sha256", "")
    )
    if expected_pretrained_cfg is not None and initialization.get(
        "pretrained_cfg"
    ) != dict(expected_pretrained_cfg):
        raise ValueError(
            "aligned OOF checkpoint pretrained source differs from protocol"
        )
    checkpoint_model = checkpoint.get("model")
    if not isinstance(checkpoint_model, Mapping):
        raise ValueError("aligned OOF checkpoint has no evaluation model state")
    observed_checkpoint_digest = canonical_checkpoint_submodule_sha256(
        checkpoint_model, prefix="encoder."
    )
    if (
        len(initial_digest) != 64
        or any(character not in "0123456789abcdef" for character in initial_digest)
        or checkpoint_digest != initial_digest
        or observed_checkpoint_digest != initial_digest
        or (
            expected_initial_tensor_sha256 is not None
            and initial_digest != str(expected_initial_tensor_sha256)
        )
    ):
        raise ValueError(
            "aligned OOF checkpoint changed or cannot identify its frozen encoder"
        )
    return dict(initialization)


class MS75AlignedSwinProxy(nn.Module):
    """Leakage-free Swin-L OOF proxy with the final model's local capacity."""

    checkpoint_role = ALIGNED_OOF_CHECKPOINT_ROLE
    oof_proxy_role = ALIGNED_OOF_ROLE
    target_successful_optimizer_updates = ALIGNED_OOF_SUCCESSFUL_UPDATES

    def __init__(
        self,
        *,
        num_classes: int,
        encoder_config: Mapping[str, Any],
        decoder_channels: int = 256,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.encoder = build_pyramid_encoder(
            encoder_config,
            default_channels=(192, 384, 768, 1536),
            default_in_channels=4,
        )
        self.decoder = UPerDecoder(
            tuple(int(value) for value in self.encoder.out_channels),
            channels=int(decoder_channels),
            pool_scales=tuple(int(value) for value in pool_scales),
        )
        self.classifier = nn.Conv2d(
            int(decoder_channels), self.num_classes, kernel_size=1
        )
        self.encoder_initial_tensor_sha256 = canonical_module_tensor_sha256(
            self.encoder
        )
        self.encoder_initialization_provenance = {
            "schema_version": 1,
            "scope": "generic_pretraining_only",
            "provider": "timm",
            "initialization": ALIGNED_OOF_ENCODER_INITIALIZATION,
            "configured_encoder": ALIGNED_OOF_ENCODER,
            "pretrained": True,
            "pretrained_cfg": timm_pretrained_cfg_identity(self.encoder),
            "encoder_initial_tensor_sha256": (
                self.encoder_initial_tensor_sha256
            ),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MS75AlignedSwinProxy":
        model = config["model"]
        aligned = model["aligned_proxy"]
        return cls(
            num_classes=int(model["num_classes"]),
            encoder_config=aligned["encoder"],
            decoder_channels=int(aligned.get("decoder_channels", 256)),
            pool_scales=aligned.get("pool_scales", (1, 2, 3, 6)),
        )

    def forward(
        self,
        ms: torch.Tensor,
        sar: torch.Tensor | None = None,
        sar_valid: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        del sar, sar_valid
        features = self.encoder(ms)
        decoded = self.decoder(features)
        logits = self.classifier(decoded)
        if logits.shape[-2:] != ms.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=ms.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        zero = logits.sum(dim=1, keepdim=True) * 0.0
        return {
            "logits": logits,
            "ms_logits": logits,
            "sar_logits": logits * 0.0,
            "sar_gate": zero,
            "sar_scale": zero.mean().reshape(1),
        }


__all__ = [
    "MS75AlignedSwinProxy",
    "build_aligned_execution_provenance",
    "canonical_checkpoint_submodule_sha256",
    "canonical_module_tensor_sha256",
    "canonical_tensor_mapping_sha256",
    "timm_pretrained_cfg_identity",
    "validate_aligned_checkpoint_initialization",
    "validate_aligned_execution_provenance",
]
