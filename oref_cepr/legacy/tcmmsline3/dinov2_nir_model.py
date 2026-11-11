"""DINOv2-B/14 RGB expert with a separately gated NIR patch adapter.

The FBP reader supplies normalized B/G/R/NIR crops.  This module restores
reflectance, reorders the visible bands to native RGB, applies the frozen
DINOv2 RGB normalization, and independently normalizes NIR with the FBP
statistics.  A 14x14 NIR patch adapter is initialized from the mean of the
pretrained RGB patch weights and is injected through a scalar gate whose
initial effective value is exactly 0.1.

The 512x512 crop is reflect-padded to DINOv2's native 518x518 support.  DPT
features are read after transformer blocks 2, 5, 8, and 11 (zero based), and
the final 24-class logits are cropped back to the original 512x512 support.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


DINOV2_VITB14_OFFICIAL_SHA256 = (
    "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
)
DINOV2_RGB_MEAN = (0.485, 0.456, 0.406)
DINOV2_RGB_STD = (0.229, 0.224, 0.225)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tuple(
    values: Sequence[float], *, length: int, name: str, positive: bool = False
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values")
    if positive and any(value <= 0.0 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class _ResidualConvUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class _DPTReassemble(nn.Module):
    """Project equal-resolution ViT maps onto four DPT spatial scales."""

    def __init__(self, input_channels: int, output_channels: int, scale: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(input_channels, output_channels, 1)
        if scale == 4:
            self.resize = nn.ConvTranspose2d(
                output_channels, output_channels, kernel_size=4, stride=4
            )
        elif scale == 2:
            self.resize = nn.ConvTranspose2d(
                output_channels, output_channels, kernel_size=2, stride=2
            )
        elif scale == 1:
            self.resize = nn.Identity()
        elif scale == -2:
            self.resize = nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        else:  # pragma: no cover - constructor is private and fixed by DPTDecoder
            raise ValueError(f"unsupported DPT reassembly scale: {scale}")
        self.norm = nn.GroupNorm(_groups(output_channels), output_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.resize(self.projection(value))))


class _DPTFusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.skip = _ResidualConvUnit(channels)
        self.output = _ResidualConvUnit(channels)

    def forward(
        self, value: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        if skip is not None:
            value = F.interpolate(
                value, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            value = value + self.skip(skip)
        return self.output(value)


class DINOv2NIRDPTDecoder(nn.Module):
    """Compact DPT-style reassemble-and-refine semantic decoder."""

    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        reassemble_channels: Sequence[int] = (96, 192, 384, 768),
        fusion_channels: int = 256,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in reassemble_channels)
        if len(channels) != 4 or any(value <= 0 for value in channels):
            raise ValueError("DPT reassemble_channels must contain four positives")
        if int(fusion_channels) <= 0:
            raise ValueError("DPT fusion_channels must be positive")
        self.reassemble = nn.ModuleList(
            [
                _DPTReassemble(input_channels, output, scale)
                for output, scale in zip(channels, (4, 2, 1, -2))
            ]
        )
        self.scratch = nn.ModuleList(
            [
                nn.Conv2d(value, int(fusion_channels), 3, padding=1, bias=False)
                for value in channels
            ]
        )
        self.refine = nn.ModuleList(
            [_DPTFusionBlock(int(fusion_channels)) for _ in range(4)]
        )
        head_channels = max(int(fusion_channels) // 2, 32)
        self.head = nn.Sequential(
            nn.Conv2d(int(fusion_channels), head_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(head_channels), head_channels),
            nn.GELU(),
            nn.Conv2d(head_channels, int(num_classes), 1),
        )

    def forward(
        self, features: Sequence[torch.Tensor], *, output_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(features) != 4:
            raise ValueError("DPT decoder requires exactly four feature maps")
        pyramid = [
            lateral(reassemble(feature))
            for feature, reassemble, lateral in zip(
                features, self.reassemble, self.scratch
            )
        ]
        value = self.refine[3](pyramid[3])
        value = self.refine[2](value, pyramid[2])
        value = self.refine[1](value, pyramid[1])
        value = self.refine[0](value, pyramid[0])
        logits = self.head(value)
        logits = F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )
        return logits, value


def _build_timm_backbone(
    *,
    model_name: str,
    image_size: int,
    pretrained: bool,
    pretrained_file: str | None,
    pretrained_file_sha256: str | None,
) -> nn.Module:
    try:
        import timm
    except ImportError as error:  # pragma: no cover - deployment dependency
        raise RuntimeError("DINOv2NIR requires timm") from error

    create_arguments: dict[str, Any] = {
        "pretrained": bool(pretrained),
        "img_size": int(image_size),
        "num_classes": 0,
    }
    if pretrained_file is not None:
        path = Path(pretrained_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing DINOv2 pretrained file: {path}")
        expected = str(pretrained_file_sha256 or "").lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("DINOv2 pretrained_file_sha256 must be lowercase SHA256")
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"DINOv2 pretrained SHA256 mismatch: expected={expected}, observed={observed}"
            )
        # The pretrained configuration retains timm's official DINOv2 state
        # filter while replacing only the remote source with the pinned file.
        create_arguments["pretrained_cfg_overlay"] = {"file": str(path)}
    elif pretrained:
        raise ValueError(
            "formal DINOv2NIR construction requires a pinned local pretrained_file"
        )
    return timm.create_model(str(model_name), **create_arguments)


class DINOv2NIRSegmentation(nn.Module):
    """Native RGB DINOv2-B/14 plus a gated, independent NIR patch adapter."""

    def __init__(
        self,
        *,
        num_classes: int = 24,
        backbone_model: str = "vit_base_patch14_dinov2",
        pretrained: bool = True,
        pretrained_file: str | None = None,
        pretrained_file_sha256: str | None = None,
        input_size: int = 512,
        padded_size: int = 518,
        patch_size: int = 14,
        feature_indices: Sequence[int] = (2, 5, 8, 11),
        reassemble_channels: Sequence[int] = (96, 192, 384, 768),
        decoder_channels: int = 256,
        initial_nir_gate: float = 0.1,
        ms_mean: Sequence[float] = (0.363, 0.390, 0.371, 0.506),
        ms_std: Sequence[float] = (0.231, 0.231, 0.244, 0.254),
        rgb_mean: Sequence[float] = DINOV2_RGB_MEAN,
        rgb_std: Sequence[float] = DINOV2_RGB_STD,
        gradient_checkpointing: bool = True,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if int(num_classes) != 24:
            raise ValueError("DINOv2NIR is defined for exactly 24 classes")
        self.input_size = int(input_size)
        self.padded_size = int(padded_size)
        self.patch_size = int(patch_size)
        if self.input_size != 512 or self.padded_size != 518 or self.patch_size != 14:
            raise ValueError("DINOv2NIR requires 512 -> 518 padding and patch size 14")
        if (self.padded_size - self.input_size) % 2:
            raise ValueError("DINOv2NIR padding must be symmetric")
        if self.padded_size % self.patch_size:
            raise ValueError("padded DINOv2 input must be divisible by patch size")
        self.pad_pixels = (self.padded_size - self.input_size) // 2
        indices = tuple(int(value) for value in feature_indices)
        if indices != (2, 5, 8, 11):
            raise ValueError("DINOv2NIR DPT features must be blocks 2, 5, 8, 11")
        if not 0.0 < float(initial_nir_gate) < 1.0:
            raise ValueError("initial_nir_gate must be in (0, 1)")
        self.feature_indices = indices
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.num_classes = int(num_classes)

        ms_mean_values = _finite_tuple(ms_mean, length=4, name="ms_mean")
        ms_std_values = _finite_tuple(
            ms_std, length=4, name="ms_std", positive=True
        )
        rgb_mean_values = _finite_tuple(rgb_mean, length=3, name="rgb_mean")
        rgb_std_values = _finite_tuple(
            rgb_std, length=3, name="rgb_std", positive=True
        )
        self.register_buffer(
            "ms_mean",
            torch.tensor(ms_mean_values, dtype=torch.float32).reshape(1, 4, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "ms_std",
            torch.tensor(ms_std_values, dtype=torch.float32).reshape(1, 4, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "rgb_mean",
            torch.tensor(rgb_mean_values, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor(rgb_std_values, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=True,
        )

        self.backbone = backbone or _build_timm_backbone(
            model_name=str(backbone_model),
            image_size=self.padded_size,
            pretrained=bool(pretrained),
            pretrained_file=pretrained_file,
            pretrained_file_sha256=pretrained_file_sha256,
        )
        patch_projection = getattr(
            getattr(self.backbone, "patch_embed", None), "proj", None
        )
        if not isinstance(patch_projection, nn.Conv2d):
            raise ValueError("DINOv2 backbone lacks a Conv2d RGB patch projection")
        if tuple(patch_projection.kernel_size) != (14, 14):
            raise ValueError("DINOv2 RGB patch projection must use 14x14 patches")
        if patch_projection.in_channels != 3:
            raise ValueError("DINOv2 RGB patch projection must remain native 3-channel")
        if len(getattr(self.backbone, "blocks", ())) < 12:
            raise ValueError("DINOv2-B backbone must contain at least 12 blocks")
        if int(getattr(self.backbone, "num_prefix_tokens", 1)) != 1:
            raise ValueError("DINOv2NIR currently supports one CLS prefix token")
        embed_dim = int(patch_projection.out_channels)
        self.nir_patch_adapter = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=14,
            stride=14,
            bias=patch_projection.bias is not None,
        )
        with torch.no_grad():
            self.nir_patch_adapter.weight.copy_(
                patch_projection.weight.detach().mean(dim=1, keepdim=True)
            )
            if self.nir_patch_adapter.bias is not None:
                self.nir_patch_adapter.bias.zero_()
        initial_logit = math.log(float(initial_nir_gate) / (1.0 - float(initial_nir_gate)))
        self.nir_gate_logit = nn.Parameter(
            torch.tensor(initial_logit, dtype=torch.float32)
        )
        self.decoder = DINOv2NIRDPTDecoder(
            input_channels=embed_dim,
            num_classes=self.num_classes,
            reassemble_channels=reassemble_channels,
            fusion_channels=int(decoder_channels),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DINOv2NIRSegmentation":
        model = config["model"]
        normalization = config["data"]["normalization"]["ms"]
        return cls(
            num_classes=int(model["num_classes"]),
            backbone_model=str(model["backbone_model"]),
            pretrained=bool(model["pretrained"]),
            pretrained_file=str(model["pretrained_file"]),
            pretrained_file_sha256=str(model["pretrained_file_sha256"]),
            input_size=int(model["input_size"]),
            padded_size=int(model["padded_size"]),
            patch_size=int(model["patch_size"]),
            feature_indices=model["feature_indices"],
            reassemble_channels=model["reassemble_channels"],
            decoder_channels=int(model["decoder_channels"]),
            initial_nir_gate=float(model["initial_nir_gate"]),
            ms_mean=normalization["mean"],
            ms_std=normalization["std"],
            rgb_mean=model["rgb_normalization"]["mean"],
            rgb_std=model["rgb_normalization"]["std"],
            gradient_checkpointing=bool(model["gradient_checkpointing"]),
        )

    @property
    def nir_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.nir_gate_logit)

    def prepare_inputs(self, ms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ms.ndim != 4 or ms.shape[1] != 4:
            raise ValueError("DINOv2NIR input must have shape Bx4x512x512")
        if tuple(ms.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError("DINOv2NIR input crop must be exactly 512x512")
        if not ms.is_floating_point():
            raise TypeError("DINOv2NIR input must be floating point")
        ms_mean = self.ms_mean.to(device=ms.device, dtype=ms.dtype)
        ms_std = self.ms_std.to(device=ms.device, dtype=ms.dtype)
        reflectance = ms * ms_std + ms_mean
        # The dataset contract is B/G/R/NIR; DINOv2 must see R/G/B.
        rgb = reflectance[:, (2, 1, 0)]
        rgb_mean = self.rgb_mean.to(device=ms.device, dtype=ms.dtype)
        rgb_std = self.rgb_std.to(device=ms.device, dtype=ms.dtype)
        rgb = (rgb - rgb_mean) / rgb_std
        # Keep NIR independent from RGB ImageNet statistics. This expression
        # intentionally documents the second normalization even though it is
        # algebraically equal to the reader-provided normalized NIR channel.
        nir = (reflectance[:, 3:4] - ms_mean[:, 3:4]) / ms_std[:, 3:4]
        padding = (self.pad_pixels,) * 4
        return F.pad(rgb, padding, mode="reflect"), F.pad(
            nir, padding, mode="reflect"
        )

    @staticmethod
    def _as_tokens(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 3:
            return value
        if value.ndim == 4:
            return value.flatten(2).transpose(1, 2)
        raise RuntimeError("DINOv2 patch embedding returned an unsupported shape")

    def _backbone_features(
        self, rgb: torch.Tensor, nir: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        rgb_tokens = self._as_tokens(self.backbone.patch_embed(rgb))
        nir_tokens = self.nir_patch_adapter(nir).flatten(2).transpose(1, 2)
        if rgb_tokens.shape != nir_tokens.shape:
            raise RuntimeError("RGB and NIR patch-token grids are misaligned")
        value = rgb_tokens + self.nir_gate.to(rgb_tokens.dtype) * nir_tokens
        value = self.backbone._pos_embed(value)
        value = self.backbone.patch_drop(value)
        value = self.backbone.norm_pre(value)
        selected: list[torch.Tensor] = []
        for index, block in enumerate(self.backbone.blocks):
            if self.gradient_checkpointing and self.training and value.requires_grad:
                value = activation_checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
            if index in self.feature_indices:
                normalized = self.backbone.norm(value)
                patches = normalized[:, 1:]
                grid = self.padded_size // self.patch_size
                if patches.shape[1] != grid * grid:
                    raise RuntimeError("DINOv2 patch token count changed")
                selected.append(
                    patches.reshape(patches.shape[0], grid, grid, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        if len(selected) != 4:
            raise RuntimeError("DINOv2 did not expose all four DPT features")
        return tuple(selected)

    def forward(
        self,
        ms: torch.Tensor,
        sar: torch.Tensor | None = None,
        sar_valid: torch.Tensor | None = None,
        time_gap_days: torch.Tensor | None = None,
        ms_context: torch.Tensor | None = None,
        ms_context_valid: torch.Tensor | None = None,
        **unused: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Compatibility-only arguments must not influence this pure-MS expert.
        del sar, sar_valid, time_gap_days, ms_context, ms_context_valid, unused
        rgb, nir = self.prepare_inputs(ms)
        features = self._backbone_features(rgb, nir)
        padded_logits, decoder_embedding = self.decoder(
            features, output_size=(self.padded_size, self.padded_size)
        )
        p = self.pad_pixels
        logits = padded_logits[..., p:-p, p:-p]
        if tuple(logits.shape[-2:]) != (self.input_size, self.input_size):
            raise RuntimeError("DINOv2NIR output crop is not 512x512")
        # The protocol validator fixes every SAR/private/auxiliary objective to
        # zero, so allocating dense compatibility tensors here would waste
        # tens of MiB per crop and make the 24-GiB smoke less representative.
        # The shared trainer only reads sar_gate/sar_scale unconditionally.
        return {
            "logits": logits,
            "ms_logits": logits,
            "flat_logits": logits,
            "ms_shared": decoder_embedding,
            "sar_gate": logits.new_zeros((logits.shape[0], 1, 1, 1)),
            "sar_scale": logits.new_zeros(1),
            "nir_gate": self.nir_gate.reshape(1),
        }

    def predict_proba(self, ms: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.forward(ms, **kwargs)["logits"].softmax(dim=1)


__all__ = [
    "DINOV2_RGB_MEAN",
    "DINOV2_RGB_STD",
    "DINOV2_VITB14_OFFICIAL_SHA256",
    "DINOv2NIRDPTDecoder",
    "DINOv2NIRSegmentation",
]
