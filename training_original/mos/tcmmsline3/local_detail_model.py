"""Pure four-band local-detail expert for the FBP 24-class protocol.

The model deliberately receives only a local B/G/R/NIR target crop.  A
ConvNeXt-Base/UPer path supplies semantic evidence inside that crop, while a
stride-2 spectral-detail path preserves boundaries, texture and small objects.
The detail path can only add a bounded 24-class residual.  Its final
classifier is zero initialized, so a newly constructed model is exactly the
coarse UPer model at every pixel.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import ConvNeXtFeatures
from .ms75_model import UPerDecoder


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class _ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _ConvNormAct(channels, channels),
            _ConvNormAct(channels, channels, activation=False),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class SpectralDetailBranch(nn.Module):
    """Stride-2 B/G/R/NIR and spectral-index detail encoder."""

    def __init__(
        self,
        channels: int,
        *,
        ms_mean: Sequence[float],
        ms_std: Sequence[float],
    ) -> None:
        super().__init__()
        mean = tuple(float(value) for value in ms_mean)
        std = tuple(float(value) for value in ms_std)
        if len(mean) != 4 or len(std) != 4:
            raise ValueError("MS normalization must contain four B/G/R/NIR values")
        if not all(math.isfinite(value) for value in (*mean, *std)):
            raise ValueError("MS normalization values must be finite")
        if any(value <= 0 for value in std):
            raise ValueError("MS standard deviations must be positive")
        if int(channels) <= 0:
            raise ValueError("detail channels must be positive")

        self.register_buffer(
            "ms_mean",
            torch.tensor(mean, dtype=torch.float32).reshape(1, 4, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "ms_std",
            torch.tensor(std, dtype=torch.float32).reshape(1, 4, 1, 1),
            persistent=True,
        )
        self.encoder = nn.Sequential(
            _ConvNormAct(8, int(channels), stride=2),
            _ResidualBlock(int(channels)),
            _ResidualBlock(int(channels)),
        )

    @staticmethod
    def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return (numerator / denominator.abs().clamp_min(1e-4)).clamp(-1.0, 1.0)

    def spectral_evidence(self, normalized_ms: torch.Tensor) -> torch.Tensor:
        if normalized_ms.ndim != 4 or normalized_ms.shape[1] != 4:
            raise ValueError("local detail MS must have shape Bx4xHxW")
        mean = self.ms_mean.to(device=normalized_ms.device, dtype=normalized_ms.dtype)
        std = self.ms_std.to(device=normalized_ms.device, dtype=normalized_ms.dtype)
        reflectance = normalized_ms * std + mean
        blue, green, red, nir = reflectance.split(1, dim=1)
        ndvi = self._safe_ratio(nir - red, nir + red)
        ndwi = self._safe_ratio(green - nir, green + nir)
        savi = self._safe_ratio(1.5 * (nir - red), nir + red + 0.5)
        rgb_mean = (blue + green + red) / 3.0
        brightness_mean = mean[:, :3].mean(dim=1, keepdim=True)
        brightness_std = std[:, :3].mean(dim=1, keepdim=True).clamp_min(1e-4)
        brightness = ((rgb_mean - brightness_mean) / brightness_std).clamp(-5.0, 5.0)
        # The learned channels retain the normalized B/G/R/NIR contract; only
        # the four physical indices need the de-normalized reflectance above.
        return torch.cat([normalized_ms, ndvi, ndwi, savi, brightness], dim=1)

    def forward(self, normalized_ms: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.spectral_evidence(normalized_ms))


class LocalDetailCNXBSegmentation(nn.Module):
    """ConvNeXt-Base local expert with a bounded stride-2 detail residual."""

    def __init__(
        self,
        *,
        num_classes: int = 24,
        pretrained: bool = True,
        pretrained_file_sha256: str | None = None,
        backbone_variant: str = "base",
        decoder_channels: int = 192,
        detail_channels: int = 48,
        detail_fusion_channels: int = 64,
        max_detail_logit_scale: float = 0.5,
        initial_detail_logit_scale: float = 0.1,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        ms_mean: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        ms_std: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if int(num_classes) != 24:
            raise ValueError("LocalDetail-CNXB is defined for exactly 24 classes")
        if str(backbone_variant) not in {"tiny", "base"}:
            raise ValueError("backbone_variant must be tiny or base")
        if int(decoder_channels) <= 0:
            raise ValueError("decoder_channels must be positive")
        if int(detail_channels) <= 0:
            raise ValueError("detail_channels must be positive")
        if int(detail_fusion_channels) <= 0:
            raise ValueError("detail_fusion_channels must be positive")
        scales = tuple(int(value) for value in pool_scales)
        if not scales or any(value <= 0 for value in scales):
            raise ValueError("pool_scales must contain positive integers")
        if len(set(scales)) != len(scales):
            raise ValueError("pool_scales must be unique")
        maximum = float(max_detail_logit_scale)
        initial = float(initial_detail_logit_scale)
        if not math.isfinite(maximum) or not math.isfinite(initial):
            raise ValueError("detail logit scales must be finite")
        if not 0.0 < initial < maximum:
            raise ValueError(
                "initial_detail_logit_scale must be in (0, max_detail_logit_scale)"
            )

        self.num_classes = int(num_classes)
        self.max_detail_logit_scale = maximum
        backbone_arguments: dict[str, Any] = {
            "variant": str(backbone_variant),
            "pretrained": bool(pretrained),
        }
        if pretrained_file_sha256 is not None:
            backbone_arguments["pretrained_file_sha256"] = (
                pretrained_file_sha256
            )
        self.backbone = ConvNeXtFeatures(**backbone_arguments)
        self.decoder = UPerDecoder(
            tuple(self.backbone.channels),
            channels=int(decoder_channels),
            pool_scales=scales,
        )
        self.coarse_classifier = nn.Conv2d(int(decoder_channels), self.num_classes, 1)

        self.detail_branch = SpectralDetailBranch(
            int(detail_channels), ms_mean=ms_mean, ms_std=ms_std
        )
        self.coarse_detail_projection = _ConvNormAct(
            int(decoder_channels),
            int(detail_fusion_channels),
            kernel_size=1,
        )
        self.spectral_detail_projection = _ConvNormAct(
            int(detail_channels),
            int(detail_fusion_channels),
            kernel_size=1,
        )
        self.detail_fusion = nn.Sequential(
            _ConvNormAct(2 * int(detail_fusion_channels), int(detail_fusion_channels)),
            _ResidualBlock(int(detail_fusion_channels)),
        )
        self.detail_residual_classifier = nn.Conv2d(
            int(detail_fusion_channels), self.num_classes, 1
        )
        self.boundary_classifier = nn.Conv2d(int(detail_fusion_channels), 1, 1)
        self.spectral_aux_classifier = nn.Conv2d(
            int(detail_channels), self.num_classes, 1
        )
        nn.init.zeros_(self.detail_residual_classifier.weight)
        nn.init.zeros_(self.detail_residual_classifier.bias)

        initial_fraction = initial / maximum
        self.detail_scale_logit = nn.Parameter(
            torch.tensor(
                math.log(initial_fraction / (1.0 - initial_fraction)),
                dtype=torch.float32,
            )
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LocalDetailCNXBSegmentation":
        model = config["model"]
        data = config["data"]
        normalization = data["normalization"]["ms"]
        return cls(
            num_classes=int(model["num_classes"]),
            pretrained=bool(model.get("pretrained", True)),
            pretrained_file_sha256=model.get("pretrained_file_sha256"),
            backbone_variant=str(model.get("backbone_variant", "base")),
            decoder_channels=int(model.get("decoder_channels", 192)),
            detail_channels=int(model.get("detail_channels", 48)),
            detail_fusion_channels=int(model.get("detail_fusion_channels", 64)),
            max_detail_logit_scale=float(model.get("max_detail_logit_scale", 0.5)),
            initial_detail_logit_scale=float(
                model.get("initial_detail_logit_scale", 0.1)
            ),
            pool_scales=model.get("pool_scales", (1, 2, 3, 6)),
            ms_mean=normalization["mean"],
            ms_std=normalization["std"],
        )

    def _validate_ms(self, ms: torch.Tensor) -> None:
        if ms.ndim != 4 or ms.shape[1] != 4:
            raise ValueError("local detail MS must have shape Bx4xHxW")
        if not ms.is_floating_point():
            raise TypeError("local detail MS must be floating point")
        if ms.shape[-2] % 32 or ms.shape[-1] % 32:
            raise ValueError("local detail MS height and width must be divisible by 32")

    def forward(
        self,
        ms: torch.Tensor,
        sar: torch.Tensor,
        sar_valid: torch.Tensor,
        time_gap_days: torch.Tensor | None = None,
        ms_context: torch.Tensor | None = None,
        ms_context_valid: torch.Tensor | None = None,
        **unused: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # These arguments exist only for the shared training/evaluation call
        # signature.  Neither their values nor their shapes can influence this
        # pure four-band local expert.
        del sar, sar_valid, time_gap_days, ms_context, ms_context_valid, unused
        self._validate_ms(ms)
        output_size = tuple(int(value) for value in ms.shape[-2:])

        local_features = tuple(self.backbone(ms))
        coarse_embedding = self.decoder(local_features)
        coarse_logits_quarter = self.coarse_classifier(coarse_embedding)

        spectral_detail = self.detail_branch(ms)
        coarse_detail = F.interpolate(
            coarse_embedding,
            size=spectral_detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        detail_embedding = self.detail_fusion(
            torch.cat(
                [
                    self.coarse_detail_projection(coarse_detail),
                    self.spectral_detail_projection(spectral_detail),
                ],
                dim=1,
            )
        )

        raw_detail_residual = self.detail_residual_classifier(detail_embedding)
        detail_scale = (
            torch.sigmoid(self.detail_scale_logit) * self.max_detail_logit_scale
        ).to(raw_detail_residual.dtype)
        detail_correction = detail_scale * torch.tanh(raw_detail_residual)
        coarse_logits_half = F.interpolate(
            coarse_logits_quarter,
            size=detail_embedding.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        refined_logits_half = coarse_logits_half + detail_correction
        baseline_logits = F.interpolate(
            coarse_logits_half,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        logits = F.interpolate(
            refined_logits_half,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        boundary_logits_half = self.boundary_classifier(detail_embedding)
        boundary_logits = F.interpolate(
            boundary_logits_half,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        spectral_aux_logits = F.interpolate(
            self.spectral_aux_classifier(spectral_detail),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        zero_logits = torch.zeros_like(logits)
        zero_shared = torch.zeros_like(coarse_embedding)
        native_height = max(output_size[0] // 4, 1)
        native_width = max(output_size[1] // 4, 1)
        sar_native_logits = ms.new_zeros(
            (ms.shape[0], self.num_classes, native_height, native_width)
        )
        return {
            "logits": logits,
            "ms_logits": logits,
            "flat_logits": logits,
            "coarse_logits": baseline_logits,
            "baseline_logits": baseline_logits,
            "coarse_logits_stride4": coarse_logits_quarter,
            "detail_residual": raw_detail_residual,
            "detail_correction": detail_correction,
            "detail_scale": detail_scale.reshape(1),
            "boundary_logits": boundary_logits,
            "boundary_logits_stride2": boundary_logits_half,
            "spectral_aux_logits": spectral_aux_logits,
            "ms_shared": coarse_embedding,
            "sar_logits": zero_logits,
            "sar_native_logits": sar_native_logits,
            "sar_shared": zero_shared,
            "sar_private": zero_shared,
            "sar_gate": zero_shared[:, :1],
            "sar_scale": logits.new_zeros(1),
            "sar_present": torch.zeros(ms.shape[0], dtype=torch.bool, device=ms.device),
        }


__all__ = ["LocalDetailCNXBSegmentation", "SpectralDetailBranch"]
