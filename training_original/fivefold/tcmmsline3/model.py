from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        activation: bool = True,
    ):
        padding = dilation * (kernel_size // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, dilation=dilation),
            ConvNormAct(channels, channels, dilation=dilation, activation=False),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class MSBackbone(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(4, width),
            ResidualBlock(width),
        )
        self.down1 = nn.Sequential(
            ConvNormAct(width, width * 2, stride=2),
            ResidualBlock(width * 2),
        )
        self.down2 = nn.Sequential(
            ConvNormAct(width * 2, width * 4, stride=2),
            ResidualBlock(width * 4),
            ResidualBlock(width * 4, dilation=2),
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fine = self.stem(inputs)
        middle = self.down1(fine)
        coarse = self.down2(middle)
        return fine, middle, coarse


class SARBackbone(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        channels = width * 4
        self.encoder = nn.Sequential(
            ConvNormAct(5, width * 2),
            ResidualBlock(width * 2),
            ConvNormAct(width * 2, channels),
            ResidualBlock(channels),
            ResidualBlock(channels, dilation=2),
        )

    def forward(
        self, inputs: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        valid_channel = valid[:, None].to(inputs.dtype)
        vv, vh = inputs[:, 0:1], inputs[:, 1:2]
        derived = torch.cat(
            [vv, vh, vv - vh, 0.5 * (vv + vh), valid_channel], dim=1
        )
        features = self.encoder(derived)
        return features * valid_channel


class SARNativeSegmentation(nn.Module):
    """Standalone SAR semantic model that never depends on MS features.

    The classifier stays on the native SAR support grid.  A bilinearly
    upsampled view is returned only so the operational MS-grid evaluator can
    measure the map that SAR alone would produce; training and model selection
    use ``sar_native_logits``.
    """

    def __init__(self, *, num_classes: int = 24, width: int = 32):
        super().__init__()
        self.num_classes = int(num_classes)
        self.width = int(width)
        channels = self.width * 4
        self.sar_backbone = SARBackbone(self.width)
        self.native_decoder = nn.Sequential(
            ResidualBlock(channels),
            ConvNormAct(channels, self.width * 2),
            ResidualBlock(self.width * 2),
            nn.Conv2d(self.width * 2, self.num_classes, 1),
        )

    @classmethod
    def from_config(cls, config: dict) -> "SARNativeSegmentation":
        model = config["model"]
        return cls(
            num_classes=int(model["num_classes"]),
            width=int(model["width"]),
        )

    def forward(
        self,
        ms: torch.Tensor,
        sar: torch.Tensor,
        sar_valid: torch.Tensor,
        time_gap_days: torch.Tensor | None = None,
        ms_context: torch.Tensor | None = None,
        ms_context_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del ms_context, ms_context_valid, time_gap_days
        features = self.sar_backbone(sar, sar_valid)
        native_logits = self.native_decoder(features)
        valid = sar_valid[:, None].to(native_logits.dtype)
        native_logits = native_logits * valid
        logits = F.interpolate(
            native_logits,
            size=ms.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        zero_logits = torch.zeros_like(logits)
        zero_features = torch.zeros_like(features)
        return {
            "logits": logits,
            "ms_logits": zero_logits,
            "sar_logits": logits,
            "sar_native_logits": native_logits,
            "ms_shared": zero_features,
            "sar_shared": features,
            "sar_private": zero_features,
            "sar_gate": valid,
            "sar_scale": torch.ones(1, dtype=logits.dtype, device=logits.device),
            "sar_present": sar_valid.flatten(1).any(dim=1),
        }


class RegionCrossAttention(nn.Module):
    def __init__(self, channels: int, region_grid: int, heads: int):
        super().__init__()
        if channels % heads:
            raise ValueError("attention channels must be divisible by heads")
        self.region_grid = region_grid
        self.ms_norm = nn.LayerNorm(channels)
        self.sar_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=0.0, batch_first=True
        )
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
        )

    def forward(
        self,
        ms_features: torch.Tensor,
        sar_features: torch.Tensor,
        sar_valid: torch.Tensor,
    ) -> torch.Tensor:
        grid = self.region_grid
        ms_regions = F.adaptive_avg_pool2d(ms_features, (grid, grid))
        sar_regions = F.adaptive_avg_pool2d(sar_features, (grid, grid))
        ms_tokens = ms_regions.flatten(2).transpose(1, 2)
        sar_tokens = sar_regions.flatten(2).transpose(1, 2)
        attended, _ = self.attention(
            self.ms_norm(ms_tokens),
            self.sar_norm(sar_tokens),
            self.sar_norm(sar_tokens),
            need_weights=False,
        )
        attended = attended.transpose(1, 2).reshape(
            ms_features.shape[0], ms_features.shape[1], grid, grid
        )
        attended = F.interpolate(
            attended,
            size=ms_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        has_sar = sar_valid.flatten(1).any(dim=1).to(attended.dtype)
        return self.output(attended) * has_sar[:, None, None, None]


class Decoder(nn.Module):
    def __init__(self, width: int, num_classes: int):
        super().__init__()
        self.middle = nn.Sequential(
            ConvNormAct(width * 6, width * 2),
            ResidualBlock(width * 2),
        )
        self.fine = nn.Sequential(
            ConvNormAct(width * 3, width),
            ResidualBlock(width),
        )
        self.classifier = nn.Conv2d(width, num_classes, 1)

    def forward(
        self,
        coarse: torch.Tensor,
        middle_skip: torch.Tensor,
        fine_skip: torch.Tensor,
    ) -> torch.Tensor:
        value = F.interpolate(
            coarse, size=middle_skip.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.middle(torch.cat([value, middle_skip], dim=1))
        value = F.interpolate(
            value, size=fine_skip.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.fine(torch.cat([value, fine_skip], dim=1))
        return self.classifier(value)


class LowRankAdapter2d(nn.Module):
    """A zero-initialized modality-specific residual adapter."""

    def __init__(self, channels: int, bottleneck: int):
        super().__init__()
        if bottleneck <= 0 or bottleneck >= channels:
            raise ValueError("adapter bottleneck must be in (0, channels)")
        self.down = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.norm = nn.GroupNorm(_groups(bottleneck), bottleneck)
        self.up = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.activation = nn.GELU()
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.up(self.activation(self.norm(self.down(inputs))))


class BoundedRegionAlignment(nn.Module):
    """Learn a smooth, physically bounded SAR offset field.

    Offsets are predicted on a coarse region grid and then interpolated to the
    fusion support.  This prevents unconstrained attention from searching the
    whole scene for class-similar but geographically unrelated locations.
    """

    def __init__(
        self,
        channels: int,
        *,
        region_grid: int,
        max_offset_pixels: float,
        enabled: bool,
    ):
        super().__init__()
        self.region_grid = int(region_grid)
        self.max_offset_pixels = float(max_offset_pixels)
        self.enabled = bool(enabled)
        hidden = max(channels // 4, 16)
        self.offset = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    @staticmethod
    def _base_grid(
        batch: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1)[None].expand(
            batch, -1, -1, -1
        )

    def forward(
        self,
        ms_features: torch.Tensor,
        sar_features: torch.Tensor,
        sar_valid: torch.Tensor,
        sar_private: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, _, height, width = sar_features.shape
        if sar_private.shape[-2:] != (height, width):
            raise ValueError(
                "shared and private SAR features must use the same grid"
            )
        if not self.enabled or self.max_offset_pixels <= 0:
            offsets = torch.zeros(
                batch,
                2,
                self.region_grid,
                self.region_grid,
                device=sar_features.device,
                dtype=sar_features.dtype,
            )
            valid = sar_valid[:, None].to(sar_features.dtype)
            return (
                sar_features * valid,
                sar_private * valid,
                sar_valid,
                offsets,
            )

        pooled = F.adaptive_avg_pool2d(
            torch.cat([ms_features, sar_features], dim=1),
            (self.region_grid, self.region_grid),
        )
        offsets = torch.tanh(self.offset(pooled)) * self.max_offset_pixels
        dense_offsets = F.interpolate(
            offsets,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        normalized = torch.empty_like(dense_offsets)
        normalized[:, 0] = dense_offsets[:, 0] * (2.0 / max(width - 1, 1))
        normalized[:, 1] = dense_offsets[:, 1] * (2.0 / max(height - 1, 1))
        grid = self._base_grid(
            batch,
            height,
            width,
            device=sar_features.device,
            dtype=sar_features.dtype,
        )
        grid = grid + normalized.permute(0, 2, 3, 1)
        aligned = F.grid_sample(
            sar_features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        aligned_private = F.grid_sample(
            sar_private,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        valid = F.grid_sample(
            sar_valid[:, None].to(sar_features.dtype),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0].bool()
        valid_channel = valid[:, None].to(sar_features.dtype)
        return (
            aligned * valid_channel,
            aligned_private * valid_channel,
            valid,
            offsets,
        )


class ClassRegionRouter(nn.Module):
    """Choose a bounded correction policy independently for each class-region."""

    def __init__(
        self,
        *,
        channels: int,
        num_classes: int,
        region_grid: int,
        hidden: int,
        include_both: bool = False,
        initial_both_probability: float = 0.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.region_grid = int(region_grid)
        self.include_both = bool(include_both)
        self.expert_count = 4 if self.include_both else 3
        input_channels = channels * 3 + self.num_classes + 4
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            ResidualBlock(hidden),
            nn.Conv2d(
                hidden,
                self.expert_count * self.num_classes,
                1,
            ),
        )
        output = self.network[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        if self.include_both and initial_both_probability > 0:
            other_probability = (1.0 - initial_both_probability) / 3.0
            both_bias = math.log(initial_both_probability / other_probability)
            with torch.no_grad():
                output.bias.reshape(self.expert_count, self.num_classes)[3].fill_(
                    both_bias
                )

    @staticmethod
    def _soft_boundary(probability: torch.Tensor) -> torch.Tensor:
        dx = F.pad(
            (probability[:, :, :, 1:] - probability[:, :, :, :-1])
            .abs()
            .amax(dim=1, keepdim=True),
            (1, 0, 0, 0),
        )
        dy = F.pad(
            (probability[:, :, 1:, :] - probability[:, :, :-1, :])
            .abs()
            .amax(dim=1, keepdim=True),
            (0, 0, 1, 0),
        )
        return (dx + dy).clamp_(0.0, 1.0)

    def forward(
        self,
        *,
        ms_shared: torch.Tensor,
        sar_shared: torch.Tensor,
        ms_logits: torch.Tensor,
        sar_valid: torch.Tensor,
        time_gap_days: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = ms_shared.shape[-2:]
        probability = F.interpolate(
            ms_logits.softmax(dim=1),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        entropy = -(
            probability.clamp_min(1e-7) * probability.clamp_min(1e-7).log()
        ).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(self.num_classes)
        boundary = self._soft_boundary(probability)
        valid_channel = sar_valid[:, None].to(ms_shared.dtype)
        if time_gap_days is None:
            gap = torch.zeros(
                ms_shared.shape[0],
                1,
                1,
                1,
                device=ms_shared.device,
                dtype=ms_shared.dtype,
            )
        else:
            gap = time_gap_days.to(ms_shared.dtype).reshape(-1, 1, 1, 1)
            gap = (gap / 20.0).clamp(0.0, 1.0)
        gap = gap.expand(-1, -1, *size)
        router_input = torch.cat(
            [
                ms_shared,
                sar_shared,
                (ms_shared - sar_shared).abs(),
                probability,
                entropy,
                boundary,
                valid_channel,
                gap,
            ],
            dim=1,
        )
        pooled = F.adaptive_avg_pool2d(
            router_input, (self.region_grid, self.region_grid)
        )
        logits = self.network(pooled).reshape(
            ms_shared.shape[0],
            self.expert_count,
            self.num_classes,
            self.region_grid,
            self.region_grid,
        )
        region_valid = (
            F.adaptive_avg_pool2d(
                valid_channel,
                (self.region_grid, self.region_grid),
            )
            > 0.5
        )
        invalid = ~region_valid
        null_logits = torch.where(
            invalid,
            torch.zeros_like(logits[:, 0]),
            logits[:, 0],
        )
        non_null_logits = torch.where(
            invalid[:, None],
            torch.full_like(logits[:, 1:], -1e4),
            logits[:, 1:],
        )
        logits = torch.cat([null_logits[:, None], non_null_logits], dim=1)
        return logits, region_valid, boundary


class LogitResidualExpert(nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            ResidualBlock(hidden),
            nn.Conv2d(hidden, num_classes, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MSGuidedRegionFusion(nn.Module):
    """Dual-stream model with coarse region-level SAR interaction."""

    def __init__(
        self,
        *,
        num_classes: int = 24,
        width: int = 32,
        fusion_mode: str = "region",
        region_grid: int = 4,
        attention_heads: int = 4,
        max_sar_residual_scale: float = 1.0,
    ):
        super().__init__()
        if fusion_mode not in {
            "ms_only",
            "sar_only",
            "concat",
            "region",
            "local_residual",
        }:
            raise ValueError(f"unsupported fusion mode: {fusion_mode}")
        self.num_classes = num_classes
        self.width = width
        self.fusion_mode = fusion_mode
        channels = width * 4

        self.ms_backbone = MSBackbone(width)
        self.sar_backbone = SARBackbone(width)
        self.ms_shared = nn.Conv2d(channels, channels, 1, bias=False)
        self.sar_shared = nn.Conv2d(channels, channels, 1, bias=False)
        self.sar_private = nn.Conv2d(channels, channels, 1, bias=False)
        self.region_attention = RegionCrossAttention(
            channels, region_grid, attention_heads
        )
        self.local_gate = nn.Sequential(
            ConvNormAct(channels * 2 + 1, channels),
            nn.Conv2d(channels, 1, 1),
        )
        self.concat_projection = ConvNormAct(channels * 2, channels)
        initial_scale = 0.1 / max(max_sar_residual_scale, 1e-6)
        initial_scale = min(max(initial_scale, 1e-4), 1 - 1e-4)
        self.sar_scale_logit = nn.Parameter(
            torch.tensor(math.log(initial_scale / (1 - initial_scale)))
        )
        self.max_sar_residual_scale = float(max_sar_residual_scale)
        self.decoder = Decoder(width, num_classes)
        self.sar_classifier = nn.Conv2d(channels, num_classes, 1)

    @classmethod
    def from_config(cls, config: dict) -> "MSGuidedRegionFusion":
        model = config["model"]
        return cls(
            num_classes=int(model["num_classes"]),
            width=int(model["width"]),
            fusion_mode=str(model["fusion_mode"]),
            region_grid=int(model["region_grid"]),
            attention_heads=int(model["attention_heads"]),
            max_sar_residual_scale=float(model["max_sar_residual_scale"]),
        )

    def set_fusion_mode(self, mode: str) -> None:
        if mode not in {
            "ms_only",
            "sar_only",
            "concat",
            "region",
            "local_residual",
        }:
            raise ValueError(mode)
        self.fusion_mode = mode

    def forward(
        self,
        ms: torch.Tensor,
        sar: torch.Tensor,
        sar_valid: torch.Tensor,
        time_gap_days: torch.Tensor | None = None,
        ms_context: torch.Tensor | None = None,
        ms_context_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del ms_context, ms_context_valid, time_gap_days
        fine, middle, ms_coarse = self.ms_backbone(ms)
        ms_logits = self.decoder(ms_coarse, middle, fine)
        scale = (
            torch.sigmoid(self.sar_scale_logit) * self.max_sar_residual_scale
        )
        if self.fusion_mode == "ms_only":
            zeros = torch.zeros_like(ms_coarse)
            sar_native_logits = torch.zeros(
                ms.shape[0],
                self.num_classes,
                ms_coarse.shape[-2],
                ms_coarse.shape[-1],
                dtype=ms.dtype,
                device=ms.device,
            )
            return {
                "logits": ms_logits,
                "ms_logits": ms_logits,
                "sar_logits": torch.zeros_like(ms_logits),
                "sar_native_logits": sar_native_logits,
                "ms_shared": self.ms_shared(ms_coarse),
                "sar_shared": zeros,
                "sar_private": zeros,
                "sar_gate": zeros[:, :1],
                "sar_scale": scale.reshape(1),
                "sar_present": torch.zeros(
                    ms.shape[0], dtype=torch.bool, device=ms.device
                ),
            }

        sar_features = self.sar_backbone(sar, sar_valid)
        if sar_features.shape[-2:] != ms_coarse.shape[-2:]:
            sar_features = F.interpolate(
                sar_features,
                size=ms_coarse.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            sar_valid_coarse = F.interpolate(
                sar_valid[:, None].float(),
                size=ms_coarse.shape[-2:],
                mode="nearest",
            )[:, 0].bool()
        else:
            sar_valid_coarse = sar_valid

        ms_shared = self.ms_shared(ms_coarse)
        sar_shared = self.sar_shared(sar_features) * sar_valid_coarse[:, None]
        sar_private = self.sar_private(sar_features) * sar_valid_coarse[:, None]
        valid_channel = sar_valid_coarse[:, None].to(ms.dtype)
        gate = torch.sigmoid(
            self.local_gate(torch.cat([ms_shared, sar_shared, valid_channel], dim=1))
        )
        gate = gate * valid_channel
        if self.fusion_mode == "sar_only":
            fused_coarse = sar_shared + sar_private
            fine = torch.zeros_like(fine)
            middle = torch.zeros_like(middle)
        elif self.fusion_mode == "concat":
            fused_coarse = self.concat_projection(
                torch.cat([ms_coarse, sar_shared + sar_private], dim=1)
            )
        elif self.fusion_mode == "local_residual":
            # A geographically aligned control: SAR may only correct the
            # corresponding coarse MS location, and the gate controls the
            # complete SAR residual. This deliberately avoids global token
            # mixing so the value of paired SAR can be measured cleanly.
            fused_coarse = ms_coarse + scale * gate * (
                sar_shared + sar_private
            )
        else:
            region_context = self.region_attention(
                ms_shared, sar_shared, sar_valid_coarse
            )
            fused_coarse = ms_coarse + scale * (
                gate * sar_private + region_context
            )

        logits = self.decoder(fused_coarse, middle, fine)
        sar_native_logits = self.sar_classifier(sar_features)
        sar_logits = F.interpolate(
            sar_native_logits,
            size=ms.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return {
            "logits": logits,
            "ms_logits": ms_logits,
            "sar_logits": sar_logits,
            "sar_native_logits": sar_native_logits,
            "ms_shared": ms_shared,
            "sar_shared": sar_shared,
            "sar_private": sar_private,
            "sar_gate": gate,
            "sar_scale": scale.reshape(1),
            "sar_present": sar_valid_coarse.flatten(1).any(dim=1),
        }


class SymmSafeFusion(nn.Module):
    """Symmetric representation with an asymmetric, MS-anchored decision.

    MS supplies the complete semantic map. SAR-specialized and collaborative
    experts may only add bounded, class-conditioned coarse logit residuals.
    The router has an explicit Null choice and missing SAR returns the exact
    MS path.
    """

    valid_fusion_modes = {"symm_safe", "ms_only"}
    valid_routing_modes = {
        "learned",
        "uniform",
        "null",
        "sar",
        "collaborative",
        "both",
    }

    def __init__(
        self,
        *,
        num_classes: int = 24,
        width: int = 32,
        fusion_mode: str = "symm_safe",
        adapter_bottleneck: int = 16,
        expert_hidden: int = 64,
        router_hidden: int = 64,
        router_grid: int = 8,
        routing_mode: str = "learned",
        max_sar_residual_scale: float = 1.0,
        initial_sar_residual_scale: float = 0.1,
        boundary_attenuation: float = 0.0,
        bounded_alignment_enabled: bool = False,
        max_alignment_offset_pixels: float = 1.0,
        router_include_both: bool = False,
        router_initial_both_probability: float = 0.0,
    ):
        super().__init__()
        if fusion_mode not in self.valid_fusion_modes:
            raise ValueError(f"unsupported SymmSafe fusion mode: {fusion_mode}")
        if routing_mode not in self.valid_routing_modes:
            raise ValueError(f"unsupported routing mode: {routing_mode}")
        if not 0.0 <= boundary_attenuation <= 1.0:
            raise ValueError("boundary_attenuation must be in [0, 1]")
        if not 0.0 <= router_initial_both_probability < 1.0:
            raise ValueError("router_initial_both_probability must be in [0, 1)")
        if router_initial_both_probability > 0 and not router_include_both:
            raise ValueError(
                "router_initial_both_probability requires router_include_both"
            )
        if routing_mode == "both" and not router_include_both:
            raise ValueError("both routing requires router_include_both=true")
        if not 0.0 < initial_sar_residual_scale < max_sar_residual_scale:
            raise ValueError(
                "initial_sar_residual_scale must be in "
                "(0, max_sar_residual_scale)"
            )

        self.num_classes = int(num_classes)
        self.width = int(width)
        self.fusion_mode = str(fusion_mode)
        self.routing_mode = str(routing_mode)
        self.router_grid = int(router_grid)
        self.boundary_attenuation = float(boundary_attenuation)
        self.router_include_both = bool(router_include_both)
        channels = self.width * 4

        self.ms_backbone = MSBackbone(self.width)
        self.sar_backbone = SARBackbone(self.width)
        self.decoder = Decoder(self.width, self.num_classes)
        self.ms_adapter = LowRankAdapter2d(channels, adapter_bottleneck)
        self.sar_adapter = LowRankAdapter2d(channels, adapter_bottleneck)
        # The same projection is applied to both modalities, providing a
        # common semantic space without erasing modality-specific adapters.
        self.shared_projector = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
        )
        self.alignment = BoundedRegionAlignment(
            channels,
            region_grid=self.router_grid,
            max_offset_pixels=max_alignment_offset_pixels,
            enabled=bounded_alignment_enabled,
        )
        self.sar_expert = LogitResidualExpert(
            channels * 2,
            expert_hidden,
            self.num_classes,
        )
        self.collaborative_expert = LogitResidualExpert(
            channels * 4,
            expert_hidden,
            self.num_classes,
        )
        self.router = ClassRegionRouter(
            channels=channels,
            num_classes=self.num_classes,
            region_grid=self.router_grid,
            hidden=router_hidden,
            include_both=self.router_include_both,
            initial_both_probability=router_initial_both_probability,
        )
        self.sar_classifier = nn.Conv2d(channels, self.num_classes, 1)

        ratio = initial_sar_residual_scale / max_sar_residual_scale
        self.sar_scale_logit = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)))
        )
        self.max_sar_residual_scale = float(max_sar_residual_scale)

    @classmethod
    def from_config(cls, config: dict) -> "SymmSafeFusion":
        model = config["model"]
        return cls(
            num_classes=int(model["num_classes"]),
            width=int(model["width"]),
            fusion_mode=str(model["fusion_mode"]),
            adapter_bottleneck=int(model.get("adapter_bottleneck", 16)),
            expert_hidden=int(model.get("expert_hidden", 64)),
            router_hidden=int(model.get("router_hidden", 64)),
            router_grid=int(model.get("router_grid", 8)),
            routing_mode=str(model.get("routing_mode", "learned")),
            max_sar_residual_scale=float(model["max_sar_residual_scale"]),
            initial_sar_residual_scale=float(
                model.get("initial_sar_residual_scale", 0.1)
            ),
            boundary_attenuation=float(
                model.get("boundary_attenuation", 0.0)
            ),
            bounded_alignment_enabled=bool(
                model.get("bounded_alignment_enabled", False)
            ),
            max_alignment_offset_pixels=float(
                model.get("max_alignment_offset_pixels", 1.0)
            ),
            router_include_both=bool(model.get("router_include_both", False)),
            router_initial_both_probability=float(
                model.get("router_initial_both_probability", 0.0)
            ),
        )

    def set_fusion_mode(self, mode: str) -> None:
        if mode not in self.valid_fusion_modes:
            raise ValueError(mode)
        self.fusion_mode = mode

    def set_routing_mode(self, mode: str) -> None:
        if mode not in self.valid_routing_modes:
            raise ValueError(mode)
        self.routing_mode = mode

    def _routing_weights(
        self,
        logits: torch.Tensor,
        region_valid: torch.Tensor,
        boundary: torch.Tensor,
    ) -> torch.Tensor:
        if self.routing_mode == "learned":
            weights = logits.softmax(dim=1)
        else:
            weights = torch.zeros_like(logits)
            if self.routing_mode == "uniform":
                if self.router_include_both:
                    weights[:, 3] = 1.0
                else:
                    weights[:, 1:] = 0.5
            elif self.routing_mode == "null":
                weights[:, 0] = 1.0
            elif self.routing_mode == "sar":
                weights[:, 1] = 1.0
            elif self.routing_mode == "collaborative":
                weights[:, 2] = 1.0
            elif self.routing_mode == "both":
                if not self.router_include_both:
                    raise ValueError("both routing requires router_include_both=true")
                weights[:, 3] = 1.0

        invalid = ~region_valid
        non_null = torch.where(
            invalid[:, None],
            torch.zeros_like(weights[:, 1:]),
            weights[:, 1:],
        )
        null = torch.where(
            invalid,
            torch.ones_like(weights[:, 0]),
            weights[:, 0],
        )

        if self.boundary_attenuation > 0:
            region_boundary = F.adaptive_avg_pool2d(
                boundary,
                (self.router_grid, self.router_grid),
            )
            keep = 1.0 - self.boundary_attenuation * region_boundary
            non_null = non_null * keep[:, None]
            null = 1.0 - non_null.sum(dim=1)
        return torch.cat([null[:, None], non_null], dim=1)

    def _empty_safe_outputs(
        self,
        *,
        ms_logits: torch.Tensor,
        ms_shared: torch.Tensor,
        sar_scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = ms_logits.shape[0]
        coarse_size = ms_shared.shape[-2:]
        zero_coarse = torch.zeros_like(ms_shared)
        zero_native = torch.zeros(
            batch,
            self.num_classes,
            *coarse_size,
            device=ms_logits.device,
            dtype=ms_logits.dtype,
        )
        router_logits = torch.full(
            (
                batch,
                self.router.expert_count,
                self.num_classes,
                self.router_grid,
                self.router_grid,
            ),
            -1e4,
            device=ms_logits.device,
            dtype=ms_logits.dtype,
        )
        router_logits[:, 0] = 0
        router_weights = torch.zeros_like(router_logits)
        router_weights[:, 0] = 1
        region_valid = torch.zeros(
            batch,
            1,
            self.router_grid,
            self.router_grid,
            device=ms_logits.device,
            dtype=torch.bool,
        )
        return {
            "logits": ms_logits,
            "ms_logits": ms_logits,
            "sar_logits": torch.zeros_like(ms_logits),
            "sar_native_logits": zero_native,
            "ms_shared": ms_shared,
            "sar_shared": zero_coarse,
            "sar_private": zero_coarse,
            "sar_gate": torch.zeros(
                batch,
                1,
                *coarse_size,
                device=ms_logits.device,
                dtype=ms_logits.dtype,
            ),
            "sar_scale": sar_scale.reshape(1),
            "sar_present": torch.zeros(
                batch, dtype=torch.bool, device=ms_logits.device
            ),
            "router_logits": router_logits,
            "router_weights": router_weights,
            "router_region_valid": region_valid,
            "counterfactual_sar_logits": ms_logits,
            "counterfactual_collab_logits": ms_logits,
            "counterfactual_both_logits": ms_logits,
            "delta_sar_logits": zero_native,
            "delta_collab_logits": zero_native,
            "alignment_offsets": torch.zeros(
                batch,
                2,
                self.router_grid,
                self.router_grid,
                device=ms_logits.device,
                dtype=ms_logits.dtype,
            ),
        }

    def forward(
        self,
        ms: torch.Tensor,
        sar: torch.Tensor,
        sar_valid: torch.Tensor,
        time_gap_days: torch.Tensor | None = None,
        ms_context: torch.Tensor | None = None,
        ms_context_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del ms_context, ms_context_valid
        fine, middle, ms_coarse = self.ms_backbone(ms)
        ms_logits = self.decoder(ms_coarse, middle, fine)
        ms_adapted = self.ms_adapter(ms_coarse)
        ms_shared = self.shared_projector(ms_adapted)
        scale = (
            torch.sigmoid(self.sar_scale_logit) * self.max_sar_residual_scale
        )
        if self.fusion_mode == "ms_only":
            return self._empty_safe_outputs(
                ms_logits=ms_logits,
                ms_shared=ms_shared,
                sar_scale=scale,
            )

        sar_features = self.sar_backbone(sar, sar_valid)
        if sar_features.shape[-2:] != ms_coarse.shape[-2:]:
            sar_features = F.interpolate(
                sar_features,
                size=ms_coarse.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            sar_valid_coarse = F.interpolate(
                sar_valid[:, None].float(),
                size=ms_coarse.shape[-2:],
                mode="nearest",
            )[:, 0].bool()
        else:
            sar_valid_coarse = sar_valid

        sar_adapted = self.sar_adapter(sar_features)
        sar_shared = self.shared_projector(sar_adapted)
        sar_shared = sar_shared * sar_valid_coarse[:, None]
        (
            aligned_sar,
            aligned_private,
            aligned_valid,
            offsets,
        ) = self.alignment(
            ms_shared,
            sar_shared,
            sar_valid_coarse,
            sar_adapted,
        )

        sar_delta = self.sar_expert(
            torch.cat([aligned_sar, aligned_private], dim=1)
        )
        collaborative_delta = self.collaborative_expert(
            torch.cat(
                [
                    ms_shared,
                    aligned_sar,
                    (ms_shared - aligned_sar).abs(),
                    ms_shared * aligned_sar,
                ],
                dim=1,
            )
        )
        valid_channel = aligned_valid[:, None].to(ms.dtype)
        sar_delta = sar_delta * valid_channel
        collaborative_delta = collaborative_delta * valid_channel

        router_logits, region_valid, boundary = self.router(
            ms_shared=ms_shared,
            sar_shared=aligned_sar,
            ms_logits=ms_logits,
            sar_valid=aligned_valid,
            time_gap_days=time_gap_days,
        )
        router_weights = self._routing_weights(
            router_logits,
            region_valid,
            boundary,
        )
        sar_weight = F.interpolate(
            router_weights[:, 1]
            + (0.5 * router_weights[:, 3] if self.router_include_both else 0.0),
            size=ms_coarse.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        collaborative_weight = F.interpolate(
            router_weights[:, 2]
            + (0.5 * router_weights[:, 3] if self.router_include_both else 0.0),
            size=ms_coarse.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        residual_coarse = (
            sar_weight * sar_delta
            + collaborative_weight * collaborative_delta
        )
        residual = F.interpolate(
            residual_coarse,
            size=ms_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        logits = ms_logits + scale * residual

        sar_residual = F.interpolate(
            sar_delta,
            size=ms_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        collaborative_residual = F.interpolate(
            collaborative_delta,
            size=ms_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        both_residual = 0.5 * (sar_residual + collaborative_residual)
        sar_native_logits = (
            self.sar_classifier(aligned_private) * valid_channel
        )
        sar_logits = F.interpolate(
            sar_native_logits,
            size=ms_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        non_null = sar_weight + collaborative_weight
        return {
            "logits": logits,
            "ms_logits": ms_logits,
            "sar_logits": sar_logits,
            "sar_native_logits": sar_native_logits,
            "ms_shared": ms_shared,
            "sar_shared": aligned_sar,
            "sar_private": aligned_private,
            "sar_gate": non_null.mean(dim=1, keepdim=True),
            "sar_scale": scale.reshape(1),
            "sar_present": aligned_valid.flatten(1).any(dim=1),
            "router_logits": router_logits,
            "router_weights": router_weights,
            "router_region_valid": region_valid,
            "counterfactual_sar_logits": ms_logits + scale * sar_residual,
            "counterfactual_collab_logits": (
                ms_logits + scale * collaborative_residual
            ),
            "counterfactual_both_logits": ms_logits + scale * both_residual,
            "delta_sar_logits": sar_delta,
            "delta_collab_logits": collaborative_delta,
            "alignment_offsets": offsets,
        }


LAND_COVER_FAMILIES: tuple[tuple[int, ...], ...] = (
    (0, 11, 17, 18, 19),  # built-up: industrial/urban/rural/stadium/square
    (1, 2, 3, 4),  # cropland: paddy/irrigated/dry/garden
    (5, 6, 7),  # woody vegetation: arbor/shrub/park
    (8, 9),  # meadow: natural/artificial
    (10, 12, 13, 14),  # water: river/lake/pond/fish pond
    (15, 16),  # snow/bare land
    (20, 21, 22, 23),  # transport: road/overpass/railway/airport
)


def fine_to_family_map(
    families: Sequence[Sequence[int]] = LAND_COVER_FAMILIES,
) -> tuple[int, ...]:
    """Return the deterministic 24-class to coarse-family mapping."""
    mapping = [-1] * 24
    for family_id, class_ids in enumerate(families):
        for class_id in class_ids:
            if not 0 <= int(class_id) < len(mapping):
                raise ValueError(f"invalid fine class ID: {class_id}")
            if mapping[int(class_id)] >= 0:
                raise ValueError(f"fine class {class_id} occurs in two families")
            mapping[int(class_id)] = family_id
    if any(value < 0 for value in mapping):
        raise ValueError("coarse families must cover all 24 fine classes")
    return tuple(mapping)


class ConvNeXtFeatures(nn.Module):
    """ImageNet ConvNeXt adapted from RGB to GF-2 B/G/R/NIR.

    GF-2 files have a verified B,G,R,NIR band order.  The pretrained RGB
    filters are therefore copied as B<-B, G<-G, R<-R, while NIR starts from
    the RGB-filter mean.  This preserves the semantic initialization instead
    of silently feeding the bands in the wrong order.
    """

    _CHANNELS = {
        "tiny": (96, 192, 384, 768),
        "base": (128, 256, 512, 1024),
    }

    def __init__(
        self,
        *,
        variant: str,
        pretrained: bool,
        pretrained_file_sha256: str | None = None,
    ):
        super().__init__()
        variant = str(variant).lower()
        if variant not in self._CHANNELS:
            raise ValueError("ConvNeXt variant must be tiny or base")
        try:
            from torchvision.models import (
                ConvNeXt_Base_Weights,
                ConvNeXt_Tiny_Weights,
                convnext_base,
                convnext_tiny,
            )
        except ImportError as error:  # pragma: no cover - environment contract
            raise RuntimeError(
                "strong_ms_context requires torchvision compatible with torch"
            ) from error

        if variant == "tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            constructor = convnext_tiny
        else:
            weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            constructor = convnext_base
        expected_sha256 = (
            str(pretrained_file_sha256).lower()
            if pretrained_file_sha256 is not None
            else None
        )
        if expected_sha256 is not None:
            if (
                not pretrained
                or len(expected_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_sha256
                )
            ):
                raise ValueError(
                    "pretrained_file_sha256 requires pretrained=true and "
                    "64 hexadecimal characters"
                )
            assert weights is not None
            checkpoint = (
                Path(torch.hub.get_dir())
                / "checkpoints"
                / Path(urlparse(weights.url).path).name
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    "hash-bound ConvNeXt checkpoint is not cached: "
                    f"{checkpoint}"
                )
            digest = hashlib.sha256()
            with checkpoint.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            observed_sha256 = digest.hexdigest()
            if observed_sha256 != expected_sha256:
                raise ValueError(
                    "ConvNeXt pretrained checkpoint SHA256 differs from config: "
                    f"expected {expected_sha256}, observed {observed_sha256}"
                )
        network = constructor(weights=weights)
        self.channels = self._CHANNELS[variant]
        self.variant = variant
        old = network.features[0][0]
        replacement = nn.Conv2d(
            4,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=old.bias is not None,
        )
        with torch.no_grad():
            # torchvision RGB weights -> input B,G,R,NIR.
            replacement.weight[:, 0].copy_(old.weight[:, 2])
            replacement.weight[:, 1].copy_(old.weight[:, 1])
            replacement.weight[:, 2].copy_(old.weight[:, 0])
            replacement.weight[:, 3].copy_(old.weight.mean(dim=1))
            if old.bias is not None:
                replacement.bias.copy_(old.bias)
        network.features[0][0] = replacement
        self.features = network.features

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        value = inputs
        # Each stage is preceded by its stem/downsampling layer.
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in {1, 3, 5, 7}:
                outputs.append(value)
        return outputs


class ConvNeXtTinyFeatures(ConvNeXtFeatures):
    """Backward-compatible ConvNeXt-Tiny feature extractor."""

    channels = (96, 192, 384, 768)

    def __init__(self, *, pretrained: bool):
        super().__init__(variant="tiny", pretrained=pretrained)


class ContextFPNDecoder(nn.Module):
    """Fuse local detail with the geographically corresponding context crop."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        channels: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(value, channels, 1) for value in encoder_channels]
        )
        self.context_merge = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(channels * 2, channels, kernel_size=1),
                    ResidualBlock(channels),
                )
                for _ in encoder_channels
            ]
        )
        self.smooth = nn.ModuleList(
            [ResidualBlock(channels) for _ in encoder_channels]
        )
        self.fuse = nn.Sequential(
            ConvNormAct(channels * len(encoder_channels), channels),
            ResidualBlock(channels),
            ConvNormAct(channels, embedding_dim),
        )

    @staticmethod
    def _aligned_context(
        context: torch.Tensor,
        output_size: tuple[int, int],
        local_to_context_ratio: float,
    ) -> torch.Tensor:
        height, width = context.shape[-2:]
        crop_height = max(1, min(height, round(height * local_to_context_ratio)))
        crop_width = max(1, min(width, round(width * local_to_context_ratio)))
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        value = context[
            :, :, top : top + crop_height, left : left + crop_width
        ]
        return F.interpolate(
            value, size=output_size, mode="bilinear", align_corners=False
        )

    def forward(
        self,
        local_features: Sequence[torch.Tensor],
        context_features: Sequence[torch.Tensor],
        *,
        local_to_context_ratio: float,
    ) -> torch.Tensor:
        pyramid: list[torch.Tensor] = []
        for local, context, lateral, merge in zip(
            local_features,
            context_features,
            self.lateral,
            self.context_merge,
        ):
            local_value = lateral(local)
            context_value = lateral(
                self._aligned_context(
                    context, local.shape[-2:], local_to_context_ratio
                )
            )
            pyramid.append(merge(torch.cat([local_value, context_value], dim=1)))

        for index in range(len(pyramid) - 2, -1, -1):
            pyramid[index] = pyramid[index] + F.interpolate(
                pyramid[index + 1],
                size=pyramid[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pyramid = [smooth(value) for smooth, value in zip(self.smooth, pyramid)]
        output_size = pyramid[0].shape[-2:]
        return self.fuse(
            torch.cat(
                [
                    value
                    if value.shape[-2:] == output_size
                    else F.interpolate(
                        value,
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    for value in pyramid
                ],
                dim=1,
            )
        )


class LightweightContextPyramid(nn.Module):
    """Low-memory encoder for a wider, lower-resolution context view.

    The high-capacity ConvNeXt encoder is reserved for the local crop.  This
    branch keeps the complete surrounding view available without running a
    second ConvNeXt-Base and therefore avoids doubling its activation memory.
    """

    def __init__(self, width: int = 64):
        super().__init__()
        if width <= 0:
            raise ValueError("context encoder width must be positive")
        self.channels = tuple(width * (2**index) for index in range(4))
        self.stem = nn.Sequential(
            nn.Conv2d(4, width, kernel_size=4, stride=4, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            ResidualBlock(width),
        )
        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList([ResidualBlock(width)])
        for index in range(1, 4):
            in_channels = self.channels[index - 1]
            out_channels = self.channels[index]
            self.downsamples.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=2,
                        stride=2,
                        bias=False,
                    ),
                    nn.GroupNorm(_groups(out_channels), out_channels),
                    nn.GELU(),
                )
            )
            self.stages.append(
                nn.Sequential(
                    ResidualBlock(out_channels),
                    ResidualBlock(out_channels, dilation=2),
                )
            )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        value = self.stages[0](self.stem(inputs))
        outputs = [value]
        for downsample, stage in zip(self.downsamples, self.stages[1:]):
            value = stage(downsample(value))
            outputs.append(value)
        return outputs


class RegionalContextFPNDecoder(nn.Module):
    """Fuse local features, aligned center context and full-region tokens."""

    def __init__(
        self,
        local_channels: Sequence[int],
        context_channels: Sequence[int],
        channels: int,
        embedding_dim: int,
        *,
        context_token_grid: int = 4,
    ):
        super().__init__()
        if len(local_channels) != len(context_channels):
            raise ValueError("local and context pyramids must have equal depth")
        if context_token_grid <= 0:
            raise ValueError("context token grid must be positive")
        self.context_token_grid = int(context_token_grid)
        self.local_lateral = nn.ModuleList(
            [nn.Conv2d(value, channels, 1) for value in local_channels]
        )
        self.context_lateral = nn.ModuleList(
            [nn.Conv2d(value, channels, 1) for value in context_channels]
        )
        self.context_merge = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(channels * 3, channels, kernel_size=1),
                    ResidualBlock(channels),
                )
                for _ in local_channels
            ]
        )
        self.smooth = nn.ModuleList(
            [ResidualBlock(channels) for _ in local_channels]
        )
        self.fuse = nn.Sequential(
            ConvNormAct(channels * len(local_channels), channels),
            ResidualBlock(channels),
            ConvNormAct(channels, embedding_dim),
        )

    def forward(
        self,
        local_features: Sequence[torch.Tensor],
        context_features: Sequence[torch.Tensor],
        *,
        local_to_context_ratio: float,
    ) -> torch.Tensor:
        pyramid: list[torch.Tensor] = []
        for local, context, local_lateral, context_lateral, merge in zip(
            local_features,
            context_features,
            self.local_lateral,
            self.context_lateral,
            self.context_merge,
        ):
            local_value = local_lateral(local)
            context_value = context_lateral(context)
            aligned = ContextFPNDecoder._aligned_context(
                context_value,
                local_value.shape[-2:],
                local_to_context_ratio,
            )
            grid = min(
                self.context_token_grid,
                context_value.shape[-2],
                context_value.shape[-1],
            )
            regional = F.adaptive_avg_pool2d(context_value, (grid, grid))
            regional = F.interpolate(
                regional,
                size=local_value.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid.append(
                merge(torch.cat([local_value, aligned, regional], dim=1))
            )

        for index in range(len(pyramid) - 2, -1, -1):
            pyramid[index] = pyramid[index] + F.interpolate(
                pyramid[index + 1],
                size=pyramid[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pyramid = [smooth(value) for smooth, value in zip(self.smooth, pyramid)]
        output_size = pyramid[0].shape[-2:]
        return self.fuse(
            torch.cat(
                [
                    value
                    if value.shape[-2:] == output_size
                    else F.interpolate(
                        value,
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    for value in pyramid
                ],
                dim=1,
            )
        )


class RadiometricSpectralRefiner(nn.Module):
    """Refine fine classes from scaled GF-2 radiometry and band indices.

    The main ConvNeXt path is free to learn appearance and context.  This
    branch keeps a short, explicit path from B/G/R/NIR radiometry
    and vegetation/water indices to the within-family class decision.  Its
    output is zero-initialized so a warm-started strong-MS checkpoint has
    exactly the same prediction before fine-tuning.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        width: int,
        ms_mean: Sequence[float],
        ms_std: Sequence[float],
        max_logit_scale: float,
        initial_logit_scale: float,
    ):
        super().__init__()
        if len(ms_mean) != 4 or len(ms_std) != 4:
            raise ValueError(
                "radiometric spectral refinement requires four-band stats"
            )
        if width <= 0:
            raise ValueError("spectral refiner width must be positive")
        if not 0 < initial_logit_scale < max_logit_scale:
            raise ValueError(
                "initial spectral scale must be in (0, max_logit_scale)"
            )
        self.num_classes = int(num_classes)
        self.max_logit_scale = float(max_logit_scale)
        self.register_buffer(
            "ms_mean",
            torch.tensor(ms_mean, dtype=torch.float32).reshape(1, 4, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "ms_std",
            torch.tensor(ms_std, dtype=torch.float32).reshape(1, 4, 1, 1),
            persistent=True,
        )
        # Eight spectral channels: B/G/R/NIR, NDVI, NDWI, brightness and SAVI.
        self.encoder = nn.Sequential(
            ConvNormAct(8, width, kernel_size=7, stride=4),
            ResidualBlock(width),
            ResidualBlock(width, dilation=2),
        )
        self.context_fusion = nn.Sequential(
            ConvNormAct(width * 3, width, kernel_size=1),
            ResidualBlock(width),
        )
        self.classifier = nn.Conv2d(width, self.num_classes, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        initial_fraction = initial_logit_scale / max_logit_scale
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        self.family_scale_logits = nn.Parameter(
            torch.full((len(LAND_COVER_FAMILIES),), initial_logit)
        )

    @staticmethod
    def _normalized_difference(
        first: torch.Tensor, second: torch.Tensor
    ) -> torch.Tensor:
        denominator = first + second
        safe_denominator = torch.where(
            denominator.abs() < 1e-4,
            torch.full_like(denominator, 1e-4),
            denominator,
        )
        return ((first - second) / safe_denominator).clamp(-2.0, 2.0)

    def _spectral_channels(self, normalized_ms: torch.Tensor) -> torch.Tensor:
        reflectance = normalized_ms * self.ms_std + self.ms_mean
        blue, green, red, nir = reflectance.split(1, dim=1)
        ndvi = self._normalized_difference(nir, red)
        ndwi = self._normalized_difference(green, nir)
        brightness = reflectance.mean(dim=1, keepdim=True)
        savi = (1.5 * (nir - red) / (nir + red + 0.5).clamp_min(1e-4)).clamp(
            -2.0, 2.0
        )
        return torch.cat(
            [reflectance, ndvi, ndwi, brightness, savi], dim=1
        )

    def forward(
        self,
        normalized_ms: torch.Tensor,
        normalized_context: torch.Tensor,
        *,
        local_to_context_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.encoder(self._spectral_channels(normalized_ms))
        context = self.encoder(self._spectral_channels(normalized_context))
        aligned_context = ContextFPNDecoder._aligned_context(
            context,
            local.shape[-2:],
            local_to_context_ratio,
        )
        features = self.context_fusion(
            torch.cat(
                [local, aligned_context, (local - aligned_context).abs()],
                dim=1,
            )
        )
        residual = self.classifier(features)

        # Only redistribute probability among members of the same coarse
        # family.  Coarse-family evidence remains owned by the main model.
        centered = torch.zeros_like(residual)
        scales = torch.sigmoid(self.family_scale_logits) * self.max_logit_scale
        for class_ids in LAND_COVER_FAMILIES:
            indices = list(class_ids)
            family_residual = residual[:, indices]
            centered[:, indices] = (
                family_residual - family_residual.mean(dim=1, keepdim=True)
            )
        return centered, scales


class FamilySubtypeExpert(nn.Module):
    """Lightweight classifier dedicated to one coarse land-cover family."""

    def __init__(self, in_channels: int, width: int, num_children: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvNormAct(in_channels, width, kernel_size=1),
            ResidualBlock(width),
        )
        self.classifier = nn.Conv2d(width, num_children, 1)
        # A newly attached refiner must be an exact no-op for a warm-started
        # checkpoint.  The auxiliary loss still gives this layer gradients on
        # the first update; deeper expert layers start learning thereafter.
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class FamilyRegionSubtypeRefiner(nn.Module):
    """Use soft semantic regions to refine only sibling fine classes.

    Each family receives the local decoder embedding plus family-weighted
    context pooled at several spatial grids.  A family expert can redistribute
    logits among its children, but centering makes it unable to change the
    coarse-family evidence.  The final residual is also bounded and softly
    gated by the model's current family probability.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        width: int,
        region_grids: Sequence[int],
        max_logit_scale: float,
        initial_logit_scale: float,
        detach_family_gate: bool = True,
        scale_override: float | None = None,
        use_shape_features: bool = False,
        use_subtype_features: bool = False,
        uncertainty_margin_threshold: float | None = None,
        defer_uncertainty_gate_to_fine: bool = False,
    ):
        super().__init__()
        grids = tuple(int(value) for value in region_grids)
        if width <= 0:
            raise ValueError("family region refiner width must be positive")
        if not grids or any(value <= 0 for value in grids):
            raise ValueError("family region grids must be positive")
        if len(set(grids)) != len(grids):
            raise ValueError("family region grids must be unique")
        if not 0 < initial_logit_scale < max_logit_scale:
            raise ValueError(
                "initial family region scale must be in "
                "(0, max_logit_scale)"
            )
        if scale_override is not None and not (
            0.0 <= float(scale_override) <= max_logit_scale
        ):
            raise ValueError(
                "family region scale override must be in [0, max_logit_scale]"
            )
        if uncertainty_margin_threshold is not None and not (
            0.0 < float(uncertainty_margin_threshold) <= 1.0
        ):
            raise ValueError(
                "family region uncertainty margin threshold must be in (0, 1]"
            )
        if defer_uncertainty_gate_to_fine and uncertainty_margin_threshold is None:
            raise ValueError(
                "deferred family uncertainty gating requires a margin threshold"
            )
        self.region_grids = grids
        self.max_logit_scale = float(max_logit_scale)
        self.detach_family_gate = bool(detach_family_gate)
        self.use_shape_features = bool(use_shape_features)
        self.use_subtype_features = bool(use_subtype_features)
        self.uncertainty_margin_threshold = (
            None
            if uncertainty_margin_threshold is None
            else float(uncertainty_margin_threshold)
        )
        self.defer_uncertainty_gate_to_fine = bool(
            defer_uncertainty_gate_to_fine
        )
        self.scale_override = (
            None if scale_override is None else float(scale_override)
        )
        # Per grid, use the pooled context and its absolute contrast with the
        # local embedding.  Experts are processed sequentially to avoid a
        # B x family x embedding x H x W activation tensor.
        expert_channels = int(embedding_dim) * (1 + 2 * len(grids))
        # Shape channels describe the soft family occupancy, its local edge,
        # and coarse-to-local occupancy contrast at every configured scale.
        # They deliberately contain no ground-truth geometry at inference.
        shape_channels = 2 + 2 * len(grids) if self.use_shape_features else 0
        self.experts = nn.ModuleList(
            [
                FamilySubtypeExpert(
                    expert_channels
                    + shape_channels
                    + (len(class_ids) + 2 if self.use_subtype_features else 0),
                    int(width),
                    len(class_ids),
                )
                for class_ids in LAND_COVER_FAMILIES
            ]
        )
        initial_fraction = initial_logit_scale / max_logit_scale
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        self.family_scale_logits = nn.Parameter(
            torch.full((len(LAND_COVER_FAMILIES),), initial_logit)
        )

    @staticmethod
    def _soft_region_context(
        embeddings: torch.Tensor,
        family_gate: torch.Tensor,
        grid: int,
    ) -> torch.Tensor:
        pooled_gate = F.adaptive_avg_pool2d(family_gate, (grid, grid))
        pooled_embedding = F.adaptive_avg_pool2d(
            embeddings * family_gate,
            (grid, grid),
        )
        context = pooled_embedding / pooled_gate.clamp_min(1e-6)
        return F.interpolate(
            context,
            size=embeddings.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def apply_fine_uncertainty_gate(
        residual: torch.Tensor,
        fine_logits: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Gate an upsampled residual on the exact final-resolution margin."""

        if residual.shape != fine_logits.shape or residual.shape[1] != 24:
            raise ValueError(
                "fine uncertainty gating requires matching 24-class tensors"
            )
        if not 0.0 < float(threshold) <= 1.0:
            raise ValueError("fine uncertainty threshold must be in (0, 1]")
        gated = torch.zeros_like(residual)
        for class_ids in LAND_COVER_FAMILIES:
            indices = list(class_ids)
            probability = fine_logits[:, indices].detach().softmax(dim=1)
            top_two = probability.topk(2, dim=1).values
            margin = top_two[:, :1] - top_two[:, 1:2]
            uncertainty = (
                (float(threshold) - margin) / float(threshold)
            ).clamp_(0.0, 1.0)
            gated[:, indices] = residual[:, indices] * uncertainty
        return gated

    def forward(
        self,
        embeddings: torch.Tensor,
        family_logits: torch.Tensor,
        fine_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if embeddings.shape[-2:] != family_logits.shape[-2:]:
            raise ValueError("family logits must share the embedding grid")
        if (
            self.use_subtype_features
            or self.uncertainty_margin_threshold is not None
        ) and fine_logits is None:
            raise ValueError(
                "fine logits are required when subtype features or uncertainty "
                "gating are enabled"
            )
        if fine_logits is not None and fine_logits.shape[-2:] != embeddings.shape[-2:]:
            raise ValueError("fine logits must share the embedding grid")
        family_probability = family_logits.softmax(dim=1)
        family_gate = (
            family_probability.detach()
            if self.detach_family_gate
            else family_probability
        )
        if self.scale_override is None:
            scales = (
                torch.sigmoid(self.family_scale_logits)
                * self.max_logit_scale
            )
        else:
            scales = self.family_scale_logits.new_full(
                (len(LAND_COVER_FAMILIES),),
                self.scale_override,
            )

        batch, _, height, width = embeddings.shape
        # Classifier convolutions may run in FP16/BF16 under autocast while
        # frozen decoder embeddings remain FP32.  Anchor the assembled logits
        # to family_logits and cast each expert result before indexed writes.
        raw_residual = family_logits.new_zeros(
            (batch, 24, height, width)
        )
        scaled_residual = torch.zeros_like(raw_residual)
        effective_gate_means: list[torch.Tensor] = []
        for family_id, (class_ids, expert) in enumerate(
            zip(LAND_COVER_FAMILIES, self.experts)
        ):
            gate = family_gate[:, family_id : family_id + 1]
            expert_inputs = [embeddings]
            occupancy_maps: list[torch.Tensor] = []
            for grid in self.region_grids:
                context = self._soft_region_context(
                    embeddings,
                    gate,
                    grid,
                )
                expert_inputs.extend([context, (embeddings - context).abs()])
                if self.use_shape_features:
                    occupancy = F.interpolate(
                        F.adaptive_avg_pool2d(gate, (grid, grid)),
                        size=embeddings.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    occupancy_maps.extend(
                        [occupancy, (gate - occupancy).abs()]
                    )
            if self.use_shape_features:
                local_edge = (
                    gate - F.avg_pool2d(gate, 3, stride=1, padding=1)
                ).abs()
                expert_inputs.extend([gate, local_edge, *occupancy_maps])
            subtype_probability: torch.Tensor | None = None
            confidence_margin: torch.Tensor | None = None
            if (
                self.use_subtype_features
                or self.uncertainty_margin_threshold is not None
            ):
                assert fine_logits is not None
                indices = list(class_ids)
                subtype_probability = (
                    fine_logits[:, indices].detach().softmax(dim=1)
                )
                top_two = subtype_probability.topk(2, dim=1).values
                confidence_margin = top_two[:, :1] - top_two[:, 1:2]
            if self.use_subtype_features:
                assert subtype_probability is not None
                assert confidence_margin is not None
                entropy = -(
                    subtype_probability
                    * subtype_probability.clamp_min(1e-6).log()
                ).sum(dim=1, keepdim=True) / math.log(len(indices))
                expert_inputs.extend(
                    [subtype_probability, entropy, confidence_margin]
                )
            effective_gate = gate
            reported_gate = gate
            if self.uncertainty_margin_threshold is not None:
                assert confidence_margin is not None
                # Preserve confident Base decisions exactly.  Corrections fade
                # linearly to zero as the frozen sibling Top-2 margin reaches
                # the configured threshold and remain active only where the
                # Base classifier is genuinely ambiguous.
                threshold = self.uncertainty_margin_threshold
                uncertainty_gate = (
                    (threshold - confidence_margin) / threshold
                ).clamp_(0.0, 1.0)
                reported_gate = gate * uncertainty_gate
                if not self.defer_uncertainty_gate_to_fine:
                    effective_gate = reported_gate
            residual = expert(torch.cat(expert_inputs, dim=1)).to(
                raw_residual.dtype
            )
            residual = residual - residual.mean(dim=1, keepdim=True)
            indices = list(class_ids)
            raw_residual[:, indices] = residual
            scaled_residual[:, indices] = (
                residual
                * effective_gate.to(residual.dtype)
                * scales[family_id].to(residual.dtype)
            )
            effective_gate_means.append(reported_gate.mean())

        gate_means = torch.stack(effective_gate_means)
        return raw_residual, scaled_residual, scales, gate_means


class StrongMSContextSegmentation(nn.Module):
    """Strong MS anchor with large context, hierarchy, boundary and prototypes.

    This is intentionally an MS-only semantic anchor.  It establishes whether
    the 24-class problem is learnable with a capable pretrained representation
    before SAR is allowed to contribute at feature level in the next ablation.
    """

    def __init__(
        self,
        *,
        num_classes: int = 24,
        pretrained: bool = True,
        decoder_channels: int = 192,
        embedding_dim: int = 96,
        prototypes_per_class: int = 3,
        prototype_temperature: float = 10.0,
        max_prototype_logit_scale: float = 0.25,
        local_to_context_ratio: float = 0.5,
        enable_spectral_refinement: bool = False,
        spectral_width: int = 64,
        max_spectral_logit_scale: float = 0.5,
        initial_spectral_logit_scale: float = 0.1,
        enable_family_region_refinement: bool = False,
        family_region_width: int = 64,
        family_region_grids: Sequence[int] = (1, 2, 4),
        max_family_region_logit_scale: float = 0.5,
        initial_family_region_logit_scale: float = 0.1,
        detach_family_region_gate: bool = True,
        family_region_scale_override: float | None = None,
        family_region_use_shape_features: bool = False,
        family_region_use_subtype_features: bool = False,
        family_region_uncertainty_margin_threshold: float | None = None,
        family_region_condition_on_spectral_logits: bool = False,
        family_region_fine_uncertainty_gate: bool = False,
        ms_mean: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        ms_std: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        if num_classes != 24:
            raise ValueError("strong MS hierarchy is defined for 24 classes")
        if prototypes_per_class <= 0:
            raise ValueError("prototypes_per_class must be positive")
        if not 0 < local_to_context_ratio <= 1:
            raise ValueError("local_to_context_ratio must be in (0, 1]")
        self.num_classes = num_classes
        self.num_families = len(LAND_COVER_FAMILIES)
        self.local_to_context_ratio = float(local_to_context_ratio)
        self.prototype_temperature = float(prototype_temperature)
        self.max_prototype_logit_scale = float(max_prototype_logit_scale)
        self.enable_spectral_refinement = bool(enable_spectral_refinement)
        self.enable_family_region_refinement = bool(
            enable_family_region_refinement
        )
        self.family_region_condition_on_spectral_logits = bool(
            family_region_condition_on_spectral_logits
        )
        self.family_region_fine_uncertainty_gate = bool(
            family_region_fine_uncertainty_gate
        )

        self.backbone = ConvNeXtTinyFeatures(pretrained=pretrained)
        self.decoder = ContextFPNDecoder(
            self.backbone.channels,
            int(decoder_channels),
            int(embedding_dim),
        )
        self.fine_classifier = nn.Conv2d(embedding_dim, num_classes, 1)
        self.family_classifier = nn.Conv2d(
            embedding_dim, self.num_families, 1
        )
        self.boundary_classifier = nn.Conv2d(embedding_dim, 1, 1)
        self.prototypes = nn.Parameter(
            torch.empty(num_classes, prototypes_per_class, embedding_dim)
        )
        nn.init.normal_(self.prototypes, std=0.02)
        initial_fraction = min(
            max(0.10 / max(self.max_prototype_logit_scale, 1e-6), 1e-4),
            1.0 - 1e-4,
        )
        self.prototype_scale_logit = nn.Parameter(
            torch.tensor(math.log(initial_fraction / (1.0 - initial_fraction)))
        )
        self.spectral_refiner = (
            RadiometricSpectralRefiner(
                num_classes=num_classes,
                width=int(spectral_width),
                ms_mean=ms_mean,
                ms_std=ms_std,
                max_logit_scale=float(max_spectral_logit_scale),
                initial_logit_scale=float(initial_spectral_logit_scale),
            )
            if self.enable_spectral_refinement
            else None
        )
        self.family_region_refiner = (
            FamilyRegionSubtypeRefiner(
                embedding_dim=int(embedding_dim),
                width=int(family_region_width),
                region_grids=family_region_grids,
                max_logit_scale=float(max_family_region_logit_scale),
                initial_logit_scale=float(initial_family_region_logit_scale),
                detach_family_gate=bool(detach_family_region_gate),
                scale_override=family_region_scale_override,
                use_shape_features=bool(family_region_use_shape_features),
                use_subtype_features=bool(family_region_use_subtype_features),
                uncertainty_margin_threshold=(
                    family_region_uncertainty_margin_threshold
                ),
                defer_uncertainty_gate_to_fine=(
                    self.family_region_fine_uncertainty_gate
                ),
            )
            if self.enable_family_region_refinement
            else None
        )
        mapping = fine_to_family_map()
        self.register_buffer(
            "fine_to_family",
            torch.tensor(mapping, dtype=torch.long),
            persistent=True,
        )

    @classmethod
    def from_config(cls, config: dict) -> "StrongMSContextSegmentation":
        model = config["model"]
        data = config["data"]
        context_size = int(data.get("context_size", data["patch_size"]))
        return cls(
            num_classes=int(model["num_classes"]),
            pretrained=bool(model.get("pretrained", True)),
            decoder_channels=int(model.get("decoder_channels", 192)),
            embedding_dim=int(model.get("embedding_dim", 96)),
            prototypes_per_class=int(model.get("prototypes_per_class", 3)),
            prototype_temperature=float(
                model.get("prototype_temperature", 10.0)
            ),
            max_prototype_logit_scale=float(
                model.get("max_prototype_logit_scale", 0.25)
            ),
            local_to_context_ratio=float(data["patch_size"]) / context_size,
            enable_spectral_refinement=bool(
                model.get("enable_spectral_refinement", False)
            ),
            spectral_width=int(model.get("spectral_width", 64)),
            max_spectral_logit_scale=float(
                model.get("max_spectral_logit_scale", 0.5)
            ),
            initial_spectral_logit_scale=float(
                model.get("initial_spectral_logit_scale", 0.1)
            ),
            enable_family_region_refinement=bool(
                model.get("enable_family_region_refinement", False)
            ),
            family_region_width=int(model.get("family_region_width", 64)),
            family_region_grids=model.get(
                "family_region_grids", [1, 2, 4]
            ),
            max_family_region_logit_scale=float(
                model.get("max_family_region_logit_scale", 0.5)
            ),
            initial_family_region_logit_scale=float(
                model.get("initial_family_region_logit_scale", 0.1)
            ),
            detach_family_region_gate=bool(
                model.get("detach_family_region_gate", True)
            ),
            family_region_scale_override=(
                None
                if model.get("family_region_scale_override") is None
                else float(model["family_region_scale_override"])
            ),
            family_region_use_shape_features=bool(
                model.get("family_region_use_shape_features", False)
            ),
            family_region_use_subtype_features=bool(
                model.get("family_region_use_subtype_features", False)
            ),
            family_region_uncertainty_margin_threshold=(
                None
                if model.get(
                    "family_region_uncertainty_margin_threshold"
                ) is None
                else float(
                    model["family_region_uncertainty_margin_threshold"]
                )
            ),
            family_region_condition_on_spectral_logits=bool(
                model.get(
                    "family_region_condition_on_spectral_logits", False
                )
            ),
            family_region_fine_uncertainty_gate=bool(
                model.get("family_region_fine_uncertainty_gate", False)
            ),
            ms_mean=data["normalization"]["ms"]["mean"],
            ms_std=data["normalization"]["ms"]["std"],
        )

    def _prototype_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized_embeddings = F.normalize(embeddings, dim=1)
        normalized_prototypes = F.normalize(self.prototypes, dim=-1)
        similarities = torch.einsum(
            "behw,cke->bckhw", normalized_embeddings, normalized_prototypes
        )
        return torch.logsumexp(
            similarities * self.prototype_temperature, dim=2
        ) - math.log(self.prototypes.shape[1])

    def _compose_hierarchy(
        self, fine_logits: torch.Tensor, family_logits: torch.Tensor
    ) -> torch.Tensor:
        # Frozen branches can retain FP32 outputs under AMP while the newly
        # trainable FCR path is autocast to FP16/BF16. Indexed assignment does
        # not perform implicit casts, so make the hierarchy contract explicit.
        family_log_probability = F.log_softmax(
            family_logits, dim=1
        ).to(fine_logits.dtype)
        output = torch.empty_like(fine_logits)
        for family_id, class_ids in enumerate(LAND_COVER_FAMILIES):
            indices = list(class_ids)
            child_log_probability = F.log_softmax(
                fine_logits[:, indices], dim=1
            ).to(fine_logits.dtype)
            output[:, indices] = (
                child_log_probability
                + family_log_probability[:, family_id : family_id + 1]
            )
        return output

    def _decode_embeddings(
        self,
        ms: torch.Tensor,
        ms_context: torch.Tensor,
    ) -> torch.Tensor:
        local_features = self.backbone(ms)
        context_features = self.backbone(ms_context)
        return self.decoder(
            local_features,
            context_features,
            local_to_context_ratio=self.local_to_context_ratio,
        )

    def forward(
        self,
        ms: torch.Tensor,
        sar: torch.Tensor,
        sar_valid: torch.Tensor,
        time_gap_days: torch.Tensor | None = None,
        ms_context: torch.Tensor | None = None,
        ms_context_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del time_gap_days
        if ms_context is None:
            ms_context = ms
        if ms_context_valid is not None:
            ms_context = ms_context * ms_context_valid[:, None].to(ms.dtype)
        embeddings = self._decode_embeddings(ms, ms_context)
        flat_logits = self.fine_classifier(embeddings)
        if self.max_prototype_logit_scale > 0:
            prototype_logits = self._prototype_logits(embeddings)
            prototype_scale = (
                torch.sigmoid(self.prototype_scale_logit)
                * self.max_prototype_logit_scale
            )
            flat_logits = flat_logits + prototype_scale * prototype_logits
        else:
            prototype_scale = flat_logits.new_zeros(())
        family_logits = self.family_classifier(embeddings)
        family_region_child_logits: torch.Tensor | None = None
        family_region_scales: torch.Tensor | None = None
        family_region_gate_means: torch.Tensor | None = None
        family_region_baseline_flat_logits: torch.Tensor | None = None
        family_region_refined_logits: torch.Tensor | None = None
        family_region_baseline_logits: torch.Tensor | None = None
        family_region_refined_flat_logits: torch.Tensor | None = None
        family_region_fine_residual: torch.Tensor | None = None
        spectral_residual_coarse: torch.Tensor | None = None
        spectral_scales: torch.Tensor | None = None
        scaled_spectral_residual_coarse: torch.Tensor | None = None
        if self.spectral_refiner is not None:
            spectral_residual_coarse, spectral_scales = self.spectral_refiner(
                ms,
                ms_context,
                local_to_context_ratio=self.local_to_context_ratio,
            )
            scaled_spectral_residual_coarse = torch.zeros_like(
                spectral_residual_coarse
            )
            for family_id, class_ids in enumerate(LAND_COVER_FAMILIES):
                indices = list(class_ids)
                scaled_spectral_residual_coarse[:, indices] = (
                    spectral_residual_coarse[:, indices]
                    * spectral_scales[family_id]
                )
        if self.family_region_refiner is not None:
            family_region_baseline_flat_logits = flat_logits.detach()
            family_region_condition_logits = flat_logits
            if self.family_region_condition_on_spectral_logits:
                if scaled_spectral_residual_coarse is None:
                    raise RuntimeError(
                        "spectral-conditioned family refinement requires the "
                        "spectral refiner"
                    )
                family_region_condition_logits = (
                    flat_logits + scaled_spectral_residual_coarse.detach()
                )
            (
                family_region_residual,
                scaled_family_region_residual,
                family_region_scales,
                family_region_gate_means,
            ) = self.family_region_refiner(
                embeddings,
                family_logits,
                family_region_condition_logits,
            )
            # This low-resolution detached view gives each expert an explicit
            # sibling-only objective without retaining another full-size logit
            # tensor.  The bounded path below is what affects inference.
            family_region_child_logits = (
                family_region_condition_logits.detach()
                + family_region_residual
            )
            family_region_refined_flat_logits = (
                flat_logits + scaled_family_region_residual
            )
            if self.family_region_fine_uncertainty_gate:
                family_region_fine_residual = scaled_family_region_residual
            else:
                flat_logits = family_region_refined_flat_logits
        size = ms.shape[-2:]
        flat_logits_fine = F.interpolate(
            flat_logits, size=size, mode="bilinear", align_corners=False
        )
        family_logits_fine = F.interpolate(
            family_logits, size=size, mode="bilinear", align_corners=False
        )
        spectral_aux_logits: torch.Tensor | None = None
        if spectral_residual_coarse is not None:
            assert spectral_scales is not None
            assert scaled_spectral_residual_coarse is not None
            if family_region_baseline_flat_logits is not None:
                family_region_baseline_logits = self._compose_hierarchy(
                    family_region_baseline_flat_logits
                    + scaled_spectral_residual_coarse.detach(),
                    family_logits.detach(),
                )
                family_region_refined_logits = self._compose_hierarchy(
                    family_region_refined_flat_logits
                    + scaled_spectral_residual_coarse,
                    family_logits,
                )
            spectral_residual = F.interpolate(
                spectral_residual_coarse,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            scaled_spectral_residual = torch.zeros_like(spectral_residual)
            for family_id, class_ids in enumerate(LAND_COVER_FAMILIES):
                indices = list(class_ids)
                scaled_spectral_residual[:, indices] = (
                    spectral_residual[:, indices] * spectral_scales[family_id]
                )
            # The final path is bounded by family-specific scales.  The
            # detached auxiliary view teaches the new branch to correct the
            # existing checkpoint without moving the old classifier itself.
            spectral_aux_logits = self._compose_hierarchy(
                flat_logits_fine.detach() + spectral_residual,
                family_logits_fine.detach(),
            )
            flat_logits_fine = flat_logits_fine + scaled_spectral_residual
        elif family_region_baseline_flat_logits is not None:
            family_region_baseline_logits = self._compose_hierarchy(
                family_region_baseline_flat_logits,
                family_logits.detach(),
            )
            family_region_refined_logits = self._compose_hierarchy(
                family_region_refined_flat_logits, family_logits
            )
        if family_region_fine_residual is not None:
            assert self.family_region_refiner is not None
            threshold = (
                self.family_region_refiner.uncertainty_margin_threshold
            )
            if threshold is None:
                raise RuntimeError(
                    "fine family uncertainty gate requires a threshold"
                )
            fine_residual = F.interpolate(
                family_region_fine_residual,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            fine_residual = (
                self.family_region_refiner.apply_fine_uncertainty_gate(
                    fine_residual,
                    flat_logits_fine,
                    threshold,
                )
            )
            flat_logits_fine = flat_logits_fine + fine_residual
        logits = self._compose_hierarchy(
            flat_logits_fine, family_logits_fine
        )
        boundary_logits = F.interpolate(
            self.boundary_classifier(embeddings),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        sar_native_logits = torch.zeros(
            ms.shape[0],
            self.num_classes,
            sar.shape[-2],
            sar.shape[-1],
            dtype=ms.dtype,
            device=ms.device,
        )
        zero_shared = torch.zeros_like(embeddings)
        outputs = {
            "logits": logits,
            "ms_logits": logits,
            "sar_logits": torch.zeros_like(logits),
            "sar_native_logits": sar_native_logits,
            "ms_shared": embeddings,
            "sar_shared": zero_shared,
            "sar_private": zero_shared,
            "sar_gate": zero_shared[:, :1],
            "sar_scale": torch.zeros(1, dtype=ms.dtype, device=ms.device),
            "sar_present": torch.zeros(
                ms.shape[0], dtype=torch.bool, device=ms.device
            ),
            "flat_logits": flat_logits_fine,
            "family_logits": family_logits_fine,
            "boundary_logits": boundary_logits,
            "region_embeddings": embeddings,
            "class_prototypes": self.prototypes,
            "fine_to_family": self.fine_to_family,
            "prototype_scale": prototype_scale.reshape(1),
        }
        if spectral_aux_logits is not None and spectral_scales is not None:
            outputs["spectral_aux_logits"] = spectral_aux_logits
            outputs["spectral_scales"] = spectral_scales
        if (
            family_region_child_logits is not None
            and family_region_scales is not None
            and family_region_gate_means is not None
        ):
            outputs["family_region_child_logits"] = (
                family_region_child_logits
            )
            outputs["family_region_scales"] = family_region_scales
            outputs["family_region_gate_means"] = family_region_gate_means
            assert family_region_baseline_logits is not None
            assert family_region_refined_logits is not None
            outputs["family_region_baseline_logits"] = (
                family_region_baseline_logits
            )
            outputs["family_region_refined_logits"] = (
                family_region_refined_logits
            )
        return outputs


class StrongMS70Segmentation(StrongMSContextSegmentation):
    """Higher-capacity pure-MS anchor for the official FBP 120/30 protocol.

    A ConvNeXt-Base local path learns detailed appearance.  A deliberately
    lightweight context pyramid processes the complete medium-range crop and
    contributes both geographically aligned center features and coarse
    regional tokens.  This keeps Line3 focused on spectral and family-aware
    fine classification rather than whole-scene expert routing.
    """

    def __init__(
        self,
        *,
        backbone_variant: str = "base",
        context_encoder_width: int = 64,
        context_token_grid: int = 4,
        **kwargs,
    ):
        pretrained = bool(kwargs.pop("pretrained", True))
        super().__init__(pretrained=False, **kwargs)
        self.backbone = ConvNeXtFeatures(
            variant=backbone_variant,
            pretrained=pretrained,
        )
        self.context_backbone = LightweightContextPyramid(
            int(context_encoder_width)
        )
        self.decoder = RegionalContextFPNDecoder(
            self.backbone.channels,
            self.context_backbone.channels,
            int(kwargs.get("decoder_channels", 192)),
            int(kwargs.get("embedding_dim", 96)),
            context_token_grid=int(context_token_grid),
        )

    @classmethod
    def from_config(cls, config: dict) -> "StrongMS70Segmentation":
        model = config["model"]
        data = config["data"]
        context_size = int(data.get("context_size", data["patch_size"]))
        return cls(
            num_classes=int(model["num_classes"]),
            pretrained=bool(model.get("pretrained", True)),
            backbone_variant=str(model.get("backbone_variant", "base")),
            context_encoder_width=int(
                model.get("context_encoder_width", 64)
            ),
            context_token_grid=int(model.get("context_token_grid", 4)),
            decoder_channels=int(model.get("decoder_channels", 192)),
            embedding_dim=int(model.get("embedding_dim", 96)),
            prototypes_per_class=int(model.get("prototypes_per_class", 3)),
            prototype_temperature=float(
                model.get("prototype_temperature", 10.0)
            ),
            max_prototype_logit_scale=float(
                model.get("max_prototype_logit_scale", 0.25)
            ),
            local_to_context_ratio=float(data["patch_size"]) / context_size,
            enable_spectral_refinement=bool(
                model.get("enable_spectral_refinement", True)
            ),
            spectral_width=int(model.get("spectral_width", 64)),
            max_spectral_logit_scale=float(
                model.get("max_spectral_logit_scale", 0.5)
            ),
            initial_spectral_logit_scale=float(
                model.get("initial_spectral_logit_scale", 0.1)
            ),
            enable_family_region_refinement=bool(
                model.get("enable_family_region_refinement", False)
            ),
            family_region_width=int(model.get("family_region_width", 64)),
            family_region_grids=model.get(
                "family_region_grids", [1, 2, 4]
            ),
            max_family_region_logit_scale=float(
                model.get("max_family_region_logit_scale", 0.5)
            ),
            initial_family_region_logit_scale=float(
                model.get("initial_family_region_logit_scale", 0.1)
            ),
            detach_family_region_gate=bool(
                model.get("detach_family_region_gate", True)
            ),
            family_region_scale_override=(
                None
                if model.get("family_region_scale_override") is None
                else float(model["family_region_scale_override"])
            ),
            family_region_use_shape_features=bool(
                model.get("family_region_use_shape_features", False)
            ),
            family_region_use_subtype_features=bool(
                model.get("family_region_use_subtype_features", False)
            ),
            family_region_uncertainty_margin_threshold=(
                None
                if model.get(
                    "family_region_uncertainty_margin_threshold"
                ) is None
                else float(
                    model["family_region_uncertainty_margin_threshold"]
                )
            ),
            family_region_condition_on_spectral_logits=bool(
                model.get(
                    "family_region_condition_on_spectral_logits", False
                )
            ),
            family_region_fine_uncertainty_gate=bool(
                model.get("family_region_fine_uncertainty_gate", False)
            ),
            ms_mean=data["normalization"]["ms"]["mean"],
            ms_std=data["normalization"]["ms"]["std"],
        )

    def _decode_embeddings(
        self,
        ms: torch.Tensor,
        ms_context: torch.Tensor,
    ) -> torch.Tensor:
        local_features = self.backbone(ms)
        context_features = self.context_backbone(ms_context)
        return self.decoder(
            local_features,
            context_features,
            local_to_context_ratio=self.local_to_context_ratio,
        )


def build_model(config: dict) -> nn.Module:
    architecture = str(config["model"].get("architecture", "ms_sar_fusion"))
    if architecture == "hrnet_w32_ocr":
        from .hrnet_ocr_model import HRNetW32OCRSegmentation

        return HRNetW32OCRSegmentation.from_config(config)
    if architecture == "local_detail_cnxb":
        from .local_detail_model import LocalDetailCNXBSegmentation

        return LocalDetailCNXBSegmentation.from_config(config)
    if architecture == "ms75_multiscale":
        # Kept in a separate module so the established Line2/Line3 model
        # checkpoint namespaces remain untouched.
        from .ms75_model import MS75Segmentation

        return MS75Segmentation.from_config(config)
    if architecture == "ms75_oof_aligned_swin":
        from .ms75_aligned_proxy import MS75AlignedSwinProxy

        return MS75AlignedSwinProxy.from_config(config)
    if architecture == "ms_sar_fusion":
        return MSGuidedRegionFusion.from_config(config)
    if architecture == "sar_native":
        return SARNativeSegmentation.from_config(config)
    if architecture == "symm_safe_fusion":
        return SymmSafeFusion.from_config(config)
    if architecture == "strong_ms_context":
        return StrongMSContextSegmentation.from_config(config)
    if architecture == "ms70_context":
        return StrongMS70Segmentation.from_config(config)
    raise ValueError(f"unsupported model architecture: {architecture}")
