"""Pure-MS Line3-MS75 model with the complete CASTR 70.1564 anchor.

The complete local/context/structure/learned-router graph deliberately mirrors
the archived ``fbponlyms`` checkpoint.  This module performs no checkpoint or
filesystem I/O; callers explicitly provide already-loaded state dictionaries.

Line3 arrays are B/G/R/NIR.  The anchor was trained with NIR/R/G/B tensors, so
both exact-anchor inputs are reordered before encoding.  The newly trained
short and spectral paths continue to receive B/G/R/NIR.  The exact CASTR
context pyramid is reused as the Line3 whole-scene feature view.

Portions of ``ConvNormAct``, ``TinyPyramidEncoder``,
``TimmPyramidEncoder``, ``PyramidPooling``, ``UPerDecoder``, and the complete
CASTR context/structure/learned-router forward are adapted from fbponlyms
(Copyright (c) 2026 MiaoKiyama), under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .ms75_castr import (
    AnchorContextState,
    CrossEvidenceFusion,
    EvidenceRouter,
    SparseStructureBlock,
)


LAND_COVER_FAMILIES: tuple[tuple[int, ...], ...] = (
    (0, 11, 17, 18, 19),  # built-up
    (1, 2, 3, 4),  # cropland
    (5, 6, 7),  # woody vegetation
    (8, 9),  # meadow
    (10, 12, 13, 14),  # water
    (15, 16),  # snow / bare land
    (20, 21, 22, 23),  # transport
)

# Exact five-group order used by the archived CASTR structure expert.  It is
# intentionally separate from Line3's seven-family classification hierarchy.
CASTR_STRUCTURE_GROUPS: dict[str, tuple[int, ...]] = {
    "water": (10, 12, 13, 14),
    "vegetation": (1, 2, 3, 4, 5, 6, 7, 8, 9),
    "built": (0, 11, 17, 18, 19),
    "transport": (20, 21, 22, 23),
    "snow_bare": (15, 16),
}


class _ProjectedGradientClamp(torch.autograd.Function):
    """Clamp in forward and discard only optimizer-outward boundary gradients.

    A conventional straight-through clamp preserves *both* gradient directions
    at a boundary.  With an exact-zero residual gate, a temporarily harmful new
    branch therefore makes Adam accumulate a negative raw parameter even though
    the forward value remains zero.  The accumulated distance can keep the
    branch disabled long after its auxiliary objective has made it useful.

    This projected-gradient variant keeps gradients that move the raw value
    back into the legal interval and suppresses gradients whose optimizer step
    would move it farther out.  It consequently preserves the exact-zero anchor
    identity without creating an invisible negative (or positive) gate debt.
    """

    @staticmethod
    def forward(
        ctx: Any,
        value: torch.Tensor,
        minimum: float,
        maximum: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.minimum = float(minimum)
        ctx.maximum = float(maximum)
        return value.clamp(min=ctx.minimum, max=ctx.maximum)

    @staticmethod
    def backward(
        ctx: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (value,) = ctx.saved_tensors
        # Optimizers subtract the gradient.  At the lower boundary a positive
        # gradient would therefore move the raw value below the interval; at
        # the upper boundary a negative gradient would move it above it.
        outward = ((value <= ctx.minimum) & (gradient > 0)) | (
            (value >= ctx.maximum) & (gradient < 0)
        )
        return gradient.masked_fill(outward, 0.0), None, None


def _straight_through_clamp(
    value: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Clamp the forward value with a boundary-safe recovery gradient.

    MS75 deliberately initializes the context residual and hierarchy mixture
    at the exact anchor identity (zero).  A regular clamp can permanently
    strand either raw parameter just below zero after an early noisy update,
    because the out-of-range derivative is then zero.  The straight-through
    projected-gradient form keeps the same bounded forward contract, prevents
    new out-of-range debt, and lets later evidence move a temporarily saturated
    parameter back into the valid interval.
    """

    return _ProjectedGradientClamp.apply(value, float(minimum), float(maximum))


def fine_to_family_map(
    families: Sequence[Sequence[int]] = LAND_COVER_FAMILIES,
) -> tuple[int, ...]:
    """Validate and return the deterministic 24-class family mapping."""

    mapping = [-1] * 24
    for family_id, class_ids in enumerate(families):
        for class_id_value in class_ids:
            class_id = int(class_id_value)
            if not 0 <= class_id < len(mapping):
                raise ValueError(f"invalid fine class ID: {class_id}")
            if mapping[class_id] >= 0:
                raise ValueError(f"fine class {class_id} occurs in two families")
            mapping[class_id] = family_id
    if any(value < 0 for value in mapping):
        raise ValueError("coarse families must cover all 24 fine classes")
    return tuple(mapping)


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2, 1):
        if channels % value == 0 and channels // value >= 2:
            return value
    return 1


class ConvNormAct(nn.Sequential):
    """Convolution used by the checkpoint-compatible local decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        if padding is None:
            padding = dilation * (kernel_size // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class TinyPyramidEncoder(nn.Module):
    """Four-stage offline encoder for CPU smoke tests."""

    def __init__(
        self,
        in_channels: int = 4,
        channels: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        current = int(in_channels)
        for channel in channels:
            stages.append(
                nn.Sequential(
                    ConvNormAct(current, int(channel), 3, stride=2),
                    ConvNormAct(int(channel), int(channel), 3),
                )
            )
            current = int(channel)
        self.stages = nn.ModuleList(stages)
        self.out_channels = tuple(int(value) for value in channels)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        value = inputs
        for stage in self.stages:
            value = stage(value)
            outputs.append(value)
        return tuple(outputs)


def adapt_rgb_stem_to_multispectral(
    model: nn.Module,
    band_order: tuple[str, ...] = ("NIR", "R", "G", "B"),
) -> str:
    """Adapt the first RGB convolution without changing spectral semantics."""

    normalized_order = tuple(str(value).upper() for value in band_order)
    if len(normalized_order) != 4 or set(normalized_order) != {
        "NIR",
        "R",
        "G",
        "B",
    }:
        raise ValueError(
            "band_order must be a permutation of NIR/R/G/B, "
            f"got {normalized_order}"
        )
    stem_name = ""
    stem: nn.Conv2d | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            stem_name, stem = name, module
            break
    if stem is None:
        raise RuntimeError("could not locate a three-channel input convolution")
    if stem.groups != 1:
        raise RuntimeError(f"unsupported grouped RGB stem at {stem_name!r}")

    replacement = nn.Conv2d(
        in_channels=4,
        out_channels=stem.out_channels,
        kernel_size=stem.kernel_size,
        stride=stem.stride,
        padding=stem.padding,
        dilation=stem.dilation,
        groups=stem.groups,
        bias=stem.bias is not None,
        padding_mode=stem.padding_mode,
    ).to(device=stem.weight.device, dtype=stem.weight.dtype)
    with torch.no_grad():
        rgb = {
            "R": stem.weight[:, 0],
            "G": stem.weight[:, 1],
            "B": stem.weight[:, 2],
        }
        rgb["NIR"] = stem.weight.mean(dim=1)
        replacement.weight.copy_(
            torch.stack([rgb[name] for name in normalized_order], dim=1)
        )
        if stem.bias is not None:
            replacement.bias.copy_(stem.bias)

    parent_name, _, child_name = stem_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)
    return stem_name


class TimmPyramidEncoder(nn.Module):
    """Optional timm feature encoder with fbponlyms-compatible state keys."""

    def __init__(
        self,
        model_name: str,
        in_channels: int = 4,
        pretrained: bool = False,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        gradient_checkpointing: bool = False,
        img_size: int | None = 512,
        pretrained_file: str | None = None,
        prefer_official_url: bool = False,
        band_order: tuple[str, ...] = ("NIR", "R", "G", "B"),
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError(
                "timm is required only for the full Swin experiment; "
                "the tiny encoder remains fully offline"
            ) from error
        create_options: dict[str, Any] = {}
        if img_size is not None:
            create_options["img_size"] = int(img_size)
        if pretrained_file:
            create_options["pretrained_cfg_overlay"] = {
                "file": str(pretrained_file)
            }
        elif prefer_official_url:
            # timm prefers Hugging Face whenever an hf_hub_id is present,
            # even when the same official weights have a GitHub release URL.
            # The production server cannot reliably reach HF, so explicitly
            # select timm's still-authenticated official URL source without
            # permitting an untracked local pretrained file.
            create_options["pretrained_cfg_overlay"] = {"hf_hub_id": None}
        load_rgb_then_adapt = bool(pretrained) and int(in_channels) == 4
        # Attribute name ``model`` is a checkpoint compatibility contract.
        self.model = timm.create_model(
            model_name,
            pretrained=bool(pretrained),
            features_only=True,
            in_chans=3 if load_rgb_then_adapt else int(in_channels),
            out_indices=out_indices,
            **create_options,
        )
        if load_rgb_then_adapt:
            adapt_rgb_stem_to_multispectral(self.model, band_order=band_order)
        if gradient_checkpointing and hasattr(
            self.model, "set_grad_checkpointing"
        ):
            self.model.set_grad_checkpointing(True)
        self.out_channels = tuple(
            int(value) for value in self.model.feature_info.channels()
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        for feature, channels in zip(self.model(inputs), self.out_channels):
            if feature.ndim != 4:
                raise RuntimeError(
                    "expected a four-dimensional feature map, "
                    f"got {feature.shape}"
                )
            if feature.shape[1] == channels:
                outputs.append(feature)
            elif feature.shape[-1] == channels:
                outputs.append(feature.permute(0, 3, 1, 2).contiguous())
            else:
                raise RuntimeError(
                    f"cannot locate the {channels}-channel axis in {feature.shape}"
                )
        return tuple(outputs)


def build_pyramid_encoder(
    config: Mapping[str, Any],
    *,
    default_channels: tuple[int, ...],
    default_in_channels: int = 4,
) -> nn.Module:
    """Construct a tiny offline encoder or optional timm encoder."""

    kind = str(config.get("type", "tiny")).lower()
    if kind == "tiny":
        return TinyPyramidEncoder(
            in_channels=int(config.get("in_channels", default_in_channels)),
            channels=tuple(
                int(value) for value in config.get("channels", default_channels)
            ),
        )
    if kind == "timm":
        return TimmPyramidEncoder(
            model_name=str(config["name"]),
            in_channels=int(config.get("in_channels", default_in_channels)),
            pretrained=bool(config.get("pretrained", False)),
            out_indices=tuple(
                int(value) for value in config.get("out_indices", (0, 1, 2, 3))
            ),
            gradient_checkpointing=bool(
                config.get("gradient_checkpointing", False)
            ),
            img_size=(
                int(config["img_size"])
                if config.get("img_size") is not None
                else None
            ),
            pretrained_file=(
                str(config["pretrained_file"])
                if config.get("pretrained_file")
                else None
            ),
            prefer_official_url=bool(
                config.get("prefer_official_url", False)
            ),
            band_order=tuple(
                str(value)
                for value in config.get("band_order", ("NIR", "R", "G", "B"))
            ),
        )
    raise ValueError(f"unknown encoder type: {kind}")


class PyramidPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scales: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.scales = scales
        branch_channels = max(out_channels // len(scales), 16)
        self.branches = nn.ModuleList(
            [
                ConvNormAct(
                    in_channels, branch_channels, kernel_size=1, padding=0
                )
                for _ in scales
            ]
        )
        merged_channels = in_channels + branch_channels * len(scales)
        self.bottleneck = ConvNormAct(
            merged_channels, out_channels, kernel_size=3
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        values = [inputs]
        for scale, branch in zip(self.scales, self.branches):
            pooled = F.adaptive_avg_pool2d(inputs, output_size=(scale, scale))
            pooled = branch(pooled)
            values.append(
                F.interpolate(
                    pooled,
                    (height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return self.bottleneck(torch.cat(values, dim=1))


class UPerDecoder(nn.Module):
    """Compact UPer decoder; names and layout match the CASTR anchor."""

    def __init__(
        self,
        in_channels: tuple[int, ...],
        channels: int = 256,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("UPerDecoder requires at least two feature levels")
        self.ppm = PyramidPooling(in_channels[-1], channels, pool_scales)
        self.lateral = nn.ModuleList(
            [
                ConvNormAct(value, channels, kernel_size=1, padding=0)
                for value in in_channels[:-1]
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [ConvNormAct(channels, channels, kernel_size=3) for _ in in_channels[:-1]]
        )
        self.fusion = ConvNormAct(
            channels * len(in_channels), channels, kernel_size=3
        )
        self.out_channels = channels

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        laterals = [
            layer(feature)
            for layer, feature in zip(self.lateral, features[:-1])
        ]
        laterals.append(self.ppm(features[-1]))
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        outputs = [
            conv(value) for conv, value in zip(self.fpn_convs, laterals[:-1])
        ]
        outputs.append(laterals[-1])
        target_size = outputs[0].shape[-2:]
        outputs = [
            value
            if value.shape[-2:] == target_size
            else F.interpolate(
                value, target_size, mode="bilinear", align_corners=False
            )
            for value in outputs
        ]
        return self.fusion(torch.cat(outputs, dim=1))


class SpectralIndexPyramid(nn.Module):
    """High-resolution B/G/R/NIR plus NDVI/NDWI/SAVI/brightness pyramid."""

    def __init__(
        self,
        channels: tuple[int, ...],
        ms_mean: Sequence[float],
        ms_std: Sequence[float],
    ) -> None:
        super().__init__()
        if len(ms_mean) != 4 or len(ms_std) != 4:
            raise ValueError("MS normalization must contain four B/G/R/NIR values")
        if any(float(value) <= 0 for value in ms_std):
            raise ValueError("MS standard deviations must be positive")
        self.register_buffer(
            "ms_mean",
            torch.tensor(tuple(float(value) for value in ms_mean)).reshape(
                1, 4, 1, 1
            ),
            persistent=True,
        )
        self.register_buffer(
            "ms_std",
            torch.tensor(tuple(float(value) for value in ms_std)).reshape(
                1, 4, 1, 1
            ),
            persistent=True,
        )
        self.encoder = TinyPyramidEncoder(in_channels=8, channels=channels)
        self.out_channels = self.encoder.out_channels

    @staticmethod
    def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        safe = denominator.abs().clamp_min(1e-4)
        return (numerator / safe).clamp(-1.0, 1.0)

    def compute_indices(self, normalized_ms: torch.Tensor) -> torch.Tensor:
        mean = self.ms_mean.to(dtype=normalized_ms.dtype)
        std = self.ms_std.to(dtype=normalized_ms.dtype)
        reflectance = normalized_ms * std + mean
        blue, green, red, nir = reflectance.split(1, dim=1)
        ndvi = self._safe_ratio(nir - red, nir + red)
        ndwi = self._safe_ratio(green - nir, green + nir)
        savi = self._safe_ratio(1.5 * (nir - red), nir + red + 0.5)
        rgb_mean = (blue + green + red) / 3.0
        mean_brightness = mean[:, :3].mean(dim=1, keepdim=True)
        std_brightness = std[:, :3].mean(dim=1, keepdim=True).clamp_min(1e-4)
        brightness = ((rgb_mean - mean_brightness) / std_brightness).clamp(
            -5.0, 5.0
        )
        return torch.cat([ndvi, ndwi, savi, brightness], dim=1)

    def forward(self, normalized_ms: torch.Tensor) -> tuple[torch.Tensor, ...]:
        spectral_input = torch.cat(
            [normalized_ms, self.compute_indices(normalized_ms)], dim=1
        )
        return self.encoder(spectral_input)


class SoftScaleFusion(nn.Module):
    """Fuse local/short/global/spectral features with continuous weights."""

    VIEW_COUNT = 4

    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        channels: int,
        initial_weights: tuple[float, float, float, float] = (
            0.50,
            0.20,
            0.15,
            0.15,
        ),
    ) -> None:
        super().__init__()
        if len(initial_weights) != self.VIEW_COUNT or any(
            value <= 0 for value in initial_weights
        ):
            raise ValueError("all four initial fusion weights must be positive")
        total = float(sum(initial_weights))
        normalized = tuple(float(value) / total for value in initial_weights)
        self.projections = nn.ModuleList(
            [
                ConvNormAct(value, channels, kernel_size=1, padding=0)
                for value in in_channels
            ]
        )
        # A local ROI alone cannot distinguish identical-looking patches in
        # different scene contexts.  Preserve one full-support descriptor for
        # each contextual view and add it to that view's aligned ROI.  These
        # modules are outside the frozen anchor and therefore cannot affect
        # the exact anchor path while context_residual_scale is zero.
        self.summary_projections = nn.ModuleList(
            [
                ConvNormAct(in_channels[1], channels, kernel_size=1, padding=0),
                ConvNormAct(in_channels[2], channels, kernel_size=1, padding=0),
            ]
        )
        self.gate = nn.Conv2d(channels * self.VIEW_COUNT, self.VIEW_COUNT, 1)
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad():
            self.gate.bias.copy_(
                torch.tensor([math.log(value) for value in normalized])
            )
        self.refine = ConvNormAct(channels, channels, kernel_size=3)

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        view_present: torch.Tensor,
        full_context_features: tuple[torch.Tensor, torch.Tensor],
        gate_logit_delta: Callable[[torch.Tensor], torch.Tensor] | None = None,
        spectral_value_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if view_present.ndim != 2 or view_present.shape[1] != self.VIEW_COUNT:
            raise ValueError("view_present must have shape [batch, 4]")
        target_size = features[0].shape[-2:]
        projected: list[torch.Tensor] = []
        for projection, feature in zip(self.projections, features):
            value = projection(feature)
            if value.shape[-2:] != target_size:
                value = F.interpolate(
                    value,
                    target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(value)
        for projected_index, projection, full_feature in zip(
            (1, 2), self.summary_projections, full_context_features
        ):
            summary = projection(F.adaptive_avg_pool2d(full_feature, 1))
            projected[projected_index] = projected[projected_index] + summary
        projected_views = torch.cat(projected, dim=1)
        gate_logits = self.gate(projected_views)
        if gate_logit_delta is not None:
            delta = gate_logit_delta(projected_views)
            if delta.shape != gate_logits.shape:
                raise ValueError("fusion gate delta must match the four-view logits")
            gate_logits = gate_logits + delta.to(gate_logits.dtype)
        unavailable = ~view_present[:, :, None, None]
        gate_logits = gate_logits.masked_fill(
            unavailable, torch.finfo(gate_logits.dtype).min
        )
        weights = gate_logits.softmax(dim=1)
        # Compute the original four-view gate before changing any value.  The
        # optional condition may only transform the projected spectral branch;
        # its returned value is shared by the fusion and spectral-only path.
        if spectral_value_transform is not None:
            spectral_value = spectral_value_transform(projected_views, projected[-1])
            if spectral_value.shape != projected[-1].shape:
                raise ValueError("spectral value transform must preserve the spectral shape")
            projected[-1] = spectral_value.to(projected[-1].dtype)
        evidence = sum(
            weight * value
            for weight, value in zip(weights.split(1, dim=1), projected)
        )
        return (
            self.refine(evidence),
            weights,
            projected[-1],
            projected[1],
            projected[2],
        )


class MS75Segmentation(nn.Module):
    """Strong four-band Line3 model for the strict 120/30 FBP protocol.

    The complete CASTR local/context/structure/learned-router anchor remains an
    exact independent path.  New evidence is a bounded residual initialized to
    zero, and the hierarchical probability is mixed by a bounded convex
    coefficient also initialized to zero.  Consequently, loading every anchor
    tensor reproduces the archived raw logits before staged training begins.
    """

    ANCHOR_PREFIXES = (
        "local_encoder.",
        "decoder.",
        "local_head.",
        "context_encoder.",
        "context_projection.",
        "global_head.",
        "structure_input.",
        "structure_blocks.",
        "context_fusion.",
        "structure_fusion.",
        "context_refine.",
        "structure_refine.",
        "context_head.",
        "structure_head.",
        "router.",
    )
    ARCHIVED_ANCHOR_TENSOR_COUNT = 596

    def __init__(
        self,
        *,
        num_classes: int = 24,
        local_encoder_config: Mapping[str, Any] | None = None,
        short_encoder_config: Mapping[str, Any] | None = None,
        context_encoder_config: Mapping[str, Any] | None = None,
        global_encoder_config: Mapping[str, Any] | None = None,
        spectral_channels: Sequence[int] = (16, 32, 64, 128),
        decoder_channels: int = 256,
        fusion_channels: int = 64,
        contrast_channels: int = 64,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
        evidence_channels: int = 128,
        attention_channels: int = 128,
        attention_heads: int = 4,
        max_context_grid: int = 16,
        structure_groups: Sequence[str] = tuple(CASTR_STRUCTURE_GROUPS),
        structure_layers: int = 2,
        structure_dilations: Sequence[int] = (1, 2, 4, 8),
        router_signal_mode: str = "local_entropy",
        router_detach_signals: bool = False,
        freeze_context_batch_norm_stats: bool = False,
        max_context_residual_scale: float = 1.0,
        initial_context_residual_scale: float = 0.0,
        max_hierarchy_mix: float = 0.75,
        initial_hierarchy_mix: float = 0.0,
        hierarchy_spatial_delta: float = 0.10,
        ms_mean: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        ms_std: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if int(num_classes) != 24:
            raise ValueError("the Line3 family hierarchy requires exactly 24 classes")
        if not 0 <= initial_context_residual_scale <= max_context_residual_scale:
            raise ValueError("invalid initial context residual scale")
        if not 0 <= initial_hierarchy_mix <= max_hierarchy_mix <= 1:
            raise ValueError("hierarchy mix must satisfy 0 <= initial <= max <= 1")
        if hierarchy_spatial_delta < 0:
            raise ValueError("hierarchy_spatial_delta must be non-negative")
        self.num_classes = 24
        self.num_families = len(LAND_COVER_FAMILIES)
        self.max_context_residual_scale = float(max_context_residual_scale)
        self.max_hierarchy_mix = float(max_hierarchy_mix)
        self.hierarchy_spatial_delta = float(hierarchy_spatial_delta)
        self.freeze_context_batch_norm_stats = bool(
            freeze_context_batch_norm_stats
        )

        local_config = dict(
            local_encoder_config
            or {"type": "tiny", "channels": (16, 32, 64, 128)}
        )
        short_config = dict(
            short_encoder_config
            or {"type": "tiny", "channels": (8, 16, 32, 64)}
        )
        context_config = dict(
            context_encoder_config
            or global_encoder_config
            or {"type": "tiny", "channels": (8, 16, 32, 64)}
        )
        if int(local_config.get("in_channels", 4)) != 4:
            raise ValueError("local encoder must consume exactly four MS bands")
        if int(short_config.get("in_channels", 4)) != 4:
            raise ValueError("short encoder must consume exactly four MS bands")
        if int(context_config.get("in_channels", 4)) != 4:
            raise ValueError("context encoder must consume exactly four MS bands")
        if int(evidence_channels) <= 0:
            raise ValueError("evidence_channels must be positive")
        if int(structure_layers) <= 0:
            raise ValueError("structure_layers must be positive")

        # The following names reproduce the complete external CASTR checkpoint
        # contract.  Do not nest them under another module: the archived state
        # dictionary uses these exact top-level prefixes.
        self.local_encoder = build_pyramid_encoder(
            local_config, default_channels=(32, 64, 128, 256)
        )
        self.decoder = UPerDecoder(
            tuple(self.local_encoder.out_channels),
            channels=int(decoder_channels),
            pool_scales=tuple(int(value) for value in pool_scales),
        )
        self.local_head = nn.Conv2d(
            int(decoder_channels), self.num_classes, kernel_size=1
        )

        self.context_encoder = build_pyramid_encoder(
            context_config, default_channels=(16, 32, 64, 128)
        )
        self.context_projection = ConvNormAct(
            int(self.context_encoder.out_channels[-1]),
            int(evidence_channels),
            kernel_size=1,
            padding=0,
        )
        self.global_head = nn.Conv2d(
            int(evidence_channels), self.num_classes, kernel_size=1
        )
        group_names = tuple(str(value) for value in structure_groups)
        group_matrix = torch.zeros(
            len(group_names), self.num_classes, dtype=torch.float32
        )
        for row, name in enumerate(group_names):
            if name not in CASTR_STRUCTURE_GROUPS:
                raise ValueError(f"unknown CASTR structure group: {name}")
            group_matrix[row, list(CASTR_STRUCTURE_GROUPS[name])] = 1.0
        self.register_buffer("group_matrix", group_matrix, persistent=False)
        self.register_buffer(
            "_active_experts",
            torch.tensor((True, True, True), dtype=torch.bool),
            persistent=False,
        )
        self.structure_input = ConvNormAct(
            int(evidence_channels) + len(group_names),
            int(evidence_channels),
            kernel_size=1,
            padding=0,
        )
        self.structure_blocks = nn.Sequential(
            *[
                SparseStructureBlock(
                    int(evidence_channels),
                    tuple(int(value) for value in structure_dilations),
                )
                for _ in range(int(structure_layers))
            ]
        )
        anchor_fusion_arguments = {
            "local_channels": int(self.local_encoder.out_channels[-1]),
            "evidence_channels": int(evidence_channels),
            "decoder_channels": int(decoder_channels),
            "attention_channels": int(attention_channels),
            "num_heads": int(attention_heads),
            "max_context_grid": int(max_context_grid),
        }
        self.context_fusion = CrossEvidenceFusion(**anchor_fusion_arguments)
        self.structure_fusion = CrossEvidenceFusion(**anchor_fusion_arguments)
        self.context_refine = ConvNormAct(
            int(decoder_channels) * 2, int(decoder_channels), kernel_size=3
        )
        self.structure_refine = ConvNormAct(
            int(decoder_channels) * 2, int(decoder_channels), kernel_size=3
        )
        self.context_head = nn.Conv2d(
            int(decoder_channels), self.num_classes, kernel_size=1
        )
        self.structure_head = nn.Conv2d(
            int(decoder_channels), self.num_classes, kernel_size=1
        )
        self.router = EvidenceRouter(
            int(decoder_channels),
            signal_mode=str(router_signal_mode),
            detach_signals=bool(router_detach_signals),
        )

        self.short_encoder = build_pyramid_encoder(
            short_config, default_channels=(16, 32, 64, 128)
        )
        spectral_tuple = tuple(int(value) for value in spectral_channels)
        self.spectral_encoder = SpectralIndexPyramid(
            spectral_tuple, ms_mean=ms_mean, ms_std=ms_std
        )
        level_count = len(tuple(self.local_encoder.out_channels))
        view_level_counts = (
            len(tuple(self.short_encoder.out_channels)),
            len(tuple(self.context_encoder.out_channels)),
            len(tuple(self.spectral_encoder.out_channels)),
        )
        if level_count < 2 or any(value != level_count for value in view_level_counts):
            raise ValueError("all four encoders must expose the same number of levels")

        self.level_fusions = nn.ModuleList(
            [
                SoftScaleFusion(
                    (
                        int(self.local_encoder.out_channels[index]),
                        int(self.short_encoder.out_channels[index]),
                        int(self.context_encoder.out_channels[index]),
                        int(self.spectral_encoder.out_channels[index]),
                    ),
                    int(fusion_channels),
                )
                for index in range(level_count)
            ]
        )
        self.context_aggregate = ConvNormAct(
            int(fusion_channels) * level_count,
            int(decoder_channels),
            kernel_size=3,
        )
        self.spectral_aggregate = ConvNormAct(
            int(fusion_channels) * level_count,
            int(decoder_channels),
            kernel_size=3,
        )
        self.short_aggregate = ConvNormAct(
            int(fusion_channels) * level_count,
            int(decoder_channels),
            kernel_size=3,
        )
        self.global_aggregate = ConvNormAct(
            int(fusion_channels) * level_count,
            int(decoder_channels),
            kernel_size=3,
        )
        self.context_aux_head = nn.Conv2d(
            int(decoder_channels), self.num_classes, 1
        )
        self.spectral_aux_head = nn.Conv2d(
            int(decoder_channels), self.num_classes, 1
        )
        self.short_aux_head = nn.Conv2d(
            int(decoder_channels), self.num_classes, 1
        )
        self.global_aux_head = nn.Conv2d(
            int(decoder_channels), self.num_classes, 1
        )
        self.context_residual_scale = nn.Parameter(
            torch.tensor(float(initial_context_residual_scale))
        )

        self.family_head = nn.Conv2d(
            int(decoder_channels), self.num_families, 1
        )
        self.family_child_heads = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(
                        int(decoder_channels) + 1,
                        int(decoder_channels),
                        kernel_size=1,
                        padding=0,
                    ),
                    nn.Conv2d(int(decoder_channels), len(class_ids), 1),
                )
                for class_ids in LAND_COVER_FAMILIES
            ]
        )
        self.family_contrast_projection = nn.Conv2d(
            int(decoder_channels), int(contrast_channels), 1
        )
        self.hierarchy_mix_gate = nn.Conv2d(int(decoder_channels), 1, 1)
        nn.init.zeros_(self.hierarchy_mix_gate.weight)
        nn.init.zeros_(self.hierarchy_mix_gate.bias)
        self.hierarchy_mix_base = nn.Parameter(
            torch.tensor(float(initial_hierarchy_mix))
        )

        mapping = fine_to_family_map()
        family_order = tuple(
            class_id
            for class_ids in LAND_COVER_FAMILIES
            for class_id in class_ids
        )
        inverse_order = [0] * self.num_classes
        for family_position, class_id in enumerate(family_order):
            inverse_order[class_id] = family_position
        self.register_buffer(
            "fine_to_family",
            torch.tensor(mapping, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "family_inverse_order",
            torch.tensor(inverse_order, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "local_band_permutation",
            torch.tensor((3, 2, 1, 0), dtype=torch.long),
            persistent=False,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MS75Segmentation":
        """Build from a Line3 config without requiring config.py changes."""

        model_root = dict(config.get("model", {}))
        model = dict(model_root.get("ms75", model_root))
        data = dict(config.get("data", {}))
        normalization = dict(data.get("normalization", {})).get("ms", {})
        mean = normalization.get("mean", data.get("mean", (0.0,) * 4))
        std = normalization.get("std", data.get("std", (1.0,) * 4))
        return cls(
            num_classes=int(model.get("num_classes", 24)),
            local_encoder_config=model.get("local_encoder"),
            short_encoder_config=model.get("short_encoder"),
            context_encoder_config=model.get("context_encoder"),
            global_encoder_config=model.get("global_encoder"),
            spectral_channels=model.get("spectral_channels", (16, 32, 64, 128)),
            decoder_channels=int(model.get("decoder_channels", 256)),
            fusion_channels=int(model.get("fusion_channels", 64)),
            contrast_channels=int(model.get("contrast_channels", 64)),
            pool_scales=model.get("pool_scales", (1, 2, 3, 6)),
            evidence_channels=int(model.get("evidence_channels", 128)),
            attention_channels=int(model.get("attention_channels", 128)),
            attention_heads=int(model.get("attention_heads", 4)),
            max_context_grid=int(model.get("max_context_grid", 16)),
            structure_groups=model.get(
                "structure_groups", tuple(CASTR_STRUCTURE_GROUPS)
            ),
            structure_layers=int(model.get("structure_layers", 2)),
            structure_dilations=model.get(
                "structure_dilations", (1, 2, 4, 8)
            ),
            router_signal_mode=str(
                model.get("router_signal_mode", "local_entropy")
            ),
            router_detach_signals=bool(
                model.get("router_detach_signals", False)
            ),
            freeze_context_batch_norm_stats=bool(
                model.get("freeze_context_batch_norm_stats", False)
            ),
            max_context_residual_scale=float(
                model.get("max_context_residual_scale", 1.0)
            ),
            initial_context_residual_scale=float(
                model.get("initial_context_residual_scale", 0.0)
            ),
            max_hierarchy_mix=float(model.get("max_hierarchy_mix", 0.75)),
            initial_hierarchy_mix=float(
                model.get("initial_hierarchy_mix", 0.0)
            ),
            hierarchy_spatial_delta=float(
                model.get("hierarchy_spatial_delta", 0.10)
            ),
            ms_mean=mean,
            ms_std=std,
        )

    def to_local_anchor_order(self, ms_bgrn: torch.Tensor) -> torch.Tensor:
        """Convert normalized Line3 B/G/R/NIR to anchor NIR/R/G/B."""

        self._validate_ms(ms_bgrn, "ms")
        return ms_bgrn.index_select(1, self.local_band_permutation)

    @torch.no_grad()
    def project_constrained_parameters_(self) -> "MS75Segmentation":
        """Project scalar residual gates after an optimizer update.

        Boundary-aware gradients prevent an outward update *at* a bound, but a
        sufficiently large step can still overshoot a bound from the interior.
        Training calls this method after every successful optimizer step and
        before EMA so neither checkpoints nor the EMA can accumulate raw gate
        debt.
        """

        self.context_residual_scale.clamp_(
            min=0.0, max=self.max_context_residual_scale
        )
        self.hierarchy_mix_base.clamp_(min=0.0, max=self.max_hierarchy_mix)
        return self

    def train(self, mode: bool = True) -> "MS75Segmentation":
        """Set mode while preserving imported context BatchNorm statistics.

        The complete anchor's ResNet context encoder was authenticated with
        its archived running statistics.  Formal MS75 uses a small local batch,
        so updating those buffers would silently alter the anchor even while
        its parameters are staged-frozen.  BatchNorm affine parameters and all
        convolutions remain trainable; only running-stat updates are disabled.
        """

        super().train(mode)
        if mode and self.freeze_context_batch_norm_stats:
            for module in self.context_encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    @staticmethod
    def _validate_ms(value: torch.Tensor, name: str) -> None:
        if value.ndim != 4 or value.shape[1] != 4:
            raise ValueError(
                f"{name} must have shape [batch, 4, height, width], "
                f"got {tuple(value.shape)}"
            )

    @classmethod
    def _unwrap_state_dict(
        cls, state: Mapping[str, Any]
    ) -> Mapping[str, torch.Tensor]:
        value: Mapping[str, Any] = state
        for candidate in ("model", "state_dict", "model_state_dict"):
            nested = value.get(candidate)
            if isinstance(nested, Mapping):
                value = nested
                break
        return value  # type: ignore[return-value]

    def load_local_anchor_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> tuple[str, ...]:
        """Load the complete CASTR anchor from an in-memory checkpoint.

        The method intentionally accepts no path.  It cannot read or mutate
        the fbponlyms project and makes the isolation boundary auditable.  The
        historical method name is retained for caller compatibility.
        """

        unwrapped = self._unwrap_state_dict(state)
        selected: dict[str, torch.Tensor] = {}
        for raw_key, tensor in unwrapped.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            key = str(raw_key)
            for wrapper in ("module.", "_orig_mod."):
                if key.startswith(wrapper):
                    key = key[len(wrapper) :]
            if key.startswith(self.ANCHOR_PREFIXES):
                selected[key] = tensor
        expected = {
            key
            for key in self.state_dict()
            if key.startswith(self.ANCHOR_PREFIXES)
        }
        missing = sorted(expected.difference(selected))
        unexpected = sorted(set(selected).difference(expected))
        if strict and (missing or unexpected):
            raise RuntimeError(
                "complete anchor checkpoint mismatch: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        self.load_state_dict(selected, strict=False)
        return tuple(sorted(expected.intersection(selected)))

    def load_complete_anchor_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> tuple[str, ...]:
        """Explicit alias for :meth:`load_local_anchor_state_dict`."""

        return self.load_local_anchor_state_dict(state, strict=strict)

    @staticmethod
    def _apply_valid_mask(
        value: torch.Tensor,
        valid: torch.Tensor | None,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if valid is None:
            return value, torch.ones(
                value.shape[0], dtype=torch.bool, device=value.device
            )
        if valid.ndim == 4 and valid.shape[1] == 1:
            valid = valid[:, 0]
        if valid.ndim != 3 or valid.shape[0] != value.shape[0]:
            raise ValueError(f"{name} must have shape [batch, height, width]")
        if valid.shape[-2:] != value.shape[-2:]:
            valid = F.interpolate(
                valid[:, None].float(), value.shape[-2:], mode="nearest"
            )[:, 0].bool()
        else:
            valid = valid.bool()
        return value * valid[:, None].to(value.dtype), valid.flatten(1).any(dim=1)

    @staticmethod
    def _resolve_local_box(
        box: torch.Tensor | None,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
        required: bool,
    ) -> torch.Tensor:
        """Validate normalized ``[y0, x0, y1, x1]`` ROI boxes.

        A missing box is accepted only for legacy ``ms_context``/``ms``
        fallback views, where the only honest interpretation is the complete
        support.  Explicit short/global inputs must always declare alignment.
        """

        if box is None:
            if required:
                raise ValueError(
                    f"{name} is required when its explicit context view is supplied"
                )
            return torch.tensor(
                (0.0, 0.0, 1.0, 1.0), device=device, dtype=dtype
            )[None].expand(batch, -1)
        if box.ndim != 2 or tuple(box.shape) != (batch, 4):
            raise ValueError(
                f"{name} must have shape [batch, 4], got {tuple(box.shape)}"
            )
        normalized = box.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError(f"{name} contains a non-finite coordinate")
        if bool(((normalized < 0.0) | (normalized > 1.0)).any()):
            raise ValueError(f"{name} coordinates must lie inside [0, 1]")
        y0, x0, y1, x1 = normalized.unbind(dim=1)
        if bool(((y1 <= y0) | (x1 <= x0)).any()):
            raise ValueError(f"{name} must satisfy y0 < y1 and x0 < x1")
        return normalized

    @staticmethod
    def _sample_normalized_roi(
        feature: torch.Tensor,
        boxes: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Sample continuous normalized boxes using pixel-centre geometry."""

        output_height, output_width = (int(value) for value in output_size)
        if output_height <= 0 or output_width <= 0:
            raise ValueError("ROI output dimensions must be positive")
        if tuple(boxes.shape) != (feature.shape[0], 4):
            raise ValueError("ROI boxes do not match the feature batch")
        boxes = boxes.to(device=feature.device, dtype=feature.dtype)
        y0, x0, y1, x1 = boxes.unbind(dim=1)
        unit_y = (
            torch.arange(output_height, device=feature.device, dtype=feature.dtype)
            + 0.5
        ) / float(output_height)
        unit_x = (
            torch.arange(output_width, device=feature.device, dtype=feature.dtype)
            + 0.5
        ) / float(output_width)
        sample_y = y0[:, None, None] + (
            y1 - y0
        )[:, None, None] * unit_y[None, :, None]
        sample_x = x0[:, None, None] + (
            x1 - x0
        )[:, None, None] * unit_x[None, None, :]
        grid_y = sample_y.expand(-1, output_height, output_width)
        grid_x = sample_x.expand(-1, output_height, output_width)
        grid = 2.0 * torch.stack((grid_x, grid_y), dim=-1) - 1.0
        return F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def _resolve_context_inputs(
        self,
        ms: torch.Tensor,
        ms_context: torch.Tensor | None,
        ms_context_valid: torch.Tensor | None,
        ms_short: torch.Tensor | None,
        ms_short_valid: torch.Tensor | None,
        ms_global: torch.Tensor | None,
        ms_global_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        short = ms_short if ms_short is not None else ms_context
        short_valid = (
            ms_short_valid if ms_short is not None else ms_context_valid
        )
        if short is None:
            short, short_valid = ms, None
        global_value = ms_global if ms_global is not None else ms_context
        global_valid = (
            ms_global_valid if ms_global is not None else ms_context_valid
        )
        if global_value is None:
            global_value, global_valid = short, short_valid
        self._validate_ms(short, "ms_short/ms_context")
        self._validate_ms(global_value, "ms_global/ms_context")
        short, short_present = self._apply_valid_mask(
            short, short_valid, "short validity mask"
        )
        global_value, global_present = self._apply_valid_mask(
            global_value, global_valid, "global validity mask"
        )
        return short, short_present, global_value, global_present

    @staticmethod
    def _resize_levels(
        levels: Sequence[torch.Tensor], size: tuple[int, int]
    ) -> torch.Tensor:
        return torch.cat(
            [
                value
                if value.shape[-2:] == size
                else F.interpolate(
                    value, size, mode="bilinear", align_corners=False
                )
                for value in levels
            ],
            dim=1,
        )

    def _compose_hierarchy(
        self,
        child_logits: torch.Tensor,
        family_logits: torch.Tensor,
    ) -> torch.Tensor:
        family_log_probability = F.log_softmax(family_logits, dim=1)
        pieces: list[torch.Tensor] = []
        for family_id, class_ids in enumerate(LAND_COVER_FAMILIES):
            indices = list(class_ids)
            pieces.append(
                F.log_softmax(child_logits[:, indices], dim=1)
                + family_log_probability[:, family_id : family_id + 1]
            )
        family_order_logits = torch.cat(pieces, dim=1)
        return family_order_logits.index_select(1, self.family_inverse_order)

    def _family_child_logits(
        self,
        features: torch.Tensor,
        family_logits: torch.Tensor,
    ) -> torch.Tensor:
        family_probability = family_logits.softmax(dim=1)
        pieces = [
            head(
                torch.cat(
                    [features, family_probability[:, family_id : family_id + 1]],
                    dim=1,
                )
            )
            for family_id, head in enumerate(self.family_child_heads)
        ]
        family_order_logits = torch.cat(pieces, dim=1)
        return family_order_logits.index_select(1, self.family_inverse_order)

    def encode_complete_anchor_context(
        self, context_bgrn: torch.Tensor
    ) -> AnchorContextState:
        """Encode one whole-scene B/G/R/NIR thumbnail exactly as CASTR.

        The archived model consumes normalized NIR/R/G/B.  The Line3 data
        contract is normalized B/G/R/NIR, so the same deterministic
        permutation used by the complete anchor is applied here.
        """

        self._validate_ms(context_bgrn, "complete anchor context")
        context_features = self.context_encoder(
            self.to_local_anchor_order(context_bgrn)
        )
        context_deep = self.context_projection(context_features[-1])
        global_logits = self.global_head(context_deep)
        group_probabilities = torch.einsum(
            "bchw,gc->bghw", global_logits.softmax(dim=1), self.group_matrix
        )
        structure_input = self.structure_input(
            torch.cat([context_deep, group_probabilities], dim=1)
        )
        structure_deep = self.structure_blocks(structure_input)
        return AnchorContextState(
            context_features=tuple(context_features),
            context_deep=context_deep,
            structure_deep=structure_deep,
            global_logits=global_logits,
        )

    @staticmethod
    def _match_anchor_batch(
        value: torch.Tensor, batch: int
    ) -> torch.Tensor:
        if value.shape[0] == batch:
            return value
        if value.shape[0] == 1:
            return value.expand(batch, -1, -1, -1)
        raise ValueError(
            f"cannot match anchor context batch {value.shape[0]} to local batch {batch}"
        )

    def _forward_complete_anchor_nrgb(
        self,
        image_nrgb: torch.Tensor,
        boxes: torch.Tensor,
        context_state: AnchorContextState,
        local_features_override: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        tuple[torch.Tensor, ...],
        torch.Tensor,
    ]:
        """Execute the archived learned-router graph without Line3 branches."""

        input_size = image_nrgb.shape[-2:]
        local_features = (self.local_encoder(image_nrgb) if local_features_override is None
                          else local_features_override)
        decoder_feature = self.decoder(local_features)
        local_logits_low = self.local_head(decoder_feature)
        local_logits = F.interpolate(
            local_logits_low,
            input_size,
            mode="bilinear",
            align_corners=False,
        )
        batch = image_nrgb.shape[0]
        context_deep = self._match_anchor_batch(
            context_state.context_deep, batch
        )
        structure_deep = self._match_anchor_batch(
            context_state.structure_deep, batch
        )

        context_evidence = self.context_fusion(
            local_features[-1],
            context_deep,
            boxes,
            decoder_feature.shape[-2:],
        )
        context_feature = self.context_refine(
            torch.cat([decoder_feature, context_evidence], dim=1)
        )
        context_delta_low = self.context_head(context_feature)

        structure_evidence = self.structure_fusion(
            local_features[-1],
            structure_deep,
            boxes,
            decoder_feature.shape[-2:],
        )
        structure_feature = self.structure_refine(
            torch.cat([decoder_feature, structure_evidence], dim=1)
        )
        structure_delta_low = self.structure_head(structure_feature)

        context_delta = F.interpolate(
            context_delta_low,
            input_size,
            mode="bilinear",
            align_corners=False,
        )
        structure_delta = F.interpolate(
            structure_delta_low,
            input_size,
            mode="bilinear",
            align_corners=False,
        )
        context_logits = local_logits + context_delta
        context_logits_low = local_logits_low + context_delta_low
        structure_logits = local_logits + structure_delta
        structure_logits_low = local_logits_low + structure_delta_low

        router_local_logits = (
            local_logits
            if self.router.signal_mode == "local_entropy"
            else local_logits_low
        )
        router_logits_low = self.router(
            decoder_feature,
            context_feature,
            structure_feature,
            router_local_logits,
            context_logits_low,
            structure_logits_low,
        )
        router_logits = F.interpolate(
            router_logits_low,
            input_size,
            mode="bilinear",
            align_corners=False,
        )
        router_logits = router_logits.masked_fill(
            ~self._active_experts.view(1, 3, 1, 1),
            torch.finfo(router_logits.dtype).min,
        )
        router_weights = router_logits.softmax(dim=1)
        logits = (
            local_logits
            + router_weights[:, 1:2] * context_delta
            + router_weights[:, 2:3] * structure_delta
        )
        outputs = {
            "logits": logits,
            "local_logits": local_logits,
            "context_logits": context_logits,
            "third_logits": structure_logits,
            "structure_logits": structure_logits,
            "router_weights": router_weights,
            "router_logits": router_logits,
            "active_experts": self._active_experts,
            "global_logits": self._match_anchor_batch(
                context_state.global_logits, batch
            ),
        }
        return outputs, tuple(local_features), decoder_feature

    def forward_complete_anchor(
        self,
        local_bgrn: torch.Tensor,
        boxes: torch.Tensor,
        context_state: AnchorContextState,
    ) -> dict[str, torch.Tensor]:
        """Public exact learned-router forward for protocol verification."""

        self._validate_ms(local_bgrn, "complete anchor local")
        normalized_boxes = self._resolve_local_box(
            boxes,
            batch=local_bgrn.shape[0],
            device=local_bgrn.device,
            dtype=local_bgrn.dtype,
            name="complete anchor boxes",
            required=True,
        )
        outputs, _, _ = self._forward_complete_anchor_nrgb(
            self.to_local_anchor_order(local_bgrn),
            normalized_boxes,
            context_state,
        )
        return outputs

    def forward_complete_anchor_from_multiscale_inputs(
        self,
        ms: torch.Tensor,
        *,
        ms_context: torch.Tensor | None = None,
        ms_context_valid: torch.Tensor | None = None,
        ms_short: torch.Tensor | None = None,
        ms_short_valid: torch.Tensor | None = None,
        ms_global: torch.Tensor | None = None,
        ms_global_valid: torch.Tensor | None = None,
        local_box_global: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run only the authenticated CASTR path from a normal MS75 batch.

        This mirrors the global-input masking and normalized-box resolution of
        :meth:`forward`, but deliberately skips every new short/spectral/family
        branch.  It is used by the immutable 70.1564 teacher so no moving
        student tensor can silently become its own no-harm target.
        """

        self._validate_ms(ms, "ms")
        _, _, global_value, _ = self._resolve_context_inputs(
            ms,
            ms_context,
            ms_context_valid,
            ms_short,
            ms_short_valid,
            ms_global,
            ms_global_valid,
        )
        global_box = self._resolve_local_box(
            local_box_global,
            batch=ms.shape[0],
            device=ms.device,
            dtype=ms.dtype,
            name="local_box_global",
            required=ms_global is not None,
        )
        context_state = self.encode_complete_anchor_context(global_value)
        return self.forward_complete_anchor(ms, global_box, context_state)

    def forward(
        self,
        ms: torch.Tensor,
        sar: torch.Tensor | None = None,
        sar_valid: torch.Tensor | None = None,
        time_gap_days: torch.Tensor | None = None,
        ms_context: torch.Tensor | None = None,
        ms_context_valid: torch.Tensor | None = None,
        ms_short: torch.Tensor | None = None,
        ms_short_valid: torch.Tensor | None = None,
        ms_global: torch.Tensor | None = None,
        ms_global_valid: torch.Tensor | None = None,
        local_box_short: torch.Tensor | None = None,
        local_box_global: torch.Tensor | None = None,
        complete_anchor_context_state: AnchorContextState | None = None,
        local_features_override: tuple[torch.Tensor, ...] | None = None,
        fusion_gate_deltas: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        fusion_evidence_deltas: Sequence[Callable[[torch.Tensor], torch.Tensor] | None] | None = None,
        fusion_spectral_transforms: Sequence[Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        del sar_valid, time_gap_days
        self._validate_ms(ms, "ms")
        short, short_present, global_value, global_present = (
            self._resolve_context_inputs(
                ms,
                ms_context,
                ms_context_valid,
                ms_short,
                ms_short_valid,
                ms_global,
                ms_global_valid,
            )
        )
        short_box = self._resolve_local_box(
            local_box_short,
            batch=ms.shape[0],
            device=ms.device,
            dtype=ms.dtype,
            name="local_box_short",
            required=ms_short is not None,
        )
        global_box = self._resolve_local_box(
            local_box_global,
            batch=ms.shape[0],
            device=ms.device,
            dtype=ms.dtype,
            name="local_box_global",
            required=ms_global is not None,
        )

        if complete_anchor_context_state is None:
            anchor_context_state = self.encode_complete_anchor_context(
                global_value
            )
        else:
            if not isinstance(
                complete_anchor_context_state, AnchorContextState
            ):
                raise TypeError(
                    "complete_anchor_context_state must be AnchorContextState"
                )
            anchor_context_state = complete_anchor_context_state
            global_present = torch.ones_like(global_present)
        anchor_outputs, local_features, local_decoder_feature = (
            self._forward_complete_anchor_nrgb(
                self.to_local_anchor_order(ms),
                global_box,
                anchor_context_state,
                local_features_override=local_features_override,
            )
        )
        short_features = self.short_encoder(short)
        global_features = tuple(
            self._match_anchor_batch(value, ms.shape[0])
            for value in anchor_context_state.context_features
        )
        spectral_features = self.spectral_encoder(ms)
        view_present = torch.stack(
            [
                torch.ones_like(short_present),
                short_present,
                global_present,
                torch.ones_like(short_present),
            ],
            dim=1,
        )
        evidence_levels: list[torch.Tensor] = []
        spectral_levels: list[torch.Tensor] = []
        short_levels: list[torch.Tensor] = []
        global_levels: list[torch.Tensor] = []
        scale_weights: list[torch.Tensor] = []
        if fusion_gate_deltas is not None and len(fusion_gate_deltas) != len(self.level_fusions):
            raise ValueError("one fusion gate controller is required per feature scale")
        if fusion_evidence_deltas is not None and len(fusion_evidence_deltas) != len(self.level_fusions):
            raise ValueError("one optional evidence controller is required per feature scale")
        if fusion_spectral_transforms is not None and len(fusion_spectral_transforms) != len(self.level_fusions):
            raise ValueError("one optional spectral controller is required per feature scale")
        for level_index, (fusion, local, short_level, global_level, spectral) in enumerate(zip(
            self.level_fusions,
            local_features,
            short_features,
            global_features,
            spectral_features,
        )):
            short_roi = self._sample_normalized_roi(
                short_level, short_box, local.shape[-2:]
            )
            global_roi = self._sample_normalized_roi(
                global_level, global_box, local.shape[-2:]
            )
            evidence, weights, spectral_value, short_value, global_value_level = fusion(
                (local, short_roi, global_roi, spectral),
                view_present,
                (short_level, global_level),
                gate_logit_delta=None if fusion_gate_deltas is None else fusion_gate_deltas[level_index],
                spectral_value_transform=None if fusion_spectral_transforms is None else fusion_spectral_transforms[level_index],
            )
            if fusion_evidence_deltas is not None and fusion_evidence_deltas[level_index] is not None:
                delta = fusion_evidence_deltas[level_index](evidence)
                if delta.shape != evidence.shape:
                    raise ValueError("evidence delta must match the optical feature")
                evidence = evidence + delta.to(evidence.dtype)
            evidence_levels.append(evidence)
            spectral_levels.append(spectral_value)
            short_levels.append(short_value)
            global_levels.append(global_value_level)
            scale_weights.append(weights)

        decoder_size = local_decoder_feature.shape[-2:]
        context_evidence = self.context_aggregate(
            self._resize_levels(evidence_levels, decoder_size)
        )
        spectral_evidence = self.spectral_aggregate(
            self._resize_levels(spectral_levels, decoder_size)
        )
        short_evidence = self.short_aggregate(
            self._resize_levels(short_levels, decoder_size)
        )
        global_evidence = self.global_aggregate(
            self._resize_levels(global_levels, decoder_size)
        )
        context_scale = _straight_through_clamp(
            self.context_residual_scale,
            minimum=0.0,
            maximum=self.max_context_residual_scale,
        )
        scale_for_feature = context_scale.to(dtype=context_evidence.dtype)
        fused_feature = (
            local_decoder_feature + scale_for_feature * context_evidence
        )

        family_logits_low = self.family_head(fused_feature)
        family_child_logits_low = self._family_child_logits(
            fused_feature, family_logits_low
        )
        output_size = ms.shape[-2:]
        def interpolate(value: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                value, output_size, mode="bilinear", align_corners=False
            )
        local_anchor_logits = anchor_outputs["local_logits"]
        complete_anchor_logits = anchor_outputs["logits"]
        context_aux_logits = interpolate(
            self.context_aux_head(context_evidence)
        )
        scale_for_logits = context_scale.to(dtype=context_aux_logits.dtype)
        line3_residual = scale_for_logits * context_aux_logits
        # Multiplication by the exact-zero scalar has a zero value while
        # retaining the useful gate derivative.  Casting the scalar to the
        # anchor dtype prevents AMP from promoting an otherwise identical P0.
        flat_logits = complete_anchor_logits + line3_residual
        family_logits = interpolate(family_logits_low)
        family_child_logits = interpolate(family_child_logits_low)
        # Compose after interpolation so the fine-grid hierarchical
        # probabilities remain exactly normalized and aggregate to the
        # fine-grid family probabilities.
        hierarchical_logits = self._compose_hierarchy(
            family_child_logits, family_logits
        )
        spatial_mix = interpolate(self.hierarchy_mix_gate(fused_feature))
        hierarchy_base = _straight_through_clamp(
            self.hierarchy_mix_base,
            minimum=0.0,
            maximum=self.max_hierarchy_mix,
        )
        configured_spatial_amplitude = hierarchy_base.new_tensor(
            self.hierarchy_spatial_delta
        )
        spatial_amplitude = torch.minimum(
            configured_spatial_amplitude,
            torch.minimum(
                hierarchy_base,
                hierarchy_base.new_tensor(self.max_hierarchy_mix)
                - hierarchy_base,
            ),
        ).clamp_min(0.0)
        hierarchy_mix_raw = (
            hierarchy_base + spatial_amplitude * torch.tanh(spatial_mix)
        )
        hierarchy_mix = _straight_through_clamp(
            hierarchy_mix_raw,
            minimum=0.0,
            maximum=self.max_hierarchy_mix,
        )
        # Keep the probability mixture explicitly in FP32.  A fixed 1e-7
        # floor cuts the CE gradient for precisely the high-confidence sibling
        # errors MS75 is meant to repair.  float32.tiny retains useful gradients
        # throughout the practical logit range while still preventing log(0).
        flat_log_probability = F.log_softmax(flat_logits.float(), dim=1)
        hierarchy_log_probability = hierarchical_logits.float()
        flat_probability = flat_log_probability.exp()
        hierarchy_probability = hierarchy_log_probability.exp()
        hierarchy_mix_float = hierarchy_mix.float()
        probability = (
            (1.0 - hierarchy_mix_float) * flat_probability
            + hierarchy_mix_float * hierarchy_probability
        )
        probability_floor = torch.finfo(probability.dtype).tiny
        mixed_log_probability = probability.clamp_min(probability_floor).log()

        # At the exact-zero identity boundary, return the anchor log-probability
        # itself (not a round-trip through exp/log).  A detached-probability
        # companion preserves the derivative with respect to the mixture gate
        # without duplicating gradients into either classifier.
        gate_only_probability = (
            (1.0 - hierarchy_mix_float) * flat_probability.detach()
            + hierarchy_mix_float * hierarchy_probability.detach()
        )
        gate_only_log_probability = gate_only_probability.clamp_min(
            probability_floor
        ).log()
        gate_surrogate_zero = (
            gate_only_log_probability - gate_only_log_probability.detach()
        ).to(dtype=flat_logits.dtype)
        anchor_identity_logits = flat_logits + gate_surrogate_zero
        if bool((hierarchy_mix_float == 0.0).all()):
            # Do not route the all-zero P0 through log_softmax or a promoted
            # torch.where branch.  Raw values and dtype must match CASTR.
            logits = anchor_identity_logits
        else:
            logits = torch.where(
                hierarchy_mix_float == 0.0,
                anchor_identity_logits.float(),
                mixed_log_probability,
            )

        spectral_aux_logits = interpolate(
            self.spectral_aux_head(spectral_evidence)
        )
        short_aux_logits = interpolate(self.short_aux_head(short_evidence))
        global_aux_logits = interpolate(self.global_aux_head(global_evidence))
        contrast_embeddings = F.normalize(
            self.family_contrast_projection(fused_feature), dim=1
        )
        resized_scale_weights = torch.stack(
            [
                F.interpolate(
                    weights,
                    decoder_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for weights in scale_weights
            ],
            dim=1,
        )

        if sar is None:
            sar_height = max(ms.shape[-2] // 4, 1)
            sar_width = max(ms.shape[-1] // 4, 1)
        elif sar.ndim == 4:
            sar_height, sar_width = sar.shape[-2:]
        else:
            raise ValueError("sar compatibility input must be four-dimensional")
        sar_native_logits = logits.new_zeros(
            ms.shape[0], self.num_classes, sar_height, sar_width
        )
        zero_shared = torch.zeros_like(fused_feature)
        return {
            "logits": logits,
            "ms_logits": logits,
            "flat_logits": flat_logits,
            "local_anchor_logits": local_anchor_logits,
            "complete_anchor_logits": complete_anchor_logits,
            "anchor_context_logits": anchor_outputs["context_logits"],
            "anchor_structure_logits": anchor_outputs["structure_logits"],
            "anchor_router_weights": anchor_outputs["router_weights"],
            "anchor_router_logits": anchor_outputs["router_logits"],
            "anchor_global_logits": anchor_outputs["global_logits"],
            "family_logits": family_logits,
            "family_child_logits": family_child_logits,
            "hierarchical_logits": hierarchical_logits,
            "fine_to_family": self.fine_to_family,
            "family_contrast_embeddings": contrast_embeddings,
            "spectral_aux_logits": spectral_aux_logits,
            "short_aux_logits": short_aux_logits,
            "global_aux_logits": global_aux_logits,
            "context_aux_logits": context_aux_logits,
            "hierarchy_mix": hierarchy_mix,
            "hierarchy_mix_raw": hierarchy_mix_raw,
            "hierarchy_mix_base_raw": self.hierarchy_mix_base.reshape(1),
            "hierarchy_spatial_amplitude": spatial_amplitude.reshape(1),
            "context_residual_scale": context_scale.reshape(1),
            "context_residual_scale_raw": self.context_residual_scale.reshape(1),
            "view_scale_weights": resized_scale_weights,
            # Compatibility-only zero tensors: no SAR feature is consumed.
            "sar_logits": torch.zeros_like(logits),
            "sar_native_logits": sar_native_logits,
            "ms_shared": fused_feature,
            "sar_shared": zero_shared,
            "sar_private": zero_shared,
            "sar_gate": zero_shared[:, :1],
            "sar_scale": logits.new_zeros(1),
            "sar_present": torch.zeros(
                ms.shape[0], dtype=torch.bool, device=ms.device
            ),
        }

    def forward_batch(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward a current or future Line3 batch without PAN/SAR dependence."""

        return self(
            ms=batch["ms"],
            sar=batch.get("sar"),
            sar_valid=batch.get("sar_valid"),
            time_gap_days=batch.get("time_gap_days"),
            ms_context=batch.get("ms_context"),
            ms_context_valid=batch.get("ms_context_valid"),
            ms_short=batch.get("ms_short"),
            ms_short_valid=batch.get("ms_short_valid"),
            ms_global=batch.get("ms_global"),
            ms_global_valid=batch.get("ms_global_valid"),
            local_box_short=batch.get("local_box_short"),
            local_box_global=batch.get("local_box_global"),
        )


__all__ = [
    "LAND_COVER_FAMILIES",
    "MS75Segmentation",
    "TimmPyramidEncoder",
    "TinyPyramidEncoder",
    "UPerDecoder",
    "fine_to_family_map",
]
