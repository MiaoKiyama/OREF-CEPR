"""ConvNeXt with native-grid SAR detail retained before semantic interaction.

The original CROMA21 real-minus-zero path remains. A masked raw-SAR pyramid
exposes 83/42/21 grids; stride16 reads grid42 and stride32 reads grid21. These
odd-kernel CNN grids have different centres from CROMA's padded patch8 grid.
No stride8 insertion or additional task loss is introduced by this variant.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .convnext_sar_model import (
    ConvNeXtSAR, IGNORED_VIEW_KEYS, JOINT_INPUT_KEYS, MODEL_INPUT_KEYS, fixed_sar_intensity,
)
from .joint_sar_model import _validate_sar
from .ms75_sar_joint_model import align_croma_to_ms


class SupportAwareConv(nn.Module):
    """Partial convolution with fractional coverage and no intensity centring.

Padding is unobserved, not a synthetic dark measurement. Renormalisation by
the observed kernel mass avoids diluting valid features next to nodata. The
fractional coverage is propagated separately and gates the eventual residual.
"""
    def __init__(self, cin: int, cout: int, *, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.stride = stride

    def forward(self, value, support):
        if support.shape != (len(value), 1, *value.shape[-2:]):
            raise ValueError("SAR convolution support must match the feature grid")
        with torch.autocast(value.device.type, enabled=False):
            coverage = support.float().clamp(0, 1)
            mass = F.conv2d(coverage, coverage.new_ones((1, 1, 3, 3)),
                            stride=self.stride, padding=1)
        clean = torch.where(coverage > 0, value, torch.zeros_like(value))
        result = F.conv2d(clean * coverage.to(value.dtype), self.conv.weight,
                          bias=None, stride=self.stride, padding=1)
        result = result * (9. / mass.clamp_min(1e-6)).to(result.dtype)
        result = result + self.conv.bias.to(result.dtype)[None, :, None, None]
        result = F.gelu(result) * (mass > 0).to(result.dtype)
        return result, mass / 9.


class NativeSARPyramid(nn.Module):
    """Three trainable native SAR stages, retaining absolute VV/VH levels."""
    strides = (2, 4, 8)

    def __init__(self, channels=(32, 48, 64)):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        if len(self.channels) != 3 or min(self.channels) < 1:
            raise ValueError("SAR pyramid requires three positive channel counts")
        stages, current = [], 3
        for channels in self.channels:
            stages.append(nn.ModuleList((SupportAwareConv(current, channels, stride=2),
                                         SupportAwareConv(channels, channels))))
            current = channels
        self.stages = nn.ModuleList(stages)

    def forward(self, value, valid, *, enabled=True):
        if value.ndim != 4 or value.shape[1:] != (3, 166, 166):
            raise ValueError("raw SAR pyramid requires Bx3x166x166 input")
        if valid.shape != (len(value), 1, 166, 166) or valid.dtype != torch.bool:
            raise ValueError("raw SAR pyramid requires the native boolean support")
        support = valid.float()
        values, supports = [], []
        for stage in self.stages:
            for layer in stage:
                value, support = layer(value, support)
            values.append(value if enabled else value * 0.)
            supports.append(support)
        return tuple(values), tuple(supports)


def align_native_pyramid(feature, support, *, sar_stride, output_size):
    """Coverage-normalized mapping from native CNN centres to ConvNeXt.

The raw lattice is 0.5 + sar_stride*j in SAR pixel-boundary coordinates;
the optical lattice is (i+0.5)*512/N, hence (i+0.5)*166/N on SAR166. Odd
kernel3/pad1 stride2 convolutions preserve the 0.5 origin at every level.
The feature and its support are interpolated jointly, without extrapolating
valid SAR beyond its native raster. Returns value and fractional support.
"""
    if (sar_stride not in (2, 4, 8) or feature.ndim != 4
            or feature.shape[-2:] != (math.ceil(166 / sar_stride),) * 2
            or support.shape != (len(feature), 1, *feature.shape[-2:])
            or len(output_size) != 2 or any(type(n) is not int or n < 1 for n in output_size)):
        raise ValueError("native SAR feature/support lattice differs from its stride")
    with torch.autocast(feature.device.type, enabled=False):
        axes = []
        for n, source_n in zip(output_size, feature.shape[-2:]):
            centres = (torch.arange(n, device=feature.device, dtype=torch.float32) + .5) * (166. / n)
            indices = (centres - .5) / sar_stride
            axes.append(2. * (indices + .5) / source_n - 1.)
        yy, xx = torch.meshgrid(*axes, indexing="ij")
        grid = torch.stack((xx, yy), -1)[None].expand(len(feature), -1, -1, -1)
        coverage = support.float().clamp(0, 1)
        mapped_support = F.grid_sample(coverage, grid, mode="bilinear",
                                       padding_mode="zeros", align_corners=False)
        clean = torch.where(coverage > 0, feature.float(), torch.zeros_like(feature, dtype=torch.float32))
        numerator = F.grid_sample(clean * coverage, grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=False)
        mapped = torch.where(mapped_support > 0, numerator / mapped_support.clamp_min(1e-8),
                             torch.zeros_like(numerator))
    return mapped, mapped_support


class PyramidSARInteraction(nn.Module):
    def __init__(self, optical_channels, croma_channels, raw_channels, width, *, sar_stride):
        super().__init__()
        self.sar_stride = sar_stride
        self.content_value = nn.Conv2d(croma_channels, width, 1, bias=False)
        self.intensity_value = nn.Conv2d(raw_channels, width, 1, bias=False)
        self.query = nn.Conv2d(optical_channels, width, 1)
        self.output = nn.Conv2d(width, optical_channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, optical, content, raw, raw_support, observed, optical_valid):
        size = tuple(optical.shape[-2:])
        # Preserve the old CROMA sampling convention, separately from raw CNN.
        content_value = align_croma_to_ms(self.content_value(content), size)
        content_support = align_croma_to_ms(observed.float(), size) > 0
        raw_value, raw_coverage = align_native_pyramid(self.intensity_value(raw), raw_support,
            sar_stride=self.sar_stride, output_size=size)
        value = content_value * content_support + raw_value * raw_coverage
        optical_support = F.adaptive_avg_pool2d(optical_valid.float(), size)
        delta = self.output(value.to(optical.dtype) * self.query(optical).sigmoid())
        delta = delta * optical_support.to(delta.dtype)
        return optical + delta, delta


class ConvNeXtPyramidSAR(ConvNeXtSAR):
    """Original optical expert plus two native-scale SAR residual insertions."""
    def __init__(self, config, load_pretrained=True, *, backbone=None, sar_encoder=None):
        cfg = dict(config)
        cfg.update(config.get("model", {}))
        if cfg.get("variant", "pyramid") != "pyramid":
            raise ValueError("ConvNeXtPyramidSAR requires variant=pyramid")
        super().__init__(config, load_pretrained, backbone=backbone, sar_encoder=sar_encoder)
        channels = tuple(cfg.get("sar_pyramid_channels", (32, 48, 64)))
        self.sar_intensity_encoder = NativeSARPyramid(channels)
        self.interactions = nn.ModuleList(
            PyramidSARInteraction(optical_channels, self.sar_encoder.channels[-1], raw_channels,
                int(cfg.get("sar_interaction_width", 64)), sar_stride=stride)
            for optical_channels, raw_channels, stride in zip(self.backbone.channels[2:], channels[1:], (4, 8)))
        self._sar_new_ids = {id(p) for m in (self.interactions, self.sar_aux_head, self.sar_intensity_encoder)
                             for p in m.parameters()}
        self._mutable_ids = self._backbone_ids | self._sar_tail_ids | self._sar_new_ids | self._optical_new_ids
        self.provenance = {**self.provenance, "schema": "convnext_croma_native_raw_pyramid_v1",
            "variant": "pyramid", "raw_pyramid_channels": list(channels),
            "raw_pyramid_strides": [2, 4, 8], "raw_pyramid_sizes": [83, 42, 21],
            "interaction_raw_strides": [4, 8], "interaction_strides": [16, 32],
            "raw_geometry": "kernel3_pad1_stride2;centres=0.5+stride*j;support_normalized",
            "raw_support": "fractional_partial_convolution_then_joint_value_support_resampling",
            "zero_sar": "CROMA_real_zero_difference_and_all_raw_levels_exact_zero"}
        self.set_stage("optical_warmup")

    def forward(self, *, ms, sar_db, sar_valid, optical_valid, sar_mode="real", **unused):
        if set(unused) - IGNORED_VIEW_KEYS:
            raise ValueError("unexpected model inputs: " + ",".join(sorted(set(unused) - IGNORED_VIEW_KEYS)))
        _validate_sar(sar_db, sar_valid, sar_mode)
        if (len(ms) != len(sar_db) or ms.device != sar_db.device
                or optical_valid.shape != (len(ms), 1, 512, 512)
                or optical_valid.dtype != torch.bool or optical_valid.device != ms.device):
            raise ValueError("matching native MS/SAR batch and boolean optical support required")
        optical, spectral = self._optical_features(ms)
        present = sar_valid.flatten(1).any(1) & optical_valid.flatten(1).any(1)
        active = not (self.training and self.stage == "optical_warmup")
        if active:
            real, content = self._encode_pair(sar_db, sar_valid, sar_mode)
            observed = F.avg_pool2d(F.pad(sar_valid.float(), (1, 1, 1, 1)), 8, 8) > 0
            raw, supports = self.sar_intensity_encoder(fixed_sar_intensity(sar_db, sar_valid, sar_mode),
                                                       sar_valid, enabled=sar_mode == "real")
            fused, deltas = list(optical), []
            for index, raw_index, interaction in zip((2, 3), (1, 2), self.interactions):
                fused[index], delta = interaction(optical[index], content, raw[raw_index], supports[raw_index],
                    observed, optical_valid & present[:, None, None, None])
                deltas.append(delta)
            result = self._decode(tuple(fused), spectral, optical)
            result.update(sar_aux_logits=self.sar_aux_head(real), sar_aux_valid=observed,
                sar_content_difference=content, sar_intensity_feature=raw[-1],
                sar_pyramid_features=raw, sar_pyramid_supports=supports,
                sar_stride16_delta=deltas[0], sar_stride32_delta=deltas[1])
        else:
            result = self._decode(optical, spectral, optical)
            result.update(sar_aux_logits=ms.new_zeros((len(ms), 24, 21, 21)),
                sar_aux_valid=torch.zeros((len(ms), 1, 21, 21), device=ms.device, dtype=torch.bool),
                sar_stride16_delta=torch.zeros_like(optical[2]), sar_stride32_delta=torch.zeros_like(optical[3]))
        result["sar_present"] = present
        return result


__all__ = ["ConvNeXtPyramidSAR", "NativeSARPyramid", "SupportAwareConv", "PyramidSARInteraction",
           "align_native_pyramid", "MODEL_INPUT_KEYS", "JOINT_INPUT_KEYS"]
