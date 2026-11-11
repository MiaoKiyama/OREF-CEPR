"""Shared-backbone local/context ConvNeXt, with unchanged native SAR inserts.

The context is an explicitly centred 1024-pixel BGRN field resampled to512.
Its central physical512 field is sampled at each local feature-cell centre;
it is never stretched as though the whole context covered the local crop.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .convnext_sar_model import ConvNeXtSAR, JOINT_INPUT_KEYS, IGNORED_VIEW_KEYS, fixed_sar_intensity
from .joint_sar_model import _validate_sar

CONTEXT_INPUT_KEYS = ("ms_context", "context_valid", "local_box_context")
UPGRADE_INPUT_KEYS = JOINT_INPUT_KEYS + CONTEXT_INPUT_KEYS


def align_context_feature(feature, local_box, size):
    """Sample context features at local-cell physical centres (align_corners=False)."""
    if feature.ndim != 4 or local_box.shape != (len(feature), 4):
        raise ValueError("Expected BCHW context and Bx4 normalized y0,x0,y1,x1 box")
    if not bool(torch.isfinite(local_box).all()) or not bool(((local_box >= 0) & (local_box <= 1)).all()):
        raise ValueError("Context boxes must be finite and within [0,1]")
    if not bool((local_box[:, 2:] > local_box[:, :2]).all()):
        raise ValueError("Context box must have positive area")
    h, w = size
    expected = local_box.new_tensor([.25,.25,.75,.75]).expand_as(local_box)
    if not torch.equal(local_box,expected) or h % 2 or w % 2:
        raise ValueError("This native protocol requires central-half boxes and even feature sizes")
    # Resample the WHOLE context before cropping. Cropping its central half
    # first would clamp interpolation at the crop edges and shift boundary
    # samples. This is equivalent to the physical-coordinate grid, without
    # grid_sample's unsupported deterministic CUDA backward.
    with torch.autocast(feature.device.type, enabled=False):
        full = F.interpolate(feature.float(),size=(2*h,2*w),mode="bilinear",align_corners=False)
        return full[:,:,h//2:3*h//2,w//2:3*w//2].to(feature.dtype)


class ConvNeXtContextSAR(ConvNeXtSAR):
    """The new context projections belong to the optical optimizer/EMA union."""
    input_keys = UPGRADE_INPUT_KEYS

    def __init__(self, config, load_pretrained=True, *, backbone=None, sar_encoder=None):
        cfg = dict(config)
        cfg.update(config.get("model", {}))
        if cfg.get("context_source_size", 1024) != 1024:
            raise ValueError("This protocol fixes a centred1024 context for local512")
        super().__init__(config, load_pretrained, backbone=backbone, sar_encoder=sar_encoder)
        self.context_projections = nn.ModuleList(nn.Conv2d(c, c, 1, bias=False)
                                                for c in self.backbone.channels[2:])
        for module in self.context_projections:
            nn.init.zeros_(module.weight)
        extra = {id(p) for p in self.context_projections.parameters()}
        self._optical_new_ids |= extra
        self._mutable_ids |= extra
        self.provenance = dict(self.provenance, schema="convnext_centered_context1024_sar_v1",
            context_source_size=1024, context_input_size=512, local_box_context=[.25,.25,.75,.75],
            context_backbone="same_parameter_objects_as_local", context_strides=[16,32],
            context_projection="single_zero_initialized_terminal_1x1", context_geometry="physical_cell_centres")
        self.set_stage(self.stage)

    def _context_features(self, ms, ms_context, context_valid, local_box_context):
        if (ms_context.shape != ms.shape or tuple(ms_context.shape[1:]) != (4,512,512)
                or context_valid.shape != (len(ms),1,512,512) or context_valid.dtype != torch.bool
                or ms_context.device != ms.device or context_valid.device != ms.device
                or local_box_context.device != ms.device):
            raise ValueError("Context must share the batch/device and use native B4x512x512 with B1 bool support")
        expected = local_box_context.new_tensor([.25,.25,.75,.75]).expand(len(ms),-1)
        if local_box_context.shape != expected.shape or not torch.equal(local_box_context, expected):
            raise ValueError("Centred1024 context requires the exact central-half physical box")
        optical, spectral = super()._optical_features(ms)
        # Invalid padded values cannot carry arbitrary information into the encoder.
        clean = torch.where(context_valid, ms_context, torch.zeros_like(ms_context))
        context = self.backbone(self.rgb_input(clean))
        out = list(optical)
        for i, projection in zip((2,3), self.context_projections):
            size = tuple(optical[i].shape[-2:])
            aligned = align_context_feature(context[i], local_box_context, size)
            support = align_context_feature(context_valid.float(), local_box_context, size)
            out[i] = optical[i] + projection(aligned).to(optical[i].dtype) * support.to(optical[i].dtype)
        return tuple(out), spectral

    def forward_optical(self, *, ms, ms_context, context_valid, local_box_context, **unused):
        optical, spectral = self._context_features(ms, ms_context, context_valid, local_box_context)
        return self._decode(optical, spectral, optical)

    def forward(self, *, ms, ms_context, context_valid, local_box_context, sar_db, sar_valid,
                optical_valid, sar_mode="real", **unused):
        if set(unused) - IGNORED_VIEW_KEYS:
            raise ValueError("Unexpected model inputs: " + ",".join(sorted(set(unused) - IGNORED_VIEW_KEYS)))
        if sar_mode == "off":
            return self.forward_optical(ms=ms, ms_context=ms_context, context_valid=context_valid,
                                        local_box_context=local_box_context)
        _validate_sar(sar_db, sar_valid, sar_mode)
        if (len(ms) != len(sar_db) or ms.device != sar_db.device or optical_valid.shape != (len(ms),1,512,512)
                or optical_valid.dtype != torch.bool or optical_valid.device != ms.device):
            raise ValueError("Matching native MS/SAR and optical support required")
        optical, spectral = self._context_features(ms, ms_context, context_valid, local_box_context)
        present = sar_valid.flatten(1).any(1) & optical_valid.flatten(1).any(1)
        active = not (self.training and self.stage == "optical_warmup")
        if active:
            real, content = self._encode_pair(sar_db, sar_valid, sar_mode)
            observed = F.avg_pool2d(F.pad(sar_valid.float(), (1,1,1,1)), 8, 8) > 0
            intensity = self.sar_intensity_encoder(fixed_sar_intensity(sar_db, sar_valid, sar_mode))
            intensity = F.avg_pool2d(F.pad(intensity * sar_valid, (1,1,1,1)), 8, 8)
            intensity = intensity * observed * (1. if sar_mode == "real" else 0.)
            fused, deltas = list(optical), []
            for index, interaction in zip((2,3), self.interactions):
                fused[index], delta = interaction(optical[index], content, intensity, observed,
                                                  optical_valid & present[:,None,None,None])
                deltas.append(delta)
            result = self._decode(tuple(fused), spectral, optical)
            result.update(sar_aux_logits=self.sar_aux_head(real), sar_aux_valid=observed,
                          sar_content_difference=content, sar_intensity_feature=intensity,
                          sar_stride16_delta=deltas[0], sar_stride32_delta=deltas[1])
        else:
            result = self._decode(optical, spectral, optical)
            result.update(sar_aux_logits=ms.new_zeros((len(ms),24,21,21)),
                          sar_aux_valid=torch.zeros((len(ms),1,21,21),device=ms.device,dtype=torch.bool),
                          sar_stride16_delta=torch.zeros_like(optical[2]), sar_stride32_delta=torch.zeros_like(optical[3]))
        result["sar_present"] = present
        return result

    def forward_batch(self, batch):
        return self(**{key:batch[key] for key in self.input_keys})


__all__ = ["ConvNeXtContextSAR", "align_context_feature", "CONTEXT_INPUT_KEYS", "UPGRADE_INPUT_KEYS"]
