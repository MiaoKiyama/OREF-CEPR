"""One optical encoding, with disjoint optical-off and SAR-on gradients.

The deployed model has exactly the ordinary ConvNeXtSAR forward. Only training
is paired: optical parameters learn from SAR-off GT, while SAR parameters learn
through a decoder whose *parameters* are detached, not through a no-grad decode.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.func import functional_call

from .convnext_sar_model import ConvNeXtSAR, JOINT_INPUT_KEYS, IGNORED_VIEW_KEYS, fixed_sar_intensity
from .joint_sar_model import _validate_sar


class ConvNeXtPairedSAR(ConvNeXtSAR):
    def __init__(self, config, load_pretrained=True, **kwargs):
        super().__init__(config, load_pretrained=load_pretrained, **kwargs)
        self.provenance = dict(self.provenance, paired_training=True,
            optical_objective="full_domain_SAR_off_GT",
            sar_objective="same_crop_SAR_on_GT_detached_optical_and_decoder_parameters",
            optical_encoding_passes=1, inference="ordinary_single_SAR_on_forward")

    def gradient_parameter_sets(self):
        """The two sets are disjoint and cover the complete mutable inventory."""
        optical_ids = self._backbone_ids | self._optical_new_ids
        sar_ids = self._sar_tail_ids | self._sar_new_ids
        if optical_ids & sar_ids or optical_ids | sar_ids != self._mutable_ids:
            raise RuntimeError("paired optical/SAR parameter ownership differs")
        return {"optical": [p for p in self.parameters() if id(p) in optical_ids],
                "sar": [p for p in self.parameters() if id(p) in sar_ids]}

    def forward_paired(self, *, ms, sar_db, sar_valid, optical_valid,
                       sar_mode="real", **unused):
        if set(unused) - IGNORED_VIEW_KEYS:
            raise ValueError("unexpected paired inputs: " + ",".join(sorted(set(unused)-IGNORED_VIEW_KEYS)))
        _validate_sar(sar_db, sar_valid, sar_mode)
        if (len(ms) != len(sar_db) or ms.device != sar_db.device
                or optical_valid.shape != (len(ms), 1, 512, 512)
                or optical_valid.dtype != torch.bool or optical_valid.device != ms.device):
            raise ValueError("paired MS/SAR batch, device and support must agree")
        optical, spectral = self._optical_features(ms)
        off = self._decode(optical, spectral, optical)
        present = sar_valid.flatten(1).any(1) & optical_valid.flatten(1).any(1)
        if self.stage == "optical_warmup":
            return {"off": off, "on": None, "sar_present": present}

        # Detaching features also detaches the optical input to each query. The
        # query projection itself belongs to the trainable SAR interaction.
        optical_detached = tuple(x.detach() for x in optical)
        spectral_detached = tuple(x.detach() for x in spectral)
        real, content = self._encode_pair(sar_db, sar_valid, sar_mode)
        observed = F.avg_pool2d(F.pad(sar_valid.float(), (1, 1, 1, 1)), 8, 8) > 0
        intensity = self.sar_intensity_encoder(fixed_sar_intensity(sar_db, sar_valid, sar_mode))
        intensity = F.avg_pool2d(F.pad(intensity * sar_valid, (1, 1, 1, 1)), 8, 8)
        intensity = intensity * observed * (1. if sar_mode == "real" else 0.)
        fused, deltas = list(optical_detached), []
        for index, interaction in zip((2, 3), self.interactions):
            fused[index], delta = interaction(optical_detached[index], content, intensity,
                observed, optical_valid & present[:, None, None, None])
            deltas.append(delta)

        parameters = {name: value.detach() for name, value in self.decoder.named_parameters()}
        buffers = {name: value.detach() for name, value in self.decoder.named_buffers()}
        # No torch.no_grad here: decoder Jacobians must reach SAR feature values.
        logits = functional_call(self.decoder, (parameters, buffers),
                                 (tuple(fused), spectral_detached), strict=True)
        on = dict(logits=logits, sar_aux_logits=self.sar_aux_head(real),
                  sar_aux_valid=observed, sar_content_difference=content,
                  sar_intensity_feature=intensity,
                  sar_stride16_delta=deltas[0], sar_stride32_delta=deltas[1])
        return {"off": off, "on": on, "sar_present": present}

    def forward_paired_batch(self, batch):
        return self.forward_paired(**{key: batch[key] for key in JOINT_INPUT_KEYS})


__all__ = ["ConvNeXtPairedSAR"]
