"""Isolated old-MS75/CROMA joint expert; no immutable-probability fallback.

The caller strictly loads the authenticated original MS75 topology first.
Two hooks modify semantic tensors inside that owned optical instance; its
original forward, hierarchy and auxiliary outputs remain authoritative.
Hooks carry no persistent batch state after forward and are never installed
on the caller's production model. Local Swin stays frozen. CROMA uses its
authenticated native166/pad1/patch8 implementation, with a trainable tail.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .joint_sar_model import JointCromaSAR, _validate_sar
from .ms75_model import MS75Segmentation

MODEL_INPUT_KEYS = ("ms", "ms_short", "ms_short_valid", "ms_global",
                    "ms_global_valid", "local_box_short", "local_box_global")
JOINT_INPUT_KEYS = (*MODEL_INPUT_KEYS, "sar_db", "sar_valid", "optical_valid")
MS_SIZE, SAR_SIZE, PADDED_SAR_SIZE, SAR_PATCH = 512, 166, 168, 8
TOKEN_SIZE = 21
FROZEN_OPTICAL_PREFIXES = ("local_encoder.", "short_encoder.",
                           "context_encoder.", "spectral_encoder.")


def align_croma_to_ms(feature: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Native CROMA token centres onto an unpadded MS semantic lattice.

    MS centres: (i+.5)*512/N in pixel-boundary coordinates. CROMA centres
    are 8*(j+.5)-1 on unpadded SAR166, so j=(1+centre*166/512)/8-.5.
    This is NOT the DINO518 transform or the raw-CNN stride8 transform.
    """
    if (feature.ndim != 4 or feature.shape[-2:] != (21, 21)
            or not feature.is_floating_point() or len(output_size) != 2
            or any(type(n) is not int or n < 1 for n in output_size)):
        raise ValueError("CROMA alignment requires a native21 lattice and positive output size")
    with torch.autocast(feature.device.type, enabled=False):
        axes = []
        for n in output_size:
            padded = 1.0 + (torch.arange(n, device=feature.device, dtype=torch.float32) + .5) * (166.0 / n)
            axes.append(2.0 * padded / 168.0 - 1.0)
        yy, xx = torch.meshgrid(*axes, indexing="ij")
        grid = torch.stack((xx, yy), -1)[None].expand(feature.shape[0], -1, -1, -1)
        return F.grid_sample(feature.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)


class ContentSARResidual(nn.Module):
    """Optical controls the gate; only a SAR difference supplies the value."""
    def __init__(self, optical_channels: int, sar_channels: int = 768, width: int = 64):
        super().__init__()
        self.value = nn.Conv2d(sar_channels, width, 1, bias=False)
        self.query = nn.Conv2d(optical_channels, width, 1)
        self.output = nn.Conv2d(width, optical_channels, 1, bias=False)
        # One zero terminal projection, no second zero gate; auxiliary SAR
        # supervision can train the tail on the very first warmup update.
        nn.init.zeros_(self.output.weight)

    def forward(self, optical: torch.Tensor, difference: torch.Tensor,
                optical_valid: torch.Tensor, observed: torch.Tensor):
        size = tuple(optical.shape[-2:])
        value = align_croma_to_ms(self.value(difference), size).to(optical.dtype)
        support = F.adaptive_avg_pool2d(optical_valid.float(), size)
        support = support * (align_croma_to_ms(observed.float(), size) > 0).to(support.dtype)
        gated = value * torch.sigmoid(self.query(optical))
        delta = self.output(gated) * support.to(optical.dtype)
        return optical + delta, delta, support


class MS75SARJoint(nn.Module):
    """Original MS75 with a learnable CROMA tail and two semantic injections.

    ``mutable_keys()`` and optimizer groups cover BOTH stages from the start,
    including currently frozen optical parameters. Changing stage never changes
    the optimizer's membership or the EMA/checkpoint inventory. All BN running
    statistics are fixed. ``tail_lr`` is the CROMA/short-encoder tail rate;
    ``sar_lr`` is the new interaction/auxiliary rate; old heads use optical_lr.
    """
    num_classes = 24

    def __init__(self, optical: MS75Segmentation, croma_checkpoint,
                 croma_source=None):
        super().__init__()
        if (not isinstance(optical, MS75Segmentation) or optical.num_classes != 24
                or len(optical.level_fusions) != 4):
            raise ValueError("joint expert requires the original complete 24-class MS75 topology")
        self.optical = optical
        self.sar_encoder = JointCromaSAR(croma_checkpoint, croma_source)
        channels = int(self.sar_encoder.channels[-1])
        evidence_channels = int(optical.level_fusions[-1].refine[0].out_channels)
        context_channels = int(optical.context_aggregate[0].out_channels)
        self.interactions = nn.ModuleList((ContentSARResidual(evidence_channels, channels),
                                           ContentSARResidual(context_channels, channels)))
        self.sar_aux_head = nn.Conv2d(channels, 24, 1)
        self._short_tail = self._find_short_tail()
        self._sar_parameter_names = frozenset("sar_encoder." + n for n, p in self.sar_encoder.named_parameters() if p.requires_grad)
        self._optical_parameter_names = frozenset(
            "optical." + n for n, _ in optical.named_parameters()
            if not n.startswith(FROZEN_OPTICAL_PREFIXES) or n.startswith(self._short_tail + "."))
        self._new_parameter_names = frozenset(n for n, _ in self.named_parameters()
                                              if n.startswith(("interactions.", "sar_aux_head.")))
        self._mutable_parameter_names = self._sar_parameter_names | self._optical_parameter_names | self._new_parameter_names
        self._batch_context = None
        self._injection_records = None
        self._hooks = (optical.level_fusions[-1].register_forward_hook(self._evidence_hook),
                       optical.context_aggregate.register_forward_hook(self._context_hook))
        self.provenance = {
            "schema": "old_ms75_croma_joint_native_v1", "sar": dict(self.sar_encoder.provenance),
            "interactions": ["level_fusions.3.evidence", "context_aggregate"],
            "value": "CROMA(real)-CROMA(normalized_zero); optical_query_only; bias_free_output",
            "zero_control_warning": "zero_difference_is_structural_identity_not_semantic_evidence",
            "sar_geometry": "native166_pad1_each_side_patch8_21; optical512_unpadded",
            "sar_aux_target": "optical_pixel_centres_floor((1+(i+.5)*166/512)/8), known_GT_and_optical_valid",
            "optical_tail": self._short_tail, "local_encoder": "permanently_frozen_eval",
            "batch_norm": "all_running_statistics_fixed_eval", "missing_sar": "current_same_optical_model",
        }
        self.set_stage("warmup")

    def _find_short_tail(self) -> str:
        for path in ("short_encoder.model.layer4", "short_encoder.stages.3"):
            try:
                module = self.optical.get_submodule(path)
            except AttributeError:
                continue
            if any(True for _ in module.parameters()):
                return path
        raise ValueError("cannot authenticate the MS75 short ResNet final stage")

    def set_stage(self, stage: str):
        if stage not in ("warmup", "joint"):
            raise ValueError("stage must be warmup or joint")
        self.stage = stage
        active = self._sar_parameter_names | self._new_parameter_names
        if stage == "joint":
            active = active | self._optical_parameter_names
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in active)
        self.train(self.training)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "stage"):
            self.optical.train(mode and self.stage == "joint")
            for name in ("local_encoder", "short_encoder", "context_encoder", "spectral_encoder"):
                self.optical.get_submodule(name).eval()
            if mode and self.stage == "joint":
                self.optical.get_submodule(self._short_tail).train(True)
            self.sar_encoder.train(mode)
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def mutable_keys(self) -> tuple[str, ...]:
        state_keys = set(self.state_dict())
        buffers = {name for name, _ in self.named_buffers()} & state_keys
        return tuple(sorted(self._mutable_parameter_names | buffers))

    def parameter_groups(self, optical_lr: float, sar_lr: float, tail_lr: float,
                         weight_decay: float = .01) -> list[dict[str, Any]]:
        if (any(not math.isfinite(v) or v <= 0 for v in (optical_lr, sar_lr, tail_lr))
                or not math.isfinite(weight_decay) or weight_decay < 0):
            raise ValueError("invalid optimizer rates/decay")
        bins = {}
        for name, parameter in self.named_parameters():
            if name not in self._mutable_parameter_names:
                continue
            if name in self._sar_parameter_names:
                label, rate = "sar_tail", tail_lr
            elif name.startswith("optical." + self._short_tail + "."):
                label, rate = "optical_tail", tail_lr
            elif name in self._new_parameter_names:
                label, rate = "new_modules", sar_lr
            else:
                label, rate = "optical_heads", optical_lr
            decay = parameter.ndim > 1 and not name.endswith(".bias")
            key = (label, rate, decay)
            bins.setdefault(key, []).append((name, parameter))
        groups = [{"params": [p for _, p in members], "parameter_names": [n for n, _ in members],
                   "lr": rate, "initial_lr": rate, "weight_decay": weight_decay if decay else 0.,
                   "group_name": label + ("_decay" if decay else "_no_decay")}
                  for (label, rate, decay), members in sorted(bins.items())]
        names = [n for g in groups for n in g["parameter_names"]]
        if len(names) != len(set(names)) or set(names) != self._mutable_parameter_names:
            raise RuntimeError("optimizer inventory mismatch")
        return groups

    @torch.no_grad()
    def project_constrained_parameters_(self):
        self.optical.project_constrained_parameters_()
        return self

    def _inject(self, index, value):
        if self._batch_context is None:
            return value
        expected_size = ((16, 16), (128, 128))[index]
        if tuple(value.shape[-2:]) != expected_size:
            raise ValueError("original MS75 native512 Swin patch4 semantic lattice changed")
        difference, valid, observed = self._batch_context
        result, delta, support = self.interactions[index](value, difference, valid, observed)
        self._injection_records[index] = (delta, support)
        return result

    def _evidence_hook(self, module, args, output):
        if self._batch_context is None:
            return output
        return (self._inject(0, output[0]), *output[1:])

    def _context_hook(self, module, args, output):
        return self._inject(1, output)

    def forward_optical(self, **inputs):
        if self._batch_context is not None:
            raise RuntimeError("optical-only path cannot nest inside a SAR forward")
        return self.optical(**{k: inputs[k] for k in MODEL_INPUT_KEYS if k in inputs})

    def _encode_pair(self, db, valid, mode):
        # Match stochastic state in both encodings; reference evaluation must
        # not consume a second independent dropout/DropPath realization.
        cpu_state = torch.random.get_rng_state()
        devices = [db.device.index if db.device.index is not None else torch.cuda.current_device()] if db.is_cuda else []
        device_state = torch.cuda.get_rng_state(devices[0]) if devices else None
        real = self.sar_encoder(db, valid, mode)[-1]
        with torch.random.fork_rng(devices=devices):
            torch.random.set_rng_state(cpu_state)
            if devices:
                torch.cuda.set_rng_state(device_state, devices[0])
            reference = self.sar_encoder(db, valid, "zero")[-1]
        return real, real - reference

    def forward(self, *, sar_db, sar_valid, optical_valid, sar_mode="real", **inputs):
        if self._batch_context is not None:
            raise RuntimeError("joint forward is not reentrant")
        unknown = set(inputs) - set(MODEL_INPUT_KEYS)
        if unknown:
            raise ValueError("unknown optical model inputs: " + ",".join(sorted(unknown)))
        _validate_sar(sar_db, sar_valid, sar_mode)
        ms = inputs["ms"]
        if (ms.shape != (sar_db.shape[0], 4, 512, 512) or ms.device != sar_db.device
                or optical_valid.shape != (ms.shape[0], 1, 512, 512)
                or optical_valid.dtype != torch.bool or optical_valid.device != ms.device):
            raise ValueError("joint forward requires native512 optical and boolean physical SAR coverage")
        real, difference = self._encode_pair(sar_db, sar_valid, sar_mode)
        observed = F.avg_pool2d(F.pad(sar_valid.float(), (1, 1, 1, 1)), 8, 8) > 0
        present = sar_valid.flatten(1).any(1) & optical_valid.flatten(1).any(1)
        effective = optical_valid & present[:, None, None, None]
        self._batch_context = (difference, effective, observed)
        self._injection_records = {}
        try:
            result = self.optical(**inputs)
            if set(self._injection_records) != {0, 1}:
                raise RuntimeError("both semantic injections must execute in the original MS75 forward")
            records = self._injection_records
        finally:
            self._batch_context = None
            self._injection_records = None
        result.update({"sar_aux_logits": self.sar_aux_head(real), "sar_aux_valid": observed,
                       "sar_content_difference": difference, "sar_present": present,
                       "sar_evidence_delta": records[0][0], "sar_context_delta": records[1][0],
                       "sar_evidence_coverage": records[0][1], "sar_context_coverage": records[1][1]})
        return result

    def forward_batch(self, batch: Mapping[str, torch.Tensor]):
        return self(**{k: batch[k] for k in JOINT_INPUT_KEYS if k in batch})


__all__ = ["MS75SARJoint", "ContentSARResidual", "align_croma_to_ms",
           "MODEL_INPUT_KEYS", "JOINT_INPUT_KEYS"]
