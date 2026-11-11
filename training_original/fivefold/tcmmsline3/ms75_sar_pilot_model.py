"""Isolated native-512 MS75 continuation with one raw-SAR feature interaction.

The caller authenticates the selected train96 EMA and strictly loads it into
a *new*, dedicated MS75RebuildSegmentation instance before transferring ownership
to this wrapper. No checkpoint is selected/loaded here, and no original model
implementation is modified. P shares the decoder training budget; R/Z share the
entire architecture and initial state. Z zeros normalized content, not validity.

The explicit multiview forward below mirrors MS75Segmentation.forward, with one
insertion after context_aggregate. It deliberately does not edit returned logits
or ms_shared: that would miss the flat-logit branch. HR decoding is unchanged.
Output selection against the immutable base at unsupported pixels is external;
feature masking alone is not an exact spatial fallback after decoder training.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .ms75_castr import AnchorContextState
from .ms75_model import _straight_through_clamp
from .ms75_rebuild_model import MODEL_INPUT_KEYS, MS75RebuildSegmentation


MS_SIZE, SAR_SIZE, CONTEXT_SIZE = 512, 166, 128
PILOT_INPUT_KEYS = (*MODEL_INPUT_KEYS, "sar_db", "sar_valid", "optical_valid")
FROZEN_PREFIXES = ("local_encoder.", "short_encoder.", "context_encoder.",
                   "spectral_encoder.", "detail_decoder.encoder.")
TRAINABLE_PREFIXES = (
    "decoder.", "local_head.", "context_projection.", "global_head.",
    "structure_input.", "structure_blocks.", "context_fusion.", "structure_fusion.",
    "context_refine.", "structure_refine.", "context_head.", "structure_head.",
    "router.", "level_fusions.", "context_aggregate.", "spectral_aggregate.",
    "short_aggregate.", "global_aggregate.", "context_aux_head.", "spectral_aux_head.",
    "short_aux_head.", "global_aux_head.", "family_head.", "family_child_heads.",
    "family_contrast_projection.", "hierarchy_mix_gate.",
    "detail_decoder.context_projection.", "detail_decoder.decoder.",
    "detail_decoder.classifier.",
)
TRAINABLE_SCALARS = frozenset(("context_residual_scale", "hierarchy_mix_base",
                               "detail_residual_scale"))


def _pair(values: Sequence[float], name: str, *, positive: bool = False) -> tuple[float, float]:
    values = tuple(float(v) for v in values)
    if len(values) != 2 or any(not math.isfinite(v) or (positive and v <= 0) for v in values):
        raise ValueError(f"{name} needs two finite {'positive ' if positive else ''}values")
    return values


def align_native_sar(feature: torch.Tensor, *, sar_stride: int = 8) -> torch.Tensor:
    """Sample 166 support at Swin patch4 centres, not the R18 or DINO grid.

    MS patch4 centres are 2+4*i in the 512-pixel boundary frame. Native SAR CNN
    odd kernels/padding preserve centres .5+stride*j in the 166-sample frame.
    j=((2+4*i)*166/512-.5)/stride. align_corners=False adds half a *feature*
    pixel only when converting j to grid_sample's normalized coordinates.
    Border extension follows the native reader. No 168/518 padding is present.
    """
    if (not isinstance(feature, torch.Tensor) or feature.ndim != 4
            or not feature.is_floating_point() or sar_stride not in (1, 2, 4, 8)
            or feature.shape[-2:] != (math.ceil(SAR_SIZE / sar_stride),) * 2):
        raise ValueError("incorrect native SAR lattice; expected 166 support")
    with torch.autocast(device_type=feature.device.type, enabled=False):
        centres = 2.0 + 4.0 * torch.arange(CONTEXT_SIZE, device=feature.device, dtype=torch.float32)
        indices = (centres * (SAR_SIZE / MS_SIZE) - .5) / sar_stride
        axis = 2.0 * (indices + .5) / feature.shape[-1] - 1.0
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack((xx, yy), -1)[None].expand(len(feature), -1, -1, -1)
        return F.grid_sample(feature.float(), grid, mode="bilinear",
                             padding_mode="border", align_corners=False)


class NativeSAREncoder(nn.Module):
    """166 VV/VH values -> 21 spatial features; absolute dB scale is retained."""

    def __init__(self) -> None:
        super().__init__()
        blocks, previous = [], 2
        for channels in (32, 64, 128):
            blocks.append(nn.Sequential(
                nn.Conv2d(previous, channels, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.GELU()))
            previous = channels
        self.stages = nn.Sequential(*blocks)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.stages(value)


class ContextSARInteraction(nn.Module):
    def __init__(self, context_channels: int) -> None:
        super().__init__()
        self.sar_projection = nn.Conv2d(128, 64, 1)
        self.mix = nn.Sequential(nn.Conv2d(context_channels + 65, 64, 3, padding=1), nn.GELU())
        self.output = nn.Conv2d(64, context_channels, 1)
        # One zero terminal projection, no second zero scalar gate.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context: torch.Tensor, sar: torch.Tensor,
                coverage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (context.ndim != 4 or context.shape[-2:] != (CONTEXT_SIZE,) * 2
                or sar.shape != (len(context), 128, 21, 21)
                or coverage.shape != (len(context), 1, CONTEXT_SIZE, CONTEXT_SIZE)):
            raise ValueError("MS75 context/SAR interaction requires the native 512/166 geometry")
        projected = align_native_sar(self.sar_projection(sar))
        mixed = self.mix(torch.cat((context, projected.to(context.dtype), coverage.to(context.dtype)), 1))
        delta = self.output(mixed) * coverage.to(context.dtype)
        return context + delta, projected, delta


class MS75SARPilot(nn.Module):
    """Own a dedicated, already initialized base; never use old train_stage.

    R and Z state_dict schemas are identical. Strictly copy the initialized R
    state into Z (including train-only SAR mean/std); set constructor arm='Z'.
    P has no SAR branch and ignores SAR arguments. All arms train the same
    explicit optical decoder/head whitelist. Frozen encoders stay in eval even
    after train(), disabling DropPath/dropout and running-buffer updates.
    """
    num_classes = 24

    def __init__(self, base: MS75RebuildSegmentation, *, arm: str,
                 sar_mean: Sequence[float] = (0., 0.),
                 sar_std: Sequence[float] = (1., 1.)) -> None:
        super().__init__()
        if not isinstance(base, MS75RebuildSegmentation) or base.variant not in ("r", "hr", "robust"):
            raise ValueError("pilot requires a separately initialized native MS75 rebuild base")
        if arm not in ("P", "R", "Z"):
            raise ValueError("pilot arm must be P, R or Z")
        self.optical, self.arm = base, arm
        self.multimodal = arm != "P"
        self.variant = base.variant
        if self.multimodal:
            self.register_buffer("sar_mean", torch.tensor(_pair(sar_mean, "sar_mean"), dtype=torch.float32)[None, :, None, None])
            self.register_buffer("sar_std", torch.tensor(_pair(sar_std, "sar_std", positive=True), dtype=torch.float32)[None, :, None, None])
            self.sar_encoder = NativeSAREncoder()
            self.interaction = ContextSARInteraction(base.local_head.in_channels)
        self.configure_trainable_parameters_()
        self.train(base.training)

    def configure_trainable_parameters_(self) -> dict[str, Any]:
        frozen, trainable = [], []
        for name, parameter in self.optical.named_parameters():
            is_frozen = name.startswith(FROZEN_PREFIXES)
            is_trainable = name.startswith(TRAINABLE_PREFIXES) or name in TRAINABLE_SCALARS
            if is_frozen == is_trainable:
                raise ValueError(f"unclassified/overlapping MS75 parameter: {name}")
            parameter.requires_grad_(is_trainable)
            (trainable if is_trainable else frozen).append("optical." + name)
        if self.multimodal:
            for prefix in ("sar_encoder", "interaction"):
                for name, parameter in getattr(self, prefix).named_parameters():
                    parameter.requires_grad_(True)
                    trainable.append(prefix + "." + name)
        return {"frozen": frozen, "trainable": trainable,
                "frozen_modules_eval": [v[:-1] for v in FROZEN_PREFIXES
                                        if v != "detail_decoder.encoder." or self.optical.has_high_resolution_decoder],
                "frozen_buffers": ["optical." + n for n, _ in self.optical.named_buffers()
                                   if n.startswith(FROZEN_PREFIXES)]}

    def train(self, mode: bool = True) -> "MS75SARPilot":
        super().train(mode)
        for prefix in FROZEN_PREFIXES:
            if prefix == "detail_decoder.encoder." and not self.optical.has_high_resolution_decoder:
                continue
            self.optical.get_submodule(prefix[:-1]).eval()
        return self

    @torch.no_grad()
    def project_constrained_parameters_(self) -> "MS75SARPilot":
        self.optical.project_constrained_parameters_()
        return self

    def parameter_groups(self, *, optical_lr: float, sar_lr: float,
                         weight_decay: float = .01) -> list[dict[str, Any]]:
        """Exact optimizer partition, without choosing the pilot's training rates."""
        rates = (float(optical_lr), float(sar_lr), float(weight_decay))
        if any(not math.isfinite(v) or v < 0 for v in rates) or min(rates[:2]) <= 0:
            raise ValueError("optimizer rates must be finite, lr positive and decay nonnegative")
        expected = set(self.configure_trainable_parameters_()["trainable"])
        groups: dict[tuple[bool, bool], dict[str, Any]] = {}
        for name, parameter in self.named_parameters():
            if name not in expected:
                if parameter.requires_grad:
                    raise RuntimeError(f"unexpected trainable parameter {name}")
                continue
            optical = name.startswith("optical.")
            decay = parameter.ndim > 1 and not name.endswith(".bias")
            if (optical, decay) not in groups:
                rate = optical_lr if optical else sar_lr
                groups[optical, decay] = {"params": [], "param_names": [], "lr": rate,
                                         "initial_lr": rate, "weight_decay": weight_decay if decay else 0.}
            group = groups[optical, decay]
            group["params"].append(parameter)
            group["param_names"].append(name)
        return list(groups.values())

    def normalize_sar(self, db: torch.Tensor, valid: torch.Tensor, *,
                      aux_mode: str | None = None) -> torch.Tensor:
        mode = ("zero" if self.arm == "Z" else "real") if aux_mode is None else aux_mode
        if (not self.multimodal or mode not in ("real", "zero")
                or (self.arm == "Z" and mode != "zero")
                or not isinstance(db, torch.Tensor) or not isinstance(valid, torch.Tensor)
                or db.ndim != 4 or tuple(db.shape[1:]) != (2, SAR_SIZE, SAR_SIZE)
                or not db.is_floating_point() or valid.shape != (len(db), 1, SAR_SIZE, SAR_SIZE)
                or valid.dtype != torch.bool or valid.device != db.device):
            raise ValueError("SAR requires B2x166x166 dB, boolean B1x166x166 validity and real/zero mode")
        with torch.autocast(device_type=db.device.type, enabled=False):
            if not bool(torch.isfinite(db[valid.expand_as(db)]).all()):
                raise ValueError("observed SAR values must be finite in every arm")
            clean = torch.where(valid, db.float(), self.sar_mean)
            normalized = (clean - self.sar_mean) / self.sar_std
            normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
            return torch.zeros_like(normalized) if mode == "zero" else normalized

    def forward_batch(self, batch: Mapping[str, torch.Tensor], *,
                      aux_mode: str | None = None) -> dict[str, torch.Tensor]:
        return self(**{key: batch[key] for key in PILOT_INPUT_KEYS if key in batch}, aux_mode=aux_mode)

    def forward(self, ms: torch.Tensor, *, ms_valid: torch.Tensor | None = None,
                ms_short: torch.Tensor | None = None, ms_short_valid: torch.Tensor | None = None,
                ms_global: torch.Tensor | None = None, ms_global_valid: torch.Tensor | None = None,
                local_box_short: torch.Tensor | None = None, local_box_global: torch.Tensor | None = None,
                sar_db: torch.Tensor | None = None, sar_valid: torch.Tensor | None = None,
                optical_valid: torch.Tensor | None = None, aux_mode: str | None = None,
                complete_anchor_context_state: AnchorContextState | None = None) -> dict[str, torch.Tensor]:
        if (not isinstance(ms, torch.Tensor) or ms.ndim != 4 or len(ms) < 1
                or ms.shape[1:] != (4, MS_SIZE, MS_SIZE) or not ms.is_floating_point()
                or not bool(torch.isfinite(ms).all())):
            raise ValueError("pilot requires finite native normalized B4x512x512 MS")
        views = dict(ms_valid=ms_valid, ms_short=ms_short, ms_short_valid=ms_short_valid,
                     ms_global=ms_global, ms_global_valid=ms_global_valid,
                     local_box_short=local_box_short, local_box_global=local_box_global,
                     complete_anchor_context_state=complete_anchor_context_state)
        if not self.multimodal:
            return self.optical(ms, **views)
        if any(value is None for name, value in views.items()
               if name not in ("ms_valid", "complete_anchor_context_state")):
            raise ValueError("pilot requires explicit aligned three-view inputs")
        for name, view, mask in (("ms", ms, ms_valid), ("ms_short", ms_short, ms_short_valid),
                                 ("ms_global", ms_global, ms_global_valid)):
            self.optical._validate_ms(view, name)
            if view.shape[0] != ms.shape[0] or view.device != ms.device:
                raise ValueError(f"{name} batch/device differs from local batch")
            if mask is not None:
                mask = mask[:, 0] if mask.ndim == 4 and mask.shape[1] == 1 else mask
                if mask.shape != (len(ms), *view.shape[-2:]):
                    raise ValueError(f"{name} validity mask must match its image")
        if self.training and complete_anchor_context_state is not None:
            raise ValueError("training with cached trainable context is forbidden")
        normalized = self.normalize_sar(sar_db, sar_valid, aux_mode=aux_mode)
        if normalized.device != ms.device or len(normalized) != len(ms):
            raise ValueError("SAR and MS batch/device must match")
        if (not isinstance(optical_valid, torch.Tensor)
                or optical_valid.shape != (len(ms), 1, MS_SIZE, MS_SIZE)
                or optical_valid.dtype != torch.bool or optical_valid.device != ms.device):
            raise ValueError("optical_valid requires boolean B1x512x512 aligned SAR support")
        coverage = align_native_sar(sar_valid.float(), sar_stride=1)
        coverage = coverage * F.avg_pool2d(optical_valid.float(), kernel_size=4, stride=4)
        sar_features = self.sar_encoder(normalized)
        output = self._forward_multiview(ms, sar_features, coverage, **views)
        if self.optical.has_high_resolution_decoder:
            low_logits, feature = self.optical.detail_decoder(ms, output["ms_shared"])
            detail_logits = F.interpolate(low_logits, ms.shape[-2:], mode="bilinear", align_corners=False)
            scale = _straight_through_clamp(self.optical.detail_residual_scale, minimum=0., maximum=1.)
            base_logits = output["logits"]
            logits = base_logits.float() + scale.float() * detail_logits.float()
            output.update({"logits": logits, "ms_logits": logits, "base_ms75_logits": base_logits,
                           "detail_logits": detail_logits, "detail_features": feature,
                           "detail_residual_scale": scale.reshape(1)})
        output["sar_present"] = sar_valid.flatten(1).any(1) & optical_valid.flatten(1).any(1)
        return output


    def _forward_multiview(
        self, ms: torch.Tensor, sar_features: torch.Tensor, coverage: torch.Tensor, *,
        ms_valid: torch.Tensor | None, ms_short: torch.Tensor,
        ms_short_valid: torch.Tensor, ms_global: torch.Tensor,
        ms_global_valid: torch.Tensor, local_box_short: torch.Tensor,
        local_box_global: torch.Tensor,
        complete_anchor_context_state: AnchorContextState | None,
    ) -> dict[str, torch.Tensor]:
        # Explicit mirror of MS75Segmentation.forward; sole insertion is labelled
        # by self.interaction. All original optical branches remain connected.
        del ms_valid
        base = self.optical
        ms_context = ms_context_valid = sar = None
        base._validate_ms(ms, "ms")
        short, short_present, global_value, global_present = (
            base._resolve_context_inputs(
                ms,
                ms_context,
                ms_context_valid,
                ms_short,
                ms_short_valid,
                ms_global,
                ms_global_valid,
            )
        )
        short_box = base._resolve_local_box(
            local_box_short,
            batch=ms.shape[0],
            device=ms.device,
            dtype=ms.dtype,
            name="local_box_short",
            required=ms_short is not None,
        )
        global_box = base._resolve_local_box(
            local_box_global,
            batch=ms.shape[0],
            device=ms.device,
            dtype=ms.dtype,
            name="local_box_global",
            required=ms_global is not None,
        )

        if complete_anchor_context_state is None:
            anchor_context_state = base.encode_complete_anchor_context(
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
            base._forward_complete_anchor_nrgb(
                base.to_local_anchor_order(ms),
                global_box,
                anchor_context_state,
            )
        )
        short_features = base.short_encoder(short)
        global_features = tuple(
            base._match_anchor_batch(value, ms.shape[0])
            for value in anchor_context_state.context_features
        )
        spectral_features = base.spectral_encoder(ms)
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
        for fusion, local, short_level, global_level, spectral in zip(
            base.level_fusions,
            local_features,
            short_features,
            global_features,
            spectral_features,
        ):
            short_roi = base._sample_normalized_roi(
                short_level, short_box, local.shape[-2:]
            )
            global_roi = base._sample_normalized_roi(
                global_level, global_box, local.shape[-2:]
            )
            evidence, weights, spectral_value, short_value, global_value_level = fusion(
                (local, short_roi, global_roi, spectral),
                view_present,
                (short_level, global_level),
            )
            evidence_levels.append(evidence)
            spectral_levels.append(spectral_value)
            short_levels.append(short_value)
            global_levels.append(global_value_level)
            scale_weights.append(weights)

        decoder_size = local_decoder_feature.shape[-2:]
        context_evidence = base.context_aggregate(
            base._resize_levels(evidence_levels, decoder_size)
        )
        context_evidence, sar_projected, sar_delta = self.interaction(
            context_evidence, sar_features, coverage
        )
        spectral_evidence = base.spectral_aggregate(
            base._resize_levels(spectral_levels, decoder_size)
        )
        short_evidence = base.short_aggregate(
            base._resize_levels(short_levels, decoder_size)
        )
        global_evidence = base.global_aggregate(
            base._resize_levels(global_levels, decoder_size)
        )
        context_scale = _straight_through_clamp(
            base.context_residual_scale,
            minimum=0.0,
            maximum=base.max_context_residual_scale,
        )
        scale_for_feature = context_scale.to(dtype=context_evidence.dtype)
        fused_feature = (
            local_decoder_feature + scale_for_feature * context_evidence
        )

        family_logits_low = base.family_head(fused_feature)
        family_child_logits_low = base._family_child_logits(
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
            base.context_aux_head(context_evidence)
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
        hierarchical_logits = base._compose_hierarchy(
            family_child_logits, family_logits
        )
        spatial_mix = interpolate(base.hierarchy_mix_gate(fused_feature))
        hierarchy_base = _straight_through_clamp(
            base.hierarchy_mix_base,
            minimum=0.0,
            maximum=base.max_hierarchy_mix,
        )
        configured_spatial_amplitude = hierarchy_base.new_tensor(
            base.hierarchy_spatial_delta
        )
        spatial_amplitude = torch.minimum(
            configured_spatial_amplitude,
            torch.minimum(
                hierarchy_base,
                hierarchy_base.new_tensor(base.max_hierarchy_mix)
                - hierarchy_base,
            ),
        ).clamp_min(0.0)
        hierarchy_mix_raw = (
            hierarchy_base + spatial_amplitude * torch.tanh(spatial_mix)
        )
        hierarchy_mix = _straight_through_clamp(
            hierarchy_mix_raw,
            minimum=0.0,
            maximum=base.max_hierarchy_mix,
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
            base.spectral_aux_head(spectral_evidence)
        )
        short_aux_logits = interpolate(base.short_aux_head(short_evidence))
        global_aux_logits = interpolate(base.global_aux_head(global_evidence))
        contrast_embeddings = F.normalize(
            base.family_contrast_projection(fused_feature), dim=1
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
            ms.shape[0], base.num_classes, sar_height, sar_width
        )
        zero_shared = torch.zeros_like(fused_feature)
        output = {
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
            "fine_to_family": base.fine_to_family,
            "family_contrast_embeddings": contrast_embeddings,
            "spectral_aux_logits": spectral_aux_logits,
            "short_aux_logits": short_aux_logits,
            "global_aux_logits": global_aux_logits,
            "context_aux_logits": context_aux_logits,
            "hierarchy_mix": hierarchy_mix,
            "hierarchy_mix_raw": hierarchy_mix_raw,
            "hierarchy_mix_base_raw": base.hierarchy_mix_base.reshape(1),
            "hierarchy_spatial_amplitude": spatial_amplitude.reshape(1),
            "context_residual_scale": context_scale.reshape(1),
            "context_residual_scale_raw": base.context_residual_scale.reshape(1),
            "view_scale_weights": resized_scale_weights,
            # Preserve legacy compatibility placeholders. Actual pilot SAR
            # features/deltas are exposed below under unambiguous pilot_* keys.
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

        output.update({"pilot_sar_features": sar_projected,
                       "pilot_sar_delta": sar_delta,
                       "pilot_sar_coverage": coverage})
        return output


__all__ = ["MS75SARPilot", "NativeSAREncoder", "ContextSARInteraction",
           "align_native_sar", "PILOT_INPUT_KEYS", "FROZEN_PREFIXES",
           "TRAINABLE_PREFIXES", "TRAINABLE_SCALARS", "MS_SIZE", "SAR_SIZE"]
