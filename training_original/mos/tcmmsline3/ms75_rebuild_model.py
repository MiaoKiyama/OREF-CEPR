"""Isolated, generic-initialized FBP MS75 rebuilding experiments.

``r`` keeps the complete native 24-class multiview graph. ``hr`` and
``robust`` have exactly the same parameters and forward computation; their
only experimental difference belongs in the data augmentation pipeline.
Neither architecture loads an FBP segmenter or a DINO teacher.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .ms75_castr import AnchorContextState
from .ms75_model import (
    CASTR_STRUCTURE_GROUPS, LAND_COVER_FAMILIES, MS75Segmentation,
    _straight_through_clamp, fine_to_family_map,
)


MS_MEAN = (0.363, 0.390, 0.371, 0.506)
MS_STD = (0.231, 0.231, 0.244, 0.254)
PRETRAINED_SHA256 = {
    "swin": "30762928cd6ee9229e24e26e200951b8fe635799b67db016ba747fa653b64db9",
    "resnet18": "d63eafa07a6e32a39d328e364f8c9f89d671444ecc7f02aa0f7eb8882af3dd29",
}
VARIANTS = ("r", "hr", "robust")
MODEL_INPUT_KEYS = (
    "ms", "ms_valid", "ms_short", "ms_short_valid", "ms_global",
    "ms_global_valid", "local_box_short", "local_box_global",
)
BACKBONE_PREFIXES = ("local_encoder.", "short_encoder.", "context_encoder.")
GRADIENT_AUDIT_PREFIXES = {
    "local": ("local_encoder.",), "short": ("short_encoder.",),
    "global": ("context_encoder.",), "spectral": ("spectral_encoder.",),
    "structure": ("structure_input.", "structure_blocks."),
    "router": ("router.",), "family": ("family_head.",),
    "multiview_fusion": ("level_fusions.",),
    "multiview_residual": ("context_aux_head.",),
    **{f"family_child_{i}": (f"family_child_heads.{i}.",)
       for i in range(len(LAND_COVER_FAMILIES))},
}


def _gn_conv(inputs: int, outputs: int, *, stride: int = 1, kernel: int = 3) -> nn.Sequential:
    groups = min(8, outputs)
    while outputs % groups:
        groups -= 1
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.GroupNorm(groups, outputs), nn.GELU(),
    )


class HighResolutionDetailDecoder(nn.Module):
    """Learn a half-resolution RGBN representation and decode with context.

    This is a feature decoder, not a spatial filter over existing logits.
    All normalization is per sample, without target-domain running statistics.
    """

    def __init__(self, context_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _gn_conv(4, 32, stride=2), _gn_conv(32, 32),
            _gn_conv(32, 64), _gn_conv(64, 64),
        )
        self.context_projection = _gn_conv(context_channels, 32, kernel=1)
        self.decoder = nn.Sequential(_gn_conv(96, 64), _gn_conv(64, 64))
        self.classifier = nn.Conv2d(64, 24, 1)

    def forward(self, ms: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        detail = self.encoder(ms)
        context = F.interpolate(self.context_projection(context), size=detail.shape[-2:],
                                mode="bilinear", align_corners=False)
        decoded = self.decoder(torch.cat((detail, context), dim=1))
        return self.classifier(decoded), decoded


class MS75RebuildSegmentation(MS75Segmentation):
    """Native FBP hierarchy with optional directly supervised HR decoder."""

    def __init__(self, *, variant: str = "r", num_classes: int = 24, **kwargs: Any) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown MS75 rebuild variant: {variant}")
        if type(num_classes) is not int or num_classes != 24:
            raise ValueError("MS75 rebuilding requires the native FBP 24 classes")
        if "structure_groups" in kwargs:
            raise ValueError("MS75 rebuilding retains native FBP structure groups")
        for name in ("initial_context_residual_scale", "initial_hierarchy_mix"):
            if name in kwargs and float(kwargs[name]) != 0.1:
                raise ValueError(f"MS75 rebuilding requires {name}=0.1")
            kwargs[name] = 0.1
        super().__init__(num_classes=num_classes, **kwargs)
        self.variant = variant
        self.has_high_resolution_decoder = variant in ("hr", "robust")
        if self.has_high_resolution_decoder:
            self.detail_decoder = HighResolutionDetailDecoder(self.local_head.in_channels)
            self.detail_residual_scale = nn.Parameter(torch.tensor(0.1))

    def load_local_anchor_state_dict(self, *args: Any, **kwargs: Any) -> tuple[str, ...]:
        raise ValueError("MS75 rebuilding forbids FBP/CASTR segmentation anchor initialization")

    def load_complete_anchor_state_dict(self, *args: Any, **kwargs: Any) -> tuple[str, ...]:
        raise ValueError("MS75 rebuilding forbids FBP/CASTR segmentation anchor initialization")

    @torch.no_grad()
    def project_constrained_parameters_(self) -> "MS75RebuildSegmentation":
        super().project_constrained_parameters_()
        if self.has_high_resolution_decoder:
            self.detail_residual_scale.clamp_(min=0.0, max=1.0)
        return self

    def forward(
        self, ms: torch.Tensor, *, ms_valid: torch.Tensor | None = None,
        ms_short: torch.Tensor | None = None, ms_short_valid: torch.Tensor | None = None,
        ms_global: torch.Tensor | None = None, ms_global_valid: torch.Tensor | None = None,
        local_box_short: torch.Tensor | None = None, local_box_global: torch.Tensor | None = None,
        complete_anchor_context_state: AnchorContextState | None = None,
    ) -> dict[str, torch.Tensor]:
        required = {
            "ms_short": ms_short, "ms_short_valid": ms_short_valid,
            "ms_global": ms_global, "ms_global_valid": ms_global_valid,
            "local_box_short": local_box_short, "local_box_global": local_box_global,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"MS75 rebuild requires explicit aligned three-view inputs: {missing}")
        self._validate_ms(ms, "ms")
        for name, view, mask in (("ms", ms, ms_valid), ("ms_short", ms_short, ms_short_valid),
                                 ("ms_global", ms_global, ms_global_valid)):
            self._validate_ms(view, name)
            if view.shape[0] != ms.shape[0]:
                raise ValueError(f"{name} batch differs from local batch")
            if mask is not None:
                if mask.ndim == 4 and mask.shape[1] == 1:
                    mask = mask[:, 0]
                if tuple(mask.shape) != (ms.shape[0], *view.shape[-2:]):
                    raise ValueError(f"{name} validity mask must match its image")
        if self.training and complete_anchor_context_state is not None:
            raise ValueError("training must recompute trainable context; cached anchor state is forbidden")
        output = super().forward(
            ms, ms_short=ms_short, ms_short_valid=ms_short_valid,
            ms_global=ms_global, ms_global_valid=ms_global_valid,
            local_box_short=local_box_short, local_box_global=local_box_global,
            complete_anchor_context_state=complete_anchor_context_state,
        )
        if self.has_high_resolution_decoder:
            low_logits, feature = self.detail_decoder(ms, output["ms_shared"])
            detail_logits = F.interpolate(low_logits, ms.shape[-2:], mode="bilinear", align_corners=False)
            scale = _straight_through_clamp(self.detail_residual_scale, minimum=0.0, maximum=1.0)
            base_logits = output["logits"]
            logits = base_logits.float() + scale.float() * detail_logits.float()
            output.update({"logits": logits, "ms_logits": logits,
                           "base_ms75_logits": base_logits, "detail_logits": detail_logits,
                           "detail_features": feature, "detail_residual_scale": scale.reshape(1)})
        return output

    def forward_batch(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self(**{key: batch[key] for key in MODEL_INPUT_KEYS if key in batch})


def gradient_audit_prefixes(model: MS75RebuildSegmentation) -> dict[str, tuple[str, ...]]:
    prefixes = dict(GRADIENT_AUDIT_PREFIXES)
    if model.has_high_resolution_decoder:
        prefixes.update({"hr_detail_encoder": ("detail_decoder.encoder.",),
                         "hr_context_projection": ("detail_decoder.context_projection.",),
                         "hr_decoder": ("detail_decoder.decoder.",),
                         "hr_classifier": ("detail_decoder.classifier.",)})
    return prefixes


def _pretrained_paths(config: Mapping[str, Any], *, verify_files: bool) -> dict[str, str]:
    specs = config.get("pretrained_files", {})
    if not isinstance(specs, Mapping) or set(specs) != set(PRETRAINED_SHA256):
        raise ValueError("pretrained_files must specify exactly generic swin and resnet18")
    paths = {}
    for name, expected in PRETRAINED_SHA256.items():
        spec = specs[name]
        if not isinstance(spec, Mapping) or spec.get("sha256") != expected or not spec.get("path"):
            raise ValueError(f"{name} generic pretrained identity differs from pinned MS75 recipe")
        path = Path(spec["path"])
        if verify_files:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"{name} generic pretrained bytes failed SHA256 authentication")
        paths[name] = str(path)
    return paths


def configure_swin_block_checkpointing(model: MS75RebuildSegmentation) -> None:
    """Use RNG-preserving per-block timm checkpoints, not nested stage checkpoints."""
    backbone = model.local_encoder.model
    stages = [m for m in backbone.modules() if hasattr(m, "blocks") and hasattr(m, "grad_checkpointing")]
    if (not hasattr(backbone, "grad_checkpointing") or len(stages) != 4
            or tuple(len(stage.blocks) for stage in stages) != (2, 2, 18, 2)):
        raise RuntimeError("pinned Swin-L four-stage block-checkpoint contract changed")
    backbone.grad_checkpointing = False
    for stage in stages:
        stage.grad_checkpointing = True
    model.target_gradient_checkpointing = "timm_swin_per_block_preserve_rng_state"


def build_model(config: Mapping[str, Any], *, load_pretrained: bool = True) -> MS75RebuildSegmentation:
    if config.get("dataset") != "fbp" or config.get("variant") not in VARIANTS:
        raise ValueError("MS75 rebuilding supports dataset=fbp and variant=r/hr/robust only")
    if config.get("num_classes", 24) != 24:
        raise ValueError("FBP classifier must have 24 native output classes")
    for forbidden in ("initialization_checkpoint", "ms75_anchor_checkpoint", "teacher_checkpoint",
                      "ms_initialization_checkpoint", "continuation_initialization_checkpoint"):
        if config.get(forbidden):
            raise ValueError(f"MS75 rebuilding forbids {forbidden}")
    normalization = config.get("normalization", {"mean": MS_MEAN, "std": MS_STD})
    if (tuple(normalization["mean"]) != MS_MEAN or tuple(normalization["std"]) != MS_STD
            or float(normalization.get("divisor", 255)) != 255):
        raise ValueError("MS75 rebuilding requires fixed BGRN normalization")
    paths = _pretrained_paths(config, verify_files=load_pretrained)
    common = {"type": "timm", "in_channels": 4, "pretrained": load_pretrained,
              "prefer_official_url": False, "gradient_checkpointing": True}
    local = {**common, "name": "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
             "band_order": ("NIR", "R", "G", "B"), "img_size": 512,
             "out_indices": (0, 1, 2, 3), "pretrained_file": paths["swin"]}
    residual = {**common, "name": "resnet18.a1_in1k", "img_size": None,
                "gradient_checkpointing": False, "out_indices": (1, 2, 3, 4),
                "pretrained_file": paths["resnet18"]}
    model = MS75RebuildSegmentation(
        variant=config["variant"], local_encoder_config=local,
        short_encoder_config={**residual, "band_order": ("B", "G", "R", "NIR")},
        context_encoder_config={**residual, "band_order": ("NIR", "R", "G", "B")},
        freeze_context_batch_norm_stats=True, ms_mean=MS_MEAN, ms_std=MS_STD,
    )
    configure_swin_block_checkpointing(model)
    configure_train_stage(model, 0)
    return model


def configure_train_stage(model: MS75RebuildSegmentation, epoch: int) -> dict[str, Any]:
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a nonnegative zero-based integer")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return {"epoch_zero_based": epoch, "variant": model.variant, "all_encoders_trainable": True,
            "context_batchnorm_statistics_frozen": model.freeze_context_batch_norm_stats,
            "high_resolution_decoder": model.has_high_resolution_decoder}


def parameter_groups(model: MS75RebuildSegmentation, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if config.get("variant") != model.variant or model.variant not in VARIANTS:
        raise ValueError("optimizer variant disagrees with MS75 rebuilding model")
    defaults = {"learning_rate": 1.5e-4, "backbone_lr_multiplier": 0.30, "weight_decay": 0.01}
    for name, expected in defaults.items():
        if name in config and float(config[name]) != expected:
            raise ValueError(f"MS75 rebuilding common recipe requires {name}={expected}")
    buckets: dict[tuple[bool, bool], dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        backbone = name.startswith(BACKBONE_PREFIXES)
        decay = parameter.ndim > 1 and not name.endswith(".bias")
        key = (backbone, decay)
        if key not in buckets:
            rate = defaults["learning_rate"] * (defaults["backbone_lr_multiplier"] if backbone else 1)
            buckets[key] = {"params": [], "param_names": [], "lr": rate, "initial_lr": rate,
                            "weight_decay": defaults["weight_decay"] if decay else 0.0}
        buckets[key]["params"].append(parameter)
        buckets[key]["param_names"].append(name)
    return list(buckets.values())


__all__ = ["MS75RebuildSegmentation", "HighResolutionDetailDecoder", "build_model",
           "configure_train_stage", "parameter_groups", "gradient_audit_prefixes",
           "GRADIENT_AUDIT_PREFIXES", "MODEL_INPUT_KEYS", "PRETRAINED_SHA256", "VARIANTS",
           "MS_MEAN", "MS_STD", "LAND_COVER_FAMILIES", "CASTR_STRUCTURE_GROUPS", "fine_to_family_map",
           "configure_swin_block_checkpointing"]
