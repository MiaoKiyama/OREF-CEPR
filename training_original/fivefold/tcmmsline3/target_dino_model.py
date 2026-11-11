"""Isolated five-/seven-class target-task DINO experts.

The historical FBP architecture is deliberately not edited. Its authenticated
generic RGB backbone, independent NIR adapter and DPT body are reused, replacing
only the final 24-class convolution with the native target classifier. No FBP
segmentation checkpoint is read by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .dino_web_model import DinoWebSegmentation, dino_web_parameter_groups


MS_MEAN = (0.363, 0.390, 0.371, 0.506)
MS_STD = (0.231, 0.231, 0.244, 0.254)
DATASET_CLASSES = {"gid5": 5, "whu": 7}


class TargetDinoSegmentation(DinoWebSegmentation):
    def __init__(self, *, num_classes: int, **kwargs: Any) -> None:
        if type(num_classes) is not int or num_classes not in (5, 7):
            raise ValueError("target DINO supports exactly five or seven classes")
        if kwargs.get("variant") not in ("v3l", "v2l"):
            raise ValueError("target DINO supports only the pinned v3l/v2l experts")
        super().__init__(num_classes=24, **kwargs)
        original = self.decoder.head[-1]
        if not isinstance(original, nn.Conv2d) or original.out_channels != 24:
            raise RuntimeError("historical DPT final-classifier contract changed")
        self.decoder.head[-1] = nn.Conv2d(
            original.in_channels, num_classes, original.kernel_size,
            stride=original.stride, padding=original.padding,
            dilation=original.dilation, groups=original.groups,
            bias=original.bias is not None, padding_mode=original.padding_mode,
            device=original.weight.device, dtype=original.weight.dtype,
        )
        self.num_classes = num_classes
        configure_train_stage(self, 0)

    def _forward_ms(self, ms: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._backbone_features(self.prepare_inputs(ms))
        logits, embedding = self.decoder(features, output_size=(self.padded_size,) * 2)
        if self.pad_pixels:
            p = self.pad_pixels
            logits = logits[..., p:-p, p:-p]
        if tuple(logits.shape) != (ms.shape[0], self.num_classes, 512, 512):
            raise RuntimeError("native target DINO output geometry changed")
        return {"logits": logits, "ms_logits": logits, "flat_logits": logits,
                "ms_shared": embedding, "nir_gate": self.nir_gate.reshape(1)}


def _normalization(config: Mapping[str, Any]) -> dict[str, tuple[float, ...]]:
    normalization = config.get("normalization", {"mean": MS_MEAN, "std": MS_STD})
    mean = tuple(float(v) for v in normalization["mean"])
    std = tuple(float(v) for v in normalization["std"])
    if mean != MS_MEAN or std != MS_STD or float(normalization.get("divisor", 255)) != 255.0:
        raise ValueError("target recipe retains the fixed historical B/G/R/NIR normalization")
    return {"mean": mean, "std": std}


def build_model(config: Mapping[str, Any], *, load_pretrained: bool = True) -> TargetDinoSegmentation:
    dataset = config.get("dataset")
    if dataset not in DATASET_CLASSES:
        raise ValueError("target dataset must be gid5 or whu")
    classes = DATASET_CLASSES[dataset]
    if config.get("num_classes", classes) != classes:
        raise ValueError("native classifier class count does not match target dataset")
    norm = _normalization(config)
    return TargetDinoSegmentation(
        variant=str(config["variant"]), num_classes=classes,
        pretrained=True, pretrained_file=config.get("pretrained_file"),
        pretrained_file_sha256=config.get("pretrained_sha256"),
        load_pretrained=load_pretrained,
        ms_mean=norm["mean"], ms_std=norm["std"],
        gradient_checkpointing=True,
        freeze_prefix_tokens=config["variant"] == "v3l",
    )


def _permanent(name: str, variant: str) -> bool:
    if variant == "v2l":
        return False
    if name.startswith("backbone.patch_embed.proj.rgb_projection."):
        return True
    if name in ("backbone.cls_token", "backbone.reg_token"):
        return True
    return name.startswith("backbone.blocks.") and int(name.split(".")[2]) < 12


def _staged(name: str, variant: str) -> bool:
    if variant == "v2l":
        return False
    return name.startswith("backbone.norm.") or (
        name.startswith("backbone.blocks.") and int(name.split(".")[2]) >= 12
    )


def configure_train_stage(model: TargetDinoSegmentation, epoch: int) -> dict[str, Any]:
    """Apply the original zero-based five-epoch v3 warm-start policy."""
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a nonnegative zero-based integer")
    if model.variant not in ("v3l", "v2l"):
        raise ValueError("unexpected target variant")
    permanent, staged = [], []
    for name, parameter in model.named_parameters():
        frozen = _permanent(name, model.variant)
        temporary = _staged(name, model.variant)
        parameter.requires_grad_(not frozen and (not temporary or epoch >= 5))
        if frozen:
            permanent.append(name)
        elif temporary:
            staged.append(name)
    return {"epoch_zero_based": epoch, "variant": model.variant,
            "upper_backbone_unfrozen": model.variant == "v2l" or epoch >= 5,
            "permanently_frozen_parameters": permanent,
            "staged_parameters": staged}


def parameter_groups(model: TargetDinoSegmentation, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if config["variant"] != model.variant:
        raise ValueError("optimizer variant disagrees with model")
    defaults = {"learning_rate": 2e-4, "backbone_lr_multiplier": 0.05 if model.variant == "v3l" else 0.1,
                "layer_decay": 0.9, "weight_decay": 0.05}
    # These are part of the committed historical recipe, not tuning switches.
    for name, expected in defaults.items():
        if name in config and float(config[name]) != expected:
            raise ValueError(f"target recipe requires {name}={expected}")
    defaults["freeze_modules"] = [name for name, _ in model.named_parameters()
                                  if _permanent(name, model.variant)]
    staged = [p for name, p in model.named_parameters() if _staged(name, model.variant)]
    return dino_web_parameter_groups(model, {"train": defaults}, staged_parameters=staged)


__all__ = ["TargetDinoSegmentation", "build_model", "parameter_groups", "configure_train_stage",
           "MS_MEAN", "MS_STD", "DATASET_CLASSES"]
