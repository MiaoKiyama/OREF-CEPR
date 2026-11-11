"""Layer-wise AdamW parameter groups for the DINOv2/NIR expert."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch


def _backbone_layer_id(name: str, depth: int) -> int:
    if not name.startswith("backbone."):
        return depth + 1
    remainder = name[len("backbone.") :]
    if remainder.startswith("blocks."):
        pieces = remainder.split(".")
        try:
            index = int(pieces[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"malformed DINOv2 block parameter name: {name}") from error
        if not 0 <= index < depth:
            raise ValueError(f"DINOv2 block index is outside depth {depth}: {name}")
        return index + 1
    if remainder.startswith(("norm.", "fc_norm.", "head.")):
        return depth
    return 0


def _uses_weight_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim <= 1 or name.endswith(".bias"):
        return False
    lowered = name.lower()
    return not any(
        marker in lowered
        for marker in ("pos_embed", "cls_token", "mask_token", "register_tokens")
    )


def dinov2_nir_parameter_groups(
    model: torch.nn.Module,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Partition every trainable tensor exactly once with DINO layer decay."""

    train = config["train"]
    base_lr = float(train["learning_rate"])
    backbone_lr = base_lr * float(train["backbone_lr_multiplier"])
    layer_decay = float(train["layer_decay"])
    weight_decay = float(train["weight_decay"])
    if not math.isfinite(base_lr) or base_lr <= 0.0:
        raise ValueError("DINOv2NIR learning_rate must be positive and finite")
    if not math.isfinite(backbone_lr) or backbone_lr <= 0.0:
        raise ValueError("DINOv2NIR backbone learning rate must be positive")
    if not 0.0 < layer_decay <= 1.0:
        raise ValueError("DINOv2NIR layer_decay must be in (0, 1]")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("DINOv2NIR weight_decay must be finite and non-negative")
    depth = len(getattr(getattr(model, "backbone", None), "blocks", ()))
    if depth != 12:
        raise ValueError("DINOv2-B/14 layer decay requires exactly 12 blocks")

    grouped: defaultdict[
        tuple[bool, int, bool], list[tuple[str, torch.nn.Parameter]]
    ] = (
        defaultdict(list)
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            is_backbone = name.startswith("backbone.")
            grouped[
                (
                    is_backbone,
                    _backbone_layer_id(name, depth),
                    _uses_weight_decay(name, parameter),
                )
            ].append((name, parameter))
    if not grouped:
        raise ValueError("DINOv2NIR has no trainable parameters")

    result: list[dict[str, Any]] = []
    observed_ids: set[int] = set()
    for (is_backbone, layer_id, decay), members in sorted(grouped.items()):
        if is_backbone:
            lr = backbone_lr * (layer_decay ** (depth - layer_id))
            prefix = f"backbone_layer_{layer_id:02d}"
        else:
            # New NIR adapter, gate, and DPT decoder learn at the head LR.
            lr = base_lr
            prefix = "new_modules"
        parameters = [parameter for _, parameter in members]
        overlap = observed_ids & {id(parameter) for parameter in parameters}
        if overlap:  # pragma: no cover - named_parameters is unique by default
            raise RuntimeError("DINOv2NIR optimizer parameter groups overlap")
        observed_ids.update(id(parameter) for parameter in parameters)
        result.append(
            {
                "params": parameters,
                "lr": lr,
                "weight_decay": weight_decay if decay else 0.0,
                "group_name": f"{prefix}_{'decay' if decay else 'no_decay'}",
                "parameter_names": [name for name, _ in members],
            }
        )
    expected_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if observed_ids != expected_ids:
        raise RuntimeError("DINOv2NIR optimizer omitted a trainable parameter")
    return result


__all__ = ["dinov2_nir_parameter_groups"]
