from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .fusionaware_a1 import (
    CLASS_BALANCED_FUSION_LOSS_WEIGHT,
    DUAL_FUSION_LOSS_WEIGHT,
    PRESERVE_LOSS_WEIGHT,
    REPAIR_LOSS_WEIGHT,
    fusionaware_a1_additional_loss,
)


def _area_overlap_weights(
    source_size: int,
    target_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return exact source-pixel overlap fractions for a regular target grid."""
    source_left = torch.arange(
        source_size, device=device, dtype=dtype
    )[None]
    source_right = source_left + 1.0
    target_edges = torch.linspace(
        0.0,
        float(source_size),
        target_size + 1,
        device=device,
        dtype=dtype,
    )
    target_left = target_edges[:-1, None]
    target_right = target_edges[1:, None]
    overlap = (
        torch.minimum(target_right, source_right)
        - torch.maximum(target_left, source_left)
    ).clamp_min(0.0)
    return overlap / (target_right - target_left)


def exact_area_pool2d(
    values: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Area-resample without adaptive-pooling overlap for non-integer ratios."""
    target_height, target_width = map(int, size)
    source_height, source_width = values.shape[-2:]
    if (source_height, source_width) == (target_height, target_width):
        return values
    if (
        source_height % target_height == 0
        and source_width % target_width == 0
    ):
        return F.avg_pool2d(
            values,
            kernel_size=(
                source_height // target_height,
                source_width // target_width,
            ),
            stride=(
                source_height // target_height,
                source_width // target_width,
            ),
        )
    height_weights = _area_overlap_weights(
        source_height,
        target_height,
        device=values.device,
        dtype=values.dtype,
    )
    width_weights = _area_overlap_weights(
        source_width,
        target_width,
        device=values.device,
        dtype=values.dtype,
    )
    return torch.einsum(
        "yh,bchw,xw->bcyx",
        height_weights,
        values,
        width_weights,
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    valid = target >= 0
    safe_target = target.clamp_min(0)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.to(probabilities.dtype) * valid[:, None]
    probabilities = probabilities * valid[:, None]
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = (probabilities + one_hot).sum(dim=(0, 2, 3))
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    if not present.any():
        return logits.sum() * 0.0
    return 1.0 - dice[present].mean()


def native_sar_soft_label_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sar_valid: torch.Tensor,
    num_classes: int,
    *,
    min_purity: float,
    class_weights: torch.Tensor | None = None,
    sample_eligible: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise SAR on its native grid with mixed-cell class distributions."""
    distribution, _, supervised = native_sar_targets(
        target,
        sar_valid,
        num_classes,
        size=logits.shape[-2:],
        min_purity=min_purity,
        dtype=logits.dtype,
    )
    pixel_loss = -(
        distribution * logits.log_softmax(dim=1)
    ).sum(dim=1)
    if class_weights is not None:
        expected_weight = (
            distribution
            * class_weights.to(device=logits.device, dtype=logits.dtype)[
                None, :, None, None
            ]
        ).sum(dim=1)
        pixel_loss = pixel_loss * expected_weight
    if sample_eligible is not None:
        supervised = (
            supervised
            & sample_eligible.to(
                device=supervised.device, dtype=torch.bool
            )[:, None, None]
        )
    return _masked_mean(pixel_loss, supervised)


def native_sar_targets(
    target: torch.Tensor,
    sar_valid: torch.Tensor,
    num_classes: int,
    *,
    size: tuple[int, int],
    min_purity: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate MS-grid labels into soft and hard native SAR targets."""
    valid = target >= 0
    safe_target = target.clamp_min(0)
    one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype) * valid[:, None]
    valid_fraction = exact_area_pool2d(
        valid[:, None].to(dtype), size
    )
    class_fraction = exact_area_pool2d(one_hot, size)
    distribution = class_fraction / valid_fraction.clamp_min(1e-6)

    if sar_valid.shape[-2:] != size:
        sar_valid_native = F.interpolate(
            sar_valid[:, None].float(), size=size, mode="nearest"
        )[:, 0].bool()
    else:
        sar_valid_native = sar_valid
    purity = distribution.amax(dim=1)
    supervised = (
        sar_valid_native
        & (valid_fraction[:, 0] > 0)
        & (purity >= float(min_purity))
    )
    hard_target = distribution.argmax(dim=1)
    hard_target = torch.where(
        supervised,
        hard_target,
        torch.full_like(hard_target, -1),
    )
    return distribution, hard_target, supervised


def class_prototype_alignment(
    ms_features: torch.Tensor,
    sar_features: torch.Tensor,
    target: torch.Tensor,
    sar_valid: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    size = ms_features.shape[-2:]
    labels = F.interpolate(
        target[:, None].float(), size=size, mode="nearest"
    )[:, 0].long()
    valid = F.interpolate(
        sar_valid[:, None].float(), size=size, mode="nearest"
    )[:, 0].bool()
    losses: list[torch.Tensor] = []
    for class_id in range(num_classes):
        mask = (labels == class_id) & valid
        if int(mask.sum()) < 4:
            continue
        weights = mask[:, None].to(ms_features.dtype)
        denominator = weights.sum().clamp_min(1.0)
        ms_prototype = (ms_features * weights).sum(dim=(0, 2, 3)) / denominator
        sar_prototype = (sar_features * weights).sum(dim=(0, 2, 3)) / denominator
        losses.append(
            1.0
            - F.cosine_similarity(
                ms_prototype[None], sar_prototype[None], dim=1
            ).mean()
        )
    if not losses:
        return ms_features.sum() * 0.0
    return torch.stack(losses).mean()


def private_orthogonality(
    shared: torch.Tensor, private: torch.Tensor, sar_valid: torch.Tensor
) -> torch.Tensor:
    valid = F.interpolate(
        sar_valid[:, None].float(), size=shared.shape[-2:], mode="nearest"
    )[:, 0].bool()
    shared = F.normalize(shared, dim=1, eps=1e-6)
    private = F.normalize(private, dim=1, eps=1e-6)
    cosine_squared = (shared * private).sum(dim=1).square()
    return _masked_mean(cosine_squared, valid)


def regional_utility_targets(
    *,
    ms_pixel_loss: torch.Tensor,
    sar_pixel_loss: torch.Tensor,
    collaborative_pixel_loss: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    region_grid: int,
    margin: float,
    both_pixel_loss: torch.Tensor | None = None,
    min_pixels: int = 1,
    min_utility_gap: float = 0.0,
    relative_utility: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build class-region expert targets from detached counterfactual gains."""
    safe_target = target.clamp_min(0)
    supervised = target >= 0
    class_mask = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2)
    class_mask = class_mask.to(ms_pixel_loss.dtype) * supervised[:, None]

    def pool_for_class(values: torch.Tensor) -> torch.Tensor:
        weighted = class_mask * values[:, None]
        numerator = F.adaptive_avg_pool2d(
            weighted, (region_grid, region_grid)
        )
        denominator = F.adaptive_avg_pool2d(
            class_mask, (region_grid, region_grid)
        )
        return numerator / denominator.clamp_min(1e-6)

    ms_region_loss = pool_for_class(ms_pixel_loss.detach())

    def utility(candidate: torch.Tensor) -> torch.Tensor:
        gain = pool_for_class((ms_pixel_loss - candidate).detach())
        if relative_utility:
            gain = gain / ms_region_loss.abs().clamp_min(1e-6)
        return gain

    sar_utility = utility(sar_pixel_loss)
    collaborative_utility = utility(collaborative_pixel_loss)
    null_utility = torch.zeros_like(sar_utility)
    candidates = [null_utility, sar_utility, collaborative_utility]
    if both_pixel_loss is not None:
        candidates.append(utility(both_pixel_loss))
    utilities = torch.stack(candidates, dim=1)
    best_utility, target_expert = utilities.max(dim=1)
    target_expert = torch.where(
        best_utility > float(margin),
        target_expert,
        torch.zeros_like(target_expert),
    )
    support_fraction = F.adaptive_avg_pool2d(
        class_mask, (region_grid, region_grid)
    )
    approximate_region_area = (
        target.shape[-2] * target.shape[-1] / float(region_grid**2)
    )
    class_region_valid = (
        support_fraction * approximate_region_area >= max(int(min_pixels), 1)
    )
    if min_utility_gap > 0:
        top_two = utilities.topk(k=2, dim=1).values
        utility_gap = top_two[:, 0] - top_two[:, 1]
        confident = (best_utility <= float(margin)) | (
            utility_gap >= float(min_utility_gap)
        )
        class_region_valid = class_region_valid & confident
    return target_expert, class_region_valid, utilities


def utility_routing_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    margin: float,
    target_mode: str = "hard",
    temperature: float = 1.0,
    min_pixels: int = 1,
    min_utility_gap: float = 0.0,
    relative_utility: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    router_logits = outputs.get("router_logits")
    sar_candidate = outputs.get("counterfactual_sar_logits")
    collaborative_candidate = outputs.get(
        "counterfactual_collab_logits"
    )
    both_candidate = outputs.get("counterfactual_both_logits")
    if (
        router_logits is None
        or sar_candidate is None
        or collaborative_candidate is None
    ):
        zero = outputs["logits"].sum() * 0.0
        return zero, {
            "route_accuracy": zero,
            "route_non_null_target_fraction": zero,
            "route_non_null_usage": zero,
            "route_entropy": zero,
        }

    def pixel_loss(logits: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            target,
            weight=class_weights,
            ignore_index=-1,
            reduction="none",
        )

    ms_pixel = pixel_loss(outputs["ms_logits"])
    sar_pixel = pixel_loss(sar_candidate)
    collaborative_pixel = pixel_loss(collaborative_candidate)
    both_pixel = pixel_loss(both_candidate) if both_candidate is not None else None
    route_target, route_valid, utilities = regional_utility_targets(
        ms_pixel_loss=ms_pixel,
        sar_pixel_loss=sar_pixel,
        collaborative_pixel_loss=collaborative_pixel,
        both_pixel_loss=both_pixel if router_logits.shape[1] == 4 else None,
        target=target,
        num_classes=router_logits.shape[2],
        region_grid=router_logits.shape[-1],
        margin=margin,
        min_pixels=min_pixels,
        min_utility_gap=min_utility_gap,
        relative_utility=relative_utility,
    )
    router_region_valid = outputs.get("router_region_valid")
    if router_region_valid is not None:
        route_valid = route_valid & router_region_valid.expand(
            -1,
            route_valid.shape[1],
            -1,
            -1,
    )
    if target_mode == "soft":
        non_null_utility = utilities[:, 1:]
        useful = non_null_utility.amax(dim=1) > float(margin)
        non_null_target = (
            non_null_utility / max(float(temperature), 1e-6)
        ).softmax(dim=1)
        soft_target = torch.zeros_like(utilities)
        soft_target[:, 0] = (~useful).to(utilities.dtype)
        soft_target[:, 1:] = non_null_target * useful[:, None]
        route_pixel = -(
            soft_target * router_logits.log_softmax(dim=1)
        ).sum(dim=1)
        route_target = soft_target.argmax(dim=1)
    elif target_mode == "hard":
        route_pixel = F.cross_entropy(
            router_logits,
            route_target,
            reduction="none",
        )
    else:
        raise ValueError(f"unsupported route target mode: {target_mode}")
    route_loss = _masked_mean(route_pixel, route_valid)
    weights = outputs.get("router_weights", router_logits.softmax(dim=1))
    prediction = weights.argmax(dim=1)
    route_accuracy = _masked_mean(
        (prediction == route_target).to(router_logits.dtype),
        route_valid,
    )
    target_non_null = _masked_mean(
        (route_target != 0).to(router_logits.dtype),
        route_valid,
    )
    non_null_usage = _masked_mean(
        weights[:, 1:].sum(dim=1),
        route_valid,
    )
    entropy = -(
        weights.clamp_min(1e-7) * weights.clamp_min(1e-7).log()
    ).sum(dim=1)
    route_entropy = _masked_mean(entropy, route_valid)
    return route_loss, {
        "route_accuracy": route_accuracy,
        "route_non_null_target_fraction": target_non_null,
        "route_non_null_usage": non_null_usage,
        "route_entropy": route_entropy,
    }


def lovasz_softmax_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Multiclass Lovasz-Softmax over classes present in the batch."""
    probability = logits.softmax(dim=1)
    valid = target >= 0
    if not valid.any():
        return logits.sum() * 0.0
    labels = target[valid]
    values = probability.permute(0, 2, 3, 1)[valid]
    losses: list[torch.Tensor] = []
    for class_id in range(logits.shape[1]):
        foreground = (labels == class_id).to(values.dtype)
        if not foreground.any():
            continue
        errors = (foreground - values[:, class_id]).abs()
        errors, permutation = torch.sort(errors, descending=True)
        foreground = foreground[permutation]
        total = foreground.sum()
        intersection = total - foreground.cumsum(0)
        union = total + (1.0 - foreground).cumsum(0)
        gradient = 1.0 - intersection / union.clamp_min(1e-6)
        if gradient.numel() > 1:
            gradient = torch.cat(
                [gradient[:1], gradient[1:] - gradient[:-1]]
            )
        losses.append(torch.dot(errors, gradient))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _semantic_boundary_target(
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    supervised = target >= 0
    boundary = torch.zeros_like(supervised)
    for shift_y, shift_x in ((1, 0), (0, 1)):
        if shift_y:
            left = target[:, :-1]
            right = target[:, 1:]
            valid = supervised[:, :-1] & supervised[:, 1:]
            different = valid & (left != right)
            boundary[:, :-1] |= different
            boundary[:, 1:] |= different
        else:
            left = target[:, :, :-1]
            right = target[:, :, 1:]
            valid = supervised[:, :, :-1] & supervised[:, :, 1:]
            different = valid & (left != right)
            boundary[:, :, :-1] |= different
            boundary[:, :, 1:] |= different
    return boundary, supervised


def semantic_boundary_loss(
    boundary_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    boundary, valid = _semantic_boundary_target(target)
    if not valid.any():
        return boundary_logits.sum() * 0.0
    # A production 384x384 microbatch contains more than the 65,504 finite
    # integer range of fp16.  Casting the class counts to the logits dtype can
    # therefore produce inf/inf -> NaN in the positive weight under AMP.
    # Keep both the global counts and BCE reduction in fp32; gradients still
    # flow through the explicit logits cast to the original model tensor.
    positives = (boundary & valid).sum(dtype=torch.float32)
    negatives = ((~boundary) & valid).sum(dtype=torch.float32)
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
    pixel_loss = F.binary_cross_entropy_with_logits(
        boundary_logits[:, 0].float(),
        boundary.float(),
        reduction="none",
        pos_weight=positive_weight,
    )
    return _masked_mean(pixel_loss, valid)


def hierarchy_losses(
    flat_logits: torch.Tensor,
    family_logits: torch.Tensor,
    target: torch.Tensor,
    fine_to_family: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    supervised = target >= 0
    safe_target = target.clamp_min(0)
    family_target = fine_to_family.to(target.device)[safe_target]
    family_pixel = F.cross_entropy(
        family_logits,
        family_target,
        ignore_index=-1,
        reduction="none",
    )
    family_ce = _masked_mean(family_pixel, supervised)

    fine_probability = flat_logits.softmax(dim=1)
    family_probability = family_logits.softmax(dim=1)
    aggregated = torch.zeros_like(family_probability)
    mapping = fine_to_family.to(flat_logits.device)
    for class_id in range(flat_logits.shape[1]):
        aggregated[:, mapping[class_id]] += fine_probability[:, class_id]
    midpoint = 0.5 * (aggregated + family_probability)
    epsilon = torch.finfo(flat_logits.dtype).eps
    divergence = 0.5 * (
        aggregated
        * (
            aggregated.clamp_min(epsilon).log()
            - midpoint.clamp_min(epsilon).log()
        )
        + family_probability
        * (
            family_probability.clamp_min(epsilon).log()
            - midpoint.clamp_min(epsilon).log()
        )
    ).sum(dim=1)
    return family_ce, _masked_mean(divergence, supervised)


def region_prototype_losses(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
    min_pixels: int,
    confusion_groups: list[list[int]],
    confusion_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match interior class-region embeddings to a multi-prototype memory."""
    reduced_target = F.interpolate(
        target[:, None].to(embeddings.dtype),
        size=embeddings.shape[-2:],
        mode="nearest",
    )[:, 0].long()
    boundary, _ = _semantic_boundary_target(reduced_target)
    normalized_prototypes = F.normalize(prototypes, dim=-1)
    classification_losses: list[torch.Tensor] = []
    margin_losses: list[torch.Tensor] = []
    group_by_class: dict[int, list[int]] = {}
    for group in confusion_groups:
        for class_id in group:
            group_by_class[int(class_id)] = [int(value) for value in group]

    for batch_index in range(embeddings.shape[0]):
        labels = reduced_target[batch_index]
        for class_id in labels.unique().tolist():
            class_id = int(class_id)
            if class_id < 0 or class_id >= prototypes.shape[0]:
                continue
            support = labels == class_id
            interior = support & ~boundary[batch_index]
            selected = interior if int(interior.sum()) >= min_pixels else support
            if int(selected.sum()) < min_pixels:
                continue
            region = embeddings[batch_index, :, selected].mean(dim=1)
            region = F.normalize(region, dim=0)
            similarities = torch.einsum(
                "e,cke->ck", region, normalized_prototypes
            )
            class_scores = torch.logsumexp(
                similarities * temperature, dim=1
            ) - math.log(prototypes.shape[1])
            classification_losses.append(
                F.cross_entropy(
                    class_scores[None],
                    torch.tensor(
                        [class_id], dtype=torch.long, device=target.device
                    ),
                )
            )
            group = group_by_class.get(class_id, [])
            negatives = [value for value in group if value != class_id]
            if negatives and confusion_margin > 0:
                hardest = class_scores[negatives].max()
                margin_losses.append(
                    F.relu(
                        class_scores.new_tensor(confusion_margin)
                        - class_scores[class_id]
                        + hardest
                    )
                )
    zero = embeddings.sum() * 0.0
    classification = (
        torch.stack(classification_losses).mean()
        if classification_losses
        else zero
    )
    margin = torch.stack(margin_losses).mean() if margin_losses else zero
    return classification, margin


def family_conditional_subtype_loss(
    child_logits: torch.Tensor,
    target: torch.Tensor,
    fine_to_family: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    balance_classes: bool = False,
) -> torch.Tensor:
    """Average sibling-only cross entropy over the families in a batch.

    The refiner operates on the decoder grid, so labels are sampled on that
    same grid.  Each present family is averaged once, preventing a large
    family from suppressing the gradient of a rare sibling classifier.
    """

    sampled_target = F.interpolate(
        target[:, None].float(),
        size=child_logits.shape[-2:],
        mode="nearest",
    )[:, 0].long()
    valid = sampled_target >= 0
    safe_target = sampled_target.clamp_min(0)
    mapping = fine_to_family.to(device=target.device, dtype=torch.long)
    target_family = mapping[safe_target]
    family_losses: list[torch.Tensor] = []
    for family_id in range(int(mapping.max().item()) + 1):
        indices = torch.nonzero(mapping == family_id).flatten()
        family_mask = valid & (target_family == family_id)
        if not bool(family_mask.any()):
            continue
        local_lookup = torch.full_like(mapping, -1)
        local_lookup[indices] = torch.arange(
            indices.numel(), device=target.device
        )
        local_target = local_lookup[safe_target]
        local_target = torch.where(
            family_mask,
            local_target,
            torch.full_like(local_target, -1),
        )
        pixel_loss = F.cross_entropy(
            child_logits[:, indices],
            local_target,
            weight=(
                class_weights[indices]
                if class_weights is not None
                else None
            ),
            ignore_index=-1,
            reduction="none",
        )
        if balance_classes:
            child_losses = [
                _masked_mean(pixel_loss, family_mask & (safe_target == class_id))
                for class_id in indices.tolist()
                if bool((family_mask & (safe_target == class_id)).any())
            ]
            family_losses.append(torch.stack(child_losses).mean())
        else:
            family_losses.append(_masked_mean(pixel_loss, family_mask))
    if not family_losses:
        return child_logits.sum() * 0.0
    return torch.stack(family_losses).mean()


def sibling_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    target: torch.Tensor,
    fine_to_family: torch.Tensor,
    *,
    temperature: float = 0.08,
    max_samples_per_class: int = 64,
) -> torch.Tensor:
    """Contrast only semantic siblings on the decoder feature grid.

    The old FCR objective tried to repair already-produced logits.  MS75 uses
    this loss while the representation is still being formed.  Positives are
    pixels of the same fine class and negatives are restricted to *other fine
    classes in the same family*.  The deterministic per-class cap keeps the
    quadratic similarity matrix bounded and makes interrupted runs exactly
    reproducible.
    """

    if embeddings.ndim != 4:
        raise ValueError("family contrast embeddings must be BCHW")
    if temperature <= 0:
        raise ValueError("family contrast temperature must be positive")
    if max_samples_per_class < 2:
        raise ValueError("family contrast needs at least two samples per class")
    sampled_target = F.interpolate(
        target[:, None].float(),
        size=embeddings.shape[-2:],
        mode="nearest",
    )[:, 0].long()
    flat_target = sampled_target.reshape(-1)
    flat_embedding = embeddings.permute(0, 2, 3, 1).reshape(
        -1, embeddings.shape[1]
    )
    mapping = fine_to_family.to(
        device=target.device, dtype=torch.long
    )
    family_losses: list[torch.Tensor] = []
    for family_id in range(int(mapping.max().item()) + 1):
        class_ids = torch.nonzero(mapping == family_id).flatten()
        selected_features: list[torch.Tensor] = []
        selected_labels: list[torch.Tensor] = []
        for class_id in class_ids.tolist():
            indices = torch.nonzero(
                flat_target == int(class_id), as_tuple=False
            ).flatten()
            if indices.numel() < 2:
                continue
            if indices.numel() > max_samples_per_class:
                # Even spacing is deterministic and avoids systematically
                # selecting only the top-left portion of each crop.
                positions = torch.linspace(
                    0,
                    indices.numel() - 1,
                    steps=max_samples_per_class,
                    device=indices.device,
                ).round().long()
                indices = indices[positions]
            selected_features.append(flat_embedding[indices])
            selected_labels.append(
                torch.full(
                    (indices.numel(),),
                    int(class_id),
                    dtype=torch.long,
                    device=target.device,
                )
            )
        # A family represented by just one fine class has no sibling negative.
        if len(selected_features) < 2:
            continue
        features = F.normalize(torch.cat(selected_features, dim=0), dim=1)
        labels = torch.cat(selected_labels, dim=0)
        similarity = features @ features.transpose(0, 1)
        similarity = similarity / float(temperature)
        identity = torch.eye(
            similarity.shape[0], dtype=torch.bool, device=similarity.device
        )
        positive = labels[:, None].eq(labels[None]) & ~identity
        candidates = ~identity
        # Stable log-softmax over all same-family candidates.
        masked_similarity = similarity.masked_fill(~candidates, -torch.inf)
        log_probability = masked_similarity - torch.logsumexp(
            masked_similarity, dim=1, keepdim=True
        )
        positive_count = positive.sum(dim=1)
        valid_anchor = positive_count > 0
        if bool(valid_anchor.any()):
            positive_log_probability = torch.where(
                positive,
                log_probability,
                torch.zeros_like(log_probability),
            ).sum(dim=1) / positive_count.clamp_min(1)
            family_losses.append(
                -positive_log_probability[valid_anchor].mean()
            )
    if not family_losses:
        return embeddings.sum() * 0.0
    return torch.stack(family_losses).mean()


def family_confusion_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    confusion_groups: list[list[int]],
    *,
    margin: float,
) -> torch.Tensor:
    """Separate explicitly confused siblings with a class-balanced margin."""

    sampled_target = F.interpolate(
        target[:, None].float(), size=logits.shape[-2:], mode="nearest"
    )[:, 0].long()
    valid = sampled_target >= 0
    class_losses: list[torch.Tensor] = []
    for group in confusion_groups:
        indices = torch.as_tensor(group, device=logits.device, dtype=torch.long)
        for class_id in group:
            mask = valid & (sampled_target == int(class_id))
            if not bool(mask.any()):
                continue
            true_score = logits[:, int(class_id)]
            competitors = indices[indices != int(class_id)]
            hardest = logits[:, competitors].amax(dim=1)
            class_losses.append(
                F.relu(float(margin) - true_score[mask] + hardest[mask]).mean()
            )
    if not class_losses:
        return logits.sum() * 0.0
    return torch.stack(class_losses).mean()


def family_no_harm_distillation_loss(
    refined_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    confidence_threshold: float,
    temperature: float,
) -> torch.Tensor:
    """Protect correct, confident baseline pixels while learning corrections."""

    sampled_target = F.interpolate(
        target[:, None].float(), size=refined_logits.shape[-2:], mode="nearest"
    )[:, 0].long()
    valid = sampled_target >= 0
    teacher = baseline_logits.detach()
    teacher_probability = teacher.softmax(dim=1)
    confidence, prediction = teacher_probability.max(dim=1)
    protected = (
        valid
        & (prediction == sampled_target.clamp_min(0))
        & (confidence >= float(confidence_threshold))
    )
    if not bool(protected.any()):
        return refined_logits.sum() * 0.0
    scaled_teacher = (teacher / float(temperature)).softmax(dim=1)
    scaled_student = (refined_logits / float(temperature)).log_softmax(dim=1)
    pixel_kl = F.kl_div(
        scaled_student, scaled_teacher, reduction="none"
    ).sum(dim=1) * float(temperature) ** 2
    # Macro-average over protected target classes, otherwise abundant classes
    # would dominate the no-harm constraint and erase rare-class corrections.
    class_losses = [
        pixel_kl[protected & (sampled_target == class_id)].mean()
        for class_id in sampled_target[protected].unique().tolist()
    ]
    return torch.stack(class_losses).mean()


def fixed_anchor_no_harm_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    confidence_threshold: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Protect only correct, confident predictions from an immutable anchor.

    Unlike the legacy all-pixel distillation objective, this loss deliberately
    leaves teacher errors unconstrained.  The protected KL is macro-averaged
    over ground-truth classes so abundant land-cover classes cannot suppress
    rare sibling corrections.  The returned fraction is a detached monitoring
    statistic over all target pixels at the student-logit resolution.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("fixed-anchor teacher/student logits must have equal shape")
    confidence_threshold = float(confidence_threshold)
    temperature = float(temperature)
    if not math.isfinite(confidence_threshold) or not (
        0.0 <= confidence_threshold <= 1.0
    ):
        raise ValueError("fixed-anchor confidence threshold must be in [0, 1]")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("fixed-anchor temperature must be positive")

    sampled_target = F.interpolate(
        target[:, None].float(), size=student_logits.shape[-2:], mode="nearest"
    )[:, 0].long()
    valid = sampled_target >= 0
    # Distillation arithmetic is intentionally FP32 even under CUDA AMP.  The
    # teacher remains detached while gradients flow through the cast student.
    teacher = teacher_logits.detach().float()
    student = student_logits.float()
    teacher_probability = teacher.softmax(dim=1)
    confidence, prediction = teacher_probability.max(dim=1)
    protected = (
        valid
        & (prediction == sampled_target.clamp_min(0))
        & (confidence >= confidence_threshold)
    )
    protected_fraction = protected.to(student.dtype).mean().detach()
    if not bool(protected.any()):
        return student.sum() * 0.0, protected_fraction

    scaled_teacher = (teacher / temperature).softmax(dim=1)
    scaled_student = (student / temperature).log_softmax(dim=1)
    pixel_kl = F.kl_div(
        scaled_student, scaled_teacher, reduction="none"
    ).sum(dim=1) * temperature**2
    class_losses = [
        pixel_kl[protected & (sampled_target == class_id)].mean()
        for class_id in sampled_target[protected].unique().tolist()
    ]
    return torch.stack(class_losses).mean(), protected_fraction


def _validate_fusion_truth_probability(
    target: torch.Tensor,
    ms75_truth_probability: torch.Tensor,
) -> None:
    if tuple(ms75_truth_probability.shape) != tuple(target.shape):
        raise ValueError(
            "MS75 truth probability must have the same shape as target"
        )
    supervised = target >= 0
    q_y = ms75_truth_probability.detach().float()
    invalid = supervised & (
        (~torch.isfinite(q_y)) | (q_y < 0.0) | (q_y > 1.0)
    )
    if bool(invalid.any()):
        raise ValueError(
            "supervised MS75 truth probabilities must be finite and in [0, 1]"
        )


def fusion_mixture_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    ms75_truth_probability: torch.Tensor,
    *,
    anchor_probability_weight: float = 0.5,
    validate_probability_values: bool = True,
) -> torch.Tensor:
    """NLL of the frozen-MS75/local equal-probability mixture at truth.

    ``ms75_truth_probability`` contains only ``p_MS75(y|x)`` and is detached
    evidence, never a model input.  Arithmetic is FP32 under AMP and uses
    logaddexp so exact zero producer probabilities remain well defined.
    """

    if logits.ndim != 4:
        raise ValueError("fusion mixture logits must have shape [B,C,H,W]")
    if target.ndim != 3 or tuple(target.shape) != (
        int(logits.shape[0]),
        int(logits.shape[2]),
        int(logits.shape[3]),
    ):
        raise ValueError("fusion mixture target shape differs from logits")
    if validate_probability_values:
        _validate_fusion_truth_probability(target, ms75_truth_probability)
    elif tuple(ms75_truth_probability.shape) != tuple(target.shape):
        raise ValueError(
            "MS75 truth probability must have the same shape as target"
        )
    anchor_probability_weight = float(anchor_probability_weight)
    if not math.isfinite(anchor_probability_weight) or not (
        0.0 < anchor_probability_weight < 1.0
    ):
        raise ValueError("fusion anchor probability weight must be in (0, 1)")

    supervised = target >= 0
    q_y = ms75_truth_probability.detach().float()
    safe_target = target.clamp_min(0)
    log_p_y = (
        logits.float()
        .log_softmax(dim=1)
        .gather(1, safe_target[:, None])
        .squeeze(1)
    )
    # Invalid pixels are replaced before log: multiplying a NaN/-inf by a zero
    # mask later would otherwise still contaminate the reduction.
    safe_q_y = torch.where(supervised, q_y, torch.ones_like(q_y))
    local_probability_weight = 1.0 - anchor_probability_weight
    log_mixture = torch.logaddexp(
        log_p_y + math.log(local_probability_weight),
        safe_q_y.log() + math.log(anchor_probability_weight),
    )
    return _masked_mean(-log_mixture, supervised)


def _fusionaware_a1_masked_target(
    target: torch.Tensor,
    fusion_truth_probability_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask A1-only supervision for spectrally altered samples."""

    if fusion_truth_probability_valid is None:
        raise ValueError(
            "FusionAware A1 requires fusion_truth_probability_valid"
        )
    if fusion_truth_probability_valid.dtype != torch.bool or tuple(
        fusion_truth_probability_valid.shape
    ) != (int(target.shape[0]),):
        raise ValueError(
            "fusion_truth_probability_valid must be bool with shape [B]"
        )
    valid = fusion_truth_probability_valid.to(device=target.device)
    return target.masked_fill(~valid[:, None, None], -1), valid


def segmentation_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    sar_valid: torch.Tensor,
    config: dict,
    *,
    teacher_logits: torch.Tensor | None = None,
    phase: str = "joint",
    sar_aux_eligible: torch.Tensor | None = None,
    ms75_truth_probability: torch.Tensor | None = None,
    local512_truth_probability: torch.Tensor | None = None,
    fusion_truth_probability_valid: torch.Tensor | None = None,
    validate_ms75_truth_probability: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_config = config["loss"]
    num_classes = int(config["model"]["num_classes"])
    main_logits = outputs["logits"]
    if bool(loss_config.get("force_main_losses_fp32", False)):
        main_logits = main_logits.float()
    class_weights = loss_config.get("class_weights")
    weight_tensor = None
    if class_weights:
        weight_tensor = torch.as_tensor(
            class_weights, device=target.device, dtype=outputs["logits"].dtype
        )

    final_pixel = F.cross_entropy(
        main_logits,
        target,
        weight=weight_tensor,
        ignore_index=-1,
        reduction="none",
    )
    supervised = target >= 0
    final_ce = _masked_mean(final_pixel, supervised)
    zero = main_logits.sum() * 0.0
    needs_ms_pixel = (
        float(loss_config.get("ms_aux_weight", 0.0)) > 0
        or float(loss_config.get("negative_transfer_weight", 0.0)) > 0
    )
    ms_pixel = (
        F.cross_entropy(
            outputs["ms_logits"],
            target,
            weight=weight_tensor,
            ignore_index=-1,
            reduction="none",
        )
        if needs_ms_pixel
        else None
    )
    ms_ce = _masked_mean(ms_pixel, supervised) if ms_pixel is not None else zero
    if float(loss_config.get("sar_aux_weight", 0.0)) > 0:
        sar_pixel = F.cross_entropy(
            outputs["sar_logits"],
            target,
            weight=weight_tensor,
            ignore_index=-1,
            reduction="none",
        )
        sar_valid_fine = F.interpolate(
            sar_valid[:, None].float(),
            size=target.shape[-2:],
            mode="nearest",
        )[:, 0].bool()
        sar_ce = _masked_mean(sar_pixel, supervised & sar_valid_fine)
    else:
        sar_ce = zero
    needs_native = (
        float(loss_config.get("sar_native_aux_weight", 0.0)) > 0
        or phase in {"sar_native", "sar_pretrain"}
    )
    sar_native = (
        native_sar_soft_label_loss(
            outputs["sar_native_logits"],
            target,
            sar_valid,
            num_classes,
            min_purity=float(loss_config.get("sar_native_min_purity", 0.6)),
            class_weights=weight_tensor,
            sample_eligible=sar_aux_eligible,
        )
        if needs_native
        else zero
    )
    dice = (
        soft_dice_loss(main_logits, target, num_classes)
        if float(loss_config.get("dice_weight", 0.0)) > 0
        else zero
    )
    lovasz = (
        lovasz_softmax_loss(main_logits, target)
        if "flat_logits" in outputs
        and float(loss_config.get("lovasz_weight", 0.0)) > 0
        else zero
    )
    hierarchy_enabled = (
        float(loss_config.get("family_ce_weight", 0.0)) > 0
        or float(loss_config.get("hierarchy_consistency_weight", 0.0)) > 0
    )
    if hierarchy_enabled and {
        "flat_logits",
        "family_logits",
        "fine_to_family",
    }.issubset(outputs):
        family_ce, hierarchy_consistency = hierarchy_losses(
            outputs["flat_logits"],
            outputs["family_logits"],
            target,
            outputs["fine_to_family"],
        )
    else:
        family_ce = zero
        hierarchy_consistency = zero
    boundary = (
        semantic_boundary_loss(outputs["boundary_logits"], target)
        if "boundary_logits" in outputs
        and float(loss_config.get("boundary_weight", 0.0)) > 0
        else zero
    )
    aux = (
        _masked_mean(
            F.cross_entropy(
                outputs["aux_logits"],
                target,
                weight=weight_tensor,
                ignore_index=-1,
                reduction="none",
            ),
            supervised,
        )
        if "aux_logits" in outputs
        and float(loss_config.get("aux_weight", 0.0)) > 0
        else zero
    )
    spectral_aux = (
        _masked_mean(
            F.cross_entropy(
                outputs["spectral_aux_logits"],
                target,
                weight=weight_tensor,
                ignore_index=-1,
                reduction="none",
            ),
            supervised,
        )
        if "spectral_aux_logits" in outputs
        and float(loss_config.get("spectral_aux_weight", 0.0)) > 0
        else zero
    )
    context_aux = (
        _masked_mean(
            F.cross_entropy(
                outputs["context_aux_logits"],
                target,
                weight=weight_tensor,
                ignore_index=-1,
                reduction="none",
            ),
            supervised,
        )
        if "context_aux_logits" in outputs
        and float(loss_config.get("context_aux_weight", 0.0)) > 0
        else zero
    )
    short_aux = (
        _masked_mean(
            F.cross_entropy(
                outputs["short_aux_logits"],
                target,
                weight=weight_tensor,
                ignore_index=-1,
                reduction="none",
            ),
            supervised,
        )
        if "short_aux_logits" in outputs
        and float(loss_config.get("short_aux_weight", 0.0)) > 0
        else zero
    )
    global_aux = (
        _masked_mean(
            F.cross_entropy(
                outputs["global_aux_logits"],
                target,
                weight=weight_tensor,
                ignore_index=-1,
                reduction="none",
            ),
            supervised,
        )
        if "global_aux_logits" in outputs
        and float(loss_config.get("global_aux_weight", 0.0)) > 0
        else zero
    )
    family_region_aux = (
        family_conditional_subtype_loss(
            outputs["family_region_child_logits"],
            target,
            outputs["fine_to_family"],
            class_weights=weight_tensor,
            balance_classes=bool(
                loss_config.get("family_region_class_balanced", False)
            ),
        )
        if "family_region_child_logits" in outputs
        and float(loss_config.get("family_region_aux_weight", 0.0)) > 0
        else zero
    )
    family_child = (
        family_conditional_subtype_loss(
            outputs["family_child_logits"],
            target,
            outputs["fine_to_family"],
            class_weights=weight_tensor,
            balance_classes=bool(
                loss_config.get("family_child_class_balanced", True)
            ),
        )
        if {
            "family_child_logits",
            "fine_to_family",
        }.issubset(outputs)
        and float(loss_config.get("family_child_weight", 0.0)) > 0
        else zero
    )
    family_contrast = (
        sibling_supervised_contrastive_loss(
            outputs["family_contrast_embeddings"],
            target,
            outputs["fine_to_family"],
            temperature=float(
                loss_config.get("family_contrast_temperature", 0.08)
            ),
            max_samples_per_class=int(
                loss_config.get("family_contrast_max_samples_per_class", 64)
            ),
        )
        if {
            "family_contrast_embeddings",
            "fine_to_family",
        }.issubset(outputs)
        and float(loss_config.get("family_contrast_weight", 0.0)) > 0
        else zero
    )
    family_region_confusion = (
        family_confusion_margin_loss(
            outputs["family_region_refined_logits"],
            target,
            [
                [int(value) for value in group]
                for group in loss_config.get(
                    "family_region_confusion_groups", []
                )
            ],
            margin=float(
                loss_config.get("family_region_confusion_margin", 0.0)
            ),
        )
        if "family_region_refined_logits" in outputs
        and float(loss_config.get("family_region_confusion_weight", 0.0)) > 0
        else zero
    )
    family_region_no_harm = (
        family_no_harm_distillation_loss(
            outputs["family_region_refined_logits"],
            outputs["family_region_baseline_logits"],
            target,
            confidence_threshold=float(
                loss_config.get("family_region_no_harm_confidence", 0.7)
            ),
            temperature=float(
                loss_config.get("family_region_no_harm_temperature", 2.0)
            ),
        )
        if {
            "family_region_refined_logits",
            "family_region_baseline_logits",
        }.issubset(outputs)
        and float(loss_config.get("family_region_no_harm_weight", 0.0)) > 0
        else zero
    )
    region_objective_enabled = (
        float(loss_config.get("region_prototype_weight", 0.0)) > 0
        or float(loss_config.get("confusion_margin_weight", 0.0)) > 0
    )
    if region_objective_enabled and {
        "region_embeddings",
        "class_prototypes",
    }.issubset(outputs):
        region_prototype, confusion_margin = region_prototype_losses(
            outputs["region_embeddings"],
            outputs["class_prototypes"],
            target,
            temperature=float(loss_config.get("region_temperature", 10.0)),
            min_pixels=int(loss_config.get("region_min_pixels", 4)),
            confusion_groups=[
                [int(value) for value in group]
                for group in loss_config.get("confusion_groups", [])
            ],
            confusion_margin=float(
                loss_config.get("confusion_margin", 0.0)
            ),
        )
    else:
        region_prototype = zero
        confusion_margin = zero
    prototype = (
        class_prototype_alignment(
            outputs["ms_shared"],
            outputs["sar_shared"],
            target,
            sar_valid,
            num_classes,
        )
        if float(loss_config.get("prototype_alignment_weight", 0.0)) > 0
        else zero
    )
    orthogonality = (
        private_orthogonality(
            outputs["sar_shared"], outputs["sar_private"], sar_valid
        )
        if float(loss_config.get("private_orthogonality_weight", 0.0)) > 0
        else zero
    )
    negative_transfer_reduction = str(
        loss_config.get("negative_transfer_reduction", "pixel")
    )
    negative_transfer_enabled = (
        float(loss_config.get("negative_transfer_weight", 0.0)) > 0
    )
    if negative_transfer_enabled and negative_transfer_reduction == "class_region_best":
        safe_target = target.clamp_min(0)
        class_mask = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2)
        class_mask = class_mask.to(final_pixel.dtype) * supervised[:, None]
        region_grid = int(config["model"].get("router_grid", 8))

        def pool_class_loss(pixel_loss: torch.Tensor) -> torch.Tensor:
            numerator = F.adaptive_avg_pool2d(
                class_mask * pixel_loss[:, None],
                (region_grid, region_grid),
            )
            denominator = F.adaptive_avg_pool2d(
                class_mask,
                (region_grid, region_grid),
            )
            return numerator / denominator.clamp_min(1e-6)

        final_region = pool_class_loss(final_pixel)
        ms_region = pool_class_loss(ms_pixel).detach()
        both_logits = outputs.get("counterfactual_both_logits")
        if both_logits is not None:
            both_pixel = F.cross_entropy(
                both_logits,
                target,
                weight=weight_tensor,
                ignore_index=-1,
                reduction="none",
            )
            reference_region = torch.minimum(
                ms_region,
                pool_class_loss(both_pixel).detach(),
            )
        else:
            reference_region = ms_region
        support_fraction = F.adaptive_avg_pool2d(
            class_mask,
            (region_grid, region_grid),
        )
        approximate_region_area = (
            target.shape[-2] * target.shape[-1] / float(region_grid**2)
        )
        minimum = int(loss_config.get("negative_transfer_min_pixels", 1))
        region_valid = support_fraction * approximate_region_area >= max(minimum, 1)
        negative_transfer = _masked_mean(
            F.relu(final_region - reference_region),
            region_valid,
        )
    elif negative_transfer_enabled and negative_transfer_reduction == "pixel":
        assert ms_pixel is not None
        negative_transfer = _masked_mean(
            F.relu(final_pixel - ms_pixel.detach()), supervised
        )
    elif not negative_transfer_enabled:
        negative_transfer = zero
    else:
        raise ValueError(
            "loss.negative_transfer_reduction must be pixel or class_region_best"
        )
    route_enabled = (
        phase != "expert_warmup"
        and float(loss_config.get("utility_routing_weight", 0.0)) > 0
    )
    if route_enabled:
        route_loss, route_metrics = utility_routing_loss(
            outputs,
            target,
            class_weights=weight_tensor,
            margin=float(loss_config.get("route_utility_margin", 0.0)),
            target_mode=str(loss_config.get("route_target_mode", "hard")),
            temperature=float(loss_config.get("route_utility_temperature", 1.0)),
            min_pixels=int(loss_config.get("route_min_class_region_pixels", 1)),
            min_utility_gap=float(loss_config.get("route_min_utility_gap", 0.0)),
            relative_utility=bool(loss_config.get("route_relative_utility", False)),
        )
    else:
        route_loss = zero
        route_metrics = {
            "route_accuracy": zero,
            "route_non_null_target_fraction": zero,
            "route_non_null_usage": zero,
            "route_entropy": zero,
        }
    alignment_offsets = outputs.get("alignment_offsets")
    offset_regularization = (
        alignment_offsets.abs().mean()
        if alignment_offsets is not None
        and float(loss_config.get("alignment_offset_weight", 0.0)) > 0
        else zero
    )

    fixed_anchor_no_harm = zero
    fixed_anchor_protected_fraction = zero.detach()
    fixed_anchor_no_harm_weight = float(
        loss_config.get("fixed_anchor_no_harm_weight", 0.0)
    )
    if fixed_anchor_no_harm_weight > 0:
        if teacher_logits is None:
            raise ValueError(
                "fixed_anchor_no_harm_weight requires immutable teacher logits"
            )
        fixed_anchor_no_harm, fixed_anchor_protected_fraction = (
            fixed_anchor_no_harm_distillation_loss(
                outputs["logits"],
                teacher_logits,
                target,
                confidence_threshold=float(
                    loss_config.get("fixed_anchor_no_harm_confidence", 0.7)
                ),
                temperature=float(
                    loss_config.get("fixed_anchor_no_harm_temperature", 2.0)
                ),
            )
        )

    distillation = zero
    if teacher_logits is not None and float(
        loss_config.get("teacher_distillation_weight", 0.0)
    ) > 0:
        temperature = float(loss_config["distillation_temperature"])
        teacher_probability = (teacher_logits / temperature).softmax(dim=1)
        student_log_probability = (
            outputs["ms_logits"] / temperature
        ).log_softmax(dim=1)
        pixel_kl = F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="none",
        ).sum(dim=1)
        distillation = _masked_mean(pixel_kl, supervised) * temperature**2

    fusion_mixture = zero
    fusion_mixture_weight = float(
        loss_config.get("fusion_mixture_nll_weight", 0.0)
    )
    ms75_truth_probability_mean = zero.detach()
    if fusion_mixture_weight > 0.0:
        if ms75_truth_probability is None:
            raise ValueError(
                "fusion_mixture_nll_weight requires MS75 truth probabilities"
            )
        fusion_mixture = fusion_mixture_nll(
            outputs["logits"],
            target,
            ms75_truth_probability,
            anchor_probability_weight=float(
                loss_config.get("fusion_anchor_probability", 0.5)
            ),
            validate_probability_values=validate_ms75_truth_probability,
        )
        safe_truth_probability = torch.where(
            supervised,
            ms75_truth_probability.detach().float(),
            torch.zeros_like(ms75_truth_probability, dtype=torch.float32),
        )
        ms75_truth_probability_mean = safe_truth_probability.sum() / (
            supervised.sum().clamp_min(1).to(dtype=torch.float32)
        )

    a1_zero = outputs["logits"].float().reshape(-1)[:0].sum()
    a1_additional = a1_zero
    a1_dual = a1_zero
    a1_repair = a1_zero
    a1_class_balanced = a1_zero
    a1_preserve = a1_zero
    a1_valid_fraction = a1_zero.detach()
    local512_truth_probability_mean = a1_zero.detach()
    fusionaware_a1_enabled = bool(
        loss_config.get("fusionaware_a1_enabled", False)
    )
    if fusionaware_a1_enabled:
        if fusion_mixture_weight != 0.0:
            raise ValueError(
                "FusionAware A1 forbids the legacy single-expert fusion mixture"
            )
        if ms75_truth_probability is None or local512_truth_probability is None:
            raise ValueError(
                "FusionAware A1 requires both MS75 and Local512 truth probabilities"
            )
        coefficient_contract = {
            "fusionaware_a1_dual_fusion_weight": DUAL_FUSION_LOSS_WEIGHT,
            "fusionaware_a1_repair_weight": REPAIR_LOSS_WEIGHT,
            "fusionaware_a1_class_balanced_weight": (
                CLASS_BALANCED_FUSION_LOSS_WEIGHT
            ),
            "fusionaware_a1_noharm_weight": PRESERVE_LOSS_WEIGHT,
        }
        changed_coefficients = {
            key: {
                "expected": expected,
                "observed": loss_config.get(key),
            }
            for key, expected in coefficient_contract.items()
            if loss_config.get(key) != expected
        }
        if changed_coefficients:
            raise ValueError(
                "FusionAware A1 coefficient contract changed: "
                f"{changed_coefficients}"
            )
        a1_target, a1_valid = _fusionaware_a1_masked_target(
            target, fusion_truth_probability_valid
        )
        a1_class_weights = loss_config.get("fusionaware_a1_class_weights")
        if a1_class_weights is None:
            raise ValueError(
                "FusionAware A1 requires frozen histogram-derived class weights"
            )
        a1_additional, raw_a1_components = (
            fusionaware_a1_additional_loss(
                outputs["logits"],
                a1_target,
                ms75_truth_probability,
                local512_truth_probability,
                a1_class_weights,
            )
        )
        a1_dual = raw_a1_components["dual_fusion_nll"]
        a1_repair = raw_a1_components["repair_weighted_truth_nll"]
        a1_class_balanced = raw_a1_components[
            "class_balanced_dual_fusion_nll"
        ]
        a1_preserve = raw_a1_components[
            "preserve_noharm_one_sided_huber"
        ]
        a1_valid_fraction = a1_valid.float().mean().detach()
        a1_supervised = a1_target >= 0
        a1_denominator = a1_supervised.sum().clamp_min(1).to(torch.float32)
        safe_ms75 = torch.where(
            a1_supervised,
            ms75_truth_probability.detach().float(),
            torch.zeros_like(ms75_truth_probability, dtype=torch.float32),
        )
        safe_local512 = torch.where(
            a1_supervised,
            local512_truth_probability.detach().float(),
            torch.zeros_like(local512_truth_probability, dtype=torch.float32),
        )
        ms75_truth_probability_mean = safe_ms75.sum() / a1_denominator
        local512_truth_probability_mean = (
            safe_local512.sum() / a1_denominator
        )

    components = {
        "final_ce": final_ce,
        "dice": dice,
        "lovasz": lovasz,
        "family_ce": family_ce,
        "hierarchy_consistency": hierarchy_consistency,
        "boundary": boundary,
        "aux": aux,
        "spectral_aux": spectral_aux,
        "context_aux": context_aux,
        "short_aux": short_aux,
        "global_aux": global_aux,
        "family_region_aux": family_region_aux,
        "family_child": family_child,
        "family_contrast": family_contrast,
        "family_region_confusion": family_region_confusion,
        "family_region_no_harm": family_region_no_harm,
        "region_prototype": region_prototype,
        "confusion_margin": confusion_margin,
        "ms_aux": ms_ce,
        "sar_aux": sar_ce,
        "sar_native_aux": sar_native,
        "prototype_alignment": prototype,
        "private_orthogonality": orthogonality,
        "negative_transfer": negative_transfer,
        "utility_routing": route_loss,
        "alignment_offset": offset_regularization,
        "teacher_distillation": distillation,
        "fixed_anchor_no_harm": fixed_anchor_no_harm,
        "fixed_anchor_protected_fraction": fixed_anchor_protected_fraction,
        "fusion_mixture_nll": fusion_mixture,
        "ms75_truth_probability_mean": ms75_truth_probability_mean,
        "local512_truth_probability_mean": local512_truth_probability_mean,
        "fusionaware_a1_dual_fusion_nll": a1_dual,
        "fusionaware_a1_repair_weighted_truth_nll": a1_repair,
        "fusionaware_a1_class_balanced_dual_fusion_nll": (
            a1_class_balanced
        ),
        "fusionaware_a1_preserve_noharm_one_sided_huber": a1_preserve,
        "fusionaware_a1_additional": a1_additional,
        "fusionaware_a1_valid_fraction": a1_valid_fraction,
        "sar_gate_mean": outputs["sar_gate"].mean(),
        "sar_scale": outputs["sar_scale"].mean(),
        **route_metrics,
    }
    spectral_scales = outputs.get("spectral_scales")
    if spectral_scales is not None:
        components["spectral_scale_mean"] = spectral_scales.mean()
        for family_id, scale in enumerate(spectral_scales):
            components[f"spectral_scale_family_{family_id}"] = scale
    family_region_scales = outputs.get("family_region_scales")
    if family_region_scales is not None:
        components["family_region_scale_mean"] = (
            family_region_scales.mean()
        )
        for family_id, scale in enumerate(family_region_scales):
            components[f"family_region_scale_family_{family_id}"] = scale
    family_region_gate_means = outputs.get("family_region_gate_means")
    if family_region_gate_means is not None:
        for family_id, gate_mean in enumerate(family_region_gate_means):
            components[f"family_region_gate_family_{family_id}"] = gate_mean
    context_scale = outputs.get("context_residual_scale")
    context_scale_raw = outputs.get("context_residual_scale_raw")
    if context_scale is not None:
        components["context_residual_scale"] = context_scale.mean()
    if context_scale_raw is not None:
        components["context_residual_scale_raw"] = context_scale_raw.mean()
    hierarchy_mix = outputs.get("hierarchy_mix")
    hierarchy_mix_raw = outputs.get("hierarchy_mix_raw")
    if hierarchy_mix is not None:
        hierarchy_maximum = float(
            config.get("model", {})
            .get("ms75", {})
            .get("max_hierarchy_mix", 1.0)
        )
        components["hierarchy_mix_mean"] = hierarchy_mix.mean()
        components["hierarchy_mix_zero_fraction"] = (
            hierarchy_mix <= 0.0
        ).to(hierarchy_mix.dtype).mean()
        components["hierarchy_mix_upper_fraction"] = (
            hierarchy_mix >= hierarchy_maximum
        ).to(hierarchy_mix.dtype).mean()
    if hierarchy_mix_raw is not None:
        components["hierarchy_mix_raw_mean"] = hierarchy_mix_raw.mean()
    view_scale_weights = outputs.get("view_scale_weights")
    if view_scale_weights is not None:
        if view_scale_weights.ndim != 5 or view_scale_weights.shape[2] != 4:
            raise ValueError(
                "view_scale_weights must have shape [batch, levels, 4, height, width]"
            )
        view_means = view_scale_weights.mean(dim=(0, 1, 3, 4))
        for view_name, view_mean in zip(
            ("local", "short", "global", "spectral"), view_means
        ):
            components[f"view_scale_weight_{view_name}"] = view_mean
    routing_weight = (
        0.0
        if phase == "expert_warmup"
        else float(loss_config.get("utility_routing_weight", 0.0))
    )
    total = (
        float(loss_config["cross_entropy_weight"]) * final_ce
        + float(loss_config["dice_weight"]) * dice
        + float(loss_config.get("lovasz_weight", 0.0)) * lovasz
        + float(loss_config.get("family_ce_weight", 0.0)) * family_ce
        + float(loss_config.get("hierarchy_consistency_weight", 0.0))
        * hierarchy_consistency
        + float(loss_config.get("boundary_weight", 0.0)) * boundary
        + float(loss_config.get("aux_weight", 0.0)) * aux
        + float(loss_config.get("spectral_aux_weight", 0.0)) * spectral_aux
        + float(loss_config.get("context_aux_weight", 0.0)) * context_aux
        + float(loss_config.get("short_aux_weight", 0.0)) * short_aux
        + float(loss_config.get("global_aux_weight", 0.0)) * global_aux
        + float(loss_config.get("family_region_aux_weight", 0.0))
        * family_region_aux
        + float(loss_config.get("family_child_weight", 0.0))
        * family_child
        + float(loss_config.get("family_contrast_weight", 0.0))
        * family_contrast
        + float(loss_config.get("family_region_confusion_weight", 0.0))
        * family_region_confusion
        + float(loss_config.get("family_region_no_harm_weight", 0.0))
        * family_region_no_harm
        + float(loss_config.get("region_prototype_weight", 0.0))
        * region_prototype
        + float(loss_config.get("confusion_margin_weight", 0.0))
        * confusion_margin
        + float(loss_config["ms_aux_weight"]) * ms_ce
        + float(loss_config["sar_aux_weight"]) * sar_ce
        + float(loss_config.get("sar_native_aux_weight", 0.0)) * sar_native
        + float(loss_config["prototype_alignment_weight"]) * prototype
        + float(loss_config["private_orthogonality_weight"]) * orthogonality
        + float(loss_config["negative_transfer_weight"]) * negative_transfer
        + routing_weight * route_loss
        + float(loss_config.get("alignment_offset_weight", 0.0))
        * offset_regularization
        + float(loss_config["teacher_distillation_weight"]) * distillation
        + fixed_anchor_no_harm_weight * fixed_anchor_no_harm
        + fusion_mixture_weight * fusion_mixture
        + a1_additional
    )
    if phase == "sar_native":
        total = float(loss_config["cross_entropy_weight"]) * sar_native
    elif phase == "sar_pretrain":
        native_weight = float(loss_config.get("sar_native_aux_weight", 0.0))
        if native_weight <= 0:
            raise ValueError(
                "sar_pretrain phase requires a positive sar_native_aux_weight"
            )
        total = native_weight * sar_native
    elif phase not in {"joint", "expert_warmup"}:
        raise ValueError(f"unsupported training phase: {phase}")
    components["total"] = total
    return total, components


def grouped_segmentation_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    sar_valid: torch.Tensor,
    config: dict,
    *,
    group_size: int,
    teacher_logits: torch.Tensor | None = None,
    phase: str = "joint",
    sar_aux_eligible: torch.Tensor | None = None,
    ms75_truth_probability: torch.Tensor | None = None,
    local512_truth_probability: torch.Tensor | None = None,
    fusion_truth_probability_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average ordinary losses over equal-sized slices of one model batch.

    LocalDetail-256 uses a large physical batch to keep the GPU occupied, but
    Dice and Lovasz aggregate pixels across the batch dimension.  Evaluating
    those objectives in groups of four preserves the loss semantics of the
    established LocalDetail-512 recipe while retaining a single batched model
    forward.  Configurations without ``train.loss_microbatch_size`` continue
    to call :func:`segmentation_loss` directly.
    """

    if target.ndim < 1:
        raise ValueError("grouped segmentation target must have a batch dimension")
    batch_size = int(target.shape[0])
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise ValueError("segmentation loss group_size must be an integer")
    if group_size <= 0 or group_size > batch_size:
        raise ValueError(
            "segmentation loss group_size must be in (0, physical batch size]"
        )
    if batch_size % group_size:
        raise ValueError(
            "physical batch size must be divisible by segmentation loss group_size"
        )
    if int(sar_valid.shape[0]) != batch_size:
        raise ValueError("sar_valid batch dimension differs from target")
    if teacher_logits is not None and int(teacher_logits.shape[0]) != batch_size:
        raise ValueError("teacher logits batch dimension differs from target")
    if sar_aux_eligible is not None and int(sar_aux_eligible.shape[0]) != batch_size:
        raise ValueError("sar_aux_eligible batch dimension differs from target")
    if (
        ms75_truth_probability is not None
        and int(ms75_truth_probability.shape[0]) != batch_size
    ):
        raise ValueError(
            "MS75 truth-probability batch dimension differs from target"
        )
    if (
        local512_truth_probability is not None
        and int(local512_truth_probability.shape[0]) != batch_size
    ):
        raise ValueError(
            "Local512 truth-probability batch dimension differs from target"
        )
    if fusion_truth_probability_valid is not None and tuple(
        fusion_truth_probability_valid.shape
    ) != (batch_size,):
        raise ValueError(
            "fusion truth-probability validity must have shape [B]"
        )

    # Validate the frozen evidence once for the whole physical batch.  Calling
    # the public loss normally still validates its own tensor, while grouped
    # Local256 training avoids one CUDA-to-CPU synchronization per group.
    fusion_mixture_weight = float(
        config["loss"].get("fusion_mixture_nll_weight", 0.0)
    )
    if fusion_mixture_weight > 0.0:
        if ms75_truth_probability is None:
            raise ValueError(
                "fusion_mixture_nll_weight requires MS75 truth probabilities"
            )
        _validate_fusion_truth_probability(target, ms75_truth_probability)

    losses: list[torch.Tensor] = []
    grouped_components: dict[str, list[torch.Tensor]] = {}
    for start in range(0, batch_size, group_size):
        stop = start + group_size
        group_outputs = {
            key: (
                value[start:stop]
                if torch.is_tensor(value)
                and value.ndim > 0
                and int(value.shape[0]) == batch_size
                else value
            )
            for key, value in outputs.items()
        }
        group_loss, components = segmentation_loss(
            group_outputs,
            target[start:stop],
            sar_valid[start:stop],
            config,
            teacher_logits=(
                teacher_logits[start:stop]
                if teacher_logits is not None
                else None
            ),
            phase=phase,
            sar_aux_eligible=(
                sar_aux_eligible[start:stop]
                if sar_aux_eligible is not None
                else None
            ),
            ms75_truth_probability=(
                ms75_truth_probability[start:stop]
                if ms75_truth_probability is not None
                else None
            ),
            local512_truth_probability=(
                local512_truth_probability[start:stop]
                if local512_truth_probability is not None
                else None
            ),
            fusion_truth_probability_valid=(
                fusion_truth_probability_valid[start:stop]
                if fusion_truth_probability_valid is not None
                else None
            ),
            validate_ms75_truth_probability=False,
        )
        losses.append(group_loss)
        if not grouped_components:
            grouped_components = {key: [value] for key, value in components.items()}
        else:
            if set(components) != set(grouped_components):
                raise RuntimeError("loss component keys changed between groups")
            for key, value in components.items():
                grouped_components[key].append(value)

    loss = torch.stack(losses).mean()
    averaged_components = {
        key: torch.stack(values).mean()
        for key, values in grouped_components.items()
    }
    if fusion_mixture_weight > 0.0 and ms75_truth_probability is not None:
        supervised = target >= 0
        safe_truth_probability = torch.where(
            supervised,
            ms75_truth_probability.detach().float(),
            torch.zeros_like(ms75_truth_probability, dtype=torch.float32),
        )
        averaged_components["ms75_truth_probability_mean"] = (
            safe_truth_probability.sum()
            / supervised.sum().clamp_min(1).to(dtype=torch.float32)
        )
    if bool(config["loss"].get("fusionaware_a1_enabled", False)):
        if (
            ms75_truth_probability is None
            or local512_truth_probability is None
            or fusion_truth_probability_valid is None
        ):
            raise ValueError("grouped FusionAware A1 evidence is incomplete")
        a1_target, a1_valid = _fusionaware_a1_masked_target(
            target, fusion_truth_probability_valid
        )
        a1_supervised = a1_target >= 0
        denominator = a1_supervised.sum().clamp_min(1).to(torch.float32)
        averaged_components["ms75_truth_probability_mean"] = torch.where(
            a1_supervised,
            ms75_truth_probability.detach().float(),
            torch.zeros_like(ms75_truth_probability, dtype=torch.float32),
        ).sum() / denominator
        averaged_components["local512_truth_probability_mean"] = torch.where(
            a1_supervised,
            local512_truth_probability.detach().float(),
            torch.zeros_like(local512_truth_probability, dtype=torch.float32),
        ).sum() / denominator
        averaged_components["fusionaware_a1_valid_fraction"] = (
            a1_valid.float().mean().detach()
        )
    averaged_components["total"] = loss
    return loss, averaged_components
