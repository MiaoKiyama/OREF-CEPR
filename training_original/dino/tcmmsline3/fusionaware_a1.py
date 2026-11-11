"""Loss-only building blocks for the FusionAware A1 training protocol.

The module deliberately does not compute the ordinary segmentation loss.  It
returns only the A1 additions that a training entry point may add to its
existing CE/Dice/Lovasz/etc. objective.

Only each frozen expert's probability at the ground-truth class is required:
``q75 = p_MS75(y|x)`` and ``q512 = p_Local512(y|x)``.  Both tensors are always
detached evidence and never receive gradients.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch


MS75_FUSION_WEIGHT = 0.50
LOCAL512_FUSION_WEIGHT = 0.25
STUDENT_FUSION_WEIGHT = 0.25

DUAL_FUSION_LOSS_WEIGHT = 1.00
REPAIR_LOSS_WEIGHT = 0.20
CLASS_BALANCED_FUSION_LOSS_WEIGHT = 0.10
PRESERVE_LOSS_WEIGHT = 0.25

REPAIR_BASE_THRESHOLD = 0.50
SECURE_BASE_THRESHOLD = 0.75
SECURE_MS75_THRESHOLD = 0.70
SECURE_LOCAL512_THRESHOLD = 0.55
ONE_SIDED_MARGIN = 0.05
HUBER_BETA = 0.05


def _validate_inputs(
    logits: torch.Tensor,
    target: torch.Tensor,
    q75: torch.Tensor,
    q512: torch.Tensor,
) -> torch.Tensor:
    """Validate A1 tensors and return the supervised-pixel mask."""

    if logits.ndim != 4:
        raise ValueError("A1 logits must have shape [B,C,H,W]")
    if not logits.is_floating_point():
        raise ValueError("A1 logits must be floating point")
    expected_target_shape = (
        int(logits.shape[0]),
        int(logits.shape[2]),
        int(logits.shape[3]),
    )
    if target.ndim != 3 or tuple(target.shape) != expected_target_shape:
        raise ValueError("A1 target shape differs from logits")
    if target.dtype != torch.long:
        raise ValueError("A1 target must have dtype torch.long")
    if tuple(q75.shape) != tuple(target.shape):
        raise ValueError("MS75 truth probability must have the same shape as target")
    if tuple(q512.shape) != tuple(target.shape):
        raise ValueError(
            "Local512 truth probability must have the same shape as target"
        )
    if int(logits.shape[1]) <= 1:
        raise ValueError("A1 logits must contain at least two classes")
    if target.device != logits.device:
        raise ValueError("A1 target and logits must be on the same device")
    if q75.device != logits.device or q512.device != logits.device:
        raise ValueError("A1 teacher probabilities and logits must share a device")

    supervised = target >= 0
    if bool((supervised & (target >= int(logits.shape[1]))).any()):
        raise ValueError("A1 supervised target contains an out-of-range class")
    finite_logits = torch.isfinite(logits.detach().float()).all(dim=1)
    if bool((supervised & (~finite_logits)).any()):
        raise ValueError("A1 supervised logits must be finite")

    for name, value in (("MS75", q75), ("Local512", q512)):
        detached = value.detach().float()
        invalid = supervised & (
            (~torch.isfinite(detached)) | (detached < 0.0) | (detached > 1.0)
        )
        if bool(invalid.any()):
            raise ValueError(
                f"supervised {name} truth probabilities must be finite and in [0, 1]"
            )
    return supervised


def _truth_log_probability(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    safe_target = target.clamp(min=0, max=int(logits.shape[1]) - 1)
    return (
        logits.float()
        .log_softmax(dim=1)
        .gather(1, safe_target[:, None])
        .squeeze(1)
    )


def _masked_mean_or_connected_zero(
    values: torch.Tensor, mask: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    numerator = (values * weights).sum()
    # clamp_min avoids a data-dependent host synchronization.  With an empty
    # mask the zero numerator remains connected to the student's graph.
    denominator = weights.sum().clamp_min(1.0)
    connected_zero = reference.float().reshape(-1)[:0].sum()
    return numerator / denominator + connected_zero


def _weighted_mean_or_connected_zero(
    values: torch.Tensor, weights: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    weights = weights.to(dtype=values.dtype)
    numerator = (values * weights).sum()
    denominator = weights.sum().clamp_min(torch.finfo(values.dtype).eps)
    connected_zero = reference.float().reshape(-1)[:0].sum()
    return numerator / denominator + connected_zero


def _truth_terms(
    logits: torch.Tensor,
    target: torch.Tensor,
    q75: torch.Tensor,
    q512: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    supervised = _validate_inputs(logits, target, q75, q512)
    # Sanitize before log-softmax so even its backward pass cannot be poisoned
    # by a NaN/Inf sentinel at an ignored pixel.
    safe_logits = torch.where(
        supervised[:, None], logits.float(), torch.zeros_like(logits.float())
    )
    log_p_y = _truth_log_probability(safe_logits, target)
    p_y = log_p_y.exp()

    # Ignored pixels may deliberately contain NaN sentinels.  Replace them
    # before log/arithmetic; masking a NaN afterwards would not make it safe.
    q75_safe = torch.where(
        supervised, q75.detach().float(), torch.ones_like(log_p_y)
    )
    q512_safe = torch.where(
        supervised, q512.detach().float(), torch.ones_like(log_p_y)
    )
    return supervised, log_p_y, p_y, torch.stack((q75_safe, q512_safe), dim=0)


def dual_fusion_pixel_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    q75: torch.Tensor,
    q512: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-pixel A1 mixture NLL and its supervised mask.

    The deployed probability is fixed to

    ``0.50*q75 + 0.25*q512 + 0.25*p_student``.

    All log arithmetic is FP32 and stable when either frozen probability is
    exactly zero.
    """

    supervised, log_p_y, _, teachers = _truth_terms(
        logits, target, q75, q512
    )
    q75_safe, q512_safe = teachers[0], teachers[1]
    teacher_log = torch.logaddexp(
        q75_safe.log() + math.log(MS75_FUSION_WEIGHT),
        q512_safe.log() + math.log(LOCAL512_FUSION_WEIGHT),
    )
    mixture_log = torch.logaddexp(
        teacher_log,
        log_p_y + math.log(STUDENT_FUSION_WEIGHT),
    )
    return -mixture_log, supervised


def dual_fusion_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    q75: torch.Tensor,
    q512: torch.Tensor,
) -> torch.Tensor:
    """Mean NLL of the fixed ``(.50,.25,.25)`` dual-expert mixture."""

    pixel_nll, supervised = dual_fusion_pixel_nll(logits, target, q75, q512)
    return _masked_mean_or_connected_zero(pixel_nll, supervised, logits)


def inverse_sqrt_median_class_weights(
    class_counts: torch.Tensor | Sequence[int] | Sequence[float],
    *,
    minimum: float = 0.5,
    maximum: float = 3.0,
) -> torch.Tensor:
    """Return median-centred inverse-square-root class weights.

    For each represented class, ``w_c=sqrt(median(n_positive)/n_c)`` and the
    result is clipped to ``[0.5,3.0]`` by default.  An absent class receives
    the maximum weight; it contributes nothing until indexed by a future
    sample.  The returned tensor is FP32 and stays on the input tensor's
    device when a tensor is supplied.
    """

    counts = torch.as_tensor(class_counts).detach().float()
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("class_counts must be a non-empty one-dimensional vector")
    if not torch.isfinite(counts).all() or bool((counts < 0).any()):
        raise ValueError("class_counts must be finite and non-negative")
    minimum = float(minimum)
    maximum = float(maximum)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0.0
        or maximum < minimum
    ):
        raise ValueError("class-weight clipping bounds are invalid")
    positive = counts[counts > 0]
    if positive.numel() == 0:
        raise ValueError("class_counts must contain at least one positive count")
    median = positive.median()
    safe_counts = counts.clamp_min(1.0)
    weights = torch.sqrt(median / safe_counts).clamp(minimum, maximum)
    return torch.where(
        counts > 0,
        weights,
        torch.full_like(weights, maximum),
    ).to(dtype=torch.float32)


def normalized_inverse_sqrt_median_class_weights(
    class_counts: torch.Tensor | Sequence[int] | Sequence[float],
    *,
    minimum: float = 0.5,
    maximum: float = 3.0,
) -> torch.Tensor:
    """Return frozen histogram weights normalized to empirical ``E[w]=1``.

    Clipping is applied first by :func:`inverse_sqrt_median_class_weights`.
    The expectation is then taken under the supplied pixel histogram, rather
    than over classes uniformly.  This keeps the class-balanced objective's
    scale stable while its own weighted-mean reduction remains exact.
    """

    counts = torch.as_tensor(class_counts).detach().float()
    weights = inverse_sqrt_median_class_weights(
        counts, minimum=minimum, maximum=maximum
    )
    total = counts.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        raise ValueError("class_counts must have positive finite total mass")
    expectation = (weights * counts).sum() / total
    if not bool(torch.isfinite(expectation)) or float(expectation) <= 0.0:
        raise ValueError("class-weight expectation must be finite and positive")
    return (weights / expectation).to(dtype=torch.float32)


def _validated_class_weights(
    class_weights: torch.Tensor | Sequence[float],
    *,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.as_tensor(
        class_weights, dtype=torch.float32, device=device
    ).detach()
    if weights.ndim != 1 or int(weights.numel()) != int(num_classes):
        raise ValueError("class_weights must have one value per logit class")
    if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("class_weights must be finite and positive")
    return weights


def _pixel_class_weights(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    weights = _validated_class_weights(
        class_weights,
        num_classes=int(logits.shape[1]),
        device=logits.device,
    )
    safe_target = target.clamp(min=0, max=int(logits.shape[1]) - 1)
    return weights[safe_target]


def class_balanced_dual_fusion_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    q75: torch.Tensor,
    q512: torch.Tensor,
    class_weights: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Class-weighted fixed-mixture NLL, normalized by selected weight mass."""

    pixel_nll, supervised = dual_fusion_pixel_nll(logits, target, q75, q512)
    pixel_weights = _pixel_class_weights(
        logits, target, class_weights
    ) * supervised.to(dtype=torch.float32)
    return _weighted_mean_or_connected_zero(pixel_nll, pixel_weights, logits)


def one_sided_huber(shortfall: torch.Tensor, *, beta: float = HUBER_BETA) -> torch.Tensor:
    """Elementwise Huber penalty for positive shortfall only."""

    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("Huber beta must be finite and positive")
    positive = shortfall.float().clamp_min(0.0)
    return torch.where(
        positive < beta,
        0.5 * positive.square() / beta,
        positive - 0.5 * beta,
    )


def repair_weighted_truth_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    q75: torch.Tensor,
    q512: torch.Tensor,
    class_weights: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Weighted truth-class NLL on pixels where the dual teacher is weak.

    ``qB=(2*q75+q512)/3`` and
    ``r=relu((0.5-qB)/0.5)^2``.  The exact reduction is
    ``sum(w_y*r*(-log(clamp(p_y,1e-6)))) / sum(w_y*r)``.  If no pixel needs
    repair, a graph-connected exact zero is returned.
    """

    supervised, _, p_y, teachers = _truth_terms(logits, target, q75, q512)
    q75_safe, q512_safe = teachers[0], teachers[1]
    q_base = (2.0 * q75_safe + q512_safe) / 3.0
    repair_gate = (
        ((REPAIR_BASE_THRESHOLD - q_base) / REPAIR_BASE_THRESHOLD)
        .clamp_min(0.0)
        .square()
        * supervised.to(dtype=torch.float32)
    )
    pixel_class_weight = _pixel_class_weights(
        logits, target, class_weights
    )
    repair_weight = pixel_class_weight * repair_gate
    penalty = -p_y.clamp_min(1.0e-6).log()
    return _weighted_mean_or_connected_zero(penalty, repair_weight, logits)


def preserve_noharm_one_sided_huber(
    logits: torch.Tensor,
    target: torch.Tensor,
    q75: torch.Tensor,
    q512: torch.Tensor,
    class_weights: torch.Tensor | Sequence[float],
    *,
    margin: float = ONE_SIDED_MARGIN,
    beta: float = HUBER_BETA,
) -> torch.Tensor:
    """Prevent the student from strongly contradicting secure teachers.

    A pixel is secure exactly when ``qB>=.75``, ``q75>=.70`` and
    ``q512>=.55``.  On that mask the one-sided target is
    ``p_student(y) >= qB - margin``.
    """

    margin = float(margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("one-sided margin must be finite and non-negative")
    supervised, _, p_y, teachers = _truth_terms(logits, target, q75, q512)
    q75_safe, q512_safe = teachers[0], teachers[1]
    q_base = (2.0 * q75_safe + q512_safe) / 3.0
    secure = (
        supervised
        & (q_base >= SECURE_BASE_THRESHOLD)
        & (q75_safe >= SECURE_MS75_THRESHOLD)
        & (q512_safe >= SECURE_LOCAL512_THRESHOLD)
    )
    penalty = one_sided_huber(q_base - margin - p_y, beta=beta)
    preserve_weight = _pixel_class_weights(
        logits, target, class_weights
    ) * secure.to(dtype=torch.float32)
    return _weighted_mean_or_connected_zero(
        penalty, preserve_weight, logits
    )


def fusionaware_a1_additional_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    q75: torch.Tensor,
    q512: torch.Tensor,
    class_weights: torch.Tensor | Sequence[float],
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Return the complete A1 *additional* objective and raw components.

    The fixed combination is

    ``1.00*dual + .20*repair + .10*class_balanced_dual + .25*preserve``.

    The caller remains responsible for the ordinary segmentation/base loss.
    """

    dual = dual_fusion_nll(logits, target, q75, q512)
    repair = repair_weighted_truth_nll(
        logits, target, q75, q512, class_weights
    )
    class_balanced = class_balanced_dual_fusion_nll(
        logits, target, q75, q512, class_weights
    )
    preserve = preserve_noharm_one_sided_huber(
        logits, target, q75, q512, class_weights
    )
    total = (
        DUAL_FUSION_LOSS_WEIGHT * dual
        + REPAIR_LOSS_WEIGHT * repair
        + CLASS_BALANCED_FUSION_LOSS_WEIGHT * class_balanced
        + PRESERVE_LOSS_WEIGHT * preserve
    )
    return total, {
        "dual_fusion_nll": dual,
        "repair_weighted_truth_nll": repair,
        "class_balanced_dual_fusion_nll": class_balanced,
        "preserve_noharm_one_sided_huber": preserve,
        "total_additional": total,
    }


__all__ = [
    "CLASS_BALANCED_FUSION_LOSS_WEIGHT",
    "DUAL_FUSION_LOSS_WEIGHT",
    "HUBER_BETA",
    "LOCAL512_FUSION_WEIGHT",
    "MS75_FUSION_WEIGHT",
    "ONE_SIDED_MARGIN",
    "PRESERVE_LOSS_WEIGHT",
    "REPAIR_LOSS_WEIGHT",
    "STUDENT_FUSION_WEIGHT",
    "class_balanced_dual_fusion_nll",
    "dual_fusion_nll",
    "dual_fusion_pixel_nll",
    "fusionaware_a1_additional_loss",
    "inverse_sqrt_median_class_weights",
    "normalized_inverse_sqrt_median_class_weights",
    "one_sided_huber",
    "preserve_noharm_one_sided_huber",
    "repair_weighted_truth_nll",
]
