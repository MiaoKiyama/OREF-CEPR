from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ms75_protocol import MS75_OFFICIAL_TEST_SCENE_IDS


NUM_CLASSES = 24


@dataclass(frozen=True)
class PairedDecisionThresholds:
    min_a1_vs_a0_miou_gain: float = 0.003
    min_a1_vs_base_miou_gain: float = 0.002
    min_hard_macro_iou_gain: float = 0.010
    min_non_hard_macro_iou_gain: float = -0.003
    min_any_class_iou_gain: float = -0.010
    min_positive_scenes: int = 18
    max_single_class_positive_share: float = 0.50
    max_single_scene_positive_share: float = 0.50


def _confusion_matrix(value: object, *, name: str) -> np.ndarray:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows: list[list[int]] = []
        for row in value:
            if isinstance(row, str):
                fields = row.split()
            elif isinstance(row, Sequence):
                fields = list(row)
            else:
                raise ValueError(f"{name} contains a non-row value")
            rows.append([int(item) for item in fields])
        result = np.asarray(rows, dtype=np.int64)
    else:
        raise ValueError(f"{name} must be a sequence of rows")
    if result.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(
            f"{name} must be shaped {(NUM_CLASSES, NUM_CLASSES)}, got "
            f"{result.shape}"
        )
    if np.any(result < 0):
        raise ValueError(f"{name} must contain non-negative counts")
    return result


def _metrics(confusion: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError("confusion matrix must be 24x24")
    true_positive = np.diag(matrix).astype(np.float64)
    target = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    union = target + predicted - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=union > 0,
    )
    present = target > 0
    return {
        "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
        "per_class_iou": iou,
        "present_classes": present,
        "valid_pixels": int(matrix.sum()),
    }


def _close(left: object, right: float, *, tolerance: float = 1e-10) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _validated_evaluation(
    payload: Mapping[str, Any],
    *,
    name: str,
    expected_transforms: Sequence[str],
) -> dict[str, Any]:
    if payload.get("split") != "test":
        raise ValueError(f"{name} must evaluate the official test split")
    if payload.get("evaluation_scope") != "every_pixel_of_every_scene":
        raise ValueError(f"{name} must cover every pixel of every scene")
    if list(payload.get("tta_transforms", [])) != list(expected_transforms):
        raise ValueError(f"{name} inference transforms do not match protocol")
    if int(payload.get("scene_count", -1)) != 30:
        raise ValueError(f"{name} must contain exactly 30 scenes")
    if int(payload.get("tile_stride", -1)) != 256:
        raise ValueError(f"{name} must use tile_stride=256")
    if payload.get("blend_window") != "hann":
        raise ValueError(f"{name} must use Hann overlap blending")
    if not _close(payload.get("hann_floor"), 0.05):
        raise ValueError(f"{name} must use hann_floor=0.05")
    if payload.get("max_windows_per_scene") is not None:
        raise ValueError(f"{name} cannot limit windows per scene")
    if payload.get("amp") is not True:
        raise ValueError(f"{name} must use registered AMP inference")
    if payload.get("amp_effective") is not True:
        raise ValueError(f"{name} must actually execute AMP inference")
    if payload.get("device_type") != "cuda":
        raise ValueError(f"{name} must execute on CUDA")
    if payload.get("cudnn_benchmark_effective") is not True:
        raise ValueError(
            f"{name} must reproduce the historical cuDNN benchmark setting"
        )

    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError(f"{name}.aggregate must be a mapping")
    confusion = _confusion_matrix(
        aggregate.get("confusion_matrix"), name=f"{name}.aggregate.confusion"
    )
    recomputed = _metrics(confusion)
    if not bool(np.asarray(recomputed["present_classes"], dtype=bool).all()):
        raise ValueError(
            f"{name} aggregate is not a strict-24 evaluation: at least one "
            "target class is absent"
        )
    if not _close(aggregate.get("mean_iou"), recomputed["mean_iou"]):
        raise ValueError(f"{name} aggregate mean_iou is inconsistent")
    observed_iou = np.asarray(aggregate.get("per_class_iou"), dtype=np.float64)
    if observed_iou.shape != (NUM_CLASSES,) or not np.allclose(
        observed_iou,
        recomputed["per_class_iou"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(f"{name} aggregate per_class_iou is inconsistent")

    scenes = payload.get("scenes")
    if not isinstance(scenes, Mapping) or len(scenes) != 30:
        raise ValueError(f"{name}.scenes must contain exactly 30 records")
    if set(str(value) for value in scenes) != set(MS75_OFFICIAL_TEST_SCENE_IDS):
        raise ValueError(f"{name} does not contain the exact official test scenes")
    scene_confusions: dict[str, np.ndarray] = {}
    scene_sum = np.zeros_like(confusion)
    for scene_id in sorted(str(key) for key in scenes):
        record = scenes[scene_id]
        if not isinstance(record, Mapping):
            raise ValueError(f"{name}.scenes[{scene_id}] must be a mapping")
        scene_confusion = _confusion_matrix(
            record.get("confusion_matrix"),
            name=f"{name}.scenes[{scene_id}].confusion",
        )
        scene_metrics = _metrics(scene_confusion)
        if not _close(record.get("mean_iou"), scene_metrics["mean_iou"]):
            raise ValueError(f"{name} scene {scene_id} mean_iou is inconsistent")
        scene_confusions[scene_id] = scene_confusion
        scene_sum += scene_confusion
    if not np.array_equal(scene_sum, confusion):
        raise ValueError(f"{name} scene confusions do not sum to aggregate")
    return {
        "confusion": confusion,
        "metrics": recomputed,
        "scene_confusions": scene_confusions,
    }


def _positive_share(values: np.ndarray) -> float:
    positive = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = float(positive.sum())
    return float(positive.max() / total) if total > 0.0 else 0.0


def _net_gain_share(values: np.ndarray, *, net_gain: float) -> float:
    if not np.isfinite(net_gain) or net_gain <= 0.0:
        return float("inf")
    positive = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    return float(positive.max() / net_gain) if positive.size else 0.0


def decide_paired_continuation(
    *,
    baseline: Mapping[str, Any],
    arm_a0: Mapping[str, Any],
    arm_a1: Mapping[str, Any],
    hard_class_ids: Sequence[int],
    expected_transforms: Sequence[str] = ("none",),
    thresholds: PairedDecisionThresholds = PairedDecisionThresholds(),
) -> dict[str, Any]:
    hard = tuple(sorted({int(value) for value in hard_class_ids}))
    if len(hard) != 6 or any(value < 0 or value >= NUM_CLASSES for value in hard):
        raise ValueError("hard_class_ids must contain six unique 0-based classes")
    non_hard = tuple(value for value in range(NUM_CLASSES) if value not in hard)

    validated_base = _validated_evaluation(
        baseline, name="baseline", expected_transforms=expected_transforms
    )
    validated_a0 = _validated_evaluation(
        arm_a0, name="arm_a0", expected_transforms=expected_transforms
    )
    validated_a1 = _validated_evaluation(
        arm_a1, name="arm_a1", expected_transforms=expected_transforms
    )
    base_scenes = set(validated_base["scene_confusions"])
    if set(validated_a0["scene_confusions"]) != base_scenes or set(
        validated_a1["scene_confusions"]
    ) != base_scenes:
        raise ValueError("baseline, A0, and A1 scene inventories must match")
    valid_pixels = {
        validated_base["metrics"]["valid_pixels"],
        validated_a0["metrics"]["valid_pixels"],
        validated_a1["metrics"]["valid_pixels"],
    }
    if len(valid_pixels) != 1:
        raise ValueError("baseline, A0, and A1 valid-pixel counts must match")
    for scene_id in sorted(base_scenes):
        base_target = validated_base["scene_confusions"][scene_id].sum(axis=1)
        a0_target = validated_a0["scene_confusions"][scene_id].sum(axis=1)
        a1_target = validated_a1["scene_confusions"][scene_id].sum(axis=1)
        if not np.array_equal(base_target, a0_target) or not np.array_equal(
            base_target, a1_target
        ):
            raise ValueError(
                "baseline, A0, and A1 per-scene target histograms must match"
            )

    base_miou = float(validated_base["metrics"]["mean_iou"])
    a0_miou = float(validated_a0["metrics"]["mean_iou"])
    a1_miou = float(validated_a1["metrics"]["mean_iou"])
    a1_vs_a0 = a1_miou - a0_miou
    a1_vs_base = a1_miou - base_miou
    class_delta = (
        validated_a1["metrics"]["per_class_iou"]
        - validated_a0["metrics"]["per_class_iou"]
    )
    hard_delta = float(class_delta[list(hard)].mean())
    non_hard_delta = float(class_delta[list(non_hard)].mean())

    scene_delta: dict[str, float] = {}
    scene_influence: dict[str, float] = {}
    aggregate_a0 = validated_a0["confusion"]
    aggregate_a1 = validated_a1["confusion"]
    for scene_id in sorted(base_scenes):
        scene_a0 = validated_a0["scene_confusions"][scene_id]
        scene_a1 = validated_a1["scene_confusions"][scene_id]
        scene_delta[scene_id] = float(
            _metrics(scene_a1)["mean_iou"] - _metrics(scene_a0)["mean_iou"]
        )
        leave_one_out_delta = float(
            _metrics(aggregate_a1 - scene_a1)["mean_iou"]
            - _metrics(aggregate_a0 - scene_a0)["mean_iou"]
        )
        scene_influence[scene_id] = a1_vs_a0 - leave_one_out_delta

    positive_scene_count = sum(value > 0.0 for value in scene_delta.values())
    class_positive_mass_share = _positive_share(class_delta / NUM_CLASSES)
    scene_positive_mass_share = _positive_share(
        np.asarray(list(scene_influence.values()), dtype=np.float64)
    )
    class_positive_share = _net_gain_share(
        class_delta / NUM_CLASSES, net_gain=a1_vs_a0
    )
    scene_positive_share = _net_gain_share(
        np.asarray(list(scene_influence.values()), dtype=np.float64),
        net_gain=a1_vs_a0,
    )

    checks = {
        "a1_vs_a0_miou_gain": a1_vs_a0
        >= thresholds.min_a1_vs_a0_miou_gain,
        "a1_vs_original_miou_gain": a1_vs_base
        >= thresholds.min_a1_vs_base_miou_gain,
        "hard_macro_iou_gain": hard_delta
        >= thresholds.min_hard_macro_iou_gain,
        "non_hard_macro_iou_guard": non_hard_delta
        >= thresholds.min_non_hard_macro_iou_gain,
        "all_class_iou_guard": float(class_delta.min())
        >= thresholds.min_any_class_iou_gain,
        "positive_scene_count": positive_scene_count
        >= thresholds.min_positive_scenes,
        "single_class_contribution_guard": class_positive_share
        <= thresholds.max_single_class_positive_share,
        "single_scene_contribution_guard": scene_positive_share
        <= thresholds.max_single_scene_positive_share,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "status": "eligible_for_flip4" if passed else "stopped_after_no_tta",
        "passed": passed,
        "historical_benchmark_exploratory": True,
        "official_test_opened": True,
        "expected_transforms": list(expected_transforms),
        "valid_pixels": next(iter(valid_pixels)),
        "hard_class_ids": list(hard),
        "non_hard_class_ids": list(non_hard),
        "metrics": {
            "baseline_miou": base_miou,
            "arm_a0_miou": a0_miou,
            "arm_a1_miou": a1_miou,
            "a1_vs_a0_miou_gain": a1_vs_a0,
            "a1_vs_original_miou_gain": a1_vs_base,
            "hard_macro_iou_gain": hard_delta,
            "non_hard_macro_iou_gain": non_hard_delta,
            "minimum_class_iou_gain": float(class_delta.min()),
            "positive_scene_count": positive_scene_count,
            "single_class_positive_contribution_share": class_positive_share,
            "single_scene_positive_contribution_share": scene_positive_share,
            "single_class_positive_mass_share": class_positive_mass_share,
            "single_scene_positive_mass_share": scene_positive_mass_share,
            "per_class_iou_gain": class_delta.tolist(),
            "per_scene_miou_gain": scene_delta,
            "per_scene_leave_one_out_influence": scene_influence,
        },
        "thresholds": {
            "min_a1_vs_a0_miou_gain": thresholds.min_a1_vs_a0_miou_gain,
            "min_a1_vs_base_miou_gain": thresholds.min_a1_vs_base_miou_gain,
            "min_hard_macro_iou_gain": thresholds.min_hard_macro_iou_gain,
            "min_non_hard_macro_iou_gain": (
                thresholds.min_non_hard_macro_iou_gain
            ),
            "min_any_class_iou_gain": thresholds.min_any_class_iou_gain,
            "min_positive_scenes": thresholds.min_positive_scenes,
            "max_single_class_positive_share": (
                thresholds.max_single_class_positive_share
            ),
            "max_single_scene_positive_share": (
                thresholds.max_single_scene_positive_share
            ),
        },
        "checks": checks,
    }


__all__ = [
    "NUM_CLASSES",
    "PairedDecisionThresholds",
    "decide_paired_continuation",
]
