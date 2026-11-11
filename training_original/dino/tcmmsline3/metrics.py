from __future__ import annotations

import numpy as np
import torch


def _class_boundary(
    mask: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Mark the class side of four-connected transitions inside valid data."""
    boundary = np.zeros_like(mask, dtype=bool)
    horizontal = (
        valid[:, 1:]
        & valid[:, :-1]
        & (mask[:, 1:] != mask[:, :-1])
    )
    vertical = (
        valid[1:, :]
        & valid[:-1, :]
        & (mask[1:, :] != mask[:-1, :])
    )
    boundary[:, 1:] |= horizontal & mask[:, 1:]
    boundary[:, :-1] |= horizontal & mask[:, :-1]
    boundary[1:, :] |= vertical & mask[1:, :]
    boundary[:-1, :] |= vertical & mask[:-1, :]
    return boundary


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    dilated = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(radius)):
        source = dilated
        expanded = source.copy()
        expanded[1:, :] |= source[:-1, :]
        expanded[:-1, :] |= source[1:, :]
        expanded[:, 1:] |= source[:, :-1]
        expanded[:, :-1] |= source[:, 1:]
        expanded[1:, 1:] |= source[:-1, :-1]
        expanded[1:, :-1] |= source[:-1, 1:]
        expanded[:-1, 1:] |= source[1:, :-1]
        expanded[:-1, :-1] |= source[1:, 1:]
        dilated = expanded
    return dilated


class BoundaryF1Metrics:
    """Class-aware semantic Boundary F1 with a pixel tolerance."""

    def __init__(self, num_classes: int, tolerance: int = 3):
        self.num_classes = int(num_classes)
        self.tolerance = int(tolerance)
        if self.tolerance < 0:
            raise ValueError("boundary tolerance must be non-negative")
        self.matched_prediction = np.zeros(self.num_classes, dtype=np.int64)
        self.prediction_boundary = np.zeros(self.num_classes, dtype=np.int64)
        self.matched_target = np.zeros(self.num_classes, dtype=np.int64)
        self.target_boundary = np.zeros(self.num_classes, dtype=np.int64)

    def update(
        self,
        prediction: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray,
    ) -> None:
        prediction_np = (
            prediction.detach().cpu().numpy()
            if isinstance(prediction, torch.Tensor)
            else np.asarray(prediction)
        )
        target_np = (
            target.detach().cpu().numpy()
            if isinstance(target, torch.Tensor)
            else np.asarray(target)
        )
        if prediction_np.ndim == 2:
            prediction_np = prediction_np[None]
            target_np = target_np[None]
        if prediction_np.shape != target_np.shape or prediction_np.ndim != 3:
            raise ValueError(
                "BoundaryF1Metrics expects matching [H,W] or [N,H,W] arrays"
            )
        for prediction_item, target_item in zip(prediction_np, target_np):
            valid = (target_item >= 0) & (target_item < self.num_classes)
            present = np.unique(
                np.concatenate(
                    [
                        target_item[valid].astype(np.int64, copy=False),
                        prediction_item[valid].astype(np.int64, copy=False),
                    ]
                )
            )
            for class_id in present:
                if class_id < 0 or class_id >= self.num_classes:
                    continue
                target_mask = (target_item == class_id) & valid
                prediction_mask = (prediction_item == class_id) & valid
                target_boundary = _class_boundary(target_mask, valid)
                prediction_boundary = _class_boundary(prediction_mask, valid)
                target_count = int(target_boundary.sum())
                prediction_count = int(prediction_boundary.sum())
                self.target_boundary[class_id] += target_count
                self.prediction_boundary[class_id] += prediction_count
                if prediction_count:
                    self.matched_prediction[class_id] += int(
                        (
                            prediction_boundary
                            & _dilate(target_boundary, self.tolerance)
                        ).sum()
                    )
                if target_count:
                    self.matched_target[class_id] += int(
                        (
                            target_boundary
                            & _dilate(prediction_boundary, self.tolerance)
                        ).sum()
                    )

    def add_counts(self, result: dict) -> None:
        if int(result["tolerance_pixels"]) != self.tolerance:
            raise ValueError("cannot merge Boundary F1 with another tolerance")
        for name in (
            "matched_prediction",
            "prediction_boundary",
            "matched_target",
            "target_boundary",
        ):
            values = np.asarray(result[name], dtype=np.int64)
            if values.shape != (self.num_classes,):
                raise ValueError("Boundary F1 class count mismatch")
            getattr(self, name)[:] += values

    def compute(self) -> dict:
        precision = np.divide(
            self.matched_prediction,
            self.prediction_boundary,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=self.prediction_boundary > 0,
        )
        recall = np.divide(
            self.matched_target,
            self.target_boundary,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=self.target_boundary > 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=(precision + recall) > 0,
        )
        present = self.target_boundary > 0
        micro_precision = float(
            self.matched_prediction.sum()
            / max(int(self.prediction_boundary.sum()), 1)
        )
        micro_recall = float(
            self.matched_target.sum()
            / max(int(self.target_boundary.sum()), 1)
        )
        micro_f1 = (
            2.0 * micro_precision * micro_recall
            / (micro_precision + micro_recall)
            if micro_precision + micro_recall > 0
            else 0.0
        )
        return {
            "tolerance_pixels": self.tolerance,
            "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            "micro_f1": float(micro_f1),
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "per_class_f1": f1.tolist(),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "present_classes": present.tolist(),
            "matched_prediction": self.matched_prediction.tolist(),
            "prediction_boundary": self.prediction_boundary.tolist(),
            "matched_target": self.matched_target.tolist(),
            "target_boundary": self.target_boundary.tolist(),
        }


class ConfusionMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.confusion = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(
        self, prediction: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray
    ) -> None:
        prediction_np = (
            prediction.detach().cpu().numpy()
            if isinstance(prediction, torch.Tensor)
            else np.asarray(prediction)
        )
        target_np = (
            target.detach().cpu().numpy()
            if isinstance(target, torch.Tensor)
            else np.asarray(target)
        )
        valid = (target_np >= 0) & (target_np < self.num_classes)
        encoded = (
            target_np[valid].astype(np.int64) * self.num_classes
            + prediction_np[valid].astype(np.int64)
        )
        self.confusion += np.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        confusion = self.confusion.astype(np.float64)
        true_positive = np.diag(confusion)
        target_count = confusion.sum(axis=1)
        prediction_count = confusion.sum(axis=0)
        union = target_count + prediction_count - true_positive
        present = target_count > 0
        iou = np.divide(
            true_positive,
            union,
            out=np.zeros_like(true_positive),
            where=union > 0,
        )
        precision = np.divide(
            true_positive,
            prediction_count,
            out=np.zeros_like(true_positive),
            where=prediction_count > 0,
        )
        recall = np.divide(
            true_positive,
            target_count,
            out=np.zeros_like(true_positive),
            where=target_count > 0,
        )
        overall = true_positive.sum() / max(confusion.sum(), 1.0)
        return {
            "overall_accuracy": float(overall),
            "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
            "per_class_iou": iou.tolist(),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "target_pixels": target_count.astype(np.int64).tolist(),
            "predicted_pixels": prediction_count.astype(np.int64).tolist(),
            "present_classes": present.tolist(),
            "confusion_matrix": self.confusion.tolist(),
        }
