from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        self.num_classes = int(num_classes)
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        target = np.asarray(target)
        prediction = np.asarray(prediction)
        if target.shape != prediction.shape:
            raise ValueError(
                f"Target/prediction shape mismatch: {target.shape}, {prediction.shape}."
            )
        valid = (
            (target >= 0)
            & (target < self.num_classes)
            & (prediction >= 0)
            & (prediction < self.num_classes)
        )
        if not np.all(valid):
            raise ValueError("Target or prediction contains an invalid class index.")
        encoded = self.num_classes * target.reshape(-1).astype(
            np.int64
        ) + prediction.reshape(-1).astype(np.int64)
        self.matrix += np.bincount(encoded, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def distributed_reduce(self, device: torch.device) -> None:
        if not dist.is_initialized():
            return
        tensor = torch.from_numpy(self.matrix).to(device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self.matrix = tensor.cpu().numpy()

    def summarize(self, class_names: Sequence[str]) -> dict[str, Any]:
        if len(class_names) != self.num_classes:
            raise ValueError("class_names length does not match the confusion matrix.")
        matrix = self.matrix.astype(np.float64)
        true_positive = np.diag(matrix)
        target_count = matrix.sum(axis=1)
        prediction_count = matrix.sum(axis=0)
        union = target_count + prediction_count - true_positive

        def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
            return np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=np.float64),
                where=denominator > 0,
            )

        precision = safe_divide(true_positive, prediction_count)
        recall = safe_divide(true_positive, target_count)
        f1 = safe_divide(2.0 * true_positive, target_count + prediction_count)
        iou = safe_divide(true_positive, union)
        foreground = slice(1, self.num_classes)
        per_class = {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "iou": float(iou[index]),
                "target_pixels": int(target_count[index]),
                "predicted_pixels": int(prediction_count[index]),
            }
            for index, name in enumerate(class_names)
        }
        total = float(matrix.sum())
        return {
            "foreground_mean_f1": float(f1[foreground].mean()),
            "foreground_mean_iou": float(iou[foreground].mean()),
            "mean_f1_including_background": float(f1.mean()),
            "mean_iou_including_background": float(iou.mean()),
            "pixel_accuracy": float(true_positive.sum() / total) if total else 0.0,
            "pixels": int(total),
            "per_class": per_class,
            "confusion_matrix": self.matrix.tolist(),
        }
