from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import simpson

from .data import SUBSETS


def normalized_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    normalization: str = "interocular",
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError(
            f"Expected matching [N,K,2] arrays, got {predictions.shape} and {targets.shape}."
        )
    if normalization == "interocular":
        denominator = np.linalg.norm(targets[:, 60] - targets[:, 72], axis=-1)
    elif normalization == "interpupil":
        denominator = np.linalg.norm(targets[:, 96] - targets[:, 97], axis=-1)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    if np.any(denominator <= 0):
        raise ValueError("At least one sample has zero normalization distance.")
    point_errors = np.linalg.norm(predictions - targets, axis=-1)
    return point_errors.mean(axis=-1) / denominator


def auc_failure(
    errors: np.ndarray, threshold: float = 0.1, step: float = 0.0001
) -> tuple[float, float]:
    errors = np.sort(np.asarray(errors, dtype=np.float64))
    x_axis = np.arange(0.0, threshold + step, step)
    ced = np.searchsorted(errors, x_axis, side="right") / len(errors)
    auc = float(simpson(ced, x=x_axis) / threshold)
    failure = float(1.0 - ced[-1])
    return auc, failure


def summarize_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    subset_flags: np.ndarray,
) -> dict[str, Any]:
    errors = normalized_errors(predictions, targets, normalization="interocular")
    pupil_errors = normalized_errors(predictions, targets, normalization="interpupil")
    auc, failure = auc_failure(errors)
    result: dict[str, Any] = {
        "samples": int(len(errors)),
        "nme_interocular": float(errors.mean() * 100.0),
        "nme_interpupil_diagnostic": float(pupil_errors.mean() * 100.0),
        "fr_0.10": failure * 100.0,
        "auc_0.10": auc * 100.0,
    }
    subset_flags = np.asarray(subset_flags, dtype=bool)
    if subset_flags.shape != (len(errors), len(SUBSETS)):
        raise ValueError(f"Invalid subset flag shape: {subset_flags.shape}")
    result["subsets"] = {}
    for index, name in enumerate(SUBSETS):
        selected = errors[subset_flags[:, index]]
        result["subsets"][name] = {
            "samples": int(len(selected)),
            "nme_interocular": (
                float(selected.mean() * 100.0) if len(selected) else None
            ),
        }
    return result
