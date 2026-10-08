import numpy as np

from facebench.tasks.landmark.data import SUBSETS
from facebench.tasks.landmark.metrics import (
    auc_failure,
    normalized_errors,
    summarize_metrics,
)


def _targets(samples: int = 2) -> np.ndarray:
    targets = np.zeros((samples, 98, 2), dtype=np.float64)
    targets[:, 60, 0] = 0.0
    targets[:, 72, 0] = 10.0
    targets[:, 96, 0] = 0.0
    targets[:, 97, 0] = 5.0
    return targets


def test_normalized_error_manual_value() -> None:
    targets = _targets(1)
    predictions = targets.copy()
    predictions[..., 0] += 1.0
    error = normalized_errors(predictions, targets)
    np.testing.assert_allclose(error, [0.1])


def test_auc_and_failure_units() -> None:
    auc, failure = auc_failure(np.asarray([0.0, 0.2]), threshold=0.1)
    assert 0.49 < auc < 0.51
    assert failure == 0.5


def test_validation_without_official_subsets_has_null_subset_nme() -> None:
    targets = _targets()
    flags = np.zeros((2, len(SUBSETS)), dtype=bool)
    result = summarize_metrics(targets.copy(), targets, flags)
    assert result["nme_interocular"] == 0.0
    assert result["subsets"]["blur"] == {"samples": 0, "nme_interocular": None}
