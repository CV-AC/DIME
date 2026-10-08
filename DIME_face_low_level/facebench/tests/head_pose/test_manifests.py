from pathlib import Path

import numpy as np
import scipy.io as sio

from facebench.tasks.head_pose.prepare_manifests import _mat_record


def _write_annotation(path: Path, points: np.ndarray) -> None:
    sio.savemat(
        path,
        {
            "Pose_Para": np.array([[0.1, -0.2, 0.3, 0, 0, 0, 1]]),
            "pt2d": points,
        },
    )


def test_original_aflw_visible_landmarks_are_supported(tmp_path: Path):
    image = tmp_path / "image00001.jpg"
    image.touch()
    points = np.stack(
        (np.linspace(-1, 100, 21), np.linspace(10, 200, 21)),
        axis=0,
    )
    _write_annotation(image.with_suffix(".mat"), points)

    record, reason = _mat_record(tmp_path, image, max_angle=99.0)

    assert reason is None
    assert record["landmark_bbox"] == [-1.0, 10.0, 100.0, 200.0]


def test_repackaged_68_point_annotations_are_supported(tmp_path: Path):
    image = tmp_path / "sample.jpg"
    image.touch()
    points = np.stack(
        (np.linspace(5, 105, 68), np.linspace(15, 215, 68)),
        axis=0,
    )
    _write_annotation(image.with_suffix(".mat"), points)

    record, reason = _mat_record(tmp_path, image, max_angle=99.0)

    assert reason is None
    assert record["landmark_bbox"] == [5.0, 15.0, 105.0, 215.0]
