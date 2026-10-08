from __future__ import annotations


import math
from functools import lru_cache

import cv2
import numpy as np
from skimage.transform import SimilarityTransform


def standard_face_points(size: int = 512) -> np.ndarray:
    normalized = (
        np.asarray(
            [
                [196.0, 226.0],
                [316.0, 226.0],
                [256.0, 286.0],
                [220.0, 360.4],
                [292.0, 360.4],
            ],
            dtype=np.float32,
        )
        / 256.0
        - 1.0
    )
    return (normalized + 1.0) * np.asarray([size - 1, size - 1], dtype=np.float32) / 2.0


def face_align_matrix(points: np.ndarray, size: int = 512) -> np.ndarray:
    points = np.asarray(points)
    if points.shape != (5, 2):
        raise ValueError(f"Face alignment needs five 2D points, got {points.shape}.")
    target = standard_face_points(size)
    if hasattr(SimilarityTransform, "from_estimate"):
        transform = SimilarityTransform.from_estimate(points, target)
        if not isinstance(transform, SimilarityTransform):
            raise RuntimeError(
                "Could not estimate the five-point face alignment matrix."
            )
    else:
        transform = SimilarityTransform()
        if not transform.estimate(points, target):
            raise RuntimeError(
                "Could not estimate the five-point face alignment matrix."
            )
    return transform.params


def compose_rotate_and_scale(
    angle: float,
    scale: float,
    shift_xy: np.ndarray | tuple[float, float],
    from_center: tuple[float, float],
    to_center: tuple[float, float],
) -> np.ndarray:
    cos_value, sin_value = math.cos(angle), math.sin(angle)
    from_x, from_y = from_center
    to_x, to_y = to_center
    angle_cos, angle_sin = scale * cos_value, scale * sin_value
    return np.asarray(
        [
            [
                angle_cos,
                -angle_sin,
                to_x - angle_cos * from_x + angle_sin * from_y + shift_xy[0],
            ],
            [
                angle_sin,
                angle_cos,
                to_y - angle_sin * from_x - angle_cos * from_y + shift_xy[1],
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def random_update_matrix(
    matrix: np.ndarray,
    *,
    size: int,
    shift_sigma: float,
    rotation_sigma: float,
    scale_sigma: float,
) -> np.ndarray:

    shift = np.random.uniform(-shift_sigma, shift_sigma, size=2) * size
    angle = float(np.random.normal(0.0, rotation_sigma))
    scale = float(np.random.normal(1.0, scale_sigma))
    if scale <= 0.0:
        raise RuntimeError("Random scale was non-positive; check scale_sigma.")
    center = ((size - 1) / 2.0, (size - 1) / 2.0)
    update = compose_rotate_and_scale(angle, scale, shift, center, center)
    return update @ matrix


@lru_cache(maxsize=64)
def _meshgrid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    return yy, xx


def _safe_arctanh(values: np.ndarray) -> np.ndarray:
    values = values.copy()
    values[values < -0.999] = -0.999
    values[values > 0.999] = 0.999
    return np.arctanh(values)


def _tanh_warp_transform(
    coordinates: np.ndarray,
    matrix: np.ndarray,
    warp_factor: float,
    warped_shape: tuple[int, int],
) -> np.ndarray:
    height, width = warped_shape
    coordinates = coordinates.copy()
    if warp_factor > 0.0:
        coordinates = (
            coordinates / np.asarray([width, height], dtype=coordinates.dtype) * 2.0
            - 1.0
        )
        upper = coordinates > 1.0 - warp_factor
        lower = coordinates < -1.0 + warp_factor
        coordinates[upper] = (
            _safe_arctanh((coordinates[upper] - 1.0 + warp_factor) / warp_factor)
            * warp_factor
            + 1.0
            - warp_factor
        )
        coordinates[lower] = (
            _safe_arctanh((coordinates[lower] + 1.0 - warp_factor) / warp_factor)
            * warp_factor
            - 1.0
            + warp_factor
        )
        coordinates = (
            (coordinates + 1.0)
            / 2.0
            * np.asarray([width, height], dtype=coordinates.dtype)
        )
    homogeneous = np.concatenate(
        [
            coordinates,
            np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
        ],
        axis=1,
    )
    transformed = homogeneous @ np.linalg.inv(matrix).T
    return (transformed[:, :2] / transformed[:, [2, 2]]).astype(coordinates.dtype)


def _inverted_tanh_warp_transform(
    coordinates: np.ndarray,
    matrix: np.ndarray,
    warp_factor: float,
    warped_shape: tuple[int, int],
) -> np.ndarray:
    height, width = warped_shape
    homogeneous = np.concatenate(
        [
            coordinates,
            np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
        ],
        axis=1,
    )
    transformed = homogeneous @ matrix.T
    coordinates = (transformed[:, :2] / transformed[:, [2, 2]]).astype(
        coordinates.dtype
    )
    if warp_factor > 0.0:
        coordinates = (
            coordinates / np.asarray([width, height], dtype=coordinates.dtype) * 2.0
            - 1.0
        )
        upper = coordinates > 1.0 - warp_factor
        lower = coordinates < -1.0 + warp_factor
        coordinates[upper] = (
            np.tanh((coordinates[upper] - 1.0 + warp_factor) / warp_factor)
            * warp_factor
            + 1.0
            - warp_factor
        )
        coordinates[lower] = (
            np.tanh((coordinates[lower] + 1.0 - warp_factor) / warp_factor)
            * warp_factor
            - 1.0
            + warp_factor
        )
        coordinates = (
            (coordinates + 1.0)
            / 2.0
            * np.asarray([width, height], dtype=coordinates.dtype)
        )
    return coordinates


def _forge_map(
    output_shape: tuple[int, int],
    transform,
) -> np.ndarray:
    height, width = output_shape
    yy, xx = _meshgrid(height, width)
    coordinates = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    return transform(coordinates).reshape(height, width, 2).astype(np.float32)


def forward_transform_map(
    matrix: np.ndarray,
    *,
    canvas_size: int = 512,
    warp_factor: float = 0.0,
) -> np.ndarray:
    return _forge_map(
        (canvas_size, canvas_size),
        lambda coordinates: _tanh_warp_transform(
            coordinates,
            matrix,
            warp_factor,
            (canvas_size, canvas_size),
        ),
    )


def inverse_transform_map(
    matrix: np.ndarray,
    original_shape: tuple[int, int],
    *,
    canvas_size: int = 512,
    warp_factor: float = 0.0,
) -> np.ndarray:
    return _forge_map(
        original_shape,
        lambda coordinates: _inverted_tanh_warp_transform(
            coordinates,
            matrix,
            warp_factor,
            (canvas_size, canvas_size),
        ),
    )


def remap(
    array: np.ndarray,
    transform_map: np.ndarray,
    *,
    interpolation: int,
    border_value: float | int = 0,
) -> np.ndarray:
    return cv2.remap(
        array,
        transform_map[..., 0],
        transform_map[..., 1],
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
