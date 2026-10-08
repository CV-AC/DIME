import numpy as np

from facebench.tasks.parsing.geometry import (
    face_align_matrix,
    forward_transform_map,
    inverse_transform_map,
    standard_face_points,
)


def test_standard_points_and_identity_alignment():
    points = standard_face_points(512)
    assert points.shape == (5, 2)
    matrix = face_align_matrix(points, 512)
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-5)


def test_identity_maps_have_pixel_coordinates():
    matrix = np.eye(3, dtype=np.float32)
    forward = forward_transform_map(matrix, canvas_size=16, warp_factor=0.0)
    inverse = inverse_transform_map(matrix, (16, 16), canvas_size=16, warp_factor=0.0)
    np.testing.assert_allclose(forward, inverse, atol=1e-6)
    assert tuple(forward[7, 11]) == (11.0, 7.0)


def test_tanh_forward_inverse_maps_are_finite():
    matrix = np.asarray(
        [[0.9, -0.1, 3.0], [0.1, 0.9, -2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    forward = forward_transform_map(matrix, canvas_size=32, warp_factor=0.8)
    inverse = inverse_transform_map(matrix, (24, 20), canvas_size=32, warp_factor=0.8)
    assert forward.shape == (32, 32, 2)
    assert inverse.shape == (24, 20, 2)
    assert np.isfinite(forward).all()
    assert np.isfinite(inverse).all()
