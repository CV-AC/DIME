import math

import torch

from facebench.tasks.head_pose.rotation import (
    GeodesicLoss,
    euler_to_matrix,
    matrix_to_euler,
    rotation_6d_to_matrix,
    wrap_aware_error_deg,
)


def test_euler_matrix_round_trip():
    pitch = torch.deg2rad(torch.tensor([0.0, 10.0, -30.0]))
    yaw = torch.deg2rad(torch.tensor([0.0, -20.0, 45.0]))
    roll = torch.deg2rad(torch.tensor([0.0, 30.0, -15.0]))
    matrix = euler_to_matrix(pitch, yaw, roll)
    recovered = matrix_to_euler(matrix)
    expected = torch.stack((pitch, yaw, roll), dim=1)
    torch.testing.assert_close(recovered, expected, atol=1e-6, rtol=1e-6)


def test_rotation_6d_is_so3():
    torch.manual_seed(7)
    matrix = rotation_6d_to_matrix(torch.randn(32, 6))
    identity = torch.eye(3).expand(32, -1, -1)
    torch.testing.assert_close(
        matrix @ matrix.transpose(1, 2), identity, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        torch.linalg.det(matrix), torch.ones(32), atol=1e-5, rtol=1e-5
    )


def test_geodesic_loss_known_rotation():
    identity = torch.eye(3).unsqueeze(0)
    rotated = euler_to_matrix(
        torch.tensor([math.pi / 6]), torch.tensor([0.0]), torch.tensor([0.0])
    )
    value = GeodesicLoss()(rotated, identity)
    torch.testing.assert_close(value, torch.tensor(math.pi / 6), atol=1e-6, rtol=1e-6)


def test_official_wrap_aware_error():
    prediction = torch.tensor([[179.0, -179.0, 10.0]])
    target = torch.tensor([[-179.0, 179.0, 10.0]])
    error = wrap_aware_error_deg(prediction, target)
    torch.testing.assert_close(error, torch.tensor([[2.0, 2.0, 0.0]]))
