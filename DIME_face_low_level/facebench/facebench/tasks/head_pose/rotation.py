from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_6d_to_matrix(poses: torch.Tensor) -> torch.Tensor:

    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"Expected [B,6], got {tuple(poses.shape)}")
    x = F.normalize(poses[:, 0:3], dim=1, eps=1e-8)
    z = F.normalize(torch.cross(x, poses[:, 3:6], dim=1), dim=1, eps=1e-8)
    y = torch.cross(z, x, dim=1)
    return torch.stack((x, y, z), dim=2)


def euler_to_matrix(
    pitch: torch.Tensor, yaw: torch.Tensor, roll: torch.Tensor
) -> torch.Tensor:

    pitch, yaw, roll = torch.broadcast_tensors(pitch, yaw, roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cr, sr = torch.cos(roll), torch.sin(roll)

    row0 = torch.stack((cr * cy, cr * sy * sp - sr * cp, cr * sy * cp + sr * sp), -1)
    row1 = torch.stack((sr * cy, sr * sy * sp + cr * cp, sr * sy * cp - cr * sp), -1)
    row2 = torch.stack((-sy, cy * sp, cy * cp), -1)
    return torch.stack((row0, row1, row2), -2)


def matrix_to_euler(rotation_matrices: torch.Tensor) -> torch.Tensor:

    if rotation_matrices.ndim != 3 or rotation_matrices.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [B,3,3], got {tuple(rotation_matrices.shape)}")
    r = rotation_matrices
    sy = torch.sqrt(r[:, 0, 0].square() + r[:, 1, 0].square())
    singular = sy < 1e-6
    pitch = torch.atan2(r[:, 2, 1], r[:, 2, 2])
    yaw = torch.atan2(-r[:, 2, 0], sy)
    roll = torch.atan2(r[:, 1, 0], r[:, 0, 0])
    pitch_singular = torch.atan2(-r[:, 1, 2], r[:, 1, 1])
    pitch = torch.where(singular, pitch_singular, pitch)
    roll = torch.where(singular, torch.zeros_like(roll), roll)
    return torch.stack((pitch, yaw, roll), dim=1)


class GeodesicLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        relative = torch.bmm(predicted, target.transpose(1, 2))
        cosine = (relative.diagonal(dim1=1, dim2=2).sum(1) - 1.0) / 2.0
        return torch.acos(cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)).mean()


def wrap_aware_error_deg(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    offsets = predicted.new_tensor([0.0, 360.0, -360.0, 180.0, -180.0])
    candidates = (predicted[..., None] + offsets - target[..., None]).abs()
    return candidates.min(dim=-1).values


def rotation_mae(
    predicted_rotation: torch.Tensor, target_ypr_radians: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    predicted_pyr = matrix_to_euler(predicted_rotation) * (180.0 / math.pi)
    predicted_ypr = predicted_pyr[:, [1, 0, 2]]
    target_ypr = target_ypr_radians * (180.0 / math.pi)
    errors = wrap_aware_error_deg(predicted_ypr, target_ypr)
    return predicted_ypr, target_ypr, errors
