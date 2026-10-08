from typing import Tuple, Union

import torch

from .ext.p2i_ops import p2i


def normalize_points(points: torch.Tensor, h: int, w: int) -> torch.Tensor:


    return (points + 0.5) / torch.tensor([[[w, h]]]).to(points)


def denormalize_points(normalized_points: torch.Tensor, h: int, w: int) -> torch.Tensor:


    return normalized_points * torch.tensor([[[w, h]]]).to(normalized_points) - 0.5


def points2heatmap(normalized_points, heatmap_size: Tuple[int, int], kernel_radius: float):


    batch, npoints, _ = normalized_points.shape
    out_h, out_w = heatmap_size

    points = denormalize_points(normalized_points, out_h, out_w)


    heatmap = torch.zeros(
        batch * npoints, 1, out_h, out_w).to(points)

    points_flatten = points.view(-1, 2)

    batch_inds = torch.arange(
        batch * npoints, dtype=torch.int32).cuda()

    points_color = torch.ones(
        points_flatten.size(0), 1).to(points_flatten)

    heatmap = p2i(points_flatten, points_color, batch_inds=batch_inds, background=heatmap,
                  kernel_radius=kernel_radius,
                  kernel_kind_str='gaussian_awing', reduce='max')

    heatmap = heatmap.reshape(batch, npoints, out_h, out_w)
    return heatmap


def heatmap2points(heatmap, t_scale: Union[None, float, torch.Tensor] = None):


    dtype = heatmap.dtype
    _, _, h, w = heatmap.shape


    yy, xx = torch.meshgrid(
        torch.arange(h).float(),
        torch.arange(w).float())

    yy = yy.view(1, 1, h, w).to(heatmap)
    xx = xx.view(1, 1, h, w).to(heatmap)

    if t_scale is not None:
        heatmap = (heatmap * t_scale).exp()
    heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)

    yy_coord = (yy * heatmap).sum([2, 3]) / heatmap_sum
    xx_coord = (xx * heatmap).sum([2, 3]) / heatmap_sum

    points = torch.stack([xx_coord, yy_coord], dim=-1)

    normalized_points = normalize_points(points, h, w)
    return normalized_points
