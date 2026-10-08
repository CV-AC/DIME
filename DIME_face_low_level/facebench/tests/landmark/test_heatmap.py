import math

import torch

from facebench.tasks.landmark.model import (
    continuous_gaussian_heatmap,
    heatmap_to_points,
    points_to_heatmap,
    route_a_local_soft_argmax,
)


def test_heatmap_expectation_uses_pixel_centers() -> None:
    heatmap = torch.zeros(1, 1, 8, 8)
    heatmap[0, 0, 5, 3] = 1.0
    point = heatmap_to_points(heatmap)[0, 0]
    torch.testing.assert_close(point, torch.tensor([(3.0 + 0.5) / 8, (5.0 + 0.5) / 8]))


def test_gaussian_awing_formula_and_truncation() -> None:
    size, radius = 16, 5.0

    point = torch.tensor([[[7.5 / size, 8.5 / size]]])
    target = points_to_heatmap(point, size=size, radius=radius)[0, 0]
    assert target[8, 7].item() == 1.0
    expected = math.exp(-16.0 / (2.0 * radius * radius))
    assert abs(target[8, 8].item() - expected) < 1e-6
    assert target[8, 13].item() == 0.0


def test_route_a_recovers_a_continuous_subpixel_gaussian() -> None:
    size = 16
    target = torch.tensor([[[7.75 / size, 9.15 / size]]])
    heatmap = continuous_gaussian_heatmap(target, size=size, sigma=1.0)
    decoded = route_a_local_soft_argmax(heatmap, window_size=5, temperature=10.0)
    assert torch.max(torch.abs(decoded - target)).item() * size < 0.15


def test_route_a_local_decoder_ignores_distant_background_mass() -> None:
    heatmap = torch.zeros(1, 1, 16, 16)
    heatmap[0, 0, 8, 8] = 1.0
    heatmap[0, 0, 1:5, 1:5] = 0.2
    local = route_a_local_soft_argmax(heatmap, window_size=5, temperature=10.0)
    global_point = heatmap_to_points(heatmap)
    peak = torch.tensor([[[8.5 / 16, 8.5 / 16]]])
    assert torch.linalg.vector_norm(local - peak).item() < 1e-3
    assert torch.linalg.vector_norm(global_point - peak).item() > 0.1


def test_route_a_local_decoder_handles_heatmap_edges() -> None:
    heatmap = torch.zeros(1, 1, 8, 8)
    heatmap[0, 0, 0, 0] = 1.0
    point = route_a_local_soft_argmax(heatmap, window_size=5, temperature=10.0)
    assert torch.isfinite(point).all()
    assert 0.0 < point[0, 0, 0] < 0.2
    assert 0.0 < point[0, 0, 1] < 0.2


def test_route_a_continuous_target_keeps_fractional_center() -> None:
    size = 32
    target = torch.tensor([[[10.8 / size, 18.2 / size]]])
    heatmap = continuous_gaussian_heatmap(target, size=size, sigma=1.0)
    decoded = route_a_local_soft_argmax(heatmap, window_size=5, temperature=10.0)
    assert torch.max(torch.abs(decoded - target)).item() * size < 0.15
