import torch

from facebench.tasks.landmark.spatial import (
    WFLW_HORIZONTAL_FLIP_INDEX,
    affine_consistency_losses,
    heatmaps_to_canonical,
    horizontal_flip_matrix,
    points_to_canonical,
    undo_wflw_horizontal_flip_heatmaps,
)


def test_wflw_horizontal_flip_is_an_involution() -> None:
    permutation = WFLW_HORIZONTAL_FLIP_INDEX
    assert len(permutation) == 98
    assert tuple(permutation[index] for index in permutation) == tuple(range(98))
    assert permutation[0] == 32
    assert permutation[60] == 72
    assert permutation[96] == 97


def test_wflw_heatmap_flip_tta_is_exactly_reversible() -> None:
    heatmaps = torch.randn(2, 98, 7, 9)
    restored = undo_wflw_horizontal_flip_heatmaps(
        undo_wflw_horizontal_flip_heatmaps(heatmaps)
    )
    torch.testing.assert_close(restored, heatmaps)


def test_identity_heatmap_warp_is_exact() -> None:
    heatmaps = torch.rand(2, 98, 8, 8)
    transform = torch.eye(3).expand(2, -1, -1).clone()
    warped, valid = heatmaps_to_canonical(
        heatmaps,
        transform,
        transform,
        torch.zeros(2, dtype=torch.bool),
        canvas_size=512,
    )
    torch.testing.assert_close(warped, heatmaps, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(valid, torch.ones_like(valid))


def test_identical_views_have_zero_affine_consistency_loss() -> None:
    logits = torch.randn(2, 98, 8, 8)
    points = torch.rand(2, 98, 2) * 0.8 + 0.1
    outputs = {"heatmap_logits": logits, "points": points}
    transform = torch.eye(3).expand(2, -1, -1).clone()
    losses = affine_consistency_losses(
        outputs,
        outputs,
        transform,
        transform,
        transform,
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    assert losses["heatmap_consistency"].item() < 1e-10
    assert losses["coordinate_consistency"].item() < 1e-10


def test_flipped_view_returns_to_canonical_points_and_heatmaps() -> None:
    canonical_heatmaps = torch.rand(1, 98, 8, 8)
    view_heatmaps = undo_wflw_horizontal_flip_heatmaps(canonical_heatmaps)
    identity = torch.eye(3).unsqueeze(0)
    reflection = horizontal_flip_matrix(512).unsqueeze(0)
    restored_heatmaps, _ = heatmaps_to_canonical(
        view_heatmaps,
        reflection,
        identity,
        torch.ones(1, dtype=torch.bool),
        canvas_size=512,
    )
    torch.testing.assert_close(
        restored_heatmaps, canonical_heatmaps, atol=1e-6, rtol=1e-6
    )

    canonical_points = torch.rand(1, 98, 2) * 0.8 + 0.1
    permutation = torch.as_tensor(WFLW_HORIZONTAL_FLIP_INDEX)
    view_points = canonical_points.clone()
    view_points[..., 0] = 1.0 - view_points[..., 0]
    view_points = view_points.index_select(1, permutation)
    restored_points = points_to_canonical(
        view_points,
        reflection,
        identity,
        torch.ones(1, dtype=torch.bool),
        canvas_size=512,
    )
    torch.testing.assert_close(restored_points, canonical_points, atol=1e-6, rtol=1e-6)


def test_affine_consistency_has_finite_gradients() -> None:
    first_logits = torch.randn(1, 98, 8, 8, requires_grad=True)
    second_logits = torch.randn(1, 98, 8, 8, requires_grad=True)
    first_points = torch.rand(1, 98, 2, requires_grad=True)
    second_points = torch.rand(1, 98, 2, requires_grad=True)
    identity = torch.eye(3).unsqueeze(0)
    losses = affine_consistency_losses(
        {"heatmap_logits": first_logits, "points": first_points},
        {"heatmap_logits": second_logits, "points": second_points},
        identity,
        identity,
        identity,
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
    )
    total = losses["heatmap_consistency"] + losses["coordinate_consistency"]
    total.backward()
    for tensor in (first_logits, second_logits, first_points, second_points):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
