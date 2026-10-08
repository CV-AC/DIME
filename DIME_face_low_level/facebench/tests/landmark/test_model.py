import torch

from facebench.tasks.landmark.model import (
    UPerHead,
    continuous_gaussian_heatmap,
    foreground_balanced_heatmap_loss,
    foreground_weight_map,
    landmark_losses,
    route_a_local_soft_argmax,
)


def test_uper_head_output_shape() -> None:
    head = UPerHead([16, 16, 16, 16], channels=16, num_landmarks=98).eval()
    features = [
        torch.randn(2, 16, 32, 32),
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 16, 4, 4),
    ]
    with torch.no_grad():
        logits = head(features)
    assert logits.shape == (2, 98, 32, 32)


def _route_a_objective() -> dict:
    return {
        "name": "route_a",
        "heatmap_size": 16,
        "sigma": 1.0,
        "window_size": 5,
        "temperature": 10.0,
        "loss": {
            "name": "adaptive_wing",
            "alpha": 2.1,
            "omega": 14.0,
            "epsilon": 1.0,
            "theta": 0.5,
            "foreground_weight": 10.0,
            "foreground_threshold": 0.2,
            "dilation_kernel": 3,
        },
        "collapse_detection": {"min_heatmap_std": 1e-4},
    }


def test_route_a_uses_weighted_adaptive_wing_as_only_training_objective() -> None:
    logits = torch.zeros(2, 3, 16, 16, requires_grad=True)
    outputs = {
        "heatmap_logits": logits,
        "points": torch.full((2, 3, 2), 0.5),
    }
    landmarks = torch.tensor(
        [
            [[20.0, 30.0], [31.5, 11.0], [48.0, 50.0]],
            [[15.0, 22.0], [35.0, 41.0], [52.0, 13.0]],
        ]
    )
    losses = landmark_losses(
        outputs,
        landmarks,
        canvas_size=64,
        objective=_route_a_objective(),
    )
    torch.testing.assert_close(losses["loss"], losses["heatmap"])
    assert losses["coordinate"].item() > 0.0
    assert losses["heatmap_collapsed_fraction"].item() == 1.0
    assert losses["heatmap_std"].item() == 0.0
    losses["loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_route_a_loss_map_explicitly_upweights_target_neighborhood() -> None:
    target = torch.zeros(1, 1, 9, 9)
    target[0, 0, 4, 4] = 1.0
    weights = foreground_weight_map(
        target,
        foreground_weight=10.0,
        foreground_threshold=0.2,
        dilation_kernel=3,
    )
    assert torch.all(weights[0, 0, 3:6, 3:6] == 11.0)
    assert weights[0, 0, 0, 0].item() == 1.0


def test_route_a_heatmap_can_overfit_one_continuous_target() -> None:
    point = torch.tensor([[[7.8 / 16.0, 9.2 / 16.0]]])
    target = continuous_gaussian_heatmap(point, size=16, sigma=1.0)
    logits = torch.nn.Parameter(torch.zeros_like(target))
    optimizer = torch.optim.Adam([logits], lr=0.05)
    options = _route_a_objective()["loss"]
    initial = float(foreground_balanced_heatmap_loss(logits, target, options))
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = foreground_balanced_heatmap_loss(logits, target, options)
        loss.backward()
        optimizer.step()
    decoded = route_a_local_soft_argmax(
        logits.detach(), window_size=5, temperature=10.0
    )
    assert float(loss) < initial * 0.05
    assert torch.max(torch.abs(decoded - point)).item() * 16.0 < 0.2
