import torch

from facebench.tasks.landmark.engine import evaluate_model
from facebench.tasks.landmark.model import heatmap_to_points


class _Dataset:
    def __init__(self):
        self.samples = [type("Sample", (), {"sample_id": "sample"})()]

    def __len__(self):
        return len(self.samples)


class _UniformHeatmapModel(torch.nn.Module):
    def forward(self, images):
        logits = torch.zeros(
            (images.shape[0], 98, 4, 4),
            device=images.device,
            dtype=images.dtype,
        )
        return {
            "heatmap_logits": logits,
            "points": heatmap_to_points(torch.sigmoid(logits.float())),
        }


def test_evaluation_reports_one_prediction_per_sample() -> None:
    targets = torch.full((1, 98, 2), 255.5)
    targets[0, 60] = torch.tensor([205.5, 255.5])
    targets[0, 72] = torch.tensor([305.5, 255.5])
    targets[0, 96] = torch.tensor([215.5, 255.5])
    targets[0, 97] = torch.tensor([295.5, 255.5])
    batch = {
        "image": torch.zeros((1, 3, 8, 8)),
        "transform": torch.eye(3).unsqueeze(0),
        "landmarks_original": targets,
        "subset_flags": torch.zeros((1, 6), dtype=torch.bool),
        "sample_id": ["sample"],
    }
    dataset = _Dataset()

    metrics, payload = evaluate_model(
        _UniformHeatmapModel(),
        [batch],
        dataset,
        torch.device("cpu"),
        amp_enabled=False,
        amp_dtype="bf16",
        description="test",
    )

    assert payload is not None
    assert payload["predictions"].shape == (1, 98, 2)
    assert metrics["samples"] == 1
