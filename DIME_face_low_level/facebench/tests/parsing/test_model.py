import torch

from facebench.tasks.parsing.backbones import FeatureBackbone
from facebench.tasks.parsing.model import ParsingModel, load_finetuned_checkpoint


class DummyViT(FeatureBackbone):
    def __init__(self):
        super().__init__((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        self.out_channels = (8, 8, 8, 8)
        self.pyramid_type = "vit"
        self.projection = torch.nn.Conv2d(3, 8, kernel_size=16, stride=16)

    def forward(self, images):
        feature = self.projection(images)
        return [feature, feature, feature, feature]


class DummyNative(FeatureBackbone):
    def __init__(self):
        super().__init__((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        self.out_channels = (4, 8, 16, 32)
        self.pyramid_type = "native"
        self.levels = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(3, 4, 4, 4),
                torch.nn.Conv2d(3, 8, 8, 8),
                torch.nn.Conv2d(3, 16, 16, 16),
                torch.nn.Conv2d(3, 32, 32, 32),
            ]
        )

    def forward(self, images):
        return [level(images) for level in self.levels]


def _run(backbone):
    model = ParsingModel(
        backbone,
        input_size=32,
        output_size=64,
        head_channels=8,
        num_classes=11,
        dropout=0.1,
    ).eval()
    with torch.inference_mode():
        output = model(torch.rand(2, 3, 64, 64))
    assert output.shape == (2, 11, 64, 64)
    assert torch.isfinite(output).all()


def test_vit_model_contract():
    _run(DummyViT())


def test_native_model_contract():
    _run(DummyNative())


def test_finetuned_checkpoint_prefers_ema(tmp_path):
    model = ParsingModel(
        DummyViT(),
        input_size=32,
        output_size=64,
        head_channels=8,
        num_classes=11,
        dropout=0.1,
    )
    model_state = model.state_dict()
    ema_state = {
        name: value.clone() + 1.0 if torch.is_floating_point(value) else value.clone()
        for name, value in model_state.items()
    }
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model_state, "ema": ema_state}, path)
    load_finetuned_checkpoint(model, path, use_ema=True)
    assert torch.allclose(
        model.state_dict()["head.classifier.bias"],
        ema_state["head.classifier.bias"],
    )
