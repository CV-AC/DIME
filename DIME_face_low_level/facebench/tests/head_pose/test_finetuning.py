import torch
import torch.nn as nn

from facebench.tasks.head_pose.models import HeadPoseModel
from facebench.tasks.head_pose.optim import build_optimizer, component_learning_rates


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.projection = nn.Linear(4, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.dropout(inputs))


def _config(encoder_lr: float = 1.0e-5) -> dict:
    return {
        "protocol": {
            "encoder_lr": encoder_lr,
            "head_lr": 1.0e-4,
            "optimizer": {
                "name": "adamw",
                "weight_decay": 0.05,
                "exclude_bias_and_norm_from_weight_decay": True,
            },
        },
    }


def test_frozen_encoder_stays_eval_and_optimizer_only_has_head():
    model = HeadPoseModel(
        TinyEncoder(),
        8,
        head_type="linear",
        freeze_encoder=True,
    )
    model.train()
    optimizer = build_optimizer(model, _config())

    assert not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert {group["component"] for group in optimizer.param_groups} == {"head"}
    assert component_learning_rates(optimizer) == {"head": 1.0e-4}


def test_full_finetuning_uses_separate_encoder_lr():
    model = HeadPoseModel(TinyEncoder(), 8, head_type="linear")
    optimizer = build_optimizer(model, _config(encoder_lr=2.5e-5))

    assert component_learning_rates(optimizer) == {
        "encoder": 2.5e-5,
        "head": 1.0e-4,
    }
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.05}


def test_mlp_pose_head_outputs_rotation_matrix():
    model = HeadPoseModel(
        TinyEncoder(),
        8,
        head_type="mlp",
        hidden_dim=16,
        dropout=0.1,
    )
    output = model(torch.randn(3, 4))

    assert output.shape == (3, 3, 3)
    identity = torch.eye(3).expand(3, -1, -1)
    torch.testing.assert_close(
        output @ output.transpose(1, 2),
        identity,
        atol=1e-5,
        rtol=1e-5,
    )
