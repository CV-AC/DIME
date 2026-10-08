import torch
from torch import nn

from facebench.tasks.landmark.backbones import (
    FaRLFeatureBackbone,
    _farl_visual_state_dict,
)


def test_farl_loader_drops_only_known_pretraining_modules() -> None:
    layer = nn.Linear(3, 2)
    raw = {
        "visual.weight": layer.weight.detach().clone(),
        "visual.bias": layer.bias.detach().clone(),
        "visual.mask_token": torch.zeros(1, 1, 3),
        "visual.lm_transformer.resblocks.0.attn.in_proj_weight": torch.zeros(9, 3),
        "visual.ln_lm.weight": torch.ones(3),
        "visual.lm_head.weight": torch.zeros(3, 3),
        "transformer.resblocks.0.attn.in_proj_weight": torch.zeros(9, 3),
    }

    encoder, ignored = _farl_visual_state_dict(raw)

    assert set(encoder) == {"weight", "bias"}
    assert set(ignored) == {
        "mask_token",
        "lm_transformer.resblocks.0.attn.in_proj_weight",
        "ln_lm.weight",
        "lm_head.weight",
    }
    layer.load_state_dict(encoder, strict=True)


def test_farl_loader_keeps_unknown_visual_keys_for_strict_validation() -> None:
    encoder, ignored = _farl_visual_state_dict(
        {
            "visual.weight": torch.zeros(2, 3),
            "visual.unknown_module.weight": torch.zeros(1),
        }
    )

    assert "unknown_module.weight" in encoder
    assert ignored == ()


def test_farl_dense_adapter_freezes_unused_global_projection() -> None:
    visual = nn.Module()
    visual.transformer = nn.Module()
    visual.transformer.use_checkpoint = False
    visual.ln_post = nn.LayerNorm(4)
    visual.proj = nn.Parameter(torch.zeros(4, 2))

    FaRLFeatureBackbone(visual, checkpoint_hash="test")

    assert visual.proj.requires_grad is False
    assert all(not parameter.requires_grad for parameter in visual.ln_post.parameters())
