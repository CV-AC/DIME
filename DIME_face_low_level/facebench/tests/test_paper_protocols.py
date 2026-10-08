from copy import deepcopy

import pytest
import torch
from torch import nn

from facebench.common.ema import ModelEMA
from facebench.common.features import intermediate_indices
from facebench.tasks.head_pose.config import CONFIG_ROOT as POSE_CONFIGS
from facebench.tasks.head_pose.config import load_config as load_pose
from facebench.tasks.landmark.config import CONFIG_ROOT as LANDMARK_CONFIGS
from facebench.tasks.landmark.config import load_config as load_landmark
from facebench.tasks.landmark.backbones import (
    TimmViTFeatureBackbone as LandmarkFeatures,
)
from facebench.tasks.parsing.config import CONFIG_ROOT as PARSING_CONFIGS
from facebench.tasks.parsing.config import load_config as load_parsing
from facebench.tasks.parsing.backbones import TimmViTFeatureBackbone as ParsingFeatures


def test_pose_encoders_share_training_and_selection():
    base = load_pose(POSE_CONFIGS / "base.yaml")
    for name in ["dime", "dino_vitl16", "mae_vitl16", "repvgg_b1g2"]:
        config = load_pose(POSE_CONFIGS / f"{name}.yaml")
        for section in [
            "data",
            "protocol",
            "augmentation",
            "evaluation",
            "head",
            "finetuning",
        ]:
            assert config[section] == base[section]
    assert base["protocol"]["input_size"] == 224
    assert base["protocol"]["loss"] == "SO3_geodesic"
    assert base["head"]["type"] == "linear"
    assert base["finetuning"]["encoder_mode"] == "full"
    assert base["evaluation"]["selection"] == "fixed_final_epoch_ema"
    assert not base["evaluation"]["test_time_augmentation"]
    assert (
        load_pose(POSE_CONFIGS / "dino_vitl16.yaml")["model"]["timm_model"]
        == "vit_large_patch16_dinov3.lvd1689m"
    )
    assert (
        load_pose(POSE_CONFIGS / "mae_vitl16.yaml")["model"]["timm_model"]
        == "vit_large_patch16_224.mae"
    )


def test_landmark_encoders_share_training_and_selection():
    base = load_landmark(LANDMARK_CONFIGS / "wflw/base.yaml")
    for name in ["dime", "dino", "mae", "farl_ep64"]:
        config = load_landmark(LANDMARK_CONFIGS / f"wflw/{name}_full.yaml")
        for section in ["dataset", "model", "protocol", "augmentation"]:
            assert config[section] == base[section]
        assert not config["auxiliary_training"]["enabled"]
    assert base["model"]["input_size"] == 448
    assert base["model"]["num_landmarks"] == 98
    assert base["model"]["head_channels"] == 768
    assert base["model"]["objective"]["name"] == "farl"
    assert base["protocol"]["selection_protocol"] == "farl_official_test_best"


@pytest.mark.parametrize(
    "dataset,classes,counts",
    [
        ("lapa", 11, {"train": 18176, "val": 2000, "test": 2000}),
        ("celebamask_hq", 19, {"train": 24183, "val": 2993, "test": 2824}),
    ],
)
def test_parsing_encoders_share_training_and_selection(dataset, classes, counts):
    base = load_parsing(PARSING_CONFIGS / dataset / "base.yaml")
    for name in ["dime", "dino", "mae", "farl"]:
        config = load_parsing(PARSING_CONFIGS / dataset / f"{name}.yaml")
        for section in [
            "dataset",
            "model",
            "protocol",
            "augmentation",
            "loader",
            "evaluation",
        ]:
            assert config[section] == base[section]
    assert base["dataset"]["expected_counts"] == counts
    assert base["dataset"]["num_classes"] == classes
    assert base["model"]["input_size"] == 448
    assert base["model"]["output_size"] == 512
    assert base["protocol"]["selection_split"] == "val"
    assert base["evaluation"]["split"] == "test"
    assert base["evaluation"]["use_ema"]


@pytest.mark.parametrize("backbone_type", [LandmarkFeatures, ParsingFeatures])
def test_rotary_backbones_use_native_position_encoding(backbone_type):
    from timm.models.eva import Eva

    model = Eva(
        img_size=32,
        patch_size=16,
        embed_dim=32,
        depth=24,
        num_heads=4,
        num_classes=0,
        num_reg_tokens=4,
        use_abs_pos_emb=False,
        use_rot_pos_emb=True,
        rope_type="dinov3",
    ).eval()
    adapter = backbone_type(model, "test").eval()
    assert adapter.indices == (7, 11, 15, 23)
    images = torch.randn(1, 3, 32, 32)
    expected = model.forward_intermediates(
        images,
        indices=list(adapter.indices),
        norm=False,
        intermediates_only=True,
    )
    actual = adapter(images)
    assert len(actual) == 4
    for left, right in zip(actual, expected):
        assert left.shape == (1, 32, 2, 2)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    sum(value.square().mean() for value in actual).backward()
    assert torch.isfinite(model.patch_embed.proj.weight.grad).all()


@pytest.mark.parametrize("backbone_type", [LandmarkFeatures, ParsingFeatures])
def test_existing_vit_feature_path_is_unchanged(backbone_type):
    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        img_size=32,
        patch_size=16,
        embed_dim=32,
        depth=12,
        num_heads=4,
        num_classes=0,
    ).eval()
    reference = deepcopy(model)
    images = torch.randn(1, 3, 32, 32)
    adapter = backbone_type(model, "test").eval()
    assert intermediate_indices(12) == (3, 5, 7, 11)
    actual = adapter(images)
    tokens = reference.norm_pre(
        reference.patch_drop(reference._pos_embed(reference.patch_embed(images)))
    )
    expected = []
    for index, block in enumerate(reference.blocks):
        tokens = block(tokens)
        if index in (3, 5, 7, 11):
            expected.append(
                tokens[:, reference.num_prefix_tokens :]
                .transpose(1, 2)
                .reshape(1, 32, 2, 2)
            )
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    sum(value.square().mean() for value in actual).backward()
    sum(value.square().mean() for value in expected).backward()
    torch.testing.assert_close(
        model.patch_embed.proj.weight.grad,
        reference.patch_embed.proj.weight.grad,
        rtol=0,
        atol=0,
    )


def test_shared_ema_preserves_parameter_and_buffer_updates():
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    ema = ModelEMA(model, decay=0.9)
    reference = deepcopy(ema.module)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter))
        model[1].running_mean.add_(1)
        for name, parameter in reference.named_parameters():
            parameter.mul_(0.9).add_(
                dict(model.named_parameters())[name].detach(), alpha=1.0 - 0.9
            )
        for name, buffer in reference.named_buffers():
            buffer.copy_(dict(model.named_buffers())[name])
    ema.update(model)
    for name, value in ema.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
