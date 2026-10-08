from pathlib import Path

import pytest
import torch

from DIME_VIT.config import load_config
from DIME_VIT.encoder import DIMEViTEncoder
from DIME_VIT.engine import WarmupCosineScheduler, resume_training
from DIME_VIT.model import DIMEViT, _rope_2d, _rotate_axial_pairs


@pytest.mark.parametrize("name", ["small", "base", "large"])
def test_paper_recipe_has_two_resolution_stages(name):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / f"configs/vit_{name}_patch16.yaml")
    assert config.model.img_size == 224
    assert config.train.epochs == 500
    assert config.train.high_res_start_epoch == 400
    assert config.train.high_res_size == 512
    assert config.model.pos_encoding == "rope"
    assert config.model.mask_cell_size == 1
    assert config.loss.edds_version == "v2"
    assert config.data.pair_sampling == "identity_uniform"
    assert config.loss.edds_warmup_epochs == 20
    assert config.scheduler.warmup_epochs == 40
    high_batch = config.data.high_res_batch_size or min(32, config.data.batch_size)
    assert config.data.batch_size * config.train.accum_iter % high_batch == 0


def test_rope_keeps_spatial_coordinates_when_the_grid_changes():
    grids = [(14, 14), (28, 28), (32, 32), (14, 32)]
    tables = [_rope_2d(16, grid, torch.device("cpu"), torch.float32) for grid in grids]
    for (height, width), (cos, sin) in zip(grids, tables):
        assert cos.shape == sin.shape == (1, 1, height * width, 16)
        torch.testing.assert_close(cos.square() + sin.square(), torch.ones_like(cos))
        torch.testing.assert_close(cos[0, 0, 13 * width + 13], tables[0][0][0, 0, 195])
        torch.testing.assert_close(sin[0, 0, 13 * width + 13], tables[0][1][0, 0, 195])
        torch.testing.assert_close(cos[0, 0, width, 8:], cos[0, 0, 0, 8:])
        torch.testing.assert_close(sin[0, 0, width, 8:], sin[0, 0, 0, 8:])
        torch.testing.assert_close(cos[0, 0, 1, :8], cos[0, 0, 0, :8])
        torch.testing.assert_close(sin[0, 0, 1, :8], sin[0, 0, 0, :8])
        vectors = torch.randn(2, 4, height * width, 16)
        rotated = vectors * cos + _rotate_axial_pairs(vectors) * sin
        torch.testing.assert_close(rotated.norm(dim=-1), vectors.norm(dim=-1))


def tiny_model(size=224, depth=6):
    return DIMEViT(
        img_size=size,
        embed_dim=32,
        depth=depth,
        num_heads=4,
        decoder_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        mask_cell_size=1,
        edds_version="v1",
        edds_warmup_epochs=0,
    )


def test_training_and_transfer_follow_the_actual_input_grid():
    model = tiny_model()
    parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for size, grid in [(224, 14), (512, 32), (224, 14)]:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(torch.randn(2, 3, size, size), epoch=20)
        assert output["encoder_grid"] == output["decoder_grid"] == (grid, grid)
        assert output["pred_rgb"].shape == (4, grid * grid, 16 * 16 * 3)
        assert output["mask"].shape == (1, grid * grid, 1)
        assert torch.isfinite(output["loss"])
        output["loss"].backward()
        assert torch.isfinite(model.patch_embed.proj.weight.grad).all()
        optimizer.step()
        assert {
            name: id(parameter) for name, parameter in model.named_parameters()
        } == parameters
    model.eval()
    encoder = DIMEViTEncoder(model, "test").eval()
    with torch.inference_mode():
        for size, grid in [(224, 14), (448, 28), (512, 32)]:
            images = torch.randn(1, 3, size, size)
            tokens, actual_grid = model.forward_features(images, return_tokens=True)
            assert actual_grid == (grid, grid)
            assert tokens.shape == (1, grid * grid, 32)
            torch.testing.assert_close(encoder(images), tokens.mean(dim=1))
            assert [
                feature.shape for feature in encoder.forward_intermediates(images)
            ] == [(1, 32, grid, grid)] * 4


def test_checkpoint_resume_keeps_optimizer_and_resolution_flexible(tmp_path):
    model = tiny_model(depth=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, total_updates=10, warmup_updates=2)
    model(torch.randn(2, 3, 224, 224), epoch=20)["loss"].backward()
    optimizer.step()
    scheduler.step_update(0)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": 399,
            "global_update": 1,
        },
        checkpoint,
    )
    resumed = tiny_model(size=512, depth=1)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    resumed_scheduler = WarmupCosineScheduler(
        resumed_optimizer, total_updates=10, warmup_updates=2
    )
    state = resume_training(checkpoint, resumed, resumed_optimizer, resumed_scheduler)
    assert state["epoch"] == 399 and state["global_update"] == 1
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    for before, after in zip(
        optimizer.state.values(), resumed_optimizer.state.values()
    ):
        for name in before:
            torch.testing.assert_close(after[name], before[name])
    resumed_optimizer.zero_grad(set_to_none=True)
    output = resumed(torch.randn(2, 3, 512, 512), epoch=400)
    assert output["encoder_grid"] == (32, 32)
    output["loss"].backward()
    resumed_optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in resumed.parameters())
