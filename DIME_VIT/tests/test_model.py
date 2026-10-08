import unittest

import torch

from DIME_VIT.EDDS_V2 import compute_edds_v2_loss
from DIME_VIT.model import Attention, DIMEViT, MODEL_CONFIGS, _edds_v2, _rope_2d


def tiny_model(mask_cell_size=2, attention_mode="auto", **kwargs):
    defaults = dict(
        img_size=(64, 96),
        embed_dim=64,
        depth=2,
        num_heads=4,
        decoder_dim=64,
        decoder_depth=1,
        decoder_num_heads=4,
        mask_cell_size=mask_cell_size,
        attention_mode=attention_mode,
        edds_warmup_epochs=0,
        edds_version="v2",
    )
    defaults.update(kwargs)
    return DIMEViT(**defaults)


class DIMEViTTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_deit_variant_definitions(self):
        self.assertEqual(
            (
                MODEL_CONFIGS["vit_small_patch16"]["embed_dim"],
                MODEL_CONFIGS["vit_small_patch16"]["depth"],
                MODEL_CONFIGS["vit_small_patch16"]["num_heads"],
            ),
            (384, 12, 6),
        )
        self.assertEqual(
            (
                MODEL_CONFIGS["vit_base_patch16"]["embed_dim"],
                MODEL_CONFIGS["vit_base_patch16"]["depth"],
                MODEL_CONFIGS["vit_base_patch16"]["num_heads"],
            ),
            (768, 12, 12),
        )
        self.assertEqual(
            (
                MODEL_CONFIGS["vit_large_patch16"]["embed_dim"],
                MODEL_CONFIGS["vit_large_patch16"]["depth"],
                MODEL_CONFIGS["vit_large_patch16"]["num_heads"],
            ),
            (1024, 24, 16),
        )

    def test_cell_two_forward_and_backward(self):
        model = tiny_model(mask_cell_size=2)
        images = torch.randn(2, 3, 64, 96)
        output = model(images, epoch=20)
        self.assertEqual(output["encoder_grid"], (4, 6))
        self.assertEqual(output["decoder_grid"], (2, 3))
        self.assertEqual(output["mask"].shape, (1, 6, 1))
        self.assertEqual(output["pred_rgb"].shape, (4, 6, 32 * 32 * 3))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertIsNotNone(model.patch_embed.proj.weight.grad)

    def test_cell_one_rectangular_grid(self):
        model = tiny_model(mask_cell_size=1)
        images = torch.randn(2, 3, 48, 80)
        output = model(images, epoch=0)
        self.assertEqual(output["encoder_grid"], (3, 5))
        self.assertEqual(output["decoder_grid"], (3, 5))
        self.assertEqual(output["mask"].shape, (1, 15, 1))
        self.assertEqual(output["pred_rgb"].shape, (4, 15, 16 * 16 * 3))

    def test_feature_path_only_requires_patch_alignment(self):
        model = tiny_model(mask_cell_size=2)
        images = torch.randn(2, 3, 48, 80)
        tokens, grid = model.forward_features(images, return_tokens=True)
        self.assertEqual(grid, (3, 5))
        self.assertEqual(tokens.shape, (2, 15, 64))
        self.assertEqual(model.forward_features(images).shape, (2, 64))
        with self.assertRaises(ValueError):
            model(images)

    def test_sorted_source_attention_matches_dense_mask(self):
        attention = Attention(dim=64, num_heads=4).eval()
        tokens = torch.randn(2, 8, 64)
        source_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=tokens.dtype).view(
            1, 8, 1
        )
        expanded = source_mask.expand(tokens.shape[0], -1, -1)
        order = torch.argsort(expanded.squeeze(-1), dim=-1, stable=True)
        inverse = torch.argsort(order, dim=-1)
        rope = _rope_2d(16, (2, 4), tokens.device, tokens.dtype)
        sorted_output = attention(tokens, source_mask, (order, inverse, 4), rope)
        dense_output = attention(tokens, source_mask, None, rope)
        torch.testing.assert_close(sorted_output, dense_output, atol=2e-6, rtol=2e-6)

    def test_raw_gated_attention_and_source_isolation(self):
        plain = Attention(dim=64, num_heads=4).eval()
        gated = Attention(dim=64, num_heads=4, gated=True, gate_init_bias=2.0).eval()
        gated.qkv.load_state_dict(plain.qkv.state_dict())
        gated.proj.load_state_dict(plain.proj.state_dict())

        tokens = torch.randn(2, 8, 64)
        source_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).view(1, 8, 1)
        expanded = source_mask.expand(tokens.shape[0], -1, -1)
        order = torch.argsort(expanded.squeeze(-1), dim=-1, stable=True)
        layout = (order, torch.argsort(order, dim=-1), 4)
        rope = _rope_2d(16, (2, 4), tokens.device, tokens.dtype)

        plain_output = plain(tokens, source_mask, layout, rope)
        gated_output = gated(tokens, source_mask, layout, rope)

        self.assertTrue(torch.isfinite(gated_output).all())
        self.assertFalse(torch.allclose(gated_output, plain_output))
        torch.testing.assert_close(gated.gate.bias, torch.zeros_like(gated.gate.bias))

        perturbed = tokens.clone()
        perturbed[:, source_mask[0, :, 0].bool()] += 100.0
        isolated = gated(perturbed, source_mask, layout, rope)
        visible = ~source_mask[0, :, 0].bool()
        torch.testing.assert_close(isolated[:, visible], gated_output[:, visible])

    def test_gates_are_encoder_only_and_trainable(self):
        model = tiny_model(gated_attention=True)
        self.assertTrue(all(block.attn.gate is not None for block in model.blocks))
        self.assertTrue(all(block.attn.gate is None for block in model.decoder_blocks))
        output = model(torch.randn(2, 3, 64, 96), epoch=0)
        output["loss"].backward()
        self.assertTrue(torch.isfinite(model.blocks[0].attn.gate.weight.grad).all())

    def test_multiscale_mask_is_exact_and_pair_symmetric(self):
        model = DIMEViT(
            img_size=224,
            embed_dim=64,
            depth=1,
            num_heads=4,
            decoder_dim=64,
            decoder_depth=1,
            decoder_num_heads=4,
            mask_cell_size=2,
            shared_mask=False,
            mask_strategy="multiscale",
            mask_block_sizes=[1, 2, 4],
            block_probs=[1.0, 1.0, 1.0],
        )
        decoder_mask, encoder_mask, split, _, _ = model._sample_mask(
            torch.empty(4, 3, 224, 224), None
        )
        self.assertEqual(decoder_mask.shape, (4, 49, 1))
        torch.testing.assert_close(decoder_mask.sum(1), torch.full((4, 1), 24.0))
        torch.testing.assert_close(encoder_mask.sum(1), torch.full((4, 1), 96.0))
        torch.testing.assert_close(decoder_mask[0], decoder_mask[3])
        torch.testing.assert_close(decoder_mask[1], decoder_mask[2])
        self.assertEqual(split, 100)
        self.assertIsNotNone(model._source_layout(encoder_mask, 4, split))

    def test_multiscale_forward_and_fullgraph_compile(self):
        model = tiny_model(
            mask_strategy="multiscale",
            mask_block_sizes=[1, 2],
            gated_attention=True,
        )
        images = torch.randn(2, 3, 64, 96)
        output = model(images, epoch=0)
        self.assertEqual(output["mask"].sum().item(), 3.0)
        self.assertTrue(torch.isfinite(output["loss"]))
        if hasattr(torch, "compile"):
            compiled = torch.compile(model, backend="eager", fullgraph=True)
            compiled_output = compiled(images, epoch=0)
            self.assertTrue(torch.isfinite(compiled_output["loss"]))

    def test_compiled_edds_flag_has_only_two_graph_states(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")

        model = tiny_model(edds_warmup_epochs=20)
        images = torch.randn(2, 3, 64, 96)
        compile_count = 0

        def counting_backend(graph_module, _example_inputs):
            nonlocal compile_count
            compile_count += 1
            return graph_module.forward

        compiled = torch.compile(model, backend=counting_backend, fullgraph=True)
        for _epoch in range(8):
            output = compiled(images, edds_active=False)
            self.assertEqual(output["loss_edds"].item(), 0.0)
        self.assertEqual(compile_count, 1)

        active_output = compiled(images, edds_active=True)
        self.assertTrue(torch.isfinite(active_output["loss_edds"]))
        compiled(images, edds_active=True)
        self.assertEqual(compile_count, 2)

    def test_local_edds_v2_matches_original_module(self):
        target = torch.randn(4, 7, 48)
        prediction_a = torch.randn(4, 7, 48, requires_grad=True)
        prediction_b = prediction_a.detach().clone().requires_grad_(True)
        mask = torch.randint(0, 2, (1, 7, 1), dtype=torch.float32)
        mean = torch.randn(4, 7, 1)
        std = torch.rand(4, 7, 1).add_(0.1)
        local = _edds_v2(target, prediction_a, mask, True, std, mean, 0.5)
        original = compute_edds_v2_loss(
            target_rgb=target,
            unmix_rgb=prediction_b,
            mask=mask,
            norm_pix_loss=True,
            p_std=std,
            p_mean=mean,
            sobel_q=0.5,
        )
        torch.testing.assert_close(local, original)
        local.backward()
        original.backward()
        torch.testing.assert_close(prediction_a.grad, prediction_b.grad)

    def test_reconstruction_shapes(self):
        model = tiny_model(mask_cell_size=2).eval()
        images = torch.randn(2, 3, 64, 96)
        result = model.reconstruct(images)
        self.assertEqual(result["reconstruction"].shape, images.shape)
        self.assertEqual(result["mixed"].shape, images.shape)
        self.assertEqual(result["mask_image"].shape, images.shape)

    def test_masks_do_not_promote_autocast_features_to_fp32(self):
        model = tiny_model(mask_cell_size=2)
        observed = {}

        def record_encoder_dtype(_module, inputs):
            observed["encoder"] = inputs[0].dtype

        def record_decoder_dtype(_module, inputs):
            observed["decoder"] = inputs[0].dtype

        model.blocks[0].register_forward_pre_hook(record_encoder_dtype)
        model.decoder_blocks[0].register_forward_pre_hook(record_decoder_dtype)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(torch.randn(2, 3, 64, 96), epoch=0)
        self.assertEqual(observed["encoder"], torch.bfloat16)
        self.assertEqual(observed["decoder"], torch.bfloat16)
        self.assertEqual(output["pred_rgb"].dtype, torch.bfloat16)

        self.assertEqual(output["loss"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
