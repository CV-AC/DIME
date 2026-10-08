import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from DIME_VIT.config import load_config
from DIME_VIT.model import DIMEViT, _edds_v1


def reference_loss(target, prediction, normalized=False, std=None, mean=None, rho=0.5):
    batch, length, channels = target.shape
    raw = prediction * std.detach() + mean.detach() if normalized else prediction
    targets = []
    predictions = []
    for index in range(batch):
        other = batch - 1 - index
        true = (target[index] - target[other]).detach().reshape(length, -1, 3)
        pred = (raw[index] - raw[other]).reshape(length, -1, 3)
        targets.append(true - true.mean(dim=1, keepdim=True))
        predictions.append(pred - pred.mean(dim=1, keepdim=True))
    magnitudes = torch.stack([value.flatten(1).norm(dim=1) for value in targets])
    average = magnitudes.mean()
    losses = []
    count = int(round(length * (1.0 - rho)))
    for index in range(batch):
        if magnitudes[index].mean() > 0.2 * average and count:
            terms = []
            for patch in magnitudes[index].topk(count).indices:
                true = targets[index][patch]
                pred = predictions[index][patch]
                terms.append(
                    1.0 - (pred * true).sum() / (pred.norm() * true.norm() + 1e-6)
                )
            losses.append(torch.stack(terms).mean())
        else:
            losses.append(predictions[index].sum() * 0.0)
    return torch.stack(losses).mean()


class EDDSV1Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.target = torch.randn(4, 6, 12, dtype=torch.float64)

    def test_literal_formula_and_gradient_match(self):
        for normalized in (False, True):
            a = torch.randn_like(self.target, requires_grad=True)
            b = a.detach().clone().requires_grad_(True)
            std = torch.rand(4, 6, 1, dtype=torch.float64) + 0.1
            mean = torch.randn(4, 6, 1, dtype=torch.float64)
            actual = _edds_v1(self.target, a, normalized, std, mean, 0.5)
            expected = reference_loss(self.target, b, normalized, std, mean)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            actual.backward()
            expected.backward()
            torch.testing.assert_close(a.grad, b.grad, atol=1e-12, rtol=1e-12)

    def test_zero_prediction_is_supervised_and_has_gradient(self):
        prediction = torch.zeros_like(self.target, requires_grad=True)
        loss = _edds_v1(self.target, prediction, False, None, None, 0.5)
        torch.testing.assert_close(loss, loss.new_tensor(1.0))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0.0)

    def test_empty_support_is_zero_with_finite_zero_gradient(self):
        target = torch.zeros_like(self.target)
        prediction = torch.zeros_like(target, requires_grad=True)
        loss = _edds_v1(target, prediction, False, None, None, 0.5)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_skipped_pairs_contribute_zero_to_batch_average(self):
        target = self.target.clone()
        target[1] = target[2]
        prediction = -target.clone().requires_grad_(True)
        loss = _edds_v1(target, prediction, False, None, None, 0.5)
        expected = reference_loss(target, prediction)
        torch.testing.assert_close(loss, expected)
        self.assertAlmostEqual(loss.item(), 1.0, places=6)

    def test_target_only_hard_support_ignores_unselected_predictions(self):
        target = self.target[:2].clone()
        delta = (target[0] - target[1]).reshape(6, -1, 3)
        magnitudes = (delta - delta.mean(dim=1, keepdim=True)).flatten(1).norm(dim=1)
        selected = magnitudes.topk(3).indices
        dropped = torch.ones(6, dtype=torch.bool)
        dropped[selected] = False
        prediction = torch.randn_like(target)
        changed = prediction.clone()
        changed[:, dropped] *= 1000.0
        torch.testing.assert_close(
            _edds_v1(target, prediction, False, None, None, 0.5),
            _edds_v1(target, changed, False, None, None, 0.5),
        )

    def test_supported_patches_have_equal_weight(self):
        target = self.target[:2].clone()
        delta = (target[0] - target[1]).reshape(6, -1, 3)
        magnitude = (delta - delta.mean(dim=1, keepdim=True)).flatten(1).norm(dim=1)
        selected = magnitude.topk(3).indices
        prediction = target.clone()
        prediction[:, selected[0]] *= -1.0
        loss = _edds_v1(target, prediction, False, None, None, 0.5)
        self.assertAlmostEqual(loss.item(), 2.0 / 3.0, places=6)

    def test_pair_swap_invariance(self):
        prediction = torch.randn_like(self.target)
        a = _edds_v1(self.target, prediction, False, None, None, 0.5)
        b = _edds_v1(self.target.flip(0), prediction.flip(0), False, None, None, 0.5)
        torch.testing.assert_close(a, b)

    def test_per_channel_constant_offsets_cancel(self):
        prediction = torch.randn_like(self.target)
        offsets = torch.randn(4, 6, 1, 3, dtype=torch.float64).expand(-1, -1, 4, -1)
        shifted = prediction + offsets.reshape_as(prediction)
        torch.testing.assert_close(
            _edds_v1(self.target, prediction, False, None, None, 0.5),
            _edds_v1(self.target, shifted, False, None, None, 0.5),
        )

    def test_target_statistics_are_detached(self):
        target = self.target.clone().requires_grad_(True)
        prediction = torch.randn_like(target, requires_grad=True)
        std = torch.ones(4, 6, 1, dtype=torch.float64, requires_grad=True)
        mean = torch.zeros(4, 6, 1, dtype=torch.float64, requires_grad=True)
        _edds_v1(target, prediction, True, std, mean, 0.5).backward()
        self.assertIsNone(target.grad)
        self.assertIsNone(std.grad)
        self.assertIsNone(mean.grad)
        self.assertIsNotNone(prediction.grad)

    def test_global_ddp_pair_reference_controls_support(self):
        prediction = -self.target.clone()
        local_sum = (self.target - self.target.flip(0)).norm().item()

        def larger_global_batch(totals):
            totals[0] += 1000.0 * local_sum
            totals[1] += 4.0

        with mock.patch("torch.distributed.is_initialized", return_value=True):
            with mock.patch(
                "torch.distributed.all_reduce", side_effect=larger_global_batch
            ) as collective:
                loss = _edds_v1(self.target, prediction, False, None, None, 0.5)
        collective.assert_called_once()
        self.assertEqual(loss.item(), 0.0)

    def test_numerical_gradient_check(self):
        prediction = torch.randn_like(self.target, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda value: _edds_v1(self.target, value, False, None, None, 0.5),
                (prediction,),
                fast_mode=True,
            )
        )

    def test_optimizer_can_reduce_edds(self):
        prediction = torch.nn.Parameter(torch.randn_like(self.target))
        optimizer = torch.optim.Adam([prediction], lr=0.05)
        initial = _edds_v1(self.target, prediction, False, None, None, 0.5).item()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = _edds_v1(self.target, prediction, False, None, None, 0.5)
            loss.backward()
            optimizer.step()
        final = _edds_v1(self.target, prediction, False, None, None, 0.5).item()
        self.assertLess(final, initial * 0.5)

    def test_full_model_bfloat16_optimizer_steps(self):
        model = DIMEViT(
            img_size=32,
            embed_dim=32,
            depth=1,
            num_heads=4,
            decoder_dim=32,
            decoder_depth=1,
            decoder_num_heads=4,
            edds_version="v1",
            edds_warmup_epochs=0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.patch_embed.proj.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = model(torch.randn(4, 3, 32, 32), edds_active=True)
            self.assertTrue(torch.isfinite(output["loss"]))
            self.assertGreater(output["loss_edds"].item(), 0.0)
            output["loss"].backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
        self.assertFalse(torch.equal(before, model.patch_embed.proj.weight))

    def test_compiled_formula_forward_and_backward(self):
        compiled = torch.compile(_edds_v1, backend="eager", fullgraph=True)
        prediction = torch.randn_like(self.target, requires_grad=True)
        loss = compiled(self.target, prediction, False, None, None, 0.5)
        torch.testing.assert_close(loss, reference_loss(self.target, prediction))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_configuration_selects_both_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("{}\n", encoding="utf-8")
            for version in ("v1", "v2"):
                config = load_config(path, [f"loss.edds_version={version}"])
                self.assertEqual(config.loss.edds_version, version)
            with self.assertRaisesRegex(ValueError, "edds_version"):
                load_config(path, ["loss.edds_version=other"])


if __name__ == "__main__":
    unittest.main()
