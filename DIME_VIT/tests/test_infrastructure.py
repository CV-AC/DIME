import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import lmdb
import torch

from DIME_VIT.config import load_config
from DIME_VIT.data import (
    DistributedIdentityPairBatchSampler,
    build_eval_loader,
    build_train_loader,
)
from DIME_VIT.engine import WandBLogger, WarmupCosineScheduler, clean_state_dict
from DIME_VIT.evaluate import (
    evaluate_reconstruction,
    mse_per_image,
    psnr_per_image,
    save_reconstruction_panel,
    ssim_per_image,
)
from DIME_VIT.model import DIMEViT


class InfrastructureTest(unittest.TestCase):
    def test_train_and_eval_loaders_can_share_one_lmdb(self):
        with tempfile.TemporaryDirectory() as directory:
            lmdb_path = Path(directory) / "faces.lmdb"
            env = lmdb.open(str(lmdb_path), map_size=1 << 20)
            identity_index = {0: [0, 1], 1: [2, 3]}
            with env.begin(write=True) as txn:
                txn.put(
                    b"__meta__",
                    pickle.dumps({"num_samples": 4, "jpeg_encoded": False}),
                )
                txn.put(b"__identity_index__", pickle.dumps(identity_index))
                for index in range(4):
                    txn.put(
                        f"{index:08d}".encode("ascii"),
                        pickle.dumps(
                            {
                                "image": torch.zeros(
                                    32, 32, 3, dtype=torch.uint8
                                ).numpy(),
                                "label": index // 2,
                            }
                        ),
                    )
            env.close()

            train_loader, _ = build_train_loader(
                lmdb_path,
                input_size=32,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
            )
            eval_loader = build_eval_loader(
                lmdb_path,
                input_size=32,
                num_pairs=2,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
            )

            next(iter(train_loader))
            next(iter(eval_loader))
            self.assertIsNone(train_loader.dataset._env)
            self.assertIsNone(eval_loader.dataset._env)

    def test_dotted_config_overrides_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("{}\n", encoding="utf-8")
            config = load_config(
                config_path,
                [
                    "model.img_size=[192, 256]",
                    "data.batch_size",
                    "8",
                    "compile.enabled=true",
                ],
            )
        self.assertEqual(config.model.img_size, [192, 256])
        self.assertEqual(config.data.batch_size, 8)
        self.assertTrue(config.compile.enabled)
        self.assertEqual(config.loss.edds_version, "v2")
        self.assertEqual(config.loss.lambda_diff, 0.5)
        self.assertEqual(config.loss.sobel_q, 0.5)

    def test_fullgraph_rejects_python_random_mask_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "compile.fullgraph"):
                load_config(
                    config_path,
                    [
                        "compile.enabled=true",
                        "compile.fullgraph=true",
                        "model.mask_block_sizes=[1, 2]",
                    ],
                )

    def test_fullgraph_accepts_tensor_multiscale_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("{}\n", encoding="utf-8")
            config = load_config(
                config_path,
                [
                    "compile.enabled=true",
                    "compile.fullgraph=true",
                    "model.mask_strategy=multiscale",
                    "model.mask_block_sizes=[1, 2, 4]",
                ],
            )
        self.assertTrue(config.compile.fullgraph)

    def test_pair_sampler_mirrors_identity_and_partitions_ranks(self):
        identity_index = {label: [2 * label, 2 * label + 1] for label in range(8)}
        labels = {
            index: label
            for label, indices in identity_index.items()
            for index in indices
        }
        batches = []
        rank_labels = []
        for rank in (0, 1):
            sampler = DistributedIdentityPairBatchSampler(
                identity_index,
                batch_size=4,
                dataset_size=16,
                rank=rank,
                world_size=2,
                seed=7,
                num_batches=1,
            )
            sampler.set_epoch(3)
            batch = next(iter(sampler))
            batches.append(batch)
            self.assertEqual(
                [labels[index] for index in batch],
                [labels[index] for index in reversed(batch)],
            )
            rank_labels.append({labels[index] for index in batch})
        self.assertTrue(rank_labels[0].isdisjoint(rank_labels[1]))

    def test_compiled_and_ddp_prefix_cleanup(self):
        tensor = torch.randn(2, 3)
        cleaned = clean_state_dict({"module._orig_mod.student.module.weight": tensor})
        self.assertEqual(list(cleaned), ["weight"])
        self.assertIs(cleaned["weight"], tensor)

    def test_update_indexed_warmup_cosine(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1.0)
        scheduler = WarmupCosineScheduler(
            optimizer, total_updates=6, warmup_updates=2, min_lr=0.1
        )
        rates = [scheduler.step_update(index)[0] for index in range(6)]
        self.assertAlmostEqual(rates[0], 0.5)
        self.assertAlmostEqual(rates[1], 1.0)
        self.assertAlmostEqual(rates[-1], 0.1)

    def test_reconstruction_metrics(self):
        images = torch.rand(2, 3, 32, 48)
        torch.testing.assert_close(mse_per_image(images, images), torch.zeros(2))
        self.assertTrue(torch.all(psnr_per_image(images, images) > 100))
        torch.testing.assert_close(
            ssim_per_image(images, images), torch.ones(2), atol=1e-5, rtol=1e-5
        )

    def test_evaluation_returns_full_diagnostic_panel(self):
        model = DIMEViT(
            img_size=(64, 96),
            embed_dim=64,
            depth=1,
            num_heads=4,
            decoder_dim=64,
            decoder_depth=1,
            decoder_num_heads=4,
            mask_cell_size=2,
        )
        images = torch.randn(2, 3, 64, 96)
        metrics, visuals = evaluate_reconstruction(
            model,
            [(images, torch.zeros(2, dtype=torch.long))],
            torch.device("cpu"),
            amp_dtype="fp32",
            seed=3,
            num_visuals=2,
        )
        self.assertEqual(set(metrics), {"mse", "psnr", "ssim", "mask_ratio"})
        self.assertEqual(
            set(visuals),
            {"target", "source", "mask", "mixed", "reconstruction", "error"},
        )
        self.assertEqual(visuals["error"].shape, images.shape)
        torch.testing.assert_close(visuals["source"], visuals["target"].flip(0))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "panel.png"
            save_reconstruction_panel(visuals, output)
            self.assertTrue(output.is_file())

    def test_wandb_logger_uses_custom_step_and_full_panels(self):
        class FakeRun:
            def __init__(self):
                self.metrics = []
                self.logs = []
                self.summary = {}
                self.watched = None
                self.finished = False

            def define_metric(self, *args, **kwargs):
                self.metrics.append((args, kwargs))

            def log(self, values):
                self.logs.append(values)

            def watch(self, model, **kwargs):
                self.watched = (model, kwargs)

            def finish(self):
                self.finished = True

        run = FakeRun()
        fake_wandb = types.ModuleType("wandb")
        login_calls = []
        fake_wandb.login = lambda **kwargs: login_calls.append(kwargs) or True
        fake_wandb.init = lambda **_kwargs: run
        fake_wandb.Image = lambda tensor, caption: (tensor, caption)
        config = {
            "enabled": True,
            "project": "test",
            "entity": None,
            "run_name": "run",
            "tags": [],
            "watch_model": True,
            "watch_log": "gradients",
            "watch_freq": 7,
        }
        images = torch.rand(2, 3, 8, 8)
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}), mock.patch(
            "DIME_VIT.engine.WANDB_API_KEY", "test-api-key"
        ):
            logger = WandBLogger(config, {"wandb": config})
            model = torch.nn.Linear(2, 2)
            logger.watch(model)
            logger.set_summary({"model/parameters": 6})
            logger.log({"train/loss": 1.0}, step=3)
            logger.log_reconstructions(
                target=images,
                source=images,
                mask=images,
                mixed=images,
                reconstruction=images,
                error=images,
                step=3,
            )
            logger.finish()
        self.assertEqual(len(login_calls), 1)
        self.assertTrue(login_calls[0]["key"])
        self.assertTrue(login_calls[0]["verify"])
        self.assertEqual(run.logs[0]["optimizer_step"], 3)
        self.assertEqual(len(run.logs[1]["eval/reconstructions"]), 2)
        self.assertEqual(run.watched[1]["log_freq"], 7)
        self.assertEqual(run.summary["model/parameters"], 6)
        self.assertTrue(run.finished)


if __name__ == "__main__":
    unittest.main()
