from pathlib import Path
import gc
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from facebench.tasks.head_pose.config import (
    HEAD_POSE_ROOT,
    experiment_output_dir,
    load_config,
    CONFIG_ROOT,
)
from facebench.tasks.head_pose.collect_results import collect
from facebench.tasks.head_pose.data import train_transform
from facebench.tasks.head_pose.engine import ModelEMA
from facebench.tasks.head_pose.models import RepVGGEncoder, TimmEncoder
from facebench.tasks.head_pose.optim import build_optimizer, component_learning_rates


class LayeredEncoder(nn.Module):
    num_lr_layers = 3

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def parameter_layer_id(self, name: str) -> int:
        return int(name.split(".")[1])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class TinyPoseDataset(Dataset):
    manifest_sha256 = "synthetic"

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 8, 8),
            "rotation": torch.eye(3),
            "ypr": torch.zeros(3),
            "sample_id": str(index),
        }


def test_main_configs_share_controlled_protocol():
    names = ("repvgg_b1g2.yaml", "dino_vitb16.yaml", "mae_vitb16.yaml", "dime.yaml")
    configs = [load_config(CONFIG_ROOT / name) for name in names]
    protocols = [config["protocol"] for config in configs]
    assert {protocol["name"] for protocol in protocols} == {
        "controlled_6drepnet_strong_v1"
    }
    assert {protocol["epochs"] for protocol in protocols} == {80}
    assert {protocol["effective_batch_size"] for protocol in protocols} == {64}
    assert {config["evaluation"]["selection"] for config in configs} == {
        "fixed_final_epoch_ema"
    }
    assert {config["head"]["type"] for config in configs} == {"linear"}
    assert configs[1]["model"]["pooling"] == "cls_avg_concat"
    assert configs[2]["model"]["pooling"] == "global_avg"


def test_seed_output_directory_is_explicit():
    config = load_config(CONFIG_ROOT / "dino_vitb16.yaml")
    assert (
        experiment_output_dir(config, 2)
        .as_posix()
        .endswith("outputs/head_pose/dino_vitb16/seed_002")
    )


def test_train_transform_uses_official_random_resized_crop():
    transform = train_transform(224)
    assert isinstance(transform.transforms[0], transforms.RandomResizedCrop)
    assert transform.transforms[0].scale == (0.8, 1.0)


def test_layer_decay_and_ema():
    from facebench.tasks.head_pose.models import HeadPoseModel

    model = HeadPoseModel(LayeredEncoder(), 4)
    config = {
        "protocol": {
            "encoder_lr": 1.0e-4,
            "encoder_layer_decay": 0.5,
            "head_lr": 1.0e-4,
            "optimizer": {
                "name": "adamw",
                "weight_decay": 0.05,
                "exclude_bias_and_norm_from_weight_decay": True,
            },
        }
    }
    optimizer = build_optimizer(model, config)
    rates = component_learning_rates(optimizer)
    assert rates["encoder"] == 1.0e-4
    assert rates["encoder_min"] == 2.5e-5
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    before = next(ema.module.parameters()).clone()
    ema.update(model)
    assert not torch.equal(before, next(ema.module.parameters()))


def test_repvgg_encoder_adapter_returns_official_feature_width():
    encoder = RepVGGEncoder(
        HEAD_POSE_ROOT / "third_party" / "sixdrepnet",
        "RepVGG-B1g2",
        checkpoint_path=None,
    ).eval()
    with torch.inference_mode():
        features = encoder(torch.zeros(1, 3, 64, 64))
    assert features.shape == (1, 2048)


def test_timm_encoder_uses_native_load_then_method_specific_pooling():
    dino = TimmEncoder(
        "vit_base_patch16_224.dino",
        pretrained=False,
        checkpoint_path=None,
        pooling="cls_avg_concat",
    ).eval()
    assert dino.model.global_pool == "token"
    assert isinstance(dino.model.norm, nn.LayerNorm)
    assert dino.num_features == 1536
    with torch.inference_mode():
        assert dino(torch.zeros(1, 3, 224, 224)).shape == (1, 1536)
    del dino
    gc.collect()

    mae = TimmEncoder(
        "vit_base_patch16_224.mae",
        pretrained=False,
        checkpoint_path=None,
        pooling="global_avg",
    ).eval()
    assert mae.model.global_pool == "avg"
    assert isinstance(mae.model.norm, nn.Identity)
    assert isinstance(mae.model.fc_norm, nn.LayerNorm)
    assert mae.num_features == 768
    with torch.inference_mode():
        assert mae(torch.zeros(1, 3, 224, 224)).shape == (1, 768)


def test_prefetch_keeps_native_token_architecture(monkeypatch):
    import timm

    from facebench.tasks.head_pose import prefetch_timm

    calls = []

    class DummyModel:
        num_features = 768
        global_pool = "token"

    def fake_create_model(name, **kwargs):
        calls.append((name, kwargs))
        return DummyModel()

    monkeypatch.setattr(timm, "create_model", fake_create_model)
    prefetch_timm.main()
    assert [name for name, _ in calls] == list(prefetch_timm.TIMM_MODELS)
    assert all("global_pool" not in kwargs for _, kwargs in calls)


def test_result_collector_requires_paired_splits(tmp_path: Path):
    evaluation = tmp_path / "method" / "seed_000" / "evaluation"
    evaluation.mkdir(parents=True)
    common = {
        "method": "dime_full_head_pose",
        "seed": 0,
        "samples": 1,
        "yaw_mae": 1.0,
        "pitch_mae": 2.0,
        "roll_mae": 3.0,
        "mean_mae": 2.0,
    }
    for split in ("aflw2000", "biwi"):
        (evaluation / f"{split}_metrics.json").write_text(
            json.dumps({**common, "dataset": split}), encoding="utf-8"
        )
    output = tmp_path / "results.md"
    summary = collect(tmp_path, output)
    assert summary["methods"]["dime_full_head_pose"]["seeds"] == [0]
    assert "1.000" in output.read_text(encoding="utf-8")
    assert "1.000 ± 0.000" not in output.read_text(encoding="utf-8")


def test_training_entrypoint_writes_final_ema(tmp_path: Path, monkeypatch):
    import facebench.tasks.head_pose.train as training
    from facebench.tasks.head_pose.models import HeadPoseModel

    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    train_root.mkdir()
    test_root.mkdir()
    train_manifest = tmp_path / "train.jsonl"
    test_manifest = tmp_path / "test.jsonl"
    train_manifest.touch()
    test_manifest.touch()
    config = {
        "seed": 0,
        "resume": "",
        "wandb": {"enabled": False},
        "experiment": {
            "name": "smoke",
            "output_dir": str(tmp_path / "outputs"),
        },
        "data": {
            "train_root": str(train_root),
            "train_manifest": str(train_manifest),
            "aflw2000_root": str(test_root),
            "aflw2000_manifest": str(test_manifest),
            "biwi_processed_root": str(test_root),
            "biwi_manifest": str(test_manifest),
        },
        "model": {"kind": "dino"},
        "head": {"type": "linear", "hidden_dim": 8, "dropout": 0.0},
        "finetuning": {"encoder_mode": "full"},
        "evaluation": {
            "selection": "oracle_best_on_test",
            "batch_size_per_gpu": 2,
            "num_workers_per_gpu": 0,
            "test_time_augmentation": False,
        },
        "protocol": {
            "name": "smoke",
            "controlled_main_table": True,
            "input_size": 224,
            "epochs": 2,
            "batch_size_per_gpu": 2,
            "effective_batch_size": 2,
            "num_workers": 0,
            "encoder_lr": 1.0e-4,
            "encoder_layer_decay": 1.0,
            "head_lr": 1.0e-4,
            "optimizer": {"name": "adamw", "weight_decay": 0.0},
            "scheduler": {"name": "constant", "warmup_epochs": 0},
            "ema_decay": 0.9,
            "amp": False,
            "save_every": 1,
            "log_every": 100,
        },
    }
    monkeypatch.setattr(training, "build_dataset", lambda *_: TinyPoseDataset())
    monkeypatch.setattr(
        "facebench.tasks.head_pose.oracle.build_dataset", lambda *_: TinyPoseDataset()
    )
    monkeypatch.setattr(
        training,
        "build_model",
        lambda *_: HeadPoseModel(
            nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4)),
            4,
        ),
    )
    training.train(config)
    assert (tmp_path / "outputs" / "seed_000" / "final_ema.pth").is_file()
    assert (tmp_path / "outputs" / "seed_000" / "best_oracle_ema.pth").is_file()
    selection = json.loads(
        (tmp_path / "outputs" / "seed_000" / "oracle_selection.json").read_text(
            encoding="utf-8"
        )
    )
    history = [
        json.loads(line)
        for line in (tmp_path / "outputs" / "seed_000" / "oracle_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["epoch"] for row in history] == [1, 2]
    assert selection["epoch"] in {1, 2}
    assert selection["selection"] == "oracle_best_on_test"
    selected_checkpoint = torch.load(
        tmp_path / "outputs" / "seed_000" / "best_oracle_ema.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert selected_checkpoint["epoch"] + 1 == selection["epoch"]
