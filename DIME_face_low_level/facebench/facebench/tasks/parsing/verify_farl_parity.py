from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch.nn as nn

from .backbones import FARL_IMAGE_MEAN, FARL_IMAGE_STD, FARL_OUTPUT_INDICES
from .config import apply_overrides, load_config, resolve_path
from .labels import CELEBAMASK_HQ_LABELS, LAPA_LABELS
from .model import UPerHead, ViTFeaturePyramid


def _require_text(path: Path, fragments: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise AssertionError(f"{path} is missing expected fragments: {missing}")


def verify(config: dict[str, Any]) -> dict[str, Any]:
    dataset = str(config["dataset"]["name"]).lower()
    if dataset not in {"lapa", "celebamask_hq"}:
        raise ValueError(f"Unsupported parity-audit dataset {dataset!r}.")
    source = resolve_path(config["backbone"]["farl_source"], must_exist=True)
    assert source is not None
    face_parsing = source / "farl" / "experiments" / "face_parsing"
    transformers = source / "farl" / "network" / "transformers.py"

    _require_text(
        transformers,
        [
            "return [3, 5, 7, 11]",
            "nn.ConvTranspose2d(output_channels, output_channels",
            "nn.SyncBatchNorm(output_channels)",
            "nn.GELU()",
            "nn.MaxPool2d(kernel_size=2, stride=2)",
            "[0.48145466, 0.4578275, 0.40821073]",
            "[0.26862954, 0.26130258, 0.27577711]",
        ],
    )

    if FARL_OUTPUT_INDICES != (3, 5, 7, 11):
        raise AssertionError("The implemented FaRL transformer taps changed.")
    if FARL_IMAGE_MEAN != (0.48145466, 0.4578275, 0.40821073):
        raise AssertionError("The implemented FaRL image mean changed.")
    if FARL_IMAGE_STD != (0.26862954, 0.26130258, 0.27577711):
        raise AssertionError("The implemented FaRL image std changed.")
    pyramid = ViTFeaturePyramid(2)
    if not (
        isinstance(pyramid.levels[0][0], nn.ConvTranspose2d)
        and isinstance(pyramid.levels[0][1], nn.SyncBatchNorm)
        and isinstance(pyramid.levels[0][2], nn.GELU)
        and isinstance(pyramid.levels[0][3], nn.ConvTranspose2d)
        and isinstance(pyramid.levels[1], nn.ConvTranspose2d)
        and isinstance(pyramid.levels[2], nn.Identity)
        and isinstance(pyramid.levels[3], nn.MaxPool2d)
    ):
        raise AssertionError("The implemented FaRL feature pyramid changed.")
    head = UPerHead([2, 2, 2, 2], channels=2, num_classes=3, dropout=0.1)
    if head.ppm.scales != (1, 2, 3, 6):
        raise AssertionError("The implemented UPerHead PPM scales changed.")
    if not isinstance(head.dropout, nn.Dropout2d) or head.dropout.p != 0.1:
        raise AssertionError("The implemented UPerHead dropout changed.")
    _require_text(
        face_parsing
        / "trainers"
        / ("lapa_farl.yaml" if dataset == "lapa" else "celebm_farl.yaml"),
        ["decay: 0.999", "max_epoches: 300", "eval_interval: 1"],
    )
    _require_text(
        face_parsing / "optimizers" / "refine_backbone.yaml",
        [
            "lr: 0.001",
            "betas: [0.9, 0.999]",
            "weight_decay: 0.00001",
            "lr: 0.0001",
            "milestones: [200]",
            "gamma: 0.1",
        ],
    )
    _require_text(
        face_parsing / "networks" / "farl.yaml",
        [
            "class: farl.network.MMSEG_UPerHead",
            "channels: $$head_channel",
            "out_size: [512, 512]",
        ],
    )
    _require_text(
        face_parsing / "task.py",
        ["F.cross_entropy(", "reduction='none'", "ce_loss.mean()"],
    )
    _require_text(
        face_parsing
        / "augmenters"
        / ("lapa" if dataset == "lapa" else "celebm")
        / "train.yaml",
        [
            "shift_sigma: 0.01",
            "rot_sigma: 0.314",
            "scale_sigma: 0.1",
            ("warp_factor: $$warp_factor" if dataset == "lapa" else "warp_factor: 0.0"),
            "class: RandomGray",
            "class: RandomGamma",
            "class: RandomBlur",
        ],
    )

    protocol = config["protocol"]
    model = config["model"]
    augmentation = config["augmentation"]
    expected = {
        "input_size": 448,
        "output_size": 512,
        "head_channels": 768,
        "epochs": 300,
        "encoder_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 1e-5,
        "milestones": [200],
        "gamma": 0.1,
        "ema_decay": 0.999,
        "shift_sigma": 0.01,
        "rotation_sigma": 0.314,
        "scale_sigma": 0.1,
    }
    actual = {
        "input_size": int(model["input_size"]),
        "output_size": int(model["output_size"]),
        "head_channels": int(model["head_channels"]),
        "epochs": int(protocol["epochs"]),
        "encoder_lr": float(protocol["encoder_lr"]),
        "head_lr": float(protocol["head_lr"]),
        "weight_decay": float(protocol["weight_decay"]),
        "milestones": list(protocol["milestones"]),
        "gamma": float(protocol["gamma"]),
        "ema_decay": float(protocol["ema_decay"]),
        "shift_sigma": float(augmentation["shift_sigma"]),
        "rotation_sigma": float(augmentation["rotation_sigma"]),
        "scale_sigma": float(augmentation["scale_sigma"]),
    }
    if actual != expected:
        differences = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if actual[key] != expected[key]
        }
        raise AssertionError(f"Configured FaRL protocol differs: {differences}")
    expected_classes = (
        LAPA_LABELS.num_classes
        if dataset == "lapa"
        else CELEBAMASK_HQ_LABELS.num_classes
    )
    if int(config["dataset"]["num_classes"]) != expected_classes:
        raise AssertionError("Configured class count does not match label semantics.")
    expected_warp = 0.8 if dataset == "lapa" else 0.0
    if float(augmentation["warp_factor"]) != expected_warp:
        raise AssertionError(f"{dataset} requires FaRL warp_factor={expected_warp}.")
    return {
        "status": "PASS",
        "dataset": dataset,
        "official_source": str(source),
        "verified": {
            "vit_block_indices": [3, 5, 7, 11],
            "vit_feature_pyramid": ["4x", "2x", "1x", "0.5x"],
            "clip_normalization": True,
            "uperhead_output": [512, 512],
            "pixel_cross_entropy": True,
            "ema": 0.999,
            "optimizer_and_schedule": True,
            "augmentation": True,
            "class_count": expected_classes,
        },
        "intentional_protocol_difference": (
            "This project selects checkpoints on val by default; the original "
            "FaRL trainer evaluates test every epoch. Set "
            "protocol.selection_split=test for historical reproduction."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit this configuration against the local official FaRL source."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    print(json.dumps(verify(config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
