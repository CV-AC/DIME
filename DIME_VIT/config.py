from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "name": "vit_small_patch16",
        "img_size": 224,
        "mask_ratio": 0.5,
        "mask_cell_size": 1,
        "shared_mask": True,
        "range_mask_ratio": 0.0,
        "mask_strategy": "single",
        "mask_block_sizes": [1],
        "block_probs": None,
        "pos_encoding": "rope",
        "attention_mode": "auto",
        "gated_attention": False,
        "gate_init_bias": 2.0,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "use_checkpoint": False,
    },
    "loss": {
        "norm_pix_loss": True,
        "lambda_diff": 0.5,
        "sobel_q": 0.5,
        "lambda_edds": 1.0,
        "edds_warmup_epochs": 20,
        "edds_version": "v2",
    },
    "data": {
        "path": None,
        "batch_size": 128,
        "num_workers": 8,
        "pin_memory": True,
        "subset_ratio": 1.0,
        "pair_sampling": "identity_uniform",
        "high_res_batch_size": None,
        "transform": {
            "train_crop_scale": [0.2, 1.0],
            "train_crop_ratio": [0.75, 1.3333333333333333],
            "train_hflip_prob": 0.5,
            "eval_crop_pct": 0.875,
            "interpolation": "bicubic",
            "antialias": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "train": {
        "epochs": 500,
        "stop_epoch": None,
        "high_res_start_epoch": 400,
        "high_res_size": 512,
        "accum_iter": 1,
        "seed": 42,
        "amp_dtype": "bf16",
        "grad_clip_norm": 5.0,
        "log_freq": 100,
    },
    "optimizer": {
        "name": "adamw",
        "lr": None,
        "base_lr": 8.0e-5,
        "reference_batch_size": 256,
        "weight_decay": 0.05,
        "betas": [0.9, 0.95],
        "eps": 1.0e-6,
    },
    "scheduler": {
        "warmup_epochs": 40,
        "min_lr": 0.0,
    },
    "compile": {
        "enabled": False,
        "mode": "default",
        "dynamic": False,
        "fullgraph": False,
    },
    "distributed": {
        "nodes": 8,
        "gpus_per_node": 8,
        "backend": "nccl",
        "find_unused_parameters": False,
    },
    "checkpoint": {
        "output_dir": "outputs/vit_small_patch16",
        "resume": None,
        "init_checkpoint": None,
        "save_freq": 20,
        "keep_last_n": 3,
        "best_metric": "mse",
        "maximize_best_metric": False,
    },
    "eval": {
        "enabled": True,
        "freq": 20,
        "num_pairs": 32,
        "batch_size": 16,
        "num_workers": 4,
        "seed": 0,
        "num_visuals": 8,
    },
    "wandb": {
        "enabled": False,
        "project": "dime-vit",
        "entity": None,
        "run_name": None,
        "tags": ["dime", "vit", "pretrain"],
        "watch_model": False,
        "watch_log": "gradients",
        "watch_freq": 1000,
    },
}


class ConfigNode(dict):

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__


def _to_node(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigNode({key: _to_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_node(item) for item in value]
    return value


def to_plain_dict(value: Any) -> Any:

    if isinstance(value, Mapping):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]
    return value


def _deep_merge(
    base: dict[str, Any], update: Mapping[str, Any], prefix: str = ""
) -> None:
    for key, value in update.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise KeyError(f"Unknown configuration key: {path}")
        if isinstance(base[key], dict) and isinstance(value, Mapping):
            _deep_merge(base[key], value, path)
        else:
            base[key] = value


def _parse_overrides(opts: Iterable[str] | None) -> list[tuple[str, Any]]:
    tokens = list(opts or [])
    parsed: list[tuple[str, Any]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if "=" in token:
            key, raw_value = token.split("=", 1)
            index += 1
        else:
            if index + 1 >= len(tokens):
                raise ValueError(f"Missing value for override '{token}'")
            key, raw_value = token, tokens[index + 1]
            index += 2
        parsed.append((key, yaml.safe_load(raw_value)))
    return parsed


def _set_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    node: dict[str, Any] = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"Unknown configuration key: {path}")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"Unknown configuration key: {path}")
    node[keys[-1]] = value


def _image_size_pair(value: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("model.img_size must be an integer or [height, width]")
    return int(value[0]), int(value[1])


def validate_config(config: Mapping[str, Any]) -> None:
    for name in ("nodes", "gpus_per_node"):
        value = config["distributed"][name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"distributed.{name} must be a positive integer")
    model = config["model"]
    data = config["data"]
    train = config["train"]

    height, width = _image_size_pair(model["img_size"])
    mask_cell_size = int(model["mask_cell_size"])
    if mask_cell_size not in {1, 2}:
        raise ValueError("model.mask_cell_size must be 1 or 2")
    required_divisor = 16 * mask_cell_size
    if height % required_divisor or width % required_divisor:
        raise ValueError(
            "model.img_size must be divisible by 16 * mask_cell_size "
            f"({required_divisor}); received {(height, width)}"
        )
    if not 0.0 < float(model["mask_ratio"]) < 1.0:
        raise ValueError("model.mask_ratio must be between 0 and 1")
    range_mask_ratio = float(model["range_mask_ratio"])
    if not range_mask_ratio >= 0.0:
        raise ValueError("model.range_mask_ratio must be non-negative")
    if model["mask_strategy"] not in {"single", "multiscale"}:
        raise ValueError("model.mask_strategy must be single or multiscale")
    block_sizes = model["mask_block_sizes"]
    if not isinstance(block_sizes, (list, tuple)) or not block_sizes:
        raise ValueError("model.mask_block_sizes must be a non-empty list")
    if any(float(size) < 1 or not float(size).is_integer() for size in block_sizes):
        raise ValueError("model.mask_block_sizes must contain positive integers")
    block_sizes = [int(size) for size in block_sizes]
    block_probs = model["block_probs"]
    if block_probs is not None:
        if len(block_probs) != len(block_sizes):
            raise ValueError("model.block_probs must match model.mask_block_sizes")
        probabilities = [float(value) for value in block_probs]
        if any(not value >= 0.0 for value in probabilities) or not any(probabilities):
            raise ValueError(
                "model.block_probs must be non-negative with a positive sum"
            )
    gate_init_bias = float(model["gate_init_bias"])
    if not gate_init_bias >= 0.0:
        raise ValueError("model.gate_init_bias must be non-negative")
    decoder_grid = (height // required_divisor, width // required_divisor)
    for size in block_sizes:
        if model["mask_strategy"] == "single" and size > 1:
            valid = min(decoder_grid[0] // size, decoder_grid[1] // size) >= 2
        else:
            coarse_h = (decoder_grid[0] + size - 1) // size
            coarse_w = (decoder_grid[1] + size - 1) // size
            valid = coarse_h * coarse_w >= 2
        if not valid:
            raise ValueError(
                f"model.mask_block_sizes contains {size}, which is too large "
                f"for decoder grid {decoder_grid} and strategy {model['mask_strategy']}"
            )
    if int(data["batch_size"]) < 2 or int(data["batch_size"]) % 2:
        raise ValueError(
            "data.batch_size must be even because batch.flip(0) defines pairs"
        )
    if int(train["accum_iter"]) < 1:
        raise ValueError("train.accum_iter must be at least 1")
    if int(train["epochs"]) < 1:
        raise ValueError("train.epochs must be at least 1")
    if train["stop_epoch"] is not None and not 1 <= int(train["stop_epoch"]) <= int(
        train["epochs"]
    ):
        raise ValueError("train.stop_epoch must be within [1, train.epochs]")
    if train["high_res_start_epoch"] is not None:
        if int(train["high_res_start_epoch"]) < 0:
            raise ValueError("train.high_res_start_epoch must be nonnegative or null")
        high_h, high_w = _image_size_pair(train["high_res_size"])
        if (
            min(high_h, high_w) <= 0
            or high_h % required_divisor
            or high_w % required_divisor
        ):
            raise ValueError("train.high_res_size must be positive and patch aligned")
        high_batch = int(
            data["high_res_batch_size"] or min(32, int(data["batch_size"]))
        )
        effective_local = int(data["batch_size"]) * int(train["accum_iter"])
        if high_batch < 2 or high_batch % 2 or effective_local % high_batch:
            raise ValueError(
                "data.high_res_batch_size must be even and divide batch_size * accum_iter"
            )
    if train["amp_dtype"] not in {"bf16", "fp16", "fp32", "none"}:
        raise ValueError("train.amp_dtype must be bf16, fp16, fp32, or none")
    if config["loss"]["edds_version"] not in {"v1", "v2"}:
        raise ValueError("loss.edds_version must be v1 or v2")
    if not 0.0 < float(data["subset_ratio"]) <= 1.0:
        raise ValueError("data.subset_ratio must be in (0, 1]")
    if data["pair_sampling"] not in {"identity_uniform", "image_disjoint"}:
        raise ValueError(
            "data.pair_sampling must be identity_uniform or image_disjoint"
        )
    transform = data["transform"]
    crop_scale = transform["train_crop_scale"]
    if not isinstance(crop_scale, (list, tuple)) or len(crop_scale) != 2:
        raise ValueError("data.transform.train_crop_scale must contain two values")
    crop_min, crop_max = (float(value) for value in crop_scale)
    if not 0.0 < crop_min <= crop_max <= 1.0:
        raise ValueError(
            "data.transform.train_crop_scale must satisfy 0 < min <= max <= 1"
        )
    crop_ratio = transform["train_crop_ratio"]
    if not isinstance(crop_ratio, (list, tuple)) or len(crop_ratio) != 2:
        raise ValueError("data.transform.train_crop_ratio must contain two values")
    ratio_min, ratio_max = (float(value) for value in crop_ratio)
    if not 0.0 < ratio_min <= ratio_max:
        raise ValueError("data.transform.train_crop_ratio must satisfy 0 < min <= max")
    hflip_prob = float(transform["train_hflip_prob"])
    if not 0.0 <= hflip_prob <= 1.0:
        raise ValueError("data.transform.train_hflip_prob must be in [0, 1]")
    if not 0.0 < float(transform["eval_crop_pct"]) <= 1.0:
        raise ValueError("data.transform.eval_crop_pct must be in (0, 1]")
    if str(transform["interpolation"]).lower() not in {
        "nearest",
        "bilinear",
        "bicubic",
    }:
        raise ValueError(
            "data.transform.interpolation must be nearest, bilinear, or bicubic"
        )
    if not isinstance(transform["antialias"], bool):
        raise ValueError("data.transform.antialias must be true or false")
    for name in ("mean", "std"):
        values = transform[name]
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"data.transform.{name} must contain three values")
    if any(float(value) <= 0.0 for value in transform["std"]):
        raise ValueError("data.transform.std values must be positive")
    evaluation = config["eval"]
    if evaluation["enabled"] and int(evaluation["freq"]) < 1:
        raise ValueError("eval.freq must be at least 1 when evaluation is enabled")
    if int(evaluation["batch_size"]) < 2 or int(evaluation["batch_size"]) % 2:
        raise ValueError("eval.batch_size must be a positive even number")
    if int(evaluation["num_pairs"]) < 1:
        raise ValueError("eval.num_pairs must be at least 1")
    if config["compile"]["mode"] not in {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }:
        raise ValueError("Unsupported torch.compile mode")
    wandb = config["wandb"]
    if wandb["watch_log"] not in {"gradients", "parameters", "all"}:
        raise ValueError("wandb.watch_log must be gradients, parameters, or all")
    if int(wandb["watch_freq"]) < 1:
        raise ValueError("wandb.watch_freq must be at least 1")
    if (
        wandb["enabled"]
        and wandb["watch_model"]
        and config["compile"]["enabled"]
        and config["compile"]["fullgraph"]
    ):
        raise ValueError(
            "wandb.watch_model is incompatible with compile.fullgraph=true"
        )

    stochastic_geometry = range_mask_ratio > 0.0 or (
        model["mask_strategy"] == "single" and len(block_sizes) > 1
    )
    if (
        config["compile"]["enabled"]
        and config["compile"]["fullgraph"]
        and stochastic_geometry
    ):
        raise ValueError(
            "compile.fullgraph=true requires range_mask_ratio=0; the single mask "
            "strategy also requires one mask_block_size"
        )


def load_config(path: str | Path, opts: Iterable[str] | None = None) -> ConfigNode:
    config = copy.deepcopy(DEFAULT_CONFIG)
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError("The YAML root must be a mapping")
    _deep_merge(config, loaded)
    for key, value in _parse_overrides(opts):
        _set_dotted(config, key, value)
    validate_config(config)
    return _to_node(config)


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain_dict(config), handle, sort_keys=False)
