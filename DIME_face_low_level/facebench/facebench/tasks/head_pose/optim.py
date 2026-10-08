from __future__ import annotations

import math
from bisect import bisect_right
from typing import Any

import torch
import torch.nn as nn

from .models import HeadPoseModel


def _parameter_groups(
    module: nn.Module,
    *,
    component: str,
    learning_rate: float,
    weight_decay: float,
    exclude_bias_and_norm: bool,
    layer_decay: float = 1.0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, bool], list[nn.Parameter]] = {}
    normalization_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )
    normalization_parameters = {
        id(parameter)
        for submodule in module.modules()
        if isinstance(submodule, normalization_types)
        for parameter in submodule.parameters(recurse=False)
    }
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        skip_decay = exclude_bias_and_norm and (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or id(parameter) in normalization_parameters
        )
        layer_id = (
            int(module.parameter_layer_id(name))
            if hasattr(module, "parameter_layer_id")
            else 0
        )
        grouped.setdefault((layer_id, skip_decay), []).append(parameter)

    groups: list[dict[str, Any]] = []
    num_layers = int(getattr(module, "num_lr_layers", 1))
    if not 0.0 < layer_decay <= 1.0:
        raise ValueError("protocol.encoder_layer_decay must be in (0,1].")
    for (layer_id, skip_decay), parameters in sorted(grouped.items()):
        if not 0 <= layer_id < num_layers:
            raise ValueError(
                f"{component} returned invalid layer id {layer_id}/{num_layers}."
            )
        layer_scale = layer_decay ** (num_layers - layer_id - 1)
        suffix = "no_decay" if skip_decay else "decay"
        groups.append(
            {
                "params": parameters,
                "lr": learning_rate * layer_scale,
                "weight_decay": 0.0 if skip_decay else weight_decay,
                "name": f"{component}/layer_{layer_id}/{suffix}",
                "component": component,
                "layer_id": layer_id,
                "lr_scale": layer_scale,
            }
        )
    return groups


def build_optimizer(
    model: HeadPoseModel, config: dict[str, Any]
) -> torch.optim.Optimizer:
    protocol = config["protocol"]
    options = dict(protocol.get("optimizer", {}))
    name = str(options.get("name", "adamw")).strip().lower()
    encoder_lr = float(protocol["encoder_lr"])
    head_lr = float(protocol["head_lr"])
    weight_decay = float(options.get("weight_decay", 0.05))
    exclude = bool(options.get("exclude_bias_and_norm_from_weight_decay", True))
    if encoder_lr <= 0.0 or head_lr <= 0.0:
        raise ValueError("protocol.encoder_lr and protocol.head_lr must be positive.")
    if weight_decay < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative.")

    groups: list[dict[str, Any]] = []
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        groups.extend(
            _parameter_groups(
                model.encoder,
                component="encoder",
                learning_rate=encoder_lr,
                weight_decay=weight_decay,
                exclude_bias_and_norm=exclude,
                layer_decay=float(protocol.get("encoder_layer_decay", 1.0)),
            )
        )
    groups.extend(
        _parameter_groups(
            model.head,
            component="head",
            learning_rate=head_lr,
            weight_decay=weight_decay,
            exclude_bias_and_norm=exclude,
        )
    )
    if not groups:
        raise RuntimeError("The model has no trainable parameters.")

    if name in {"adamw", "adam"}:
        betas = tuple(float(value) for value in options.get("betas", (0.9, 0.999)))
        if len(betas) != 2 or not all(0.0 <= value < 1.0 for value in betas):
            raise ValueError("optimizer.betas must contain two values in [0,1).")
        eps = float(options.get("eps", 1e-8))
        optimizer_type = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return optimizer_type(
            groups,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
            amsgrad=bool(options.get("amsgrad", False)),
        )
    if name == "sgd":
        return torch.optim.SGD(
            groups,
            lr=head_lr,
            momentum=float(options.get("momentum", 0.9)),
            nesterov=bool(options.get("nesterov", True)),
            weight_decay=0.0,
        )
    raise ValueError(f"Unsupported optimizer.name={name!r}.")


def component_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    by_component: dict[str, list[float]] = {}
    for group in optimizer.param_groups:
        component = str(group.get("component", group.get("name", "parameters")))
        by_component.setdefault(component, []).append(float(group["lr"]))
    rates: dict[str, float] = {}
    for component, values in by_component.items():
        rates[component] = max(values)
        if not all(math.isclose(value, values[0]) for value in values):
            rates[f"{component}_min"] = min(values)
    return rates


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    protocol: dict[str, Any],
    *,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    options = dict(protocol.get("scheduler", {}))
    name = str(options.get("name", "cosine")).strip().lower()
    if name not in {"constant", "multistep", "cosine", "linear"}:
        raise ValueError(f"Unsupported scheduler.name={name!r}.")
    warmup_epochs = int(options.get("warmup_epochs", 0))
    warmup_start = float(options.get("warmup_start_factor", 0.01))
    min_factor = float(options.get("min_lr_factor", 0.01))
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError("scheduler.warmup_epochs must be in [0,total_epochs).")
    if not 0.0 < warmup_start <= 1.0:
        raise ValueError("scheduler.warmup_start_factor must be in (0,1].")
    if not 0.0 <= min_factor <= 1.0:
        raise ValueError("scheduler.min_lr_factor must be in [0,1].")
    milestones = sorted(int(value) for value in options.get("milestones", ()))
    gamma = float(options.get("gamma", 0.1))

    def factor(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            progress = epoch / warmup_epochs
            return warmup_start + (1.0 - warmup_start) * progress
        if name == "constant":
            return 1.0
        if name == "multistep":
            return gamma ** bisect_right(milestones, epoch)
        decay_epochs = max(total_epochs - warmup_epochs - 1, 1)
        progress = min(max((epoch - warmup_epochs) / decay_epochs, 0.0), 1.0)
        if name == "linear":
            return 1.0 - (1.0 - min_factor) * progress
        return min_factor + 0.5 * (1.0 - min_factor) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
