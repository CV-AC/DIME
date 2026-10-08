from __future__ import annotations

import math
from bisect import bisect_right
from typing import Any

import torch


def _trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def build_optimizer(
    model: torch.nn.Module,
    protocol: dict[str, Any],
) -> torch.optim.Optimizer:

    encoder = _trainable_parameters(model.backbone)
    downstream_modules = (
        model.downstream_modules
        if hasattr(model, "downstream_modules")
        else (model.pyramid, model.head)
    )
    downstream = [
        parameter
        for module in downstream_modules
        for parameter in _trainable_parameters(module)
    ]
    if not encoder or not downstream:
        raise RuntimeError(
            "Full fine-tuning requires trainable encoder and downstream parameters."
        )

    encoder_lr = float(protocol["encoder_lr"])
    head_lr = float(protocol["head_lr"])
    if encoder_lr <= 0.0 or head_lr <= 0.0:
        raise ValueError("encoder_lr and head_lr must be positive.")
    parameter_groups = [
        {"params": encoder, "lr": encoder_lr, "name": "encoder"},
        {"params": downstream, "lr": head_lr, "name": "head"},
    ]

    options = dict(protocol.get("optimizer", {}))
    name = str(options.get("name", "adamw")).strip().lower()
    weight_decay = float(
        options.get("weight_decay", protocol.get("weight_decay", 1e-5))
    )
    if weight_decay < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative.")

    if name in {"adamw", "adam"}:
        betas = tuple(
            float(value)
            for value in options.get("betas", protocol.get("betas", (0.9, 0.999)))
        )
        if len(betas) != 2 or not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("optimizer.betas must contain two values in [0, 1).")
        eps = float(options.get("eps", 1e-8))
        if eps <= 0.0:
            raise ValueError("optimizer.eps must be positive.")
        optimizer_type = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return optimizer_type(
            parameter_groups,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=bool(options.get("amsgrad", False)),
        )

    if name == "sgd":
        momentum = float(options.get("momentum", 0.9))
        dampening = float(options.get("dampening", 0.0))
        nesterov = bool(options.get("nesterov", False))
        if momentum < 0.0 or dampening < 0.0:
            raise ValueError("optimizer momentum and dampening must be non-negative.")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov SGD requires momentum > 0 and dampening = 0.")
        return torch.optim.SGD(
            parameter_groups,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    raise ValueError(
        f"Unsupported optimizer.name={name!r}; choose adamw, adam, or sgd."
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    protocol: dict[str, Any],
    *,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:

    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")

    options = dict(protocol.get("scheduler", {}))
    name = str(options.get("name", "multistep")).strip().lower()
    aliases = {"none": "constant", "cosineannealing": "cosine"}
    name = aliases.get(name, name)
    supported = {"constant", "multistep", "cosine", "linear"}
    if name not in supported:
        raise ValueError(
            f"Unsupported scheduler.name={name!r}; choose "
            "constant, multistep, cosine, or linear."
        )

    warmup_epochs = int(options.get("warmup_epochs", 0))
    warmup_start_factor = float(options.get("warmup_start_factor", 0.01))
    if warmup_epochs < 0 or warmup_epochs >= total_epochs:
        raise ValueError("scheduler.warmup_epochs must be in [0, total_epochs).")
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("scheduler.warmup_start_factor must be in (0, 1].")

    milestones = sorted(
        int(value)
        for value in options.get("milestones", protocol.get("lr_milestones", (200,)))
    )
    if any(value <= 0 for value in milestones):
        raise ValueError("scheduler.milestones must contain positive epochs.")
    gamma = float(options.get("gamma", protocol.get("lr_gamma", 0.1)))
    if gamma <= 0.0:
        raise ValueError("scheduler.gamma must be positive.")
    min_lr_factor = float(options.get("min_lr_factor", 0.0))
    if not 0.0 <= min_lr_factor <= 1.0:
        raise ValueError("scheduler.min_lr_factor must be in [0, 1].")

    if name == "multistep" and warmup_epochs == 0:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    def factor(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            progress = epoch / warmup_epochs
            return warmup_start_factor + (1.0 - warmup_start_factor) * progress

        if name == "constant":
            return 1.0
        if name == "multistep":
            return gamma ** bisect_right(milestones, epoch)

        decay_epochs = max(total_epochs - warmup_epochs - 1, 1)
        progress = min(max((epoch - warmup_epochs) / decay_epochs, 0.0), 1.0)
        if name == "linear":
            return 1.0 - (1.0 - min_lr_factor) * progress
        return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
