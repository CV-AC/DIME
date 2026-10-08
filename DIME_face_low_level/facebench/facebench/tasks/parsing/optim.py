from __future__ import annotations

from typing import Any

import torch


def _trainable(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def build_optimizer(
    model: torch.nn.Module,
    protocol: dict[str, Any],
) -> torch.optim.Optimizer:

    encoder = _trainable(model.encoder)
    decoder = [
        parameter for module in model.decoder for parameter in _trainable(module)
    ]
    if not encoder or not decoder:
        raise RuntimeError(
            "Full fine-tuning requires trainable encoder and decoder parameters."
        )
    if set(map(id, encoder)).intersection(map(id, decoder)):
        raise RuntimeError("Encoder and decoder parameter groups overlap.")

    encoder_lr = float(protocol["encoder_lr"])
    head_lr = float(protocol["head_lr"])
    weight_decay = float(protocol.get("weight_decay", 1e-5))
    betas = tuple(float(value) for value in protocol.get("betas", (0.9, 0.999)))
    if encoder_lr <= 0.0 or head_lr <= 0.0:
        raise ValueError("encoder_lr and head_lr must be positive.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if len(betas) != 2:
        raise ValueError("betas must contain exactly two numbers.")

    name = str(protocol.get("optimizer", "adamw")).strip().lower()
    groups = [
        {"params": encoder, "lr": encoder_lr, "name": "encoder"},
        {"params": decoder, "lr": head_lr, "name": "head"},
    ]
    if name == "adamw":
        return torch.optim.AdamW(
            groups,
            betas=betas,
            eps=float(protocol.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            groups,
            momentum=float(protocol.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(protocol.get("nesterov", True)),
        )
    raise ValueError("protocol.optimizer must be adamw or sgd.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    protocol: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler:
    name = str(protocol.get("scheduler", "multistep")).strip().lower()
    if name == "multistep":
        milestones = [int(value) for value in protocol.get("milestones", (200,))]
        if not milestones or any(value <= 0 for value in milestones):
            raise ValueError("protocol.milestones must contain positive epochs.")
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(protocol.get("gamma", 0.1)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(protocol["epochs"]),
            eta_min=float(protocol.get("min_lr", 0.0)),
        )
    raise ValueError("protocol.scheduler must be multistep or cosine.")


def learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    return {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
