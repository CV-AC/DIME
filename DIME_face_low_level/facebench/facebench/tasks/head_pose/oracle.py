from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .data import build_dataset
from .rotation import rotation_mae
from .utils import autocast_context


ORACLE_SELECTION = "oracle_best_on_test"
FIXED_FINAL_SELECTION = "fixed_final_epoch_ema"
SUPPORTED_SELECTIONS = {ORACLE_SELECTION, FIXED_FINAL_SELECTION}


def selection_mode(config: dict[str, Any]) -> str:
    mode = (
        str(config.get("evaluation", {}).get("selection", ORACLE_SELECTION))
        .strip()
        .lower()
    )
    if mode not in SUPPORTED_SELECTIONS:
        raise ValueError(
            f"Unsupported evaluation.selection={mode!r}; expected one of "
            f"{sorted(SUPPORTED_SELECTIONS)}."
        )
    return mode


def default_checkpoint_name(config: dict[str, Any]) -> str:
    return (
        "best_oracle_ema.pth"
        if selection_mode(config) == ORACLE_SELECTION
        else "final_ema.pth"
    )


def build_oracle_loaders(
    config: dict[str, Any],
    *,
    rank: int,
    world_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, DataLoader]:
    evaluation = config.get("evaluation", {})
    workers = int(evaluation.get("num_workers_per_gpu", 2))
    if workers < 0:
        raise ValueError("evaluation.num_workers_per_gpu must be non-negative.")
    batch_size = int(
        evaluation.get("batch_size_per_gpu", evaluation.get("batch_size", 64))
    )
    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive.")

    loaders: dict[str, DataLoader] = {}
    for split_index, split in enumerate(("aflw2000", "biwi")):
        dataset = build_dataset(config, split)

        shard = Subset(dataset, range(rank, len(dataset), world_size))
        generator = torch.Generator().manual_seed(
            int(seed) + 10_000 + 1_000 * split_index + rank
        )
        loaders[split] = DataLoader(
            shard,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            generator=generator,
        )
    return loaders


@torch.inference_mode()
def evaluate_oracle(
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: str,
) -> tuple[dict[str, dict[str, float | int]], float]:
    model.eval()
    metrics: dict[str, dict[str, float | int]] = {}
    for split, loader in loaders.items():
        totals = torch.zeros(4, device=device, dtype=torch.float64)
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with autocast_context(amp_enabled, amp_dtype):
                rotations = model(images)
            _, _, errors = rotation_mae(
                rotations.float(),
                batch["ypr"].to(device, non_blocking=True).float(),
            )
            totals[:3] += errors.double().sum(dim=0)
            totals[3] += errors.shape[0]
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[3].item() <= 0:
            raise RuntimeError(f"Oracle evaluation split {split!r} is empty.")
        per_axis = totals[:3] / totals[3]
        metrics[split] = {
            "samples": int(totals[3].item()),
            "yaw_mae": per_axis[0].item(),
            "pitch_mae": per_axis[1].item(),
            "roll_mae": per_axis[2].item(),
            "mean_mae": per_axis.mean().item(),
        }

    score = sum(float(metrics[split]["mean_mae"]) for split in loaders) / len(loaders)
    return metrics, score
