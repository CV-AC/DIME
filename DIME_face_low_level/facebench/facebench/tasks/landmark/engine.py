from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from facebench.common.ema import ModelEMA

from .data import CANVAS_SIZE, WFLWDataset
from .metrics import summarize_metrics
from .utils import autocast_context, gather_objects, is_main_process


def _canvas_to_original(
    normalized_points: torch.Tensor, transforms: torch.Tensor
) -> torch.Tensor:
    canvas = normalized_points.float() * CANVAS_SIZE - 0.5
    ones = torch.ones((*canvas.shape[:-1], 1), device=canvas.device, dtype=canvas.dtype)
    homogeneous = torch.cat([canvas, ones], dim=-1)
    inverse = torch.linalg.inv(transforms.float())
    return torch.bmm(homogeneous, inverse.transpose(1, 2))[..., :2]


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: WFLWDataset,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: str,
    description: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    del description
    model.eval()
    local: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        transforms = batch["transform"].to(device, non_blocking=True)
        with autocast_context(amp_enabled, amp_dtype):
            outputs = model(images)
        predictions = _canvas_to_original(outputs["points"], transforms).cpu().numpy()
        targets = batch["landmarks_original"].numpy()
        flags = batch["subset_flags"].numpy()
        for sample_id, prediction, target, subset_flags in zip(
            batch["sample_id"], predictions, targets, flags
        ):
            local.append(
                {
                    "sample_id": sample_id,
                    "prediction": prediction,
                    "target": target,
                    "subset_flags": subset_flags,
                }
            )

    gathered = gather_objects(local)
    payload: dict[str, np.ndarray] | None = None
    if is_main_process():
        rows = [row for part in gathered for row in part]
        by_id = {row["sample_id"]: row for row in rows}
        if len(by_id) != len(dataset):
            raise RuntimeError(
                f"Evaluation gathered {len(by_id)} unique predictions for {len(dataset)} samples."
            )
        order = [sample.sample_id for sample in dataset.samples]
        ordered = [by_id[sample_id] for sample_id in order]
        payload = {
            "sample_ids": np.asarray(order),
            "predictions": np.stack([row["prediction"] for row in ordered]),
            "targets": np.stack([row["target"] for row in ordered]),
            "subset_flags": np.stack([row["subset_flags"] for row in ordered]),
        }
        metrics = summarize_metrics(
            payload["predictions"], payload["targets"], payload["subset_flags"]
        )
    else:
        metrics = {}
    if dist.is_initialized():
        values = [metrics]
        dist.broadcast_object_list(values, src=0)
        metrics = values[0]
    return metrics, payload


def save_predictions(path: str | Path, payload: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
