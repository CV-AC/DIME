from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .geometry import inverse_transform_map, remap
from .labels import LabelSpace
from .metrics import ConfusionMatrix
from .utils import autocast_context, is_main_process, unwrap_model


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1).")
        self.decay = float(decay)
        self.module = deepcopy(unwrap_model(model)).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = unwrap_model(model)
        source_parameters = dict(source.named_parameters())
        grouped: dict[
            tuple[torch.device, torch.dtype],
            tuple[list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        for name, parameter in self.module.named_parameters():
            target_group, source_group = grouped.setdefault(
                (parameter.device, parameter.dtype), ([], [])
            )
            target_group.append(parameter)
            source_group.append(source_parameters[name].detach())
        for target_group, source_group in grouped.values():
            torch._foreach_mul_(target_group, self.decay)
            torch._foreach_add_(target_group, source_group, alpha=1.0 - self.decay)
        source_buffers = dict(source.named_buffers())
        for name, buffer in self.module.named_buffers():
            buffer.copy_(source_buffers[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state, strict=True)


def _distributed_mean(total: float, count: int, device: torch.device) -> float:
    values = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0] / values[1].clamp_min(1.0))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema: ModelEMA,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: str,
    gradient_clip_norm: float,
    log_interval: int,
    progress: Callable[[str], None] | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    sample_count = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast_context(amp_enabled, amp_dtype):
            logits = model(images)
            loss = F.cross_entropy(logits.float(), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch {step}: {float(loss)}"
            )
        scaler.scale(loss).backward()
        if gradient_clip_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

        batch_size = int(images.shape[0])
        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
        if (
            progress is not None
            and is_main_process()
            and log_interval > 0
            and (step % log_interval == 0 or step == len(loader))
        ):
            elapsed = max(time.perf_counter() - started, 1e-6)
            progress(
                f"batch {step}/{len(loader)} | "
                f"loss={total_loss / max(sample_count, 1):.5f} | "
                f"{sample_count / elapsed:.1f} samples/s"
            )

    return {
        "loss": _distributed_mean(total_loss, sample_count, device),
        "samples": float(sample_count),
    }


def _inverse_logits(
    logits: np.ndarray,
    matrix: np.ndarray,
    original_shape: tuple[int, int],
    *,
    canvas_size: int,
    warp_factor: float,
) -> np.ndarray:
    transform_map = inverse_transform_map(
        matrix,
        original_shape,
        canvas_size=canvas_size,
        warp_factor=warp_factor,
    )
    restored = np.stack(
        [
            remap(
                class_logits,
                transform_map,
                interpolation=cv2.INTER_LINEAR,
                border_value=0.0,
            )
            for class_logits in logits
        ],
        axis=0,
    )
    return restored.argmax(axis=0).astype(np.uint8)


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    label_space: LabelSpace,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: str,
    canvas_size: int,
    warp_factor: float,
    prediction_dir: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    confusion = ConfusionMatrix(label_space.num_classes)
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with autocast_context(amp_enabled, amp_dtype):
            logits = model(images)
        logits_numpy = logits.float().cpu().numpy()
        for index, sample_id in enumerate(batch["sample_id"]):
            prediction = _inverse_logits(
                logits_numpy[index],
                np.asarray(batch["transform"][index]),
                tuple(batch["original_shape"][index]),
                canvas_size=canvas_size,
                warp_factor=warp_factor,
            )
            target = np.asarray(batch["label_original"][index])
            confusion.update(target, prediction)
            if prediction_dir is not None:
                output = prediction_dir / f"{sample_id}.png"
                if not cv2.imwrite(str(output), prediction):
                    raise OSError(f"Could not write prediction {output}.")

    confusion.distributed_reduce(device)
    metrics = confusion.summarize(label_space.names)
    if dist.is_initialized():
        values: list[dict[str, Any] | None] = [metrics if is_main_process() else None]
        dist.broadcast_object_list(values, src=0)
        assert values[0] is not None
        metrics = values[0]
    return metrics
