from __future__ import annotations

import contextlib
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import nn

from .config import WANDB_API_KEY, to_plain_dict


LOGGER = logging.getLogger("dime_vit")


def distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not distributed_ready() or dist.get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:

    while True:
        if hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:

    wrappers = ("module.", "_orig_mod.", "student.")
    cleaned: dict[str, torch.Tensor] = {}
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in wrappers:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        cleaned[key] = value
    return cleaned


def load_model_weights(
    model: nn.Module,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:

    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    incompatible = unwrap_model(model).load_state_dict(
        clean_state_dict(state_dict), strict=strict
    )
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def build_optimizer(
    model: nn.Module, config: Mapping[str, Any]
) -> torch.optim.Optimizer:

    if str(config["name"]).lower() != "adamw":
        raise ValueError("Only AdamW is supported")

    raw_model = unwrap_model(model)
    explicit_skip = set()
    if hasattr(raw_model, "no_weight_decay"):
        explicit_skip.update(raw_model.no_weight_decay())

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    position_names = ("pos_embed", "position_bias", "relative_position", "mask_token")
    for name, parameter in raw_model.named_parameters():
        if not parameter.requires_grad:
            continue
        skip_by_name = name in explicit_skip or any(
            name.endswith(f".{skip_name}") or name == skip_name
            for skip_name in explicit_skip
        )
        if (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or skip_by_name
            or any(token in name for token in position_names)
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups = [
        {"params": decay, "weight_decay": float(config["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=float(config["lr"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
    )


class WarmupCosineScheduler:

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_updates: int,
        warmup_updates: int,
        min_lr: float = 0.0,
    ) -> None:
        if total_updates < 1:
            raise ValueError("total_updates must be positive")
        self.optimizer = optimizer
        self.total_updates = int(total_updates)
        self.warmup_updates = min(int(warmup_updates), self.total_updates)
        self.min_lr = float(min_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.num_updates = 0

    def _lr_at(self, base_lr: float, update_index: int) -> float:
        if self.warmup_updates and update_index < self.warmup_updates:
            return base_lr * float(update_index + 1) / self.warmup_updates
        cosine_updates = self.total_updates - self.warmup_updates
        if cosine_updates <= 1:
            return self.min_lr
        progress = (update_index - self.warmup_updates) / (cosine_updates - 1)
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def step_update(self, update_index: int | None = None) -> list[float]:
        if update_index is None:
            update_index = self.num_updates
        lrs = [self._lr_at(base_lr, int(update_index)) for base_lr in self.base_lrs]
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        self.num_updates = int(update_index) + 1
        return lrs

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_updates": self.num_updates,
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:

        self.num_updates = int(state.get("num_updates", 0))
        saved_base_lrs = state.get("base_lrs")
        if saved_base_lrs is not None and len(saved_base_lrs) == len(self.base_lrs):
            self.base_lrs = [float(value) for value in saved_base_lrs]


def create_grad_scaler(amp_dtype: str, device: torch.device):
    enabled = amp_dtype == "fp16" and device.type == "cuda"
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype in {"none", "fp32"}:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    if device.type == "cpu" and dtype == torch.float16:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _all_ranks_finite(loss: torch.Tensor) -> bool:
    finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
    if distributed_ready():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def _reduce_sums(
    sums: Mapping[str, float], sample_count: int, device: torch.device
) -> dict[str, float]:
    names = sorted(sums)
    values = [float(sums[name]) for name in names] + [float(sample_count)]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed_ready():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    denominator = max(float(tensor[-1].item()), 1.0)
    return {
        name: float(tensor[index].item() / denominator)
        for index, name in enumerate(names)
    }


def _extract_images(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, Mapping):
        for key in ("images", "image", "inputs"):
            if key in batch:
                return batch[key]
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise TypeError("A training batch must contain an image tensor")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    epoch: int,
    *,
    accum_iter: int,
    amp_dtype: str,
    scaler=None,
    grad_clip_norm: float | None = None,
    global_update: int = 0,
    log_freq: int = 100,
    wandb_logger=None,
) -> tuple[dict[str, float], int]:

    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)
    if num_batches == 0:
        raise RuntimeError("The training loader is empty")

    tracked_names = ("loss", "loss_rgb", "loss_diff", "loss_edds", "mask_ratio")
    sums = {name: 0.0 for name in tracked_names}
    sample_count = 0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    raw_model = unwrap_model(model)
    lambda_diff = float(getattr(raw_model, "lambda_diff", 1.0))
    lambda_edds = float(getattr(raw_model, "lambda_edds", 0.0))

    edds_active = lambda_edds > 0.0 and epoch >= int(
        getattr(raw_model, "edds_warmup_epochs", 0)
    )
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for micro_step, batch in enumerate(loader):
        images = _extract_images(batch).to(device, non_blocking=True)
        batch_size = images.shape[0]
        group_start = (micro_step // accum_iter) * accum_iter
        group_size = min(accum_iter, num_batches - group_start)
        should_update = (micro_step - group_start + 1) == group_size

        sync_context = contextlib.nullcontext()
        if not should_update and hasattr(model, "no_sync"):
            sync_context = model.no_sync()

        with sync_context:
            with autocast_context(device, amp_dtype):
                output = model(images, edds_active=edds_active)
                if not isinstance(output, Mapping) or "loss" not in output:
                    raise TypeError(
                        "model.forward must return a mapping containing 'loss'"
                    )
                loss = output["loss"]

            if not _all_ranks_finite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, micro-batch {micro_step + 1}"
                )
            scaled_loss = loss / group_size
            sampler = getattr(loader, "batch_sampler", None)
            if hasattr(sampler, "num_samples") and hasattr(sampler, "batch_size"):
                group_samples = min(
                    group_size * sampler.batch_size,
                    sampler.num_samples - group_start * sampler.batch_size,
                )
                if group_samples != group_size * sampler.batch_size:
                    scaled_loss = loss * (batch_size / group_samples)
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()

        for name in tracked_names[:-1]:
            value = output.get(name)
            if value is not None:
                sums[name] += float(value.detach()) * batch_size
        mask = output.get("mask")
        if torch.is_tensor(mask):
            sums["mask_ratio"] += float(mask.detach().float().mean()) * batch_size
        sample_count += batch_size

        if should_update:
            scheduler.step_update(global_update)
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = None
            if grad_clip_norm is not None and grad_clip_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), float(grad_clip_norm)
                )
                if not _all_ranks_finite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient at optimizer update {global_update + 1}"
                    )
                grad_norm_sum += float(grad_norm)
                grad_norm_count += 1
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1

            if log_freq > 0 and (global_update == 1 or global_update % log_freq == 0):
                current = _reduce_sums(sums, sample_count, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = max(time.perf_counter() - start_time, 1.0e-6)
                world_size = dist.get_world_size() if distributed_ready() else 1
                current["lr"] = scheduler.get_last_lr()[0]
                current["images_per_second"] = sample_count * world_size / elapsed
                current["loss_diff_weighted"] = current["loss_diff"] * lambda_diff
                current["loss_edds_weighted"] = current["loss_edds"] * lambda_edds
                current["epoch"] = epoch + 1
                current["epoch_progress"] = (micro_step + 1) / num_batches
                if grad_norm is not None:
                    current["grad_norm"] = float(grad_norm)
                if device.type == "cuda":
                    gib = 1024.0**3
                    current["memory_allocated_gib"] = (
                        torch.cuda.memory_allocated(device) / gib
                    )
                    current["memory_reserved_gib"] = (
                        torch.cuda.memory_reserved(device) / gib
                    )
                    current["peak_memory_gib"] = (
                        torch.cuda.max_memory_allocated(device) / gib
                    )
                if is_main_process():
                    LOGGER.info(
                        "epoch %d | update %d | loss %.5f | rgb %.5f | "
                        "diff %.5f | edds %.5f | mask %.3f | lr %.3e | grad %.3f | %.1f img/s",
                        epoch + 1,
                        global_update,
                        current["loss"],
                        current["loss_rgb"],
                        current["loss_diff"],
                        current["loss_edds"],
                        current["mask_ratio"],
                        current["lr"],
                        float(grad_norm) if grad_norm is not None else float("nan"),
                        current["images_per_second"],
                    )
                    if wandb_logger is not None:
                        wandb_logger.log(
                            {f"train/{key}": value for key, value in current.items()},
                            step=global_update,
                        )

    metrics = _reduce_sums(sums, sample_count, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics["lr"] = scheduler.get_last_lr()[0]
    metrics["epoch_seconds"] = time.perf_counter() - start_time
    metrics["images_per_second"] = (
        sample_count
        * (dist.get_world_size() if distributed_ready() else 1)
        / max(metrics["epoch_seconds"], 1.0e-6)
    )
    metrics["loss_diff_weighted"] = metrics["loss_diff"] * lambda_diff
    metrics["loss_edds_weighted"] = metrics["loss_edds"] * lambda_edds
    if grad_norm_count:
        metrics["grad_norm"] = grad_norm_sum / grad_norm_count
    if device.type == "cuda":
        metrics["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024.0**3
    return metrics, global_update


class CheckpointManager:

    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_freq: int,
        keep_last_n: int,
        maximize: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = int(save_freq)
        self.keep_last_n = int(keep_last_n)
        self.maximize = bool(maximize)
        self.best_value = -math.inf if maximize else math.inf

    def _is_better(self, value: float) -> bool:
        return value > self.best_value if self.maximize else value < self.best_value

    @staticmethod
    def _atomic_save(checkpoint: Mapping[str, Any], path: Path) -> None:
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    def save(
        self,
        *,
        epoch: int,
        global_update: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupCosineScheduler,
        scaler,
        config: Mapping[str, Any],
        metrics: Mapping[str, float],
        best_value: float | None,
        is_last_epoch: bool = False,
    ) -> None:
        is_new_best = False
        if best_value is not None and math.isfinite(float(best_value)):
            value = float(best_value)
            is_new_best = self._is_better(value)
            if is_new_best:
                self.best_value = value

        checkpoint = {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch),
            "global_update": int(global_update),
            "best_value": self.best_value,
            "metrics": dict(metrics),
            "config": to_plain_dict(config),
        }

        self._atomic_save(checkpoint, self.output_dir / "checkpoint-latest.pth")

        periodic = self.save_freq > 0 and (
            (epoch + 1) % self.save_freq == 0 or is_last_epoch
        )
        if periodic:
            path = self.output_dir / f"checkpoint-epoch-{epoch + 1:04d}.pth"
            self._atomic_save(checkpoint, path)
            self._remove_old_periodic()

        if is_new_best:
            self._atomic_save(checkpoint, self.output_dir / "checkpoint-best.pth")
            LOGGER.info("new best checkpoint: %.6f", self.best_value)

    def _remove_old_periodic(self) -> None:
        if self.keep_last_n <= 0:
            return
        checkpoints = sorted(self.output_dir.glob("checkpoint-epoch-*.pth"))
        for path in checkpoints[: -self.keep_last_n]:
            path.unlink()


def resume_training(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler=None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_model_weights(model, checkpoint, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


class WandBLogger:

    def __init__(
        self, config: Mapping[str, Any], full_config: Mapping[str, Any]
    ) -> None:
        self.enabled = bool(config["enabled"]) and is_main_process()
        self._wandb = None
        self._run = None
        self._config = config
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb.enabled=true, but wandb is not installed"
            ) from exc
        self._wandb = wandb
        try:
            logged_in = wandb.login(key=WANDB_API_KEY, verify=True)
        except Exception as exc:
            raise RuntimeError("W&B authentication failed") from exc
        if not logged_in:
            raise RuntimeError("W&B authentication failed")
        self._run = wandb.init(
            project=config["project"],
            entity=config.get("entity"),
            name=config.get("run_name"),
            tags=list(config.get("tags", [])),
            config=to_plain_dict(full_config),
        )
        self._run.define_metric("optimizer_step")
        self._run.define_metric("train/*", step_metric="optimizer_step")
        self._run.define_metric("epoch/*", step_metric="optimizer_step")
        self._run.define_metric("eval/*", step_metric="optimizer_step")
        self._run.define_metric("eval/mse", step_metric="optimizer_step", summary="min")
        self._run.define_metric(
            "eval/psnr", step_metric="optimizer_step", summary="max"
        )
        self._run.define_metric(
            "eval/ssim", step_metric="optimizer_step", summary="max"
        )

    def watch(self, model: nn.Module) -> None:
        if self.enabled and bool(self._config.get("watch_model", False)):
            self._run.watch(
                unwrap_model(model),
                log=self._config.get("watch_log", "gradients"),
                log_freq=int(self._config.get("watch_freq", 1000)),
                log_graph=False,
            )

    def set_summary(self, values: Mapping[str, Any]) -> None:
        if self.enabled:
            self._run.summary.update(dict(values))

    def log(self, values: Mapping[str, Any], step: int) -> None:
        if self.enabled:
            payload = dict(values)
            payload["optimizer_step"] = int(step)
            self._run.log(payload)

    def log_reconstructions(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        mask: torch.Tensor,
        mixed: torch.Tensor,
        reconstruction: torch.Tensor,
        error: torch.Tensor,
        *,
        step: int,
    ) -> None:
        if not self.enabled:
            return
        panels = []
        for index in range(target.shape[0]):
            panel = (
                torch.cat(
                    [
                        target[index],
                        source[index],
                        mask[index],
                        mixed[index],
                        reconstruction[index],
                        error[index],
                    ],
                    dim=-1,
                )
                .detach()
                .cpu()
            )
            panels.append(
                self._wandb.Image(
                    panel,
                    caption=(
                        f"sample {index}: target | partner | mask | mixed | "
                        "reconstruction | 3x absolute error"
                    ),
                )
            )
        self._run.log({"eval/reconstructions": panels, "optimizer_step": int(step)})

    def finish(self) -> None:
        if self.enabled:
            self._run.finish()
