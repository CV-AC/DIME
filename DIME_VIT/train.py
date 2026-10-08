from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import load_config, save_config, to_plain_dict
from .data import build_eval_loader, build_train_loader
from .engine import (
    CheckpointManager,
    LOGGER,
    WandBLogger,
    WarmupCosineScheduler,
    build_optimizer,
    create_grad_scaler,
    load_model_weights,
    resume_training,
    train_one_epoch,
)
from .evaluate import evaluate_reconstruction, save_reconstruction_panel
from .model import build_model


def setup_distributed(backend: str) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if world_size > 1:
            raise RuntimeError("Multi-process DIME-ViT training requires CUDA or ROCm")
        device = torch.device("cpu")

    if world_size > 1:
        if backend != "nccl":
            raise ValueError(
                "CUDA and ROCm distributed training must use backend='nccl'"
            )
        dist.init_process_group(backend=backend, init_method="env://", device_id=device)
        dist.barrier()
    return rank, world_size, local_rank, device


def setup_logging(output_dir: Path, rank: int) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    level = logging.INFO if rank == 0 else logging.WARNING
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_dir / "train.log", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s | rank {rank} | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def seed_everything(seed: int, rank: int) -> None:
    seed = int(seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_from_config(config) -> torch.nn.Module:
    model_kwargs = to_plain_dict(config.model)
    name = model_kwargs.pop("name")
    model_kwargs.update(to_plain_dict(config.loss))
    return build_model(name=name, **model_kwargs)


def maybe_compile(model: torch.nn.Module, compile_config) -> torch.nn.Module:
    if not compile_config.enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile.enabled=true requires PyTorch 2.0 or newer")
    LOGGER.info(
        "torch.compile enabled | mode=%s | dynamic=%s | fullgraph=%s",
        compile_config.mode,
        compile_config.dynamic,
        compile_config.fullgraph,
    )
    return torch.compile(
        model,
        mode=compile_config.mode,
        dynamic=bool(compile_config.dynamic),
        fullgraph=bool(compile_config.fullgraph),
    )


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _resolve_resume(path: str | None, output_dir: Path) -> str | None:
    if path is None:
        return None
    if str(path).lower() == "auto":
        latest = output_dir / "checkpoint-latest.pth"
        return str(latest) if latest.exists() else None
    return str(path)


def _best_value(config, train_metrics, eval_metrics) -> float | None:
    name = str(config.checkpoint.best_metric).lower()
    if name in {"loss", "train_loss", "train/loss"}:
        return float(train_metrics["loss"])
    if eval_metrics is None:
        return None
    if name.startswith("eval/"):
        name = name.split("/", 1)[1]
    if name not in eval_metrics:
        raise KeyError(
            f"checkpoint.best_metric='{config.checkpoint.best_metric}' is not an evaluation metric"
        )
    return float(eval_metrics[name])


def _build_train_loader(config, input_size, batch_size, rank, world_size):
    return build_train_loader(
        data_path=config.data.path,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.train.seed,
        rank=rank,
        world_size=world_size,
        subset_ratio=config.data.subset_ratio,
        pair_sampling=config.data.pair_sampling,
        crop_scale=tuple(config.data.transform.train_crop_scale),
        crop_ratio=tuple(config.data.transform.train_crop_ratio),
        hflip_prob=config.data.transform.train_hflip_prob,
        interpolation=config.data.transform.interpolation,
        antialias=config.data.transform.antialias,
        mean=config.data.transform.mean,
        std=config.data.transform.std,
    )


def _build_eval_loader(config, input_size):
    return build_eval_loader(
        data_path=config.data.path,
        input_size=input_size,
        num_pairs=config.eval.num_pairs,
        batch_size=config.eval.batch_size,
        num_workers=config.eval.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.eval.seed,
        subset_ratio=config.data.subset_ratio,
        crop_pct=config.data.transform.eval_crop_pct,
        interpolation=config.data.transform.interpolation,
        antialias=config.data.transform.antialias,
        mean=config.data.transform.mean,
        std=config.data.transform.std,
    )


def train(config) -> None:
    rank, world_size, local_rank, device = setup_distributed(config.distributed.backend)
    output_dir = Path(config.checkpoint.output_dir)
    setup_logging(output_dir, rank)
    seed_everything(config.train.seed, rank)

    if config.data.path is None:
        raise ValueError("Set data.path to the face LMDB in the YAML or with --opts")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
        if os.environ.get("DIME_DISABLE_CUDNN_SDPA", "0") == "1" and hasattr(
            torch.backends.cuda, "enable_cudnn_sdp"
        ):

            torch.backends.cuda.enable_cudnn_sdp(False)
            if rank == 0:
                LOGGER.info(
                    "cuDNN SDPA disabled by DIME_DISABLE_CUDNN_SDPA; "
                    "using another supported SDPA backend"
                )

    effective_batch_size = (
        int(config.data.batch_size) * world_size * int(config.train.accum_iter)
    )
    if config.optimizer.lr is None:
        config.optimizer.lr = (
            float(config.optimizer.base_lr)
            * effective_batch_size
            / int(config.optimizer.reference_batch_size)
        )

    if rank == 0:
        save_config(config, output_dir / "config.yaml")
        LOGGER.info("output directory: %s", output_dir.resolve())
        LOGGER.info(
            "world size %d | per-device batch %d | accumulation %d | effective batch %d",
            world_size,
            config.data.batch_size,
            config.train.accum_iter,
            effective_batch_size,
        )
        LOGGER.info("learning rate: %.3e", config.optimizer.lr)
    _barrier()

    train_loader, pair_sampler = _build_train_loader(
        config, config.model.img_size, config.data.batch_size, rank, world_size
    )
    updates_per_epoch = math.ceil(len(train_loader) / int(config.train.accum_iter))

    raw_model = build_model_from_config(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
    if rank == 0:
        LOGGER.info(
            "model: %s | parameters: %.2f M", config.model.name, parameter_count / 1e6
        )
        LOGGER.info(
            "attention gate: %s | mask strategy: %s %s | EDDS: %s",
            "enabled" if config.model.gated_attention else "disabled",
            config.model.mask_strategy,
            list(config.model.mask_block_sizes),
            config.loss.edds_version,
        )

    if config.checkpoint.resume and config.checkpoint.init_checkpoint:
        raise ValueError(
            "Use either checkpoint.resume or checkpoint.init_checkpoint, not both"
        )
    if config.checkpoint.init_checkpoint:
        missing, unexpected = load_model_weights(
            raw_model, config.checkpoint.init_checkpoint, strict=True
        )
        LOGGER.info(
            "initialized model from %s (missing=%d, unexpected=%d)",
            config.checkpoint.init_checkpoint,
            len(missing),
            len(unexpected),
        )

    optimizer = build_optimizer(raw_model, config.optimizer)
    total_updates = int(config.train.epochs) * updates_per_epoch
    warmup_updates = int(config.scheduler.warmup_epochs) * updates_per_epoch
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
        min_lr=config.scheduler.min_lr,
    )
    scaler = create_grad_scaler(config.train.amp_dtype, device)

    train_model = maybe_compile(raw_model, config.compile)
    if world_size > 1:
        train_model = DistributedDataParallel(
            train_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=bool(config.distributed.find_unused_parameters),
        )

    checkpoint_manager = None
    if rank == 0:
        checkpoint_manager = CheckpointManager(
            output_dir,
            save_freq=config.checkpoint.save_freq,
            keep_last_n=config.checkpoint.keep_last_n,
            maximize=config.checkpoint.maximize_best_metric,
        )

    start_epoch = 0
    global_update = 0
    resume_path = _resolve_resume(config.checkpoint.resume, output_dir)
    if resume_path:
        checkpoint = resume_training(
            resume_path, raw_model, optimizer, scheduler, scaler=scaler
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_update = int(checkpoint.get("global_update", scheduler.num_updates))
        if checkpoint_manager is not None:
            checkpoint_manager.best_value = float(
                checkpoint.get("best_value", checkpoint_manager.best_value)
            )
        LOGGER.info(
            "resumed %s at epoch %d, optimizer update %d",
            resume_path,
            start_epoch + 1,
            global_update,
        )
    elif config.checkpoint.resume and str(config.checkpoint.resume).lower() == "auto":
        LOGGER.info("resume=auto: no latest checkpoint found, starting a new run")

    eval_loader = None
    if rank == 0 and config.eval.enabled:
        eval_loader = _build_eval_loader(config, config.model.img_size)
    _barrier()

    wandb_logger = WandBLogger(config.wandb, config)
    wandb_logger.watch(raw_model)
    wandb_logger.set_summary(
        {
            "model/name": config.model.name,
            "model/parameters": parameter_count,
            "model/gated_attention": bool(config.model.gated_attention),
            "model/mask_strategy": config.model.mask_strategy,
            "model/edds_version": config.loss.edds_version,
            "train/effective_batch_size": effective_batch_size,
        }
    )
    end_epoch = int(config.train.stop_epoch or config.train.epochs)
    active_size = config.model.img_size
    active_accum = int(config.train.accum_iter)
    try:
        for epoch in range(start_epoch, end_epoch):
            high_res_epoch = config.train.high_res_start_epoch
            if high_res_epoch is not None and epoch >= int(high_res_epoch):
                target_size = config.train.high_res_size
                if target_size != active_size:
                    high_batch = int(
                        config.data.high_res_batch_size
                        or min(32, int(config.data.batch_size))
                    )
                    active_accum = (
                        int(config.data.batch_size)
                        * int(config.train.accum_iter)
                        // high_batch
                    )
                    train_loader, pair_sampler = _build_train_loader(
                        config, target_size, high_batch, rank, world_size
                    )
                    if math.ceil(len(train_loader) / active_accum) != updates_per_epoch:
                        raise RuntimeError(
                            "Resolution transition must preserve optimizer updates per epoch"
                        )
                    if rank == 0 and config.eval.enabled:
                        eval_loader = _build_eval_loader(config, target_size)
                    active_size = target_size
                    LOGGER.info(
                        "resolution transition | input %s | per-device batch %d | accumulation %d | effective batch %d",
                        active_size,
                        high_batch,
                        active_accum,
                        effective_batch_size,
                    )
                    _barrier()
            pair_sampler.set_epoch(epoch)
            train_metrics, global_update = train_one_epoch(
                train_model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                accum_iter=active_accum,
                amp_dtype=config.train.amp_dtype,
                scaler=scaler,
                grad_clip_norm=config.train.grad_clip_norm,
                global_update=global_update,
                log_freq=config.train.log_freq,
                wandb_logger=wandb_logger if rank == 0 else None,
            )

            eval_metrics = None
            should_evaluate = bool(config.eval.enabled) and (
                (epoch + 1) % int(config.eval.freq) == 0 or epoch + 1 == end_epoch
            )
            if rank == 0:
                LOGGER.info(
                    "epoch %d complete | loss %.5f | rgb %.5f | diff %.5f | edds %.5f",
                    epoch + 1,
                    train_metrics["loss"],
                    train_metrics["loss_rgb"],
                    train_metrics["loss_diff"],
                    train_metrics["loss_edds"],
                )
                wandb_logger.log(
                    {
                        "epoch/index": epoch + 1,
                        **{
                            f"epoch/train_{key}": value
                            for key, value in train_metrics.items()
                        },
                    },
                    step=global_update,
                )

                if should_evaluate:
                    eval_metrics, visuals = evaluate_reconstruction(
                        raw_model,
                        eval_loader,
                        device,
                        amp_dtype=config.train.amp_dtype,
                        seed=config.eval.seed,
                        num_visuals=config.eval.num_visuals,
                        normalization_mean=config.data.transform.mean,
                        normalization_std=config.data.transform.std,
                    )
                    LOGGER.info(
                        "evaluation | MSE %.6f | PSNR %.3f dB | SSIM %.5f",
                        eval_metrics["mse"],
                        eval_metrics["psnr"],
                        eval_metrics["ssim"],
                    )
                    wandb_logger.log(
                        {f"eval/{key}": value for key, value in eval_metrics.items()},
                        step=global_update,
                    )
                    wandb_logger.log_reconstructions(**visuals, step=global_update)
                    save_reconstruction_panel(
                        visuals,
                        output_dir / f"reconstruction-epoch-{epoch + 1:04d}.png",
                    )

                checkpoint_metrics = {
                    f"train_{key}": value for key, value in train_metrics.items()
                }
                if eval_metrics is not None:
                    checkpoint_metrics.update(eval_metrics)
                checkpoint_manager.save(
                    epoch=epoch,
                    global_update=global_update,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    metrics=checkpoint_metrics,
                    best_value=_best_value(config, train_metrics, eval_metrics),
                    is_last_epoch=epoch + 1 == end_epoch,
                )
            _barrier()
    finally:
        try:
            wandb_logger.finish()
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
            logging.shutdown()


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DIME-ViT pre-training")
    parser.add_argument("--config", required=True, type=str, help="experiment YAML")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="dotted configuration overrides, e.g. data.batch_size 64",
    )
    return parser


def main() -> None:
    args = get_args_parser().parse_args()
    config = load_config(args.config, args.opts)
    try:
        train(config)
    finally:

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        logging.shutdown()


if __name__ == "__main__":
    main()
