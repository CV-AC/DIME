from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .config import (
    apply_overrides,
    experiment_output_dir,
    load_config,
    public_config,
    resolve_path,
)
from .data import build_dataset, build_loader
from .engine import ModelEMA, evaluate_model, train_one_epoch
from .model import build_model
from .optim import build_optimizer, build_scheduler, learning_rates
from .tracking import (
    flatten_metrics,
    format_epoch_line,
    start_tracker,
)
from .utils import (
    append_jsonl,
    barrier,
    broadcast_module,
    cleanup_distributed,
    ddp_options,
    gather_objects,
    git_commit,
    init_distributed,
    is_main_process,
    parameter_counts,
    restore_rng_state,
    rng_state,
    runtime_versions,
    set_seed,
    setup_runtime,
    unwrap_model,
    write_json,
)


LOGGER = logging.getLogger("dime_parsing")


def _configure_logging(output_dir: Path, rank: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(
            logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8")
        )
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Training checkpoint {path} must contain a dictionary.")
    return value


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_epoch: int,
    best_score: float,
    config: dict[str, Any],
    metadata: dict[str, Any],
    all_rng_states: list[Any],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_foreground_mean_f1": float(best_score),
        "model": unwrap_model(model).state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_states": all_rng_states,
        "config": public_config(config),
        "metadata": metadata,
    }


def _best_checkpoint_payload(
    *,
    ema: ModelEMA,
    epoch: int,
    best_score: float,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:

    return {
        "format_version": 2,
        "checkpoint_type": "ema_inference",
        "epoch": int(epoch),
        "best_epoch": int(epoch),
        "best_foreground_mean_f1": float(best_score),
        "ema": ema.state_dict(),
        "config": public_config(config),
        "metadata": metadata,
    }


def _load_resume(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    rank: int,
) -> tuple[int, int, float, dict[str, Any]]:
    checkpoint = _torch_load(path)
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    states = checkpoint.get("rng_states", [])
    if states:
        restore_rng_state(states[min(rank, len(states) - 1)])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("best_epoch", 0)),
        float(checkpoint.get("best_foreground_mean_f1", -math.inf)),
        dict(checkpoint.get("metadata", {}).get("wandb", {})),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a FaRL-style face-parsing model."
    )
    parser.add_argument("--config", required=True, help="Experiment YAML file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted YAML value; may be repeated.",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Resume a complete training checkpoint.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B for this run.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow a fresh run in a directory that already has run artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    if args.no_wandb:
        config["wandb"]["enabled"] = False
    resume_value = args.resume or config["experiment"].get("resume", "")
    resume_path = resolve_path(resume_value, must_exist=bool(resume_value))

    setup_runtime()
    rank, local_rank, world_size, device = init_distributed()
    del local_rank
    output_dir = experiment_output_dir(config)
    existing_artifacts = tuple(
        path
        for path in (
            output_dir / "metrics.jsonl",
            output_dir / "last.pt",
            output_dir / "best.pt",
        )
        if path.exists()
    )
    if resume_path is None and existing_artifacts and not args.allow_existing_output:
        names = ", ".join(path.name for path in existing_artifacts)
        cleanup_distributed()
        raise FileExistsError(
            f"{output_dir} already contains {names}. Resume with --resume, "
            "choose a new experiment.output_dir, or explicitly pass "
            "--allow-existing-output."
        )
    _configure_logging(output_dir, rank)
    set_seed(int(config["experiment"]["seed"]) + rank)
    tracker = None

    try:
        dataset_name = str(config["dataset"]["name"]).lower()
        selection_split = str(config["protocol"].get("selection_split", "val")).lower()
        if selection_split not in {"val", "test"}:
            raise ValueError("protocol.selection_split must be val or test.")

        train_dataset = build_dataset(config, "train")
        selection_dataset = build_dataset(config, selection_split)
        train_loader, train_sampler = build_loader(
            config, train_dataset, rank=rank, world_size=world_size
        )
        selection_loader, _ = build_loader(
            config, selection_dataset, rank=rank, world_size=world_size
        )

        backbone_name = str(config["backbone"]["name"]).lower()

        initialize_pretrained = resume_path is None and (
            world_size == 1 or rank == 0 or backbone_name == "dime"
        )
        model = build_model(config, initialize_pretrained=initialize_pretrained).to(
            device
        )
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                **ddp_options(config),
            )
        ema = ModelEMA(model, decay=float(config["protocol"].get("ema_decay", 0.999)))
        optimizer = build_optimizer(unwrap_model(model), config["protocol"])
        scheduler = build_scheduler(optimizer, config["protocol"])
        amp_enabled = bool(config["runtime"].get("amp_enabled", False))
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        start_epoch, best_epoch, best_score = 1, 0, -math.inf
        resume_tracking: dict[str, Any] = {}
        if resume_path is not None:
            start_epoch, best_epoch, best_score, resume_tracking = _load_resume(
                resume_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                rank=rank,
            )
            LOGGER.info("Resumed %s at epoch %d", resume_path, start_epoch)

        metadata = {
            "dataset": dataset_name,
            "backbone": str(config["backbone"]["name"]).lower(),
            "world_size": world_size,
            "device": str(device),
            "git_commit": git_commit(Path(config["_parsing_root"]).parent),
            "versions": runtime_versions(),
            "model": parameter_counts(unwrap_model(model)),
            "backbone_checkpoint_sha256": getattr(
                unwrap_model(model).backbone,
                "checkpoint_sha256",
                "unknown",
            ),
            "train_samples": len(train_dataset),
            "selection_split": selection_split,
            "selection_samples": len(selection_dataset),
        }
        if is_main_process():
            write_json(output_dir / "resolved_config.json", public_config(config))
            tracker = start_tracker(
                dataset=dataset_name,
                backbone=str(config["backbone"]["name"]).lower(),
                output_dir=output_dir,
                config=public_config(config),
                metadata=metadata,
                options=config["wandb"],
                resume_identity=resume_tracking,
            )
            metadata["wandb"] = {
                "run_id": tracker.run_id,
                "name": tracker.name,
                "url": tracker.url,
            }
            write_json(output_dir / "run_metadata.json", metadata)
            LOGGER.info(
                "Training %s/%s: %d train, %d %s, world=%d",
                dataset_name,
                config["backbone"]["name"],
                len(train_dataset),
                len(selection_dataset),
                selection_split,
                world_size,
            )

        epochs = int(config["protocol"]["epochs"])
        eval_interval = int(config["protocol"].get("evaluation_interval", 1))
        save_interval = int(config["experiment"].get("save_interval", 20))
        keep_periodic = bool(
            config["experiment"].get("keep_periodic_checkpoints", False)
        )
        if epochs <= 0 or eval_interval <= 0 or save_interval < 0:
            raise ValueError(
                "epochs/evaluation_interval must be positive and "
                "save_interval must be non-negative."
            )
        if start_epoch > epochs:
            raise ValueError(
                f"Checkpoint already reached epoch {start_epoch - 1}, "
                f"but protocol.epochs={epochs}."
            )
        for epoch in range(start_epoch, epochs + 1):
            epoch_started = time.perf_counter()
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)
            rates = learning_rates(optimizer)
            training = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                ema,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=str(config["runtime"].get("amp_dtype", "fp16")),
                gradient_clip_norm=float(
                    config["protocol"].get("gradient_clip_norm", 0.0)
                ),
                log_interval=int(config["experiment"].get("log_interval", 50)),
                progress=LOGGER.info,
            )

            metrics = None
            should_evaluate = epoch % eval_interval == 0 or epoch == epochs
            if should_evaluate:
                metrics = evaluate_model(
                    ema.module,
                    selection_loader,
                    selection_dataset.space,
                    device,
                    amp_enabled=amp_enabled,
                    amp_dtype=str(config["runtime"].get("amp_dtype", "fp16")),
                    canvas_size=int(config["augmentation"]["canvas_size"]),
                    warp_factor=float(config["augmentation"]["warp_factor"]),
                )
            is_best = (
                metrics is not None
                and float(metrics["foreground_mean_f1"]) > best_score
            )
            if is_best:
                best_score = float(metrics["foreground_mean_f1"])
                best_epoch = epoch

            scheduler.step()
            full_checkpoint_due = epoch == epochs or (
                save_interval > 0 and epoch % save_interval == 0
            )
            all_rng_states = gather_objects(rng_state()) if full_checkpoint_due else []
            seconds = time.perf_counter() - epoch_started
            row = {
                "epoch": epoch,
                "train": training,
                "learning_rates": rates,
                "selection_split": selection_split,
                "selection": metrics,
                "best_epoch": best_epoch,
                "best_foreground_mean_f1": best_score,
                "seconds": seconds,
            }
            if is_main_process():
                append_jsonl(output_dir / "metrics.jsonl", row)
                if is_best:
                    _save_checkpoint(
                        output_dir / "best.pt",
                        _best_checkpoint_payload(
                            ema=ema,
                            epoch=epoch,
                            best_score=best_score,
                            config=config,
                            metadata=metadata,
                        ),
                    )
                if full_checkpoint_due:
                    payload = _checkpoint_payload(
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_epoch=best_epoch,
                        best_score=best_score,
                        config=config,
                        metadata=metadata,
                        all_rng_states=all_rng_states,
                    )
                    _save_checkpoint(output_dir / "last.pt", payload)
                    if keep_periodic:
                        _save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", payload)
                LOGGER.info(
                    format_epoch_line(
                        epoch=epoch,
                        total_epochs=epochs,
                        train_loss=float(training["loss"]),
                        split=selection_split,
                        metrics=metrics,
                        best_epoch=best_epoch,
                        best_score=best_score,
                        is_best=is_best,
                        seconds=seconds,
                    )
                )
                log_payload: dict[str, Any] = {
                    "epoch": epoch,
                    "train/loss": float(training["loss"]),
                    "lr/encoder": rates["encoder"],
                    "lr/head": rates["head"],
                    "time/epoch_seconds": seconds,
                    "selection/is_best": int(is_best),
                    "best/epoch": best_epoch,
                    "best/foreground_mean_f1": best_score,
                }
                if metrics is not None:
                    log_payload.update(flatten_metrics(selection_split, metrics))
                assert tracker is not None
                tracker.log(log_payload, step=epoch)

        barrier()
        best_path = output_dir / "best.pt"
        if not best_path.is_file():
            raise RuntimeError("Training produced no best checkpoint.")
        if is_main_process():
            best_checkpoint = _torch_load(best_path)
            ema.module.load_state_dict(best_checkpoint["ema"], strict=True)
        broadcast_module(ema.module)

        final_metrics = None
        if bool(config["protocol"].get("test_after_training", True)):
            test_dataset = (
                selection_dataset
                if selection_split == "test"
                else build_dataset(config, "test")
            )
            test_loader = (
                selection_loader
                if selection_split == "test"
                else build_loader(
                    config,
                    test_dataset,
                    rank=rank,
                    world_size=world_size,
                )[0]
            )
            prediction_dir = (
                output_dir / "predictions" / "test"
                if bool(config["evaluation"].get("save_predictions", False))
                else None
            )
            final_metrics = evaluate_model(
                ema.module,
                test_loader,
                test_dataset.space,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=str(config["runtime"].get("amp_dtype", "fp16")),
                canvas_size=int(config["augmentation"]["canvas_size"]),
                warp_factor=float(config["augmentation"]["warp_factor"]),
                prediction_dir=prediction_dir,
            )
            if is_main_process():
                write_json(output_dir / "test_metrics.json", final_metrics)
                assert tracker is not None
                tracker.log(
                    {"epoch": epochs, **flatten_metrics("test", final_metrics)},
                    step=epochs,
                )
                tracker.summary(
                    {
                        "best/epoch": best_epoch,
                        "best/foreground_mean_f1": best_score,
                        "final_test/foreground_mean_f1": float(
                            final_metrics["foreground_mean_f1"]
                        ),
                        "final_test/foreground_mean_iou": float(
                            final_metrics["foreground_mean_iou"]
                        ),
                    }
                )
                LOGGER.info(
                    "Final test (best EMA): mean-F1=%.4f, mIoU=%.4f",
                    final_metrics["foreground_mean_f1"],
                    final_metrics["foreground_mean_iou"],
                )
        if is_main_process() and tracker is not None:
            tracker.finish(0)
    except Exception:
        LOGGER.exception("Training failed.")
        if is_main_process() and tracker is not None:
            tracker.finish(1)
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
