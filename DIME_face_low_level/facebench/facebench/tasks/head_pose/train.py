from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .config import (
    HEAD_POSE_ROOT,
    experiment_output_dir,
    load_config,
    public_config,
    resolve_path,
)
from .data import build_dataset, validate_data_storage
from .engine import ModelEMA
from .models import HeadPoseModel, build_model
from .optim import build_optimizer, build_scheduler, component_learning_rates
from .oracle import (
    FIXED_FINAL_SELECTION,
    ORACLE_SELECTION,
    build_oracle_loaders,
    evaluate_oracle,
    selection_mode,
)
from .rotation import GeodesicLoss
from .tracking import WandbTracker
from .utils import (
    autocast_context,
    barrier,
    cleanup_distributed,
    git_commit,
    init_distributed,
    is_main_process,
    parameter_counts,
    restore_rng_state,
    rng_state,
    runtime_versions,
    seed_worker,
    set_seed,
    setup_runtime,
    unwrap_model,
    write_json,
)


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _gather_object(value: Any) -> list[Any]:
    if not dist.is_initialized():
        return [value]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    config: dict[str, Any],
    train_loss: float,
    rank_states: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        "epoch": int(epoch),
        "model": unwrap_model(model).state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_loss": float(train_loss),
        "config": public_config(config),
        "rank_states": rank_states,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, temporary)
    os.replace(temporary, path)


def _save_evaluation_checkpoint(
    path: Path,
    *,
    epoch: int,
    ema: ModelEMA,
    config: dict[str, Any],
    train_loss: float,
    source: str,
    selection: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": int(epoch),
            "model": ema.state_dict(),
            "source": source,
            "train_loss": float(train_loss),
            "config": public_config(config),
            "selection": selection,
        },
        temporary,
    )
    os.replace(temporary, path)


def _load_resume(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    generator: torch.Generator,
    rank: int,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    ema.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    rank_states = checkpoint.get("rank_states", [])
    if rank_states:
        selected = rank_states[min(rank, len(rank_states) - 1)]
        restore_rng_state(selected["rng"])
        generator.set_state(selected["data_generator"])
    return int(checkpoint["epoch"]) + 1


def _validate_before_distributed(config: dict[str, Any]) -> None:
    data = config["data"]
    for key in ("train_manifest", "aflw2000_manifest"):
        resolve_path(data[key], must_exist=True)
    validate_data_storage(config, ("train", "aflw2000"))
    model = config["model"]
    kind = str(model["kind"]).lower()
    if kind == "dime":
        resolve_path(model["dime_source"], must_exist=True)
        resolve_path(model.get("pretrained_checkpoint"), must_exist=True)
    elif kind == "repvgg":
        resolve_path(model["repo"], must_exist=True)
        resolve_path(model.get("pretrained_checkpoint"), must_exist=True)
    elif (
        kind in {"dino", "mae"} and str(model.get("pretrained_checkpoint", "")).strip()
    ):
        resolve_path(model["pretrained_checkpoint"], must_exist=True)
    else:
        if kind not in {"dino", "mae"}:
            raise ValueError("Controlled training supports dime, dino, mae, or repvgg.")
    protocol = config["protocol"]
    mode = selection_mode(config)
    if int(protocol.get("input_size", 224)) != 224:
        raise ValueError("The controlled protocol requires 224x224 input.")
    if bool(protocol.get("controlled_main_table", True)):
        if bool(config.get("evaluation", {}).get("test_time_augmentation", False)):
            raise ValueError("The controlled benchmark forbids test-time augmentation.")
        if str(config.get("head", {}).get("type", "linear")).lower() != "linear":
            raise ValueError(
                "The controlled main comparison requires a Linear-6D head."
            )
        if (
            str(config.get("finetuning", {}).get("encoder_mode", "full")).lower()
            != "full"
        ):
            raise ValueError(
                "The controlled main comparison requires full fine-tuning."
            )
    if mode == ORACLE_SELECTION:
        resolve_path(data["biwi_manifest"], must_exist=True)
        validate_data_storage(config, ("biwi",))


def train(
    config: dict[str, Any],
    resume_override: str = "",
    seed_override: int | None = None,
) -> None:
    if seed_override is not None:
        config["seed"] = int(seed_override)
    _validate_before_distributed(config)
    setup_runtime()
    rank, _, world_size, device = init_distributed()
    protocol = config["protocol"]
    seed = int(config.get("seed", 0))
    set_seed(seed + rank)

    output_dir = experiment_output_dir(config, seed)
    resume = resume_override or str(config.get("resume", "")).strip()
    if not resume and any(
        (output_dir / name).exists()
        for name in (
            "train_metrics.jsonl",
            "oracle_metrics.jsonl",
            "latest.pth",
            "final.pth",
            "final_ema.pth",
            "best_oracle_ema.pth",
        )
    ):
        raise FileExistsError(
            f"Refusing to mix a fresh run with existing files in {output_dir}. "
            "Use --resume or move the old seed directory."
        )
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(config, "train")
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    per_gpu_batch = int(protocol["batch_size_per_gpu"])
    effective_batch = int(protocol["effective_batch_size"])
    global_micro_batch = per_gpu_batch * world_size
    if effective_batch < global_micro_batch or effective_batch % global_micro_batch:
        raise ValueError(
            "effective_batch_size must be divisible by batch_size_per_gpu * world_size."
        )
    accumulation_steps = effective_batch // global_micro_batch
    generator = torch.Generator().manual_seed(seed + rank)
    workers = int(protocol.get("num_workers", 8))
    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = build_model(config)
    if not isinstance(model, HeadPoseModel):
        raise ValueError("Official fine-tuned 6DRepNet is evaluation-only.")
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    ema = ModelEMA(model, decay=float(protocol.get("ema_decay", 0.999)))
    ema.module.to(device)
    optimizer = build_optimizer(model, config)
    epochs = int(protocol["epochs"])
    scheduler = build_scheduler(optimizer, protocol, total_epochs=epochs)
    amp_enabled = bool(protocol.get("amp", True)) and device.type == "cuda"
    amp_dtype = str(protocol.get("amp_dtype", "bf16"))
    scaler = _make_grad_scaler(amp_enabled and amp_dtype.lower() == "fp16")
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    mode = selection_mode(config)
    oracle_loaders = (
        build_oracle_loaders(
            config,
            rank=rank,
            world_size=world_size,
            seed=seed,
            device=device,
        )
        if mode == ORACLE_SELECTION
        else {}
    )

    resume_path = resolve_path(resume) if resume else None
    start_epoch = 0
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        start_epoch = _load_resume(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            generator=generator,
            rank=rank,
        )

    tracker: WandbTracker | None = None
    if is_main_process():
        method = str(config["experiment"]["name"])
        metadata = {
            "method": method,
            "seed": seed,
            "protocol": str(protocol["name"]),
            "selection": mode,
            "selection_warning": (
                "AFLW2000 and BIWI are evaluated every epoch and directly select "
                "the checkpoint; reported metrics are oracle/test-tuned."
                if mode == ORACLE_SELECTION
                else "Fixed final-epoch EMA; no test-set selection."
            ),
            "train_samples": len(dataset),
            "train_manifest_sha256": dataset.manifest_sha256,
            "data_backend": getattr(dataset, "storage_backend", "unknown"),
            "train_lmdb_logical_content_sha256": (
                getattr(dataset, "storage_logical_content_sha256", None)
            ),
            "pretrained_sha256": config["model"].get("pretrained_sha256"),
            "parameters": parameter_counts(unwrap_model(model)),
            "world_size": world_size,
            "effective_batch_size": effective_batch,
            "git_commit": git_commit(HEAD_POSE_ROOT),
            "versions": runtime_versions(),
        }
        tracker = WandbTracker.start(
            method=method,
            seed=seed,
            output_dir=output_dir,
            config=config,
            metadata=metadata,
            resume_checkpoint=str(resume_path) if resume_path else "",
        )
        metadata["wandb"] = {
            "run_id": tracker.run_id,
            "name": tracker.name,
            "url": tracker.url,
        }
        write_json(output_dir / "resolved_config.json", public_config(config))
        write_json(output_dir / "run_metadata.json", metadata)
        counts = metadata["parameters"]
        print(
            f"method={method} seed={seed} samples={len(dataset)} epochs={epochs} "
            f"world={world_size} micro={per_gpu_batch} "
            f"accumulation={accumulation_steps} effective_batch={effective_batch} "
            f"precision={amp_dtype if amp_enabled else 'fp32'} "
            f"trainable={counts['trainable']:,}/{counts['total']:,} "
            f"wandb={tracker.url or 'disabled'}"
        )

    loss_function = GeodesicLoss().to(device)
    save_every = int(protocol.get("save_every", 10))
    log_every = int(protocol.get("log_every", 100))
    grad_clip = protocol.get("grad_clip_norm")
    history_path = output_dir / "train_metrics.jsonl"
    oracle_history_path = output_dir / "oracle_metrics.jsonl"
    oracle_selection_path = output_dir / "oracle_selection.json"
    last_mean_loss = float("nan")
    best_oracle_score = float("inf")
    best_oracle_epoch: int | None = None
    if mode == ORACLE_SELECTION and start_epoch > 0:
        if (
            not oracle_selection_path.is_file()
            or not (output_dir / "best_oracle_ema.pth").is_file()
        ):
            raise RuntimeError(
                "Cannot resume oracle selection without oracle_selection.json and "
                "best_oracle_ema.pth from the completed earlier epochs."
            )
        previous_selection = json.loads(
            oracle_selection_path.read_text(encoding="utf-8")
        )
        best_oracle_score = float(previous_selection["oracle_score"])
        best_oracle_epoch = int(previous_selection["epoch"])

    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(2, device=device, dtype=torch.float64)
        epoch_start = time.time()
        epoch_lrs = component_learning_rates(optimizer)
        iterator = tqdm(
            enumerate(loader),
            total=len(loader),
            disable=not is_main_process(),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )
        for step, batch in iterator:
            group_start = (step // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(loader) - group_start)
            should_step = (step - group_start + 1) == group_size
            sync_context = (
                contextlib.nullcontext()
                if should_step or world_size == 1
                else model.no_sync()
            )
            with sync_context:
                target = batch["rotation"].to(device, non_blocking=True).float()
                with autocast_context(amp_enabled, amp_dtype):
                    predicted = model(batch["image"].to(device, non_blocking=True))
                loss = loss_function(predicted.float(), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch + 1}, step={step + 1}."
                    )
                scaled_loss = loss / group_size
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(unwrap_model(model))

            batch_size = batch["image"].shape[0]
            totals[0] += loss.detach().double() * batch_size
            totals[1] += batch_size
            if is_main_process() and (step % log_every == 0 or step + 1 == len(loader)):
                iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    encoder_lr=f"{epoch_lrs.get('encoder', 0.0):.2e}",
                    head_lr=f"{epoch_lrs['head']:.2e}",
                )

        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        elapsed = time.time() - epoch_start
        last_mean_loss = (totals[0] / totals[1]).item()
        row = {
            "epoch": epoch + 1,
            "train_geodesic_loss": last_mean_loss,
            "encoder_lr": epoch_lrs.get("encoder"),
            "encoder_lr_min": epoch_lrs.get("encoder_min", epoch_lrs.get("encoder")),
            "head_lr": epoch_lrs["head"],
            "seconds": elapsed,
            "images_per_second": float(totals[1].item() / max(elapsed, 1e-9)),
        }
        scheduler.step()

        rank_states = _gather_object(
            {"rng": rng_state(), "data_generator": generator.get_state()}
        )
        oracle_row: dict[str, Any] | None = None
        if mode == ORACLE_SELECTION:
            oracle_metrics, oracle_score = evaluate_oracle(
                ema.module,
                oracle_loaders,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            is_best = oracle_score < best_oracle_score
            if is_best:
                best_oracle_score = oracle_score
                best_oracle_epoch = epoch + 1
            oracle_row = {
                "epoch": epoch + 1,
                "selection": ORACLE_SELECTION,
                "selection_warning": "test-set-tuned oracle; not an unbiased test result",
                "score_definition": "mean(aflw2000.mean_mae, biwi.mean_mae)",
                "oracle_score": oracle_score,
                "best_oracle_score": best_oracle_score,
                "best_oracle_epoch": best_oracle_epoch,
                "is_best": is_best,
                "splits": oracle_metrics,
            }
        if is_main_process():
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            assert tracker is not None
            tracker.log_epoch(row, commit=oracle_row is None)
            if oracle_row is not None:
                tracker.log_oracle_epoch(oracle_row)
                with oracle_history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(oracle_row, ensure_ascii=False) + "\n")
                if bool(oracle_row["is_best"]):
                    _save_evaluation_checkpoint(
                        output_dir / "best_oracle_ema.pth",
                        epoch=epoch,
                        ema=ema,
                        config=config,
                        train_loss=last_mean_loss,
                        source="EMA selected by minimum AFLW2000+BIWI oracle score",
                        selection=oracle_row,
                    )
                    write_json(oracle_selection_path, oracle_row)
            print(row, flush=True)
            if oracle_row is not None:
                print(oracle_row, flush=True)
            _save_training_checkpoint(
                output_dir / "latest.pth",
                epoch=epoch,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                train_loss=last_mean_loss,
                rank_states=rank_states,
            )
            if (epoch + 1) % save_every == 0:
                _save_training_checkpoint(
                    output_dir / f"epoch_{epoch + 1:03d}.pth",
                    epoch=epoch,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    train_loss=last_mean_loss,
                    rank_states=rank_states,
                )
            if epoch + 1 == epochs:
                _save_training_checkpoint(
                    output_dir / "final.pth",
                    epoch=epoch,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    train_loss=last_mean_loss,
                    rank_states=rank_states,
                )
                _save_evaluation_checkpoint(
                    output_dir / "final_ema.pth",
                    epoch=epoch,
                    ema=ema,
                    config=config,
                    train_loss=last_mean_loss,
                    source="fixed-final-epoch EMA",
                    selection={
                        "selection": FIXED_FINAL_SELECTION,
                        "epoch": epoch + 1,
                    },
                )
        barrier()

    if tracker is not None:
        tracker.finish()
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one controlled head-pose encoder run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_config(args.config), args.resume, args.seed)


if __name__ == "__main__":
    main()
