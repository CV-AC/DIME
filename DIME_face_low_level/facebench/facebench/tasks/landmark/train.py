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

from .auxiliary_data import (
    AuxiliaryLandmarkDataset,
    DistributedEpochFractionSampler,
    auxiliary_task_schedule,
    epoch_fractions,
    inspect_auxiliary_index,
)
from .config import (
    LANDMARK_ROOT,
    experiment_output_dir,
    load_config,
    public_config,
    resolve_path,
)
from .data import EXPECTED_COUNTS, DistributedEvalSampler, build_dataset
from .engine import ModelEMA, evaluate_model
from .lapa_transfer import initialize_shared_weights
from .model import LandmarkModel, build_model, landmark_losses, task_losses
from .optim import build_optimizer, build_scheduler
from .tracking import HeatmapCollapseMonitor, WandbTracker, format_epoch_line
from .utils import (
    autocast_context,
    barrier,
    cleanup_distributed,
    gather_objects,
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


def _save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    config: dict[str, Any],
    best_epoch: int,
    best_nme: float,
    data_generators: dict[str, torch.Generator],
    collapse_monitor: HeatmapCollapseMonitor,
) -> None:
    rank_states = gather_objects(
        {
            "rng": rng_state(),
            "data_generators": {
                name: generator.get_state()
                for name, generator in data_generators.items()
            },
        }
    )
    if not is_main_process():
        return
    state: dict[str, Any] = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": public_config(config),
        "best_epoch": best_epoch,
        "best_nme": best_nme,
        "rank_states": rank_states,
        "collapse_monitor": collapse_monitor.state_dict(),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _save_evaluation_checkpoint(
    path: Path,
    *,
    epoch: int,
    ema: ModelEMA,
    config: dict[str, Any],
    metrics: dict[str, Any] | None,
) -> None:
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "ema": ema.state_dict(),
            "config": public_config(config),
            "metrics": metrics,
        },
        temporary,
    )
    os.replace(temporary, path)


def _resume(
    path: Path,
    *,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    config: dict[str, Any],
    data_generators: dict[str, torch.Generator],
    collapse_monitor: HeatmapCollapseMonitor,
    rank: int,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_config = checkpoint.get("config", {})
    saved_objective = saved_config.get("model", {}).get("objective")
    current_objective = public_config(config).get("model", {}).get("objective")
    saved_head_lr = saved_config.get("protocol", {}).get("head_lr")
    current_head_lr = config.get("protocol", {}).get("head_lr")
    if saved_objective != current_objective or saved_head_lr != current_head_lr:
        raise ValueError(
            "Resume checkpoint objective/head_lr does not match the current "
            "configuration. Old unbalanced-MSE Route-A checkpoints must not "
            "be resumed into the weighted Adaptive-Wing recipe."
        )
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    collapse_monitor.load_state_dict(checkpoint.get("collapse_monitor"))
    rank_states = checkpoint.get("rank_states")
    if rank_states:
        selected = rank_states[min(rank, len(rank_states) - 1)]
        restore_rng_state(selected["rng"])
        saved_generators = selected.get("data_generators")
        if saved_generators:
            for name, state in saved_generators.items():
                if name in data_generators:
                    data_generators[name].set_state(state)
        elif "data_generator" in selected and "wflw" in data_generators:

            data_generators["wflw"].set_state(selected["data_generator"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("best_epoch", 0)),
        float(checkpoint.get("best_nme", float("inf"))),
    )


def _metadata(config: dict[str, Any], model: LandmarkModel) -> dict[str, Any]:
    audit_path = resolve_path(config["dataset"].get("audit_file"))
    audit_manifest = "unavailable"
    if audit_path and audit_path.is_file():
        with audit_path.open("r", encoding="utf-8") as handle:
            audit_manifest = json.load(handle).get("manifest_sha256", "unavailable")
    auxiliary_manifests: dict[str, dict[str, Any]] = {}
    auxiliary = config.get("auxiliary_training", {})
    if bool(auxiliary.get("enabled", False)):
        for task in ("lapa", "300w_lp"):
            options = auxiliary.get("datasets", {}).get(task, {})
            if not bool(options.get("enabled", True)):
                continue
            index_path = resolve_path(options.get("index_file"), must_exist=True)
            if index_path is None:
                continue
            index_metadata = inspect_auxiliary_index(
                index_path,
                expected_task=task,
                validate_arrays=False,
            )
            auxiliary_manifests[task] = {
                "samples": int(index_metadata["samples"]),
                "manifest_sha256": index_metadata["manifest_sha256"],
                "index_format_version": int(index_metadata["index_format_version"]),
                "source_root": index_metadata["source_root"],
            }
    return {
        **parameter_counts(model),
        "backbone_parameters": sum(
            parameter.numel() for parameter in model.backbone.parameters()
        ),
        "pyramid_parameters": sum(
            parameter.numel() for parameter in model.pyramid.parameters()
        ),
        "head_parameters": sum(
            parameter.numel()
            for module in model.downstream_modules[1:]
            for parameter in module.parameters()
        ),
        "backbone_checkpoint_sha256": model.backbone.checkpoint_sha256,
        "dataset_manifest_sha256": audit_manifest,
        "auxiliary_dataset_manifests": auxiliary_manifests,
        "git_commit": git_commit(LANDMARK_ROOT),
        "versions": runtime_versions(),
    }


def _validate_before_distributed(
    config: dict[str, Any],
    stage: str,
    selection: str,
    initialization: str,
    resume: str,
) -> None:
    if stage not in {"benchmark", "dev", "final"}:
        raise ValueError("--stage must be benchmark, dev, or final.")
    backbone = config["backbone"]
    if backbone["name"] in {"dime", "farl"}:
        resolve_path(backbone.get("checkpoint"), must_exist=True)
    elif str(backbone.get("checkpoint", "")).strip():
        resolve_path(backbone["checkpoint"], must_exist=True)
    data_root = resolve_path(config["dataset"]["root"], must_exist=True)
    audit_path = resolve_path(config["dataset"].get("audit_file"), must_exist=True)
    assert data_root is not None and audit_path is not None
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit.get("status") != "ok" or not audit.get("decoded_all_images", False):
        raise RuntimeError(
            "The WFLW audit is missing a successful full image decode. "
            "Run python -m dime_landmark.audit_data without --skip-decode."
        )
    if Path(audit.get("root", "")).resolve() != data_root:
        raise RuntimeError("The audit report belongs to a different WFLW root.")
    for name, expected in EXPECTED_COUNTS.items():
        if int(audit.get("counts", {}).get(name, -1)) != expected:
            raise RuntimeError(f"The audit report has an invalid {name} count.")
    if audit.get("development_split", {}).get("shared_source_images") != 0:
        raise RuntimeError("The development split is not source-image-disjoint.")
    if int(config["model"].get("canvas_size", 512)) != 512:
        raise ValueError("The FaRL-compatible protocol requires canvas_size=512.")
    if int(config["model"].get("input_size", 448)) != 448:
        raise ValueError("The FaRL WFLW protocol requires input_size=448.")
    objective = config["model"].get("objective", {})
    objective_name = str(objective.get("name", "farl")).strip().lower()
    if objective_name not in {"farl", "route_a"}:
        raise ValueError("model.objective.name must be farl or route_a.")
    if objective_name == "route_a":
        expected_size = int(config["model"].get("input_size", 448)) // 4
        if int(objective.get("heatmap_size", expected_size)) != expected_size:
            raise ValueError("Route A heatmap_size must equal input_size/4.")
    auxiliary = config.get("auxiliary_training", {})
    if bool(auxiliary.get("enabled", False)):
        experiment_output_dir(config)
        enabled = []
        expected_landmarks = {"lapa": 106, "300w_lp": 68}
        for task in ("lapa", "300w_lp"):
            options = auxiliary.get("datasets", {}).get(task, {})
            if not bool(options.get("enabled", True)):
                continue
            enabled.append(task)
            if int(options.get("num_landmarks", expected_landmarks[task])) != (
                expected_landmarks[task]
            ):
                raise ValueError(
                    f"{task}.num_landmarks must be {expected_landmarks[task]}."
                )
            if task == "lapa" and int(options.get("num_parsing_classes", 11)) != 11:
                raise ValueError("lapa.num_parsing_classes must be 11.")
            root = resolve_path(options.get("root"), must_exist=True)
            index = resolve_path(options.get("index_file"), must_exist=True)
            if root is None or index is None or index.suffix != ".npz":
                raise ValueError(f"{task}.index_file must be a prepared .npz file.")
            inspect_auxiliary_index(
                index,
                expected_task=task,
                expected_root=root,
                expected_num_landmarks=expected_landmarks[task],
                require_full_decode=bool(
                    auxiliary.get("verify_decode_during_prepare", True)
                ),
                validate_arrays=False,
            )
        if not enabled:
            raise ValueError("auxiliary_training.enabled requires LaPa and/or 300W-LP.")
        sampling = auxiliary.get("sampling", {})
        epoch_fractions(sampling, enabled)
        loss_weights = auxiliary.get("loss_weights", {})
        if any(
            float(loss_weights.get(name, default)) < 0.0
            for name, default in (
                ("wflw", 1.0),
                ("lapa", 1.0),
                ("300w_lp", 0.35),
                ("lapa_parsing", 0.1),
                ("lapa_parsing_dice", 1.0),
            )
        ):
            raise ValueError("Auxiliary loss weights must be non-negative.")
        if int(auxiliary.get("num_workers_per_auxiliary_loader", 1)) < 0:
            raise ValueError("num_workers_per_auxiliary_loader must be non-negative.")
    if stage == "benchmark":
        protocol = config["protocol"]
        if protocol.get("selection_protocol") != "farl_official_test_best":
            raise ValueError(
                "benchmark stage requires selection_protocol=farl_official_test_best."
            )
        if int(protocol.get("eval_interval", 1)) != 1:
            raise ValueError(
                "FaRL historical checkpoint selection evaluates every epoch."
            )
    if stage == "final":
        resolve_path(selection, must_exist=True)
    if initialization and resume:
        raise ValueError("--initialization and --resume are mutually exclusive.")
    if initialization:
        resolve_path(initialization, must_exist=True)


def train(
    config: dict[str, Any],
    *,
    stage: str,
    selection: str = "",
    resume: str = "",
    initialization: str = "",
    output_override: str = "",
) -> None:
    _validate_before_distributed(config, stage, selection, initialization, resume)
    setup_runtime()
    rank, _, world_size, device = init_distributed()
    seed = int(config.get("seed", 0))
    set_seed(seed + rank)
    protocol = config["protocol"]

    configured_output = resolve_path(config["experiment"]["output_dir"])
    assert configured_output is not None
    if output_override:
        base_output = resolve_path(output_override)
        assert base_output is not None
    else:
        base_output = experiment_output_dir(config)

    output_dir = base_output / ("final" if stage == "benchmark" else stage)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    data_root = resolve_path(config["dataset"]["root"], must_exist=True)
    assert data_root is not None
    auxiliary = config.get("auxiliary_training", {})
    auxiliary_enabled = bool(auxiliary.get("enabled", False))
    enabled_auxiliary_tasks = [
        task
        for task in ("lapa", "300w_lp")
        if bool(auxiliary.get("datasets", {}).get(task, {}).get("enabled", True))
    ]
    fractions = (
        epoch_fractions(auxiliary.get("sampling", {}), enabled_auxiliary_tasks)
        if auxiliary_enabled
        else {"wflw": 1.0}
    )
    training_augmentation = dict(config.get("augmentation", {}))
    if auxiliary_enabled:

        training_augmentation.update(auxiliary.get("augmentation", {}))
    train_split = "dev_train" if stage == "dev" else "full_train"
    train_dataset = build_dataset(
        data_root,
        train_split,
        augmentation=training_augmentation,
    )
    if stage == "dev":
        selection_split = "dev_validation"
    elif stage == "benchmark":
        selection_split = "test"
    else:
        selection_split = None
    selection_dataset = (
        build_dataset(data_root, selection_split) if selection_split else None
    )
    if auxiliary_enabled:
        train_sampler = DistributedEpochFractionSampler(
            len(train_dataset),
            fraction=fractions["wflw"],
            num_replicas=world_size,
            rank=rank,
            seed=seed,
        )
    elif world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
    else:
        train_sampler = None
    selection_sampler = (
        DistributedEvalSampler(selection_dataset, rank, world_size)
        if world_size > 1 and selection_dataset is not None
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
    workers = int(protocol.get("num_workers", 6))
    train_datasets: dict[str, torch.utils.data.Dataset] = {"wflw": train_dataset}
    train_samplers: dict[str, Any] = {"wflw": train_sampler}
    if auxiliary_enabled:
        for task in ("lapa", "300w_lp"):
            options = auxiliary.get("datasets", {}).get(task, {})
            if not bool(options.get("enabled", True)):
                continue
            root = resolve_path(options.get("root"), must_exist=True)
            index = resolve_path(options.get("index_file"), must_exist=True)
            assert root is not None and index is not None
            dataset = AuxiliaryLandmarkDataset(
                task=task,
                root=root,
                index_file=index,
                augmentation=training_augmentation,
                landmark_box_expansion=float(
                    options.get("landmark_box_expansion", 1.25)
                ),
                parsing_enabled=(
                    task == "lapa" and bool(options.get("parsing_enabled", True))
                ),
                parsing_ignore_artificial_occlusion=bool(
                    options.get("parsing_ignore_artificial_occlusion", True)
                ),
            )
            train_datasets[task] = dataset
            train_samplers[task] = DistributedEpochFractionSampler(
                len(dataset),
                fraction=fractions[task],
                num_replicas=world_size,
                rank=rank,
                seed=seed + len(train_samplers) * 10_007,
            )

    data_generators = {
        task: torch.Generator().manual_seed(seed + rank + index * 100_003)
        for index, task in enumerate(train_datasets)
    }
    train_loaders = {
        task: DataLoader(
            dataset,
            batch_size=per_gpu_batch,
            shuffle=train_samplers[task] is None,
            sampler=train_samplers[task],
            num_workers=(
                workers
                if task == "wflw"
                else int(auxiliary.get("num_workers_per_auxiliary_loader", 1))
            ),
            pin_memory=True,
            persistent_workers=False,
            drop_last=not auxiliary_enabled,
            worker_init_fn=seed_worker,
            generator=data_generators[task],
        )
        for task, dataset in train_datasets.items()
    }
    train_loader = train_loaders["wflw"]
    task_counts = {task: len(loader) for task, loader in train_loaders.items()}
    sampling_plan = None
    if auxiliary_enabled:
        sampling_plan = {
            task: {
                "dataset_samples": len(train_datasets[task]),
                "epoch_fraction": fractions[task],
                "unique_samples_per_epoch": train_samplers[task].selected_samples,
                "ddp_padding_samples": train_samplers[task].padding_samples,
                "batches_per_rank": task_counts[task],
            }
            for task in train_loaders
        }
    selection_loader = (
        DataLoader(
            selection_dataset,
            batch_size=int(protocol.get("eval_batch_size_per_gpu", per_gpu_batch)),
            shuffle=False,
            sampler=selection_sampler,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
            worker_init_fn=seed_worker,
        )
        if selection_dataset is not None
        else None
    )

    if stage == "final":
        selection_path = resolve_path(selection, must_exist=True)
        assert selection_path is not None
        with selection_path.open("r", encoding="utf-8") as handle:
            final_epochs = int(json.load(handle)["best_epoch"])
        if final_epochs <= 0:
            raise ValueError("selection.json contains an invalid best_epoch.")
        epochs = final_epochs
    else:
        epochs = int(protocol["epochs"])

    model = build_model(config)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    initialization_report = None
    initialization_path = (
        resolve_path(initialization, must_exist=True) if initialization else None
    )
    if initialization_path is not None:
        initialization_report = initialize_shared_weights(
            model,
            initialization_path,
            target_config=config,
        )
    model.to(device)
    ema = ModelEMA(model, decay=float(protocol.get("ema_decay", 0.999)))
    optimizer = build_optimizer(model, protocol)
    scheduler = build_scheduler(optimizer, protocol, total_epochs=epochs)
    collapse_monitor = HeatmapCollapseMonitor.from_objective(
        config["model"].get("objective")
    )
    amp_enabled = bool(protocol.get("amp", False)) and device.type == "cuda"
    amp_dtype = str(protocol.get("amp_dtype", "bf16"))
    scaler = _make_grad_scaler(amp_enabled and amp_dtype.lower() == "fp16")
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    start_epoch, best_epoch, best_nme = 0, 0, float("inf")
    resume_path = resolve_path(resume) if resume else None
    if resume_path:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        start_epoch, best_epoch, best_nme = _resume(
            resume_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            data_generators=data_generators,
            collapse_monitor=collapse_monitor,
            rank=rank,
        )

    tracker = None
    if is_main_process():
        write_json(output_dir / "resolved_config.json", public_config(config))
        metadata = {
            **_metadata(config, unwrap_model(model)),
            "stage": stage,
            "planned_epochs": epochs,
            "auxiliary_sampling_plan": sampling_plan,
            "shared_initialization": initialization_report,
            "selection_file": (
                str(resolve_path(selection, must_exist=True))
                if stage == "final"
                else None
            ),
        }
        write_json(output_dir / "run_metadata.json", metadata)
        method = configured_output.name
        wandb_config = config["wandb"]
        tracker = WandbTracker.start(
            method=method,
            backbone=str(config["backbone"]["name"]),
            stage=stage,
            output_dir=output_dir,
            config=public_config(config),
            metadata=metadata,
            resume_checkpoint=str(resume_path) if resume_path else "",
            entity=str(wandb_config["entity"]),
            project=str(wandb_config["project"]),
            api_key=str(wandb_config["api_key"]),
        )
        metadata["wandb"] = {
            "entity": str(wandb_config["entity"]),
            "project": str(wandb_config["project"]),
            "run_id": tracker.run_id,
            "name": tracker.name,
            "url": tracker.url,
        }
        write_json(output_dir / "run_metadata.json", metadata)
        print(
            f"stage={stage} backbone={config['backbone']['name']} "
            f"samples={len(train_dataset)} epochs={epochs} world={world_size} "
            f"micro={per_gpu_batch} accumulation={accumulation_steps} "
            f"effective_batch={effective_batch} precision="
            f"{amp_dtype if amp_enabled else 'fp32'} "
            f"optimizer={optimizer.__class__.__name__} "
            f"scheduler={protocol.get('scheduler', {}).get('name', 'multistep')} "
            f"warmup_epochs="
            f"{protocol.get('scheduler', {}).get('warmup_epochs', 0)} "
            f"wandb_name={tracker.name} wandb_url={tracker.url}"
        )

    history_path = output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):

        epoch_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        for sampler in train_samplers.values():
            if sampler is not None:
                sampler.set_epoch(epoch)
        model.train()

        totals = torch.zeros(8, dtype=torch.float64, device=device)
        epoch_start = time.time()
        schedule = (
            auxiliary_task_schedule(task_counts, seed=seed, epoch=epoch)
            if auxiliary_enabled
            else ["wflw"] * len(train_loader)
        )
        iterators = {task: iter(loader) for task, loader in train_loaders.items()}
        cycles = {task: 0 for task in train_loaders}
        for step, task in enumerate(schedule):
            try:
                batch = next(iterators[task])
            except StopIteration:
                cycles[task] += 1
                sampler = train_samplers[task]
                if sampler is not None:
                    sampler.set_epoch(epoch * 10_000 + cycles[task])
                iterators[task] = iter(train_loaders[task])
                batch = next(iterators[task])
            batch_tasks = batch.get("task", [task])
            if any(value != task for value in batch_tasks):
                raise RuntimeError("A training batch contains mixed landmark schemas.")
            group_start = (step // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(schedule) - group_start)
            should_step = (step - group_start + 1) == group_size
            synchronization = (
                contextlib.nullcontext()
                if should_step or world_size == 1
                else model.no_sync()
            )
            with synchronization:
                with autocast_context(amp_enabled, amp_dtype):
                    images = batch["image"].to(device, non_blocking=True)
                    landmarks = batch["landmarks_canvas"].to(device, non_blocking=True)
                    outputs = model(images, task=task)

                    if auxiliary_enabled:
                        parsing_target = (
                            batch["parsing_mask"].to(device, non_blocking=True)
                            if "parsing_mask" in batch
                            else None
                        )
                        parsing_valid_mask = (
                            batch["parsing_valid_mask"].to(device, non_blocking=True)
                            if "parsing_valid_mask" in batch
                            else None
                        )
                        losses = task_losses(
                            outputs,
                            landmarks,
                            task=task,
                            auxiliary_training=auxiliary,
                            objective=config["model"].get("objective"),
                            parsing_target=parsing_target,
                            parsing_valid_mask=parsing_valid_mask,
                        )
                    else:
                        losses = landmark_losses(
                            outputs,
                            landmarks,
                            canvas_size=int(config["model"].get("canvas_size", 512)),
                            objective=config["model"].get("objective"),
                        )
                    if not torch.isfinite(losses["loss"]):
                        raise FloatingPointError(
                            f"Non-finite loss in task={task}, epoch={epoch + 1}, "
                            f"step={step + 1}."
                        )
                    scaled_loss = losses["loss"] / group_size
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            if should_step:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(unwrap_model(model))

            if task == "wflw":
                batch_size = batch["image"].shape[0]
                raw_landmark_loss = losses["loss"]
                totals[0] += raw_landmark_loss.detach().double() * batch_size
                totals[1] += losses["coordinate"].detach().double() * batch_size
                totals[2] += losses["heatmap"].detach().double() * batch_size
                totals[3] += batch_size
                if "heatmap_max" in losses:
                    totals[4] += losses["heatmap_max"].double() * batch_size
                    totals[5] += losses["heatmap_std"].double() * batch_size
                    totals[6] += losses["heatmap_peak_response"].double() * batch_size
                    totals[7] += (
                        losses["heatmap_collapsed_fraction"].double() * batch_size
                    )
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        scheduler.step()
        train_metrics = {
            "loss": float((totals[0] / totals[3]).item()),
            "coordinate_loss": float((totals[1] / totals[3]).item()),
            "heatmap_loss": float((totals[2] / totals[3]).item()),
        }
        if collapse_monitor.enabled:
            train_metrics.update(
                {
                    "heatmap_max": float((totals[4] / totals[3]).item()),
                    "heatmap_std": float((totals[5] / totals[3]).item()),
                    "heatmap_peak_response": float((totals[6] / totals[3]).item()),
                    "heatmap_collapsed_fraction": float((totals[7] / totals[3]).item()),
                }
            )
        collapse_message = collapse_monitor.update(
            epoch=epoch + 1, train_metrics=train_metrics
        )
        selection_metrics = None
        is_best = False
        if stage in {"benchmark", "dev"}:
            assert selection_loader is not None and selection_dataset is not None
            selection_metrics, _ = evaluate_model(
                ema.module,
                selection_loader,
                selection_dataset,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                description=(
                    "WFLW official test selection"
                    if stage == "benchmark"
                    else "Development validation"
                ),
            )
            candidate = float(selection_metrics["nme_interocular"])
            if is_main_process() and candidate < best_nme:
                best_nme, best_epoch, is_best = candidate, epoch + 1, True
            if dist.is_initialized():
                decision = [best_nme, best_epoch, is_best]
                dist.broadcast_object_list(decision, src=0)
                best_nme, best_epoch, is_best = decision

        row = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": selection_metrics if stage == "dev" else None,
            "official_test": selection_metrics if stage == "benchmark" else None,
            "encoder_lr": epoch_lrs[0],
            "head_lr": epoch_lrs[1],
            "seconds": time.time() - epoch_start,
            "collapse": (
                {
                    "consecutive_epochs": collapse_monitor.consecutive_epochs,
                    "triggered": collapse_message is not None,
                }
                if collapse_monitor.enabled
                else None
            ),
        }
        if is_main_process():
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            assert tracker is not None
            tracker.log_epoch(
                row,
                best_epoch=best_epoch,
                best_nme=best_nme,
                is_best=is_best,
            )
            print(
                format_epoch_line(
                    row,
                    total_epochs=epochs,
                    best_epoch=best_epoch,
                    best_nme=best_nme,
                    is_best=is_best,
                ),
                flush=True,
            )
        _save_training_checkpoint(
            output_dir / "latest.pt",
            epoch=epoch,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            best_epoch=best_epoch,
            best_nme=best_nme,
            data_generators=data_generators,
            collapse_monitor=collapse_monitor,
        )
        if stage in {"benchmark", "dev"} and is_best:
            _save_evaluation_checkpoint(
                output_dir / "best_ema.pt",
                epoch=epoch,
                ema=ema,
                config=config,
                metrics=selection_metrics,
            )
            if is_main_process():
                selection_payload = {
                    "best_epoch": best_epoch,
                    "selection_nme_interocular": best_nme,
                    "selection_metric": (
                        "EMA official-test inter-ocular NME"
                        if stage == "benchmark"
                        else "EMA validation inter-ocular NME"
                    ),
                    "selection_split": (
                        "WFLW official test"
                        if stage == "benchmark"
                        else "deterministic development validation"
                    ),
                    "protocol": (
                        "FaRL historical test-best"
                        if stage == "benchmark"
                        else "validation-selected"
                    ),
                }
                if stage == "dev":
                    selection_payload["validation_nme_interocular"] = best_nme
                else:
                    selection_payload["official_test_nme_interocular"] = best_nme
                write_json(
                    output_dir / "selection.json",
                    selection_payload,
                )
        barrier()
        if collapse_message is not None:
            if is_main_process():
                print(f"FATAL: {collapse_message}", flush=True)
                assert tracker is not None
                tracker.finish(exit_code=1)
            cleanup_distributed()
            raise RuntimeError(collapse_message)

    if stage == "final":
        _save_evaluation_checkpoint(
            output_dir / "best_ema.pt",
            epoch=epochs - 1,
            ema=ema,
            config=config,
            metrics=None,
        )
    barrier()
    if tracker is not None:
        tracker.finish()
    cleanup_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the controlled WFLW benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=("benchmark", "dev", "final"),
        required=True,
        help=(
            "benchmark reproduces FaRL's full-train/test-best protocol; "
            "dev/final provide the validation-selected alternative"
        ),
    )
    parser.add_argument("--selection", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--initialization",
        default="",
        help=(
            "Optional LaPa transfer checkpoint. Only shared backbone/FPN/UPer "
            "features are loaded; optimizer, EMA, epoch, and WFLW heads start fresh."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional isolated base output directory for this training run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        load_config(args.config),
        stage=args.stage,
        selection=args.selection,
        resume=args.resume,
        initialization=args.initialization,
        output_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
