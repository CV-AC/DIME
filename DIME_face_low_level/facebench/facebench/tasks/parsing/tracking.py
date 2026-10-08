from __future__ import annotations

import atexit
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "run"


def new_run_identity(dataset: str, backbone: str) -> tuple[str, str]:
    run_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = _slug(os.environ.get("SLURM_JOB_ID", "local"))
    name = f"{_slug(dataset)}-{_slug(backbone)}-{job_id}-" f"{timestamp}-{run_id[:8]}"
    return run_id, name


def flatten_metrics(
    split: str,
    metrics: dict[str, Any],
) -> dict[str, float]:
    payload = {
        f"{split}/foreground_mean_f1": float(metrics["foreground_mean_f1"]),
        f"{split}/foreground_mean_iou": float(metrics["foreground_mean_iou"]),
        f"{split}/pixel_accuracy": float(metrics["pixel_accuracy"]),
    }
    for name, values in metrics["per_class"].items():
        payload[f"{split}/classes/{name}/f1"] = float(values["f1"])
        payload[f"{split}/classes/{name}/iou"] = float(values["iou"])
    return payload


def format_epoch_line(
    *,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    split: str,
    metrics: dict[str, Any] | None,
    best_epoch: int,
    best_score: float,
    is_best: bool,
    seconds: float,
) -> str:
    fields = [
        f"[{epoch:03d}/{total_epochs:03d}]",
        f"train CE={train_loss:.5f}",
    ]
    if metrics is not None:
        fields.append(
            f"{split.upper()} mean-F1={metrics['foreground_mean_f1']:.4f} "
            f"mIoU={metrics['foreground_mean_iou']:.4f}"
        )
    if best_epoch > 0:
        marker = " NEW_BEST" if is_best else ""
        fields.append(f"best mean-F1={best_score:.4f}@{best_epoch:03d}{marker}")
    fields.append(f"{seconds:.1f}s")
    return " | ".join(fields)


class Tracker:
    run_id = ""
    name = "disabled"
    url = ""

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        del payload, step

    def summary(self, payload: dict[str, Any]) -> None:
        del payload

    def finish(self, exit_code: int = 0) -> None:
        del exit_code


@dataclass
class WandbTracker(Tracker):
    run: Any
    run_id: str
    name: str
    url: str
    _finished: bool = False

    @classmethod
    def start(
        cls,
        *,
        dataset: str,
        backbone: str,
        output_dir: Path,
        config: dict[str, Any],
        metadata: dict[str, Any],
        options: dict[str, Any],
        resume_identity: dict[str, Any] | None = None,
    ) -> "WandbTracker":
        entity = str(options.get("entity", "")).strip()
        project = str(options.get("project", "")).strip()
        api_key = str(options.get("api_key", "")).strip()
        if not entity or not project or not api_key:
            raise ValueError(
                "wandb.entity, wandb.project, and wandb.api_key must be set."
            )
        os.environ["WANDB_API_KEY"] = api_key
        try:
            import wandb
        except (ImportError, Exception) as exc:
            raise RuntimeError(
                "W&B is enabled but cannot be imported. Install requirements.txt "
                "in a clean environment."
            ) from exc

        resume_identity = resume_identity or {}
        run_id = str(resume_identity.get("run_id", "")).strip()
        name = str(resume_identity.get("name", "")).strip()
        is_resume = bool(run_id)
        if not is_resume:
            run_id, name = new_run_identity(dataset, backbone)
        elif not name:
            name = f"{_slug(dataset)}-{_slug(backbone)}-{run_id[:8]}"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            run = wandb.init(
                entity=entity,
                project=project,
                id=run_id,
                name=name,
                group=f"{dataset}-full-finetune",
                job_type="train",
                tags=["face-parsing", dataset, backbone, "uperhead"],
                notes="Controlled FaRL-style face-parsing backbone benchmark.",
                config={
                    "resolved_config": config,
                    "run_metadata": metadata,
                },
                dir=str(output_dir),
                mode=str(options.get("mode", "online")),
                resume="allow" if is_resume else "never",
                force=True,
                reinit="finish_previous",
                settings=wandb.Settings(
                    console="wrap",
                    silent=True,
                    init_timeout=int(options.get("init_timeout", 120)),
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                "W&B initialization failed. Check the configured key/entity/project "
                "and compute-node network access."
            ) from exc
        if run is None:
            raise RuntimeError("wandb.init returned no run.")
        run.define_metric("epoch")
        run.define_metric("train/loss", step_metric="epoch", summary="min")
        run.define_metric("lr/*", step_metric="epoch")
        run.define_metric("time/*", step_metric="epoch")
        run.define_metric("val/foreground_mean_f1", step_metric="epoch", summary="max")
        run.define_metric("val/foreground_mean_iou", step_metric="epoch", summary="max")
        run.define_metric("test/foreground_mean_f1", step_metric="epoch", summary="max")
        run.define_metric(
            "test/foreground_mean_iou", step_metric="epoch", summary="max"
        )
        run.define_metric("val/classes/*", step_metric="epoch")
        run.define_metric("test/classes/*", step_metric="epoch")

        tracker = cls(
            run=run,
            run_id=run_id,
            name=name,
            url=str(getattr(run, "url", "")),
        )
        atexit.register(tracker.finish, 1)
        return tracker

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        self.run.log(payload, step=step)

    def summary(self, payload: dict[str, Any]) -> None:
        for key, value in payload.items():
            self.run.summary[key] = value

    def finish(self, exit_code: int = 0) -> None:
        if self._finished:
            return
        self._finished = True
        self.run.finish(exit_code=exit_code)


def start_tracker(
    *,
    dataset: str,
    backbone: str,
    output_dir: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    options: dict[str, Any],
    resume_identity: dict[str, Any] | None = None,
) -> Tracker:
    if not bool(options.get("enabled", False)):
        return Tracker()
    return WandbTracker.start(
        dataset=dataset,
        backbone=backbone,
        output_dir=output_dir,
        config=config,
        metadata=metadata,
        options=options,
        resume_identity=resume_identity,
    )
