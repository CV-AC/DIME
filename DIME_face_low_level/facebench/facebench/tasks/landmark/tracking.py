from __future__ import annotations

import atexit
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WANDB_ENTITY = ""
WANDB_PROJECT = "DIME_landmark"


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "run"


def new_run_identity(method: str, stage: str) -> tuple[str, str]:

    run_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = _slug(os.environ.get("SLURM_JOB_ID", "local"))
    nonce = run_id[:8]
    name = f"{_slug(method)}-{_slug(stage)}-{job_id}-{timestamp}-{nonce}"
    return run_id, name


def epoch_log_payload(
    row: dict[str, Any],
    *,
    best_epoch: int,
    best_nme: float,
    is_best: bool,
) -> dict[str, int | float]:
    payload: dict[str, int | float] = {
        "epoch": int(row["epoch"]),
        "train/loss": float(row["train"]["loss"]),
        "train/coordinate_loss": float(row["train"]["coordinate_loss"]),
        "train/heatmap_loss": float(row["train"]["heatmap_loss"]),
        "lr/encoder": float(row["encoder_lr"]),
        "lr/head": float(row["head_lr"]),
        "time/epoch_seconds": float(row["seconds"]),
        "selection/is_best": int(is_best),
    }
    for source, destination in (
        ("heatmap_max", "diagnostics/heatmap_max"),
        ("heatmap_std", "diagnostics/heatmap_std"),
        ("heatmap_peak_response", "diagnostics/heatmap_peak_response"),
        (
            "heatmap_collapsed_fraction",
            "diagnostics/heatmap_collapsed_fraction",
        ),
    ):
        if source in row["train"]:
            payload[destination] = float(row["train"][source])
    collapse = row.get("collapse")
    if collapse is not None:
        payload["collapse/consecutive_epochs"] = int(collapse["consecutive_epochs"])
        payload["collapse/triggered"] = int(bool(collapse["triggered"]))
    if best_epoch > 0 and math.isfinite(best_nme):
        payload["best/epoch"] = int(best_epoch)
        payload["best/nme_interocular"] = float(best_nme)

    def add_metrics(prefix: str, metrics: dict[str, Any]) -> None:
        payload[f"{prefix}/nme_interocular"] = float(metrics["nme_interocular"])
        payload[f"{prefix}/nme_interpupil_diagnostic"] = float(
            metrics["nme_interpupil_diagnostic"]
        )
        payload[f"{prefix}/fr_0.10"] = float(metrics["fr_0.10"])
        payload[f"{prefix}/auc_0.10"] = float(metrics["auc_0.10"])
        for subset, subset_metrics in metrics["subsets"].items():
            payload[f"{prefix}/subsets/{subset}/nme_interocular"] = float(
                subset_metrics["nme_interocular"]
            )

    split_name = ""
    split_metrics = None
    if row.get("official_test") is not None:
        split_name, split_metrics = "test", row["official_test"]
    elif row.get("validation") is not None:
        split_name, split_metrics = "validation", row["validation"]

    if split_metrics is not None:
        add_metrics(split_name, split_metrics)
    return payload


def format_epoch_line(
    row: dict[str, Any],
    *,
    total_epochs: int,
    best_epoch: int,
    best_nme: float,
    is_best: bool,
) -> str:
    train = row["train"]
    fields = [
        f"[{int(row['epoch']):03d}/{int(total_epochs):03d}]",
        (
            f"train loss={float(train['loss']):.5f} "
            f"(coord={float(train['coordinate_loss']):.5f}, "
            f"heat={float(train['heatmap_loss']):.5f})"
        ),
    ]
    if "heatmap_max" in train:
        fields.append(
            "heatmap "
            f"max={float(train['heatmap_max']):.4f} "
            f"std={float(train['heatmap_std']):.6f} "
            f"GT={float(train['heatmap_peak_response']):.4f} "
            f"flat={100.0 * float(train['heatmap_collapsed_fraction']):.1f}%"
        )
    split_name = ""
    split_metrics = None
    if row.get("official_test") is not None:
        split_name, split_metrics = "test", row["official_test"]
    elif row.get("validation") is not None:
        split_name, split_metrics = "val", row["validation"]
    if split_metrics is not None:
        fields.append(
            f"{split_name.upper()} "
            f"NME={float(split_metrics['nme_interocular']):.3f} "
            f"FR10={float(split_metrics['fr_0.10']):.2f} "
            f"AUC10={float(split_metrics['auc_0.10']):.2f}"
        )
    if best_epoch > 0 and math.isfinite(best_nme):
        marker = " NEW_BEST" if is_best else ""
        fields.append(f"best NME={float(best_nme):.3f}@{int(best_epoch):03d}{marker}")
    fields.append(f"{float(row['seconds']):.1f}s")
    return " | ".join(fields)


@dataclass
class HeatmapCollapseMonitor:

    enabled: bool
    warmup_epochs: int
    patience: int
    fraction_threshold: float
    consecutive_epochs: int = 0

    @classmethod
    def from_objective(cls, objective: dict[str, Any] | None) -> HeatmapCollapseMonitor:
        objective = dict(objective or {})
        is_route_a = str(objective.get("name", "farl")).strip().lower() == "route_a"
        options = dict(objective.get("collapse_detection", {}))
        enabled = is_route_a and bool(options.get("enabled", True))
        warmup_epochs = int(options.get("warmup_epochs", 2))
        patience = int(options.get("patience", 2))
        fraction_threshold = float(options.get("fraction_threshold", 0.95))
        if warmup_epochs < 0 or patience <= 0:
            raise ValueError(
                "collapse_detection requires warmup_epochs>=0 and patience>0."
            )
        if not 0.0 <= fraction_threshold <= 1.0:
            raise ValueError("collapse_detection.fraction_threshold must be in [0,1].")
        return cls(
            enabled=enabled,
            warmup_epochs=warmup_epochs,
            patience=patience,
            fraction_threshold=fraction_threshold,
        )

    def update(self, *, epoch: int, train_metrics: dict[str, float]) -> str | None:
        if not self.enabled or epoch <= self.warmup_epochs:
            self.consecutive_epochs = 0
            return None
        fraction = float(train_metrics["heatmap_collapsed_fraction"])
        if not math.isfinite(fraction):
            raise FloatingPointError("Non-finite Route-A heatmap collapse diagnostic.")
        if fraction >= self.fraction_threshold:
            self.consecutive_epochs += 1
        else:
            self.consecutive_epochs = 0
        if self.consecutive_epochs < self.patience:
            return None
        return (
            "Route A heatmaps collapsed: "
            f"{fraction:.1%} have spatial std below the configured threshold "
            f"for {self.consecutive_epochs} consecutive epochs "
            f"(epoch {epoch}). Training was stopped to avoid selecting an "
            "invalid flat-heatmap checkpoint."
        )

    def state_dict(self) -> dict[str, int]:
        return {"consecutive_epochs": int(self.consecutive_epochs)}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        consecutive = int(state.get("consecutive_epochs", 0))
        if consecutive < 0:
            raise ValueError("Invalid negative collapse-monitor state.")
        self.consecutive_epochs = consecutive


@dataclass
class WandbTracker:
    run: Any
    run_id: str
    name: str
    url: str
    _finished: bool = False

    @classmethod
    def start(
        cls,
        *,
        method: str,
        backbone: str,
        stage: str,
        dataset: str = "wflw",
        output_dir: Path,
        config: dict[str, Any],
        metadata: dict[str, Any],
        resume_checkpoint: str,
        entity: str,
        project: str,
        api_key: str,
    ) -> WandbTracker:
        if not bool(config.get("wandb", {}).get("enabled", True)):
            return cls(run=_LocalRun(), run_id="", name="", url="")
        entity = entity.strip()
        project = project.strip()
        api_key = api_key.strip() or os.environ.get("WANDB_API_KEY", "").strip()
        if not entity or not project or not api_key:
            raise ValueError(
                "wandb.entity, wandb.project, and wandb.api_key must be set in "
                "configs/wflw/base.yaml."
            )
        os.environ["WANDB_API_KEY"] = api_key
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B is required for landmark training. Run "
                "bash scripts/lumi/install_deps.sh in the LUMI environment."
            ) from error

        run_id, name = new_run_identity(method, stage)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            run = wandb.init(
                entity=entity,
                project=project,
                id=run_id,
                name=name,
                group=f"{_slug(dataset)}-{stage}",
                job_type=stage,
                tags=["landmark", dataset, method, backbone, stage],
                notes=(
                    "Controlled DIME landmark benchmark on WFLW."
                    if dataset == "wflw"
                    else f"DIME landmark transfer stage on {dataset}."
                ),
                config={
                    "method": method,
                    "backbone": backbone,
                    "stage": stage,
                    "dataset": dataset,
                    "resume_checkpoint": resume_checkpoint or None,
                    "resolved_config": config,
                    "run_metadata": metadata,
                },
                dir=str(output_dir),
                mode="online",
                resume="never",
                force=True,
                reinit="finish_previous",
                settings=wandb.Settings(
                    console="wrap",
                    silent=True,
                    init_timeout=120,
                ),
            )
        except Exception as error:
            raise RuntimeError(
                "W&B online initialization failed. Check the wandb entity, "
                "project, api_key, and compute-node network access configured "
                "in configs/wflw/base.yaml."
            ) from error
        if run is None:
            raise RuntimeError("wandb.init returned no online run.")

        run.define_metric("epoch")
        run.define_metric("train/loss", step_metric="epoch", summary="min")
        run.define_metric("train/coordinate_loss", step_metric="epoch", summary="min")
        run.define_metric("train/heatmap_loss", step_metric="epoch", summary="min")
        run.define_metric("diagnostics/*", step_metric="epoch")
        run.define_metric("collapse/*", step_metric="epoch")
        run.define_metric("lr/*", step_metric="epoch")
        run.define_metric("time/*", step_metric="epoch")
        run.define_metric("test/nme_interocular", step_metric="epoch", summary="min")
        run.define_metric("test/fr_0.10", step_metric="epoch", summary="min")
        run.define_metric("test/auc_0.10", step_metric="epoch", summary="max")
        run.define_metric("test/subsets/*", step_metric="epoch", summary="min")
        run.define_metric(
            "validation/nme_interocular", step_metric="epoch", summary="min"
        )
        run.define_metric("validation/fr_0.10", step_metric="epoch", summary="min")
        run.define_metric("validation/auc_0.10", step_metric="epoch", summary="max")
        run.define_metric("validation/subsets/*", step_metric="epoch", summary="min")
        tracker = cls(
            run=run,
            run_id=run_id,
            name=name,
            url=str(getattr(run, "url", "")),
        )

        atexit.register(tracker.finish, 1)
        return tracker

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        self.run.log(payload, step=int(step))

    def log_epoch(
        self,
        row: dict[str, Any],
        *,
        best_epoch: int,
        best_nme: float,
        is_best: bool,
    ) -> None:
        payload = epoch_log_payload(
            row,
            best_epoch=best_epoch,
            best_nme=best_nme,
            is_best=is_best,
        )
        self.run.log(payload, step=int(row["epoch"]))
        if best_epoch > 0 and math.isfinite(best_nme):
            self.run.summary["best/epoch"] = int(best_epoch)
            self.run.summary["best/nme_interocular"] = float(best_nme)

    def finish(self, exit_code: int = 0) -> None:
        if self._finished:
            return
        self._finished = True
        self.run.finish(exit_code=exit_code)


class _LocalRun:
    def __init__(self):
        self.summary = {}

    def log(self, payload, step=None):
        pass

    def finish(self, exit_code=0):
        pass
