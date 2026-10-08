from __future__ import annotations

import atexit
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import public_config
from .oracle import ORACLE_SELECTION


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "run"


def new_run_identity(method: str, seed: int) -> tuple[str, str]:
    run_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = _slug(os.environ.get("SLURM_JOB_ID", "local"))
    name = f"{_slug(method)}-seed{int(seed):03d}-{job_id}-{timestamp}-{run_id[:8]}"
    return run_id, name


def _wandb_settings(config: dict[str, Any]) -> tuple[bool, str, str, str]:
    options = dict(config.get("wandb", {}))
    enabled = bool(options.get("enabled", True))
    entity = str(options.get("entity", "")).strip()
    project = str(options.get("project", "")).strip()
    mode = str(options.get("mode", "online")).strip().lower()
    if enabled and (not entity or not project):
        raise ValueError("wandb.entity and wandb.project must be configured.")
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline, or disabled.")
    configured_api_key = str(options.get("api_key", "")).strip()
    api_key_env = str(options.get("api_key_env", "WANDB_API_KEY")).strip()
    if configured_api_key:
        os.environ["WANDB_API_KEY"] = configured_api_key
    elif api_key_env != "WANDB_API_KEY" and os.environ.get(api_key_env):
        os.environ["WANDB_API_KEY"] = os.environ[api_key_env]
    if enabled and mode == "online" and not os.environ.get("WANDB_API_KEY", "").strip():
        raise RuntimeError(
            "W&B online mode requires wandb.api_key in YAML or the configured "
            f"{api_key_env} environment variable."
        )
    return enabled, entity, project, mode


@dataclass
class WandbTracker:
    run: Any | None
    run_id: str
    name: str
    url: str
    _finished: bool = False

    @classmethod
    def start(
        cls,
        *,
        method: str,
        seed: int,
        output_dir: Path,
        config: dict[str, Any],
        metadata: dict[str, Any],
        resume_checkpoint: str,
    ) -> "WandbTracker":
        enabled, entity, project, mode = _wandb_settings(config)
        run_id, name = new_run_identity(method, seed)
        if not enabled or mode == "disabled":
            return cls(None, run_id, name, "")
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install wandb before running the benchmark.") from exc
        run = wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            name=name,
            group="controlled-head-pose-strong-v1",
            job_type="train",
            tags=["head-pose", "300w-lp", method, f"seed-{seed}", "controlled"],
            notes=(
                "Controlled 300W-LP to AFLW2000/BIWI encoder comparison with "
                "a shared Linear-6D head and SO(3) loss."
            ),
            config={
                "method": method,
                "seed": int(seed),
                "resume_checkpoint": resume_checkpoint or None,
                "resolved_config": public_config(config),
                "run_metadata": metadata,
            },
            dir=str(output_dir),
            mode=mode,
            resume="never",
            force=True,
            reinit="finish_previous",
            settings=wandb.Settings(console="wrap", silent=True, init_timeout=120),
        )
        if run is None:
            raise RuntimeError("wandb.init returned no run.")
        run.define_metric("epoch")
        run.define_metric("train/geodesic_loss", step_metric="epoch", summary="min")
        run.define_metric("lr/*", step_metric="epoch")
        run.define_metric("time/*", step_metric="epoch")
        run.define_metric("throughput/*", step_metric="epoch")
        run.define_metric("oracle/*", step_metric="epoch")
        run.define_metric("oracle/score", step_metric="epoch", summary="min")
        tracker = cls(run, run_id, name, str(getattr(run, "url", "")))
        atexit.register(tracker.finish, 1)
        return tracker

    def log_epoch(self, row: dict[str, Any], *, commit: bool = True) -> None:
        if self.run is None:
            return
        payload = {
            "epoch": int(row["epoch"]),
            "train/geodesic_loss": float(row["train_geodesic_loss"]),
            "lr/encoder": row.get("encoder_lr"),
            "lr/encoder_min": row.get("encoder_lr_min"),
            "lr/head": float(row["head_lr"]),
            "time/epoch_seconds": float(row["seconds"]),
            "throughput/images_per_second": float(row["images_per_second"]),
        }
        self.run.log(payload, step=int(row["epoch"]), commit=commit)

    def log_oracle_epoch(self, row: dict[str, Any]) -> None:
        if self.run is None:
            return
        payload: dict[str, Any] = {
            "epoch": int(row["epoch"]),
            "oracle/score": float(row["oracle_score"]),
            "oracle/best_score": float(row["best_oracle_score"]),
            "oracle/best_epoch": int(row["best_oracle_epoch"]),
            "oracle/is_best": int(bool(row["is_best"])),
        }
        for split, metrics in row["splits"].items():
            for key in ("yaw_mae", "pitch_mae", "roll_mae", "mean_mae"):
                payload[f"oracle/{split}/{key}"] = float(metrics[key])
        self.run.log(payload, step=int(row["epoch"]), commit=True)
        self.run.summary["selection/mode"] = ORACLE_SELECTION
        self.run.summary["selection/warning"] = (
            "AFLW2000 and BIWI directly select the checkpoint."
        )
        self.run.summary["selection/best_epoch"] = int(row["best_oracle_epoch"])
        self.run.summary["selection/best_oracle_score"] = float(
            row["best_oracle_score"]
        )

    def finish(self, exit_code: int = 0) -> None:
        if self._finished:
            return
        self._finished = True
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


def log_evaluation_to_wandb(
    *,
    config: dict[str, Any],
    output_dir: Path,
    result: dict[str, Any],
    metrics_path: Path,
    predictions_path: Path,
) -> None:

    enabled, entity, project, mode = _wandb_settings(config)
    if not enabled or mode == "disabled":
        return
    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Cannot attach evaluation to W&B; missing {metadata_path}"
        )
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_id = str(metadata.get("wandb", {}).get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError("run_metadata.json has no W&B run ID.")
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb before evaluating the benchmark.") from exc
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must" if mode == "online" else "allow",
        mode=mode,
        dir=str(output_dir),
        reinit="finish_previous",
        settings=wandb.Settings(console="wrap", silent=True, init_timeout=120),
    )
    split = str(result["dataset"])
    payload = {
        f"test/{split}/yaw_mae": float(result["yaw_mae"]),
        f"test/{split}/pitch_mae": float(result["pitch_mae"]),
        f"test/{split}/roll_mae": float(result["roll_mae"]),
        f"test/{split}/mean_mae": float(result["mean_mae"]),
        f"test/{split}/samples": int(result["samples"]),
    }
    run.log(payload)
    for key, value in payload.items():
        run.summary[key] = value
    run.summary[f"data/{split}/manifest_sha256"] = result["dataset_manifest_sha256"]
    run.summary[f"data/{split}/storage_backend"] = result["dataset_storage_backend"]
    if result.get("dataset_lmdb_logical_content_sha256"):
        run.summary[f"data/{split}/lmdb_logical_content_sha256"] = result[
            "dataset_lmdb_logical_content_sha256"
        ]
    run.summary["checkpoint/selected_sha256"] = result["checkpoint_sha256"]
    run.summary["selection/mode"] = result["selection"]
    artifact = wandb.Artifact(
        name=f"{_slug(metadata['method'])}-seed{int(metadata['seed']):03d}-{split}",
        type="head-pose-evaluation",
        metadata=result,
    )
    artifact.add_file(str(metrics_path), name=metrics_path.name)
    artifact.add_file(str(predictions_path), name=predictions_path.name)
    run.log_artifact(artifact)
    run.finish()
