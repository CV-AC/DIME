from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import torch

from .config import apply_overrides, experiment_output_dir, load_config, resolve_path
from .data import build_dataset
from .utils import runtime_versions


def _check_backbone(config: dict[str, Any]) -> dict[str, Any]:
    options = config["backbone"]
    name = str(options["name"]).lower()
    result: dict[str, Any] = {"name": name}
    if name in {"farl", "dime"} or options.get("checkpoint"):
        checkpoint = resolve_path(options.get("checkpoint"), must_exist=True)
        assert checkpoint is not None
        result["checkpoint"] = str(checkpoint)
        result["checkpoint_bytes"] = checkpoint.stat().st_size
    if name == "farl":
        result["source"] = str(resolve_path(options["farl_source"], must_exist=True))
    elif name == "dime":
        result["source"] = str(resolve_path(options["dime_source"], must_exist=True))
    elif name in {"dino", "mae"}:
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm is required for DINO/MAE.") from exc
        result["timm"] = timm.__version__
        result["timm_model"] = str(options["timm_model"])
        result["requires_successful_timm_prefetch"] = not bool(
            options.get("checkpoint")
        )
    else:
        raise ValueError(f"Unsupported backbone {name!r}.")
    return result


def check_config(
    config_path: str,
    overrides: list[str],
    *,
    require_gpu: bool,
) -> dict[str, Any]:
    config = apply_overrides(load_config(config_path), overrides)
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm/CUDA PyTorch and a visible GPU are required. Use --allow-cpu "
            "only for filesystem/configuration checks."
        )
    datasets: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        dataset = build_dataset(config, split)
        if not dataset.samples:
            raise RuntimeError(f"{config_path}: {split} is empty.")
        checked = []
        for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
            sample = dataset.samples[index]
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            label = cv2.imread(str(sample.label_path), cv2.IMREAD_GRAYSCALE)
            if image is None or label is None:
                raise FileNotFoundError(
                    f"Could not decode preflight sample {sample.sample_id}: "
                    f"{sample.image_path}, {sample.label_path}"
                )
            checked.append(sample.sample_id)
        datasets[split] = {"count": len(dataset), "decoded": checked}

    workers = int(config["loader"].get("workers_per_gpu", 0))
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    warnings: list[str] = []
    if allocated_cpus and workers > allocated_cpus:
        warnings.append(
            f"workers_per_gpu={workers} exceeds SLURM_CPUS_PER_TASK={allocated_cpus}."
        )
    output_dir = experiment_output_dir(config)
    artifacts = [
        name
        for name in ("metrics.jsonl", "last.pt", "best.pt")
        if (output_dir / name).exists()
    ]
    if artifacts:
        warnings.append(
            f"output_dir already contains run artifacts: {', '.join(artifacts)}"
        )
    return {
        "config": str(Path(config_path).resolve()),
        "dataset": str(config["dataset"]["name"]),
        "datasets": datasets,
        "backbone": _check_backbone(config),
        "output_dir": str(output_dir),
        "effective_train_batch": int(config["loader"]["batch_size_per_gpu"])
        * int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1"))),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "versions": runtime_versions(),
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check data, manifests, dependencies and weights before a run."
    )
    parser.add_argument(
        "--config", action="append", required=True, help="YAML config; repeatable."
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted YAML value for every config.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not require a visible ROCm/CUDA GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        check_config(path, args.set, require_gpu=not args.allow_cpu)
        for path in args.config
    ]
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
