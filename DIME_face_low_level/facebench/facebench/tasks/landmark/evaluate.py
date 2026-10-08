from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import (
    LANDMARK_ROOT,
    experiment_output_dir,
    load_config,
    public_config,
    resolve_path,
)
from .data import DistributedEvalSampler, build_dataset
from .engine import evaluate_model, save_predictions
from .model import build_model, load_finetuned_checkpoint
from .utils import (
    barrier,
    cleanup_distributed,
    git_commit,
    init_distributed,
    is_main_process,
    parameter_counts,
    runtime_versions,
    seed_worker,
    sha256_file,
    setup_runtime,
    write_json,
)


def evaluate(
    config: dict,
    *,
    checkpoint_override: str = "",
    output_override: str = "",
) -> dict:
    checkpoint_value = checkpoint_override or str(
        config.get("evaluation", {}).get("checkpoint", "")
    )
    checkpoint = resolve_path(checkpoint_value, must_exist=True)
    assert checkpoint is not None
    data_root = resolve_path(config["dataset"]["root"], must_exist=True)
    assert data_root is not None
    if config["backbone"]["name"] == "dime":
        resolve_path(config["backbone"].get("checkpoint"), must_exist=True)

    setup_runtime()
    rank, _, world_size, device = init_distributed()

    initialize = config["backbone"]["name"] == "dime"
    model = build_model(config, initialize_pretrained=initialize)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    checkpoint_data = load_finetuned_checkpoint(model, checkpoint)
    model.to(device).eval()

    dataset = build_dataset(data_root, "test")
    sampler = (
        DistributedEvalSampler(dataset, rank, world_size) if world_size > 1 else None
    )
    protocol = config["protocol"]
    workers = int(protocol.get("num_workers", 6))
    loader = DataLoader(
        dataset,
        batch_size=int(protocol.get("eval_batch_size_per_gpu", 5)),
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    amp_enabled = bool(protocol.get("amp", False)) and device.type == "cuda"
    metrics, payload = evaluate_model(
        model,
        loader,
        dataset,
        device,
        amp_enabled=amp_enabled,
        amp_dtype=str(protocol.get("amp_dtype", "bf16")),
        description="WFLW test",
    )
    if is_main_process():
        base_output = experiment_output_dir(config)
        audit_path = resolve_path(config["dataset"].get("audit_file"))
        dataset_manifest = "unavailable"
        if audit_path is not None and audit_path.is_file():
            with audit_path.open("r", encoding="utf-8") as handle:
                dataset_manifest = json.load(handle).get(
                    "manifest_sha256", "unavailable"
                )
        output_dir = (
            Path(output_override).expanduser().resolve()
            if output_override
            else base_output / "final" / "evaluation"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics.update(
            {
                "dataset": "WFLW official test",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "dataset_manifest_sha256": dataset_manifest,
                "checkpoint_epoch": (
                    int(checkpoint_data["epoch"]) + 1
                    if checkpoint_data.get("epoch") is not None
                    else None
                ),
                "normalization": "GT landmarks 60-72 inter-ocular",
                "config": public_config(config),
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
                "git_commit": git_commit(LANDMARK_ROOT),
                "versions": runtime_versions(),
            }
        )
        write_json(output_dir / "test_metrics.json", metrics)
        assert payload is not None
        save_predictions(output_dir / "test_predictions.npz", payload)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    barrier()
    cleanup_distributed()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate the selected checkpoint on official WFLW test."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        load_config(args.config),
        checkpoint_override=args.checkpoint,
        output_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
