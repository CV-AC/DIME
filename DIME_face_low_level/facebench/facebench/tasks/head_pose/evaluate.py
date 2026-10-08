from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import experiment_output_dir, load_config, resolve_path
from .data import build_dataset
from .models import build_model, load_finetuned_checkpoint
from .oracle import ORACLE_SELECTION, default_checkpoint_name, selection_mode
from .rotation import rotation_mae
from .tracking import log_evaluation_to_wandb
from .utils import autocast_context, parameter_counts, sha256_file, write_json


@torch.inference_mode()
def evaluate(
    config: dict,
    split: str,
    checkpoint_override: str = "",
    *,
    seed_override: int | None = None,
    log_to_wandb: bool = True,
) -> dict:
    if seed_override is not None:
        config["seed"] = int(seed_override)
    seed = int(config.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_official = config["model"]["kind"] == "sixdrepnet_official"
    configured_selection = selection_mode(config)

    initialize_encoder = is_official or config["model"]["kind"] == "dime"
    model = build_model(config, initialize_encoder=initialize_encoder)
    checkpoint = None
    checkpoint_metadata: dict = {}
    if not is_official:
        checkpoint_value = (
            checkpoint_override
            or str(config.get("evaluation", {}).get("checkpoint", "")).strip()
        )
        if not checkpoint_value:
            checkpoint_value = str(
                experiment_output_dir(config, seed) / default_checkpoint_name(config)
            )
        checkpoint = resolve_path(checkpoint_value, must_exist=True)
        assert checkpoint is not None
        checkpoint_metadata = load_finetuned_checkpoint(model, checkpoint)
    model.to(device).eval()

    dataset = build_dataset(config, split)
    protocol = config["protocol"]
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 64)),
        shuffle=False,
        num_workers=int(protocol.get("num_workers", 8)),
        pin_memory=True,
        persistent_workers=int(protocol.get("num_workers", 8)) > 0,
    )
    amp_enabled = bool(protocol.get("amp", True)) and device.type == "cuda"
    amp_dtype = str(protocol.get("amp_dtype", "bf16"))
    errors: list[torch.Tensor] = []
    predictions: list[dict] = []

    for batch in tqdm(loader, desc=f"Evaluate {split}", unit="batch"):
        images = batch["image"].to(device, non_blocking=True)
        with autocast_context(amp_enabled, amp_dtype):
            rotations = model(images)
        pred_ypr, target_ypr, batch_errors = rotation_mae(
            rotations.float().cpu(), batch["ypr"].float()
        )
        errors.append(batch_errors)
        for sample_id, prediction, target, error in zip(
            batch["sample_id"], pred_ypr, target_ypr, batch_errors
        ):
            predictions.append(
                {
                    "sample_id": sample_id,
                    "gt_yaw": target[0].item(),
                    "gt_pitch": target[1].item(),
                    "gt_roll": target[2].item(),
                    "pred_yaw": prediction[0].item(),
                    "pred_pitch": prediction[1].item(),
                    "pred_roll": prediction[2].item(),
                    "err_yaw": error[0].item(),
                    "err_pitch": error[1].item(),
                    "err_roll": error[2].item(),
                }
            )

    all_errors = torch.cat(errors, dim=0)
    per_axis = all_errors.mean(dim=0)
    saved_selection = checkpoint_metadata.get("selection")
    if isinstance(saved_selection, dict):
        selected_mode = str(saved_selection.get("selection", configured_selection))
    elif checkpoint_override:
        selected_mode = "manual_checkpoint_override"
    else:
        selected_mode = configured_selection
    if is_official:
        selected_mode = "official_released_checkpoint"
    result = {
        "dataset": split,
        "method": str(config["experiment"]["name"]),
        "seed": seed,
        "samples": len(dataset),
        "yaw_mae": per_axis[0].item(),
        "pitch_mae": per_axis[1].item(),
        "roll_mae": per_axis[2].item(),
        "mean_mae": per_axis.mean().item(),
        "metric": "6DRepNet five-candidate wrap-aware Euler MAE (degrees)",
        "parameters": parameter_counts(model),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_storage_backend": getattr(dataset, "storage_backend", "unknown"),
        "dataset_lmdb_logical_content_sha256": (
            getattr(dataset, "storage_logical_content_sha256", None)
        ),
        "checkpoint_sha256": (
            sha256_file(checkpoint)
            if checkpoint is not None
            else str(getattr(model, "checkpoint_sha256", "official"))
        ),
        "selection": selected_mode,
        "selection_warning": (
            "AFLW2000 and BIWI were evaluated every epoch and directly selected "
            "this checkpoint; this is a test-tuned oracle result."
            if selected_mode == ORACLE_SELECTION
            else "No local test-set checkpoint selection."
        ),
        "checkpoint_epoch": (
            int(checkpoint_metadata["epoch"]) + 1
            if "epoch" in checkpoint_metadata
            else None
        ),
        "oracle_selection": checkpoint_metadata.get("selection"),
    }
    output_dir = experiment_output_dir(config, seed)
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = evaluation_dir / f"{split}_metrics.json"
    predictions_path = evaluation_dir / f"{split}_predictions.csv"
    write_json(metrics_path, result)
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    if log_to_wandb and not is_official:
        log_evaluation_to_wandb(
            config=config,
            output_dir=output_dir,
            result=result,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
        )
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AFLW2000 or BIWI.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("aflw2000", "biwi", "all"), required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    splits = ("aflw2000", "biwi") if args.dataset == "all" else (args.dataset,)
    for split in splits:
        evaluate(
            config,
            split,
            args.checkpoint,
            seed_override=args.seed,
            log_to_wandb=not args.no_wandb,
        )


if __name__ == "__main__":
    main()
