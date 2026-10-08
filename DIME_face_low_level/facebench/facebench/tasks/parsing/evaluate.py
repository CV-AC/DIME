from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from .config import (
    apply_overrides,
    experiment_output_dir,
    load_config,
    resolve_path,
)
from .data import build_dataset, build_loader
from .engine import evaluate_model
from .model import build_model, load_finetuned_checkpoint
from .utils import (
    cleanup_distributed,
    ddp_options,
    init_distributed,
    is_main_process,
    setup_runtime,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a face-parsing checkpoint.")
    parser.add_argument("--config", required=True, help="Experiment YAML file.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Fine-tuned checkpoint; overrides evaluation.checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="",
        help="Evaluation split; overrides evaluation.split.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save class-index PNG predictions.",
    )
    parser.add_argument(
        "--raw-model",
        action="store_true",
        help="Evaluate raw model weights instead of EMA weights.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted YAML value; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    checkpoint_value = args.checkpoint or config["evaluation"].get("checkpoint", "")
    checkpoint_path = resolve_path(checkpoint_value, must_exist=True)
    assert checkpoint_path is not None
    split = args.split or str(config["evaluation"].get("split", "test"))
    use_ema = not args.raw_model and bool(config["evaluation"].get("use_ema", True))

    setup_runtime()
    rank, _, world_size, device = init_distributed()
    try:
        dataset = build_dataset(config, split)
        loader, _ = build_loader(config, dataset, rank=rank, world_size=world_size)
        model = build_model(config, initialize_pretrained=False).to(device)
        if is_main_process():
            load_finetuned_checkpoint(model, checkpoint_path, use_ema=use_ema)
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                **ddp_options(config),
            )
        output_dir = experiment_output_dir(config)
        save_predictions = args.save_predictions or bool(
            config["evaluation"].get("save_predictions", False)
        )
        prediction_dir = (
            output_dir / "predictions" / split if save_predictions else None
        )
        metrics = evaluate_model(
            model,
            loader,
            dataset.space,
            device,
            amp_enabled=bool(config["runtime"].get("amp_enabled", False)),
            amp_dtype=str(config["runtime"].get("amp_dtype", "fp16")),
            canvas_size=int(config["augmentation"]["canvas_size"]),
            warp_factor=float(config["augmentation"]["warp_factor"]),
            prediction_dir=prediction_dir,
        )
        if is_main_process():
            output_path = output_dir / f"{split}_metrics.json"
            write_json(output_path, metrics)
            concise = {
                "checkpoint": str(checkpoint_path),
                "weights": "ema" if use_ema else "raw",
                "split": split,
                "foreground_mean_f1": metrics["foreground_mean_f1"],
                "foreground_mean_iou": metrics["foreground_mean_iou"],
                "pixel_accuracy": metrics["pixel_accuracy"],
                "metrics_file": str(output_path),
            }
            print(json.dumps(concise, indent=2, ensure_ascii=False))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
