from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .data import build_dataset
from .model import build_model, landmark_losses
from .utils import parameter_counts, setup_runtime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One real WFLW forward/backward for checkpoint and memory preflight."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 2:
        raise ValueError("Use batch-size >=2 for a single-process SyncBN smoke test.")
    if not torch.cuda.is_available():
        raise RuntimeError("The full 448 smoke test requires a ROCm/CUDA GPU.")

    config = load_config(args.config)
    data_root = resolve_path(config["dataset"]["root"], must_exist=True)
    assert data_root is not None
    setup_runtime()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    dataset = build_dataset(
        data_root,
        "dev_train",
        augmentation=config.get("augmentation"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    batch = next(iter(loader))
    model = build_model(config)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device).train()
    images = batch["image"].to(device)
    outputs = model(images)
    losses = landmark_losses(
        outputs,
        batch["landmarks_canvas"].to(device),
        canvas_size=int(config["model"].get("canvas_size", 512)),
        objective=config["model"].get("objective"),
    )
    losses["loss"].backward()

    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if unused:
        raise RuntimeError(
            "Trainable parameters did not participate in the landmark loss: "
            + ", ".join(unused)
        )

    encoder_gradient = next(
        (
            parameter.grad
            for parameter in model.backbone.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ),
        None,
    )
    head_gradient = next(
        (
            parameter.grad
            for parameter in model.head.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ),
        None,
    )
    if encoder_gradient is None or not torch.isfinite(encoder_gradient).all():
        raise RuntimeError("Encoder did not receive a finite gradient.")
    if head_gradient is None or not torch.isfinite(head_gradient).all():
        raise RuntimeError("Landmark head did not receive a finite gradient.")

    result = {
        "status": "ok",
        "backbone": config["backbone"]["name"],
        "batch_size": args.batch_size,
        "heatmap_shape": list(outputs["heatmap_logits"].shape),
        "loss": float(losses["loss"].detach().cpu()),
        "coordinate_loss": float(losses["coordinate"].detach().cpu()),
        "heatmap_loss": float(losses["heatmap"].detach().cpu()),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "objective": config["model"].get("objective", {}).get("name", "farl"),
        **parameter_counts(model),
    }
    for key in (
        "heatmap_max",
        "heatmap_std",
        "heatmap_peak_response",
        "heatmap_collapsed_fraction",
    ):
        if key in losses:
            result[key] = float(losses[key].cpu())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
