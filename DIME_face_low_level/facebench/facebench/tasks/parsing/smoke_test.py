from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .config import apply_overrides, load_config
from .data import build_dataset
from .engine import ModelEMA
from .model import build_model
from .optim import build_optimizer
from .utils import autocast_context, parameter_counts, set_seed, setup_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real forward/backward/update with a parsing config."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for the SyncBN head.")
    if not torch.cuda.is_available():
        raise RuntimeError("The real-model smoke test requires a ROCm/CUDA GPU.")

    config = apply_overrides(load_config(args.config), args.set)
    setup_runtime()
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    dataset = build_dataset(config, "train")
    subset = Subset(dataset, range(min(args.batch_size, len(dataset))))
    loader = DataLoader(subset, batch_size=args.batch_size, num_workers=0)
    batch = next(iter(loader))

    started = time.perf_counter()
    model = build_model(config, initialize_pretrained=True).to(device)
    ema = ModelEMA(model, decay=float(config["protocol"]["ema_decay"]))
    optimizer = build_optimizer(model, config["protocol"])
    amp_enabled = bool(config["runtime"].get("amp_enabled", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    with autocast_context(amp_enabled, str(config["runtime"].get("amp_dtype", "fp16"))):
        logits = model(images)
        loss = F.cross_entropy(logits.float(), labels)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite smoke-test loss: {float(loss)}")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    ema.update(model)
    torch.cuda.synchronize()

    result = {
        "status": "ok",
        "dataset": str(config["dataset"]["name"]),
        "backbone": str(config["backbone"]["name"]),
        "batch_size": int(images.shape[0]),
        "input_shape": list(images.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "seconds": time.perf_counter() - started,
        "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        **parameter_counts(model),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
