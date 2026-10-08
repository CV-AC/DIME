from __future__ import annotations

import argparse
import contextlib
import logging
import random
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from .config import load_config, to_plain_dict
from .data import build_eval_loader, denormalize
from .engine import WandBLogger, autocast_context, load_model_weights, unwrap_model
from .model import build_model


LOGGER = logging.getLogger("dime_vit")


def _images_from_batch(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, Mapping):
        return batch.get("images", batch.get("image"))
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise TypeError("Evaluation batches must contain an image tensor")


def mse_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().flatten(1).mean(1)


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = mse_per_image(prediction, target).clamp_min(1.0e-12)
    return -10.0 * torch.log10(mse)


def _gaussian_kernel(
    channels: int, size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    vector = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    vector = vector / vector.sum()
    kernel = torch.outer(vector, vector)
    return kernel.expand(channels, 1, size, size).contiguous()


def ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:

    height, width = prediction.shape[-2:]
    size = min(window_size, height, width)
    if size % 2 == 0:
        size -= 1
    if size < 1:
        raise ValueError("SSIM requires non-empty spatial dimensions")
    channels = prediction.shape[1]
    kernel = _gaussian_kernel(
        channels, size, sigma, prediction.device, prediction.dtype
    )
    padding = size // 2

    def filter_image(images: torch.Tensor) -> torch.Tensor:
        if padding:
            images = F.pad(images, (padding,) * 4, mode="reflect")
        return F.conv2d(images, kernel, groups=channels)

    mean_x = filter_image(prediction)
    mean_y = filter_image(target)
    mean_x_sq = mean_x.square()
    mean_y_sq = mean_y.square()
    mean_xy = mean_x * mean_y
    variance_x = filter_image(prediction.square()) - mean_x_sq
    variance_y = filter_image(target.square()) - mean_y_sq
    covariance = filter_image(prediction * target) - mean_xy

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mean_xy + c1) * (2 * covariance + c2)
    denominator = (mean_x_sq + mean_y_sq + c1) * (variance_x + variance_y + c2)
    return (numerator / denominator.clamp_min(1.0e-12)).flatten(1).mean(1)


@contextlib.contextmanager
def _deterministic_masks(device: torch.device, seed: int):
    devices = []
    if device.type == "cuda":
        devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    python_state = random.getstate()
    try:
        random.seed(seed)
        with torch.random.fork_rng(devices=devices):

            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)


@torch.inference_mode()
def evaluate_reconstruction(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    amp_dtype: str = "bf16",
    seed: int = 0,
    num_visuals: int = 8,
    normalization_mean=(0.485, 0.456, 0.406),
    normalization_std=(0.229, 0.224, 0.225),
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:

    raw_model = unwrap_model(model)
    was_training = raw_model.training
    raw_model.eval()
    sums = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "mask_ratio": 0.0}
    count = 0
    visual_chunks: dict[str, list[torch.Tensor]] = {
        "target": [],
        "source": [],
        "mask": [],
        "mixed": [],
        "reconstruction": [],
        "error": [],
    }

    with _deterministic_masks(device, seed):
        for batch in loader:
            images = _images_from_batch(batch).to(device, non_blocking=True)
            with autocast_context(device, amp_dtype):
                result = raw_model.reconstruct(images)
            if not isinstance(result, Mapping):
                raise TypeError("model.reconstruct must return a mapping")
            for required in ("reconstruction", "mixed"):
                if required not in result:
                    raise KeyError(f"model.reconstruct output is missing '{required}'")

            target = denormalize(images.float(), normalization_mean, normalization_std)
            reconstruction = denormalize(
                result["reconstruction"].float(),
                normalization_mean,
                normalization_std,
            )
            mixed = denormalize(
                result["mixed"].float(), normalization_mean, normalization_std
            )
            mask = result["mask"].float()
            if mask.shape[0] == 1:
                mask = mask.expand(target.shape[0], -1, -1)
            batch_metrics = {
                "mse": mse_per_image(reconstruction, target),
                "psnr": psnr_per_image(reconstruction, target),
                "ssim": ssim_per_image(reconstruction, target),
                "mask_ratio": mask.flatten(1).mean(1),
            }
            for name, values in batch_metrics.items():
                sums[name] += float(values.sum())
            count += target.shape[0]

            visual_count = sum(chunk.shape[0] for chunk in visual_chunks["target"])
            remaining = max(num_visuals - visual_count, 0)
            if remaining:
                take = min(remaining, target.shape[0])
                visual_chunks["target"].append(target[:take].cpu())
                visual_chunks["source"].append(target.flip(0)[:take].cpu())
                mask_image = result["mask_image"].float().clamp(0.0, 1.0)
                visual_chunks["mask"].append(mask_image[:take].cpu())
                visual_chunks["mixed"].append(mixed[:take].cpu())
                visual_chunks["reconstruction"].append(reconstruction[:take].cpu())
                error = (reconstruction - target).abs().mul(3.0).clamp(0.0, 1.0)
                visual_chunks["error"].append(error[:take].cpu())

    if was_training:
        raw_model.train()
    if count == 0:
        raise RuntimeError("The evaluation loader is empty")
    metrics = {name: value / count for name, value in sums.items()}
    visuals = {
        name: torch.cat(chunks, dim=0) if chunks else torch.empty(0)
        for name, chunks in visual_chunks.items()
    }
    return metrics, visuals


def save_reconstruction_panel(
    visuals: Mapping[str, torch.Tensor], path: str | Path
) -> None:

    if visuals["target"].numel() == 0:
        return
    order = ("target", "source", "mask", "mixed", "reconstruction", "error")
    panels = torch.cat([visuals[name] for name in order], dim=-1)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image(panels, destination, nrow=1, padding=2)


def build_model_from_config(config) -> torch.nn.Module:
    kwargs = to_plain_dict(config.model)
    name = kwargs.pop("name")
    kwargs.update(to_plain_dict(config.loss))
    return build_model(name=name, **kwargs)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate DIME-ViT reconstruction")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return parser


def main() -> None:
    args = get_args_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    config = load_config(args.config, args.opts)
    if config.data.path is None:
        raise ValueError("Set data.path to the face LMDB in the YAML or with --opts")
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/ROCm device requested, but torch.cuda.is_available() is false"
        )
    device = requested_device

    model = build_model_from_config(config).to(device)
    missing, unexpected = load_model_weights(model, args.checkpoint, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    loader = build_eval_loader(
        data_path=config.data.path,
        input_size=config.model.img_size,
        num_pairs=config.eval.num_pairs,
        batch_size=config.eval.batch_size,
        num_workers=config.eval.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.eval.seed,
        subset_ratio=config.data.subset_ratio,
        crop_pct=config.data.transform.eval_crop_pct,
        interpolation=config.data.transform.interpolation,
        antialias=config.data.transform.antialias,
        mean=config.data.transform.mean,
        std=config.data.transform.std,
    )
    metrics, visuals = evaluate_reconstruction(
        model,
        loader,
        device,
        amp_dtype=config.train.amp_dtype,
        seed=config.eval.seed,
        num_visuals=config.eval.num_visuals,
        normalization_mean=config.data.transform.mean,
        normalization_std=config.data.transform.std,
    )
    output = Path(
        args.output or Path(args.checkpoint).with_suffix(".reconstruction.png")
    )
    save_reconstruction_panel(visuals, output)
    LOGGER.info(
        "reconstruction | MSE %.6f | PSNR %.3f dB | SSIM %.5f",
        metrics["mse"],
        metrics["psnr"],
        metrics["ssim"],
    )
    LOGGER.info("saved visualization to %s", output)

    wandb_logger = WandBLogger(config.wandb, config)
    if wandb_logger.enabled:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        step = (
            int(checkpoint.get("global_update", 0))
            if isinstance(checkpoint, Mapping)
            else 0
        )
        wandb_logger.log({f"eval/{key}": value for key, value in metrics.items()}, step)
        wandb_logger.log_reconstructions(**visuals, step=step)
        wandb_logger.finish()


if __name__ == "__main__":
    main()
