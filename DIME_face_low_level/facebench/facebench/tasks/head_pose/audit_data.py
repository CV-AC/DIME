from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .config import load_config, resolve_path
from .data import IMAGENET_MEAN, IMAGENET_STD, build_dataset
from .utils import set_seed, write_json


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    image = (tensor * std + mean).clamp(0, 1)
    array = (image.permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def audit(config: dict, split: str, samples: int, output: Path) -> dict:
    set_seed(int(config.get("seed", 42)))
    dataset = build_dataset(config, split)
    count = min(samples, len(dataset))
    items = [dataset[index] for index in range(count)]
    rotations = torch.stack([item["rotation"] for item in items])
    identity = torch.eye(3).expand(count, -1, -1)
    orthogonality_error = (rotations @ rotations.transpose(1, 2) - identity).abs().max()
    determinant_error = (torch.linalg.det(rotations) - 1.0).abs().max()

    columns = 4
    tile = int(config["protocol"].get("input_size", 224))
    caption = 24
    rows = (count + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile, rows * (tile + caption)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(items):
        x = (index % columns) * tile
        y = (index // columns) * (tile + caption)
        canvas.paste(_to_pil(item["image"]), (x, y))
        draw.text((x + 3, y + tile + 3), item["sample_id"][-32:], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)

    result = {
        "split": split,
        "dataset_size": len(dataset),
        "preview_samples": count,
        "max_rotation_determinant_error": determinant_error.item(),
        "max_rotation_orthogonality_error": orthogonality_error.item(),
        "preview": str(output),
    }
    write_json(output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate labels and render crop previews."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset", choices=("train", "aflw2000", "biwi"), required=True
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else resolve_path(f"data_audit/{args.dataset}_preview.jpg")
    )
    assert output is not None
    audit(config, args.dataset, args.samples, output)


if __name__ == "__main__":
    main()
