from __future__ import annotations

import argparse
import json

from .backbones import build_backbone
from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify timm DINO/MAE weights before offline jobs."
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[
            "configs/lapa/dino.yaml",
            "configs/lapa/mae.yaml",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    for path in args.config:
        config = load_config(path)
        name = str(config["backbone"]["name"]).lower()
        if name not in {"dino", "mae"}:
            raise ValueError(f"{path} is a {name!r} config, not DINO/MAE.")
        backbone = build_backbone(config, initialize_pretrained=True)
        reports.append(
            {
                "config": path,
                "backbone": name,
                "timm_model": config["backbone"]["timm_model"],
                "sha256": backbone.checkpoint_sha256,
            }
        )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
