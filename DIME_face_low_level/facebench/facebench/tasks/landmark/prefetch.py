from __future__ import annotations

import argparse

from .backbones import build_backbone
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve and cache one pretrained backbone before a compute job."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    backbone = build_backbone(config, initialize_pretrained=True)
    parameters = sum(parameter.numel() for parameter in backbone.parameters())
    print(
        f"ready: {config['backbone']['name']} "
        f"parameters={parameters:,} checkpoint={backbone.checkpoint_sha256}"
    )


if __name__ == "__main__":
    main()
