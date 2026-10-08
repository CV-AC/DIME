from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..common.checkpoint import (
    extract_state_dict,
    resize_position_embedding,
    sha256_file,
    sha256_module,
    strip_prefixes,
    torch_load,
)
from .base import IMAGENET_MEAN, IMAGENET_STD, FeatureBackbone
from .registry import register


DEFAULT_TAPS_12 = (3, 5, 7, 11)


_CLASSIFIER_KEYS = ("head.", "fc_norm.")


@register("timm_vit")
class TimmViTBackbone(FeatureBackbone):

    def __init__(
        self,
        model_name: str,
        checkpoint: Optional[str] = None,
        input_size: int = 224,
        pretrained: bool = False,
        taps: Optional[Sequence[int]] = None,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ) -> None:
        import timm

        if checkpoint and pretrained:
            raise ValueError(
                "pass either checkpoint= (local weights) or "
                "pretrained=True (timm downloads), not both: "
                "otherwise which weights produced a number is ambiguous"
            )

        model = timm.create_model(
            model_name, pretrained=pretrained, img_size=input_size, num_classes=0
        )
        cfg = getattr(model, "pretrained_cfg", {}) or {}

        resolved_mean = tuple(mean or cfg.get("mean", IMAGENET_MEAN))
        resolved_std = tuple(std or cfg.get("std", IMAGENET_STD))

        checkpoint_hash = ""
        if checkpoint:
            path = Path(checkpoint)
            _load_into(model, path)
            checkpoint_hash = sha256_file(path)
        elif pretrained:
            checkpoint_hash = sha256_module(model)

        super().__init__(resolved_mean, resolved_std, checkpoint_hash)
        self.model = model
        self.model_name = model_name
        self.input_size = input_size

        depth = len(model.blocks)
        self.taps = tuple(taps) if taps is not None else _default_taps(depth)
        bad = [t for t in self.taps if not 0 <= t < depth]
        if bad:
            raise ValueError(f"tap index {bad} out of range for a {depth}-block model")
        self._dim = int(model.num_features)
        self._patch = int(model.patch_embed.patch_size[0])

    @property
    def feature_dims(self) -> list[int]:
        return [self._dim] * len(self.taps)

    @property
    def feature_strides(self) -> list[int]:
        return [self._patch] * len(self.taps)

    @property
    def num_layers(self) -> int:
        return len(self.model.blocks) + 1

    def forward_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        model = self.model
        tokens = model.patch_embed(images)
        height, width = model.patch_embed.grid_size
        tokens = model._pos_embed(tokens)
        tokens = model.patch_drop(tokens)
        tokens = model.norm_pre(tokens)
        prefix = int(model.num_prefix_tokens)
        out: list[torch.Tensor] = []
        for index, block in enumerate(model.blocks):
            tokens = block(tokens)
            if index in self.taps:
                spatial = (
                    tokens[:, prefix:]
                    .transpose(1, 2)
                    .reshape(tokens.shape[0], -1, height, width)
                )
                out.append(spatial)
        return out

    def forward_pooled(self, images: torch.Tensor) -> torch.Tensor:

        model = self.model
        tokens = model.patch_embed(images)
        tokens = model._pos_embed(tokens)
        tokens = model.patch_drop(tokens)
        tokens = model.norm_pre(tokens)
        for block in model.blocks:
            tokens = block(tokens)
        tokens = model.norm(tokens)
        return tokens[:, int(model.num_prefix_tokens) :].mean(dim=1)

    def parameter_layer_id(self, name: str) -> int:

        if name.startswith(("model.patch_embed", "model.pos_embed", "model.cls_token")):
            return 0
        if name.startswith("model.blocks."):
            return int(name.split(".")[2]) + 1
        return len(self.model.blocks) + 1


def _default_taps(depth: int) -> tuple[int, ...]:
    if depth == 12:
        return DEFAULT_TAPS_12

    step = max(depth // 4, 1)
    return tuple(sorted({min(depth - 1, (i + 1) * step - 1) for i in range(4)}))


def _load_into(model: nn.Module, path: Path) -> None:

    state = strip_prefixes(extract_state_dict(torch_load(path)))
    current = model.state_dict()

    if "pos_embed" in state and state["pos_embed"].shape != current["pos_embed"].shape:
        grid = model.patch_embed.grid_size
        state["pos_embed"] = resize_position_embedding(
            state["pos_embed"],
            grid[0],
            grid[1],
            prefix_tokens=int(model.num_prefix_tokens),
        )

    unexpected = {k for k in state if k not in current}
    bad_unexpected = {k for k in unexpected if not k.startswith(_CLASSIFIER_KEYS)}
    if bad_unexpected:
        raise RuntimeError(
            f"{path.name} contains {len(bad_unexpected)} keys this model has no slot "
            f"for, e.g. {sorted(bad_unexpected)[:8]}. Wrong model_name for this "
            f"checkpoint, or a checkpoint from a different architecture."
        )

    filtered = {k: v for k, v in state.items() if k in current}
    shape_mismatch = {
        k: (tuple(v.shape), tuple(current[k].shape))
        for k, v in filtered.items()
        if v.shape != current[k].shape
    }
    if shape_mismatch:
        raise RuntimeError(f"{path.name} shape mismatches: {shape_mismatch}")

    missing, unexpected_after = model.load_state_dict(filtered, strict=False)
    bad_missing = {k for k in missing if not k.startswith(_CLASSIFIER_KEYS)}
    if bad_missing or unexpected_after:
        raise RuntimeError(
            f"strict encoder load failed for {path.name}; "
            f"missing={sorted(bad_missing)[:8]}, unexpected={list(unexpected_after)[:8]}"
        )
