from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .utils import sha256_file


MODEL_MODULES = {
    "dime": "model_mim",
    "dime_v2": "model_mim_swin_v2",
    "dime_hiera": "model_mim_hiera",
    "dime_hiera_v2": "model_mim_hiera_v2",
    "dime_hiera_v2_wo_ldp": "model_mim_hiera_v2_wo_ldp",
}


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("DIME checkpoint must be a dictionary.")
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if (
            isinstance(value, dict)
            and value
            and all(isinstance(tensor, torch.Tensor) for tensor in value.values())
        ):
            return value
    if checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint
    raise KeyError(
        "Could not find model/model_state_dict/state_dict in DIME checkpoint."
    )


def _strip_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod.", "student."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        if new_key in cleaned:
            raise KeyError(f"Checkpoint prefix cleanup collision: {new_key}")
        cleaned[new_key] = value
    return cleaned


def _checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    value = checkpoint.get("config", {})
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return {key: val for key, val in vars(value).items() if not key.startswith("_")}
    return {}


def _module_for_model(model_name: str) -> str:
    if "_hiera_" in model_name and model_name.endswith("_v2_wo_ldp"):
        return MODEL_MODULES["dime_hiera_v2_wo_ldp"]
    if "_hiera_" in model_name and model_name.endswith("_v2"):
        return MODEL_MODULES["dime_hiera_v2"]
    if "_hiera_" in model_name:
        return MODEL_MODULES["dime_hiera"]
    if model_name.endswith("_v2"):
        return MODEL_MODULES["dime_v2"]
    return MODEL_MODULES["dime"]


def _build_pretraining_model(
    dime_source: Path,
    model_name: str,
    checkpoint_config: dict[str, Any],
) -> nn.Module:
    source = str(dime_source)
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module(_module_for_model(model_name))
    try:
        factory = getattr(module, model_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown DIME model {model_name!r}") from exc

    input_size = int(checkpoint_config.get("input_size", 224))
    if input_size != 224:
        raise ValueError(
            f"Head-pose protocol is 224px, but DIME checkpoint reports {input_size}px. "
            "Use the canonical 224px full-DIME checkpoint."
        )
    kwargs = {
        "img_size": input_size,
        "shared_mask": bool(checkpoint_config.get("shared_mask", False)),
        "lambda_diff": float(checkpoint_config.get("lambda_diff", 0.5)),
        "sobel_q": float(checkpoint_config.get("sobel_q", 0.5)),
        "lambda_edds": float(checkpoint_config.get("lambda_edds", 1.0)),
        "edds_warmup_epochs": int(checkpoint_config.get("edds_warmup_epochs", 0)),
        "edds_version": str(checkpoint_config.get("edds_version", "v1")),
    }
    return factory(**kwargs)


class DIMEEncoder(nn.Module):

    def __init__(self, pretraining_model: nn.Module, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.patch_embed = pretraining_model.patch_embed
        self.ldp = getattr(pretraining_model, "ldp", None)
        self.absolute_pos_embed = pretraining_model.absolute_pos_embed
        self.pos_drop = pretraining_model.pos_drop
        self.layers = pretraining_model.layers
        self.norm = pretraining_model.norm
        self.downsamples = getattr(pretraining_model, "downsamples", None)
        self.num_features = int(pretraining_model.num_features)
        self.is_hiera = self.downsamples is not None
        self.num_lr_layers = len(self.layers) + 2

    def parameter_layer_id(self, name: str) -> int:
        if name.startswith(("patch_embed.", "ldp.", "absolute_pos_embed")):
            return 0
        if name.startswith("layers."):
            try:
                return min(int(name.split(".", 2)[1]) + 1, self.num_lr_layers - 1)
            except (IndexError, ValueError):
                return 0
        if name.startswith("downsamples."):
            try:
                return min(int(name.split(".", 2)[1]) + 1, self.num_lr_layers - 1)
            except (IndexError, ValueError):
                return 0
        return self.num_lr_layers - 1

    @classmethod
    def from_checkpoint(
        cls,
        dime_source: str | Path,
        checkpoint_path: str | Path,
        explicit_model_name: str = "",
    ) -> "DIMEEncoder":
        dime_source = Path(dime_source).resolve()
        checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint = _torch_load(checkpoint_path)
        checkpoint_config = _checkpoint_config(checkpoint)
        if isinstance(checkpoint_config.get("model"), dict):
            from facebench.backbones.dime_vit import load_encoder

            encoder = load_encoder(
                checkpoint_path,
                model_name=explicit_model_name,
                checkpoint_data=checkpoint,
            )
            encoder.checkpoint_sha256 = sha256_file(checkpoint_path)
            return encoder
        model_name = (
            explicit_model_name.strip()
            or str(checkpoint_config.get("model", "")).strip()
        )
        if not model_name:
            raise ValueError(
                "DIME model name is absent from the checkpoint. Set model.dime_model "
                "in configs/dime.yaml."
            )
        model = _build_pretraining_model(dime_source, model_name, checkpoint_config)
        state = _strip_prefixes(_state_dict(checkpoint))
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Strict full-DIME checkpoint loading failed. Refusing to continue with "
                "partially initialized encoder weights."
            ) from exc
        encoder = cls(model, model_name)
        encoder.checkpoint_sha256 = sha256_file(checkpoint_path)
        return encoder

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        batch, length, _ = tokens.shape
        height = width = int(length**0.5)
        if height * width != length:
            raise RuntimeError(f"DIME patch tokens are not square: {length}")
        if self.ldp is not None:
            tokens = self.ldp(tokens, height, width, mask=None)
        tokens = self.pos_drop(tokens + self.absolute_pos_embed)

        if self.is_hiera:
            tokens = self.layers[0](tokens, height, width)
            tokens, height, width = self.downsamples[0](tokens, height, width)
            tokens = self.layers[1](tokens, height, width)
            tokens, height, width = self.downsamples[1](tokens, height, width)
            tokens = self.layers[2](tokens, height, width, group_mask=None)
            tokens, height, width = self.downsamples[2](tokens, height, width)
            tokens = self.layers[3](tokens, height, width, group_mask=None)
        else:
            for layer in self.layers:
                tokens = layer(tokens, attn_mask=None)
        return self.norm(tokens)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(images).mean(dim=1)
