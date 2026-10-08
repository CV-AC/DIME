from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .model import DIMEViT, _rope_2d, _sincos_2d, build_model


class DIMEViTEncoder(nn.Module):

    def __init__(self, model: DIMEViT, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.patch_embed = model.patch_embed
        self.blocks = model.blocks
        self.norm = model.norm
        self.pos_drop = model.pos_drop
        self.patch_size = model.patch_size
        self.pos_encoding = model.pos_encoding
        self.use_checkpoint = model.use_checkpoint
        self.num_features = self.feat_dim = model.embed_dim
        self.num_lr_layers = len(self.blocks) + 2
        self.image_mean = model.input_mean
        self.image_std = model.input_std

    _validate_images = DIMEViT._validate_images
    _run_encoder = DIMEViT._run_encoder
    forward_features = DIMEViT.forward_features

    def forward_tokens(self, images):
        return self.forward_features(images, return_tokens=True)[0]

    def forward(self, images):
        return self.forward_features(images)

    def parameter_layer_id(self, name):
        if name.startswith("patch_embed."):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return self.num_lr_layers - 1

    def forward_intermediates(self, images):

        self._validate_images(images, paired=False)
        tokens, grid = self.patch_embed(images)
        rope = None
        if self.pos_encoding == "sincos":
            tokens = tokens + _sincos_2d(
                tokens.shape[-1], grid, tokens.device, tokens.dtype
            )
        else:
            rope = _rope_2d(
                self.blocks[0].attn.head_dim, grid, tokens.device, tokens.dtype
            )
        tokens = self.pos_drop(tokens)
        depth = len(self.blocks)

        indices = (depth // 3 - 1, depth // 2 - 1, 2 * depth // 3 - 1, depth - 1)
        outputs = []
        for index, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                tokens = checkpoint(
                    lambda value, module=block: module(value, rope=rope),
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens, rope=rope)
            if index in indices:
                features = self.norm(tokens)
                outputs.append(
                    features.transpose(1, 2).reshape(images.shape[0], -1, *grid)
                )
        return outputs


def load_encoder(checkpoint_path, *, model_name="", checkpoint_data=None):

    path = Path(checkpoint_path)
    checkpoint_data = (
        torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint_data is None
        else checkpoint_data
    )
    config = checkpoint_data.get("config", {})
    model_config = dict(config.get("model", {}))
    saved_name = model_config.pop("name", "")
    if model_name and saved_name and model_name != saved_name:
        raise ValueError(f"Requested {model_name}, checkpoint contains {saved_name}")
    name = model_name or saved_name
    if not name:
        raise ValueError("DIME-ViT checkpoint must contain config.model.name")
    model_config.update(config.get("loss", {}))
    model = build_model(name, **model_config)
    state = checkpoint_data.get("model", checkpoint_data.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("DIME-ViT checkpoint must contain model weights")
    cleaned = {}
    for key, tensor in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in cleaned:
            raise ValueError(f"Duplicate checkpoint key: {key}")
        cleaned[key] = tensor
    model.load_state_dict(cleaned, strict=True)
    encoder = DIMEViTEncoder(model, name)
    transform = config.get("data", {}).get("transform", {})
    encoder.image_mean = tuple(transform.get("mean", model.input_mean))
    encoder.image_std = tuple(transform.get("std", model.input_std))
    return encoder
