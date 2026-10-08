from __future__ import annotations

import importlib
import importlib.util
import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from facebench.common.features import intermediate_indices

from .config import resolve_path
from .utils import sha256_file


DIME_MODULES = {
    "dime": "model_mim",
    "dime_v2": "model_mim_swin_v2",
    "dime_hiera": "model_mim_hiera",
    "dime_hiera_v2": "model_mim_hiera_v2",
    "dime_hiera_v2_wo_ldp": "model_mim_hiera_v2_wo_ldp",
}

FARL_VISUAL_AUXILIARY_KEYS = ("mask_token",)
FARL_VISUAL_AUXILIARY_PREFIXES = (
    "lm_transformer.",
    "ln_lm.",
    "lm_head.",
)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _module_sha256(module: nn.Module) -> str:

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return value
    if checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint
    raise KeyError("Could not find model/model_state_dict/state_dict in checkpoint.")


def _strip_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod.", "student.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        if new_key in output:
            raise KeyError(f"Checkpoint key collision after prefix cleanup: {new_key}")
        output[new_key] = value
    return output


def _farl_visual_state_dict(
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:

    visual_state = {
        key[len("visual.") :]: value
        for key, value in state.items()
        if key.startswith("visual.")
    }
    if not visual_state:
        raise KeyError(
            "FaRL checkpoint contains no visual.* weights after prefix cleanup."
        )

    def is_auxiliary(key: str) -> bool:
        return key in FARL_VISUAL_AUXILIARY_KEYS or key.startswith(
            FARL_VISUAL_AUXILIARY_PREFIXES
        )

    ignored = tuple(sorted(key for key in visual_state if is_auxiliary(key)))
    encoder_state = {
        key: value for key, value in visual_state.items() if not is_auxiliary(key)
    }
    if not encoder_state:
        raise KeyError("FaRL checkpoint contains no downstream visual encoder weights.")
    return encoder_state, ignored


def _checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    value = checkpoint.get("config", {})
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return {
            key: item for key, item in vars(value).items() if not key.startswith("_")
        }
    return {}


def _resize_position_embedding(
    position: torch.Tensor, height: int, width: int, *, prefix_tokens: int
) -> torch.Tensor:
    prefix = position[:, :prefix_tokens]
    spatial = position[:, prefix_tokens:]
    old_size = int(math.sqrt(spatial.shape[1]))
    if old_size * old_size != spatial.shape[1]:
        raise ValueError(
            f"Position embedding has non-square token count: {spatial.shape[1]}"
        )
    if (old_size, old_size) == (height, width):
        return position
    spatial = spatial.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    spatial = F.interpolate(
        spatial.float(), size=(height, width), mode="bicubic", align_corners=False
    ).to(position.dtype)
    spatial = spatial.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    return torch.cat([prefix, spatial], dim=1)


class FeatureBackbone(nn.Module):
    out_channels: tuple[int, int, int, int]
    pyramid_type: str
    checkpoint_sha256: str

    def __init__(
        self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ):
        super().__init__()
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1))
        self.checkpoint_sha256 = "framework-managed"

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.image_mean) / self.image_std


def _dime_module(model_name: str) -> str:
    if "_hiera_" in model_name and model_name.endswith("_v2_wo_ldp"):
        return DIME_MODULES["dime_hiera_v2_wo_ldp"]
    if "_hiera_" in model_name and model_name.endswith("_v2"):
        return DIME_MODULES["dime_hiera_v2"]
    if "_hiera_" in model_name:
        return DIME_MODULES["dime_hiera"]
    if model_name.endswith("_v2"):
        return DIME_MODULES["dime_v2"]
    return DIME_MODULES["dime"]


def _build_dime_model(
    source: Path, model_name: str, checkpoint_config: dict[str, Any]
) -> nn.Module:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module(_dime_module(model_name))
    try:
        factory = getattr(module, model_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown DIME model {model_name!r}") from exc
    kwargs = {
        "img_size": int(checkpoint_config.get("input_size", 224)),
        "shared_mask": bool(checkpoint_config.get("shared_mask", False)),
        "lambda_diff": float(checkpoint_config.get("lambda_diff", 0.5)),
        "sobel_q": float(checkpoint_config.get("sobel_q", 0.5)),
        "lambda_edds": float(checkpoint_config.get("lambda_edds", 1.0)),
        "edds_warmup_epochs": int(checkpoint_config.get("edds_warmup_epochs", 0)),
        "edds_version": str(checkpoint_config.get("edds_version", "v1")),
    }
    return factory(**kwargs)


class DIMEFeatureBackbone(FeatureBackbone):

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        checkpoint_hash: str,
        input_size: int,
    ):
        super().__init__(
            tuple(getattr(model, "image_mean", (0.485, 0.456, 0.406))),
            tuple(getattr(model, "image_std", (0.229, 0.224, 0.225))),
        )
        self.model_name = model_name
        self.patch_embed = model.patch_embed
        self.ldp = getattr(model, "ldp", None)
        self.absolute_pos_embed = model.absolute_pos_embed
        self.pos_drop = model.pos_drop
        self.layers = model.layers
        self.norm = model.norm
        self.downsamples = getattr(model, "downsamples", None)
        self.is_hiera = self.downsamples is not None
        embed_dim = int(model.embed_dim)
        self.out_channels = tuple(embed_dim * (2**index) for index in range(4))
        self.pyramid_type = "native"
        self.checkpoint_sha256 = checkpoint_hash
        if self.is_hiera:
            self._prepare_hiera_resolution(input_size)

    def _prepare_hiera_resolution(self, input_size: int) -> None:

        patch_size = getattr(self.patch_embed, "patch_size", 4)
        patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        grid = int(input_size) // int(patch_size)
        for stage_index, layer in enumerate(self.layers):
            resolution = (grid // (2**stage_index),) * 2
            for block in layer.blocks:
                attention = getattr(block, "attn", None)
                if attention is None:
                    continue
                attention.input_resolution = resolution
                if bool(getattr(attention, "use_v2_attention", False)):
                    attention.attn_window_size = resolution
                    attention._init_log_cpb_tables(
                        resolution, attention.pretrained_window_size
                    )

    @classmethod
    def from_checkpoint(
        cls,
        source: Path,
        checkpoint_path: Path,
        explicit_model_name: str = "",
        input_size: int = 448,
    ) -> "DIMEFeatureBackbone":
        checkpoint = _torch_load(checkpoint_path)
        config = _checkpoint_config(checkpoint)
        if isinstance(config.get("model"), dict):
            from facebench.backbones.dime_vit import (
                DIMEViTFeatureBackbone,
                load_encoder,
            )

            encoder = load_encoder(
                checkpoint_path,
                model_name=explicit_model_name,
                checkpoint_data=checkpoint,
            )
            return DIMEViTFeatureBackbone(encoder, sha256_file(checkpoint_path))
        model_name = explicit_model_name.strip() or str(config.get("model", "")).strip()
        if not model_name:
            raise ValueError(
                "DIME model name is absent from the checkpoint; set backbone.dime_model."
            )
        model = _build_dime_model(source, model_name, config)
        state = _strip_prefixes(_state_dict(checkpoint))
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Strict full-DIME checkpoint loading failed; partial initialization is forbidden."
            ) from exc
        return cls(model, model_name, sha256_file(checkpoint_path), input_size)

    def _patch_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        features = self.patch_embed.proj(images)
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        patch_norm = getattr(self.patch_embed, "norm", None)
        if patch_norm is not None:
            tokens = patch_norm(tokens)
        if self.ldp is not None:
            tokens = self.ldp(tokens, height, width, mask=None)
        position = _resize_position_embedding(
            self.absolute_pos_embed, height, width, prefix_tokens=0
        )
        return self.pos_drop(tokens + position.to(tokens.dtype)), height, width

    @staticmethod
    def _map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        tokens, height, width = self._patch_tokens(images)
        outputs: list[torch.Tensor] = []
        if self.is_hiera:
            for index, layer in enumerate(self.layers):
                tokens = layer(tokens, height, width, group_mask=None)
                if index == len(self.layers) - 1:
                    tokens = self.norm(tokens)
                outputs.append(self._map(tokens, height, width))
                if index < len(self.downsamples):
                    tokens, height, width = self.downsamples[index](
                        tokens, height, width
                    )
        else:
            for index, layer in enumerate(self.layers):
                layer.input_resolution = (height, width)
                for block in layer.blocks:
                    block.input_resolution = (height, width)
                    tokens = block(tokens, attn_mask=None)
                if index == len(self.layers) - 1:
                    tokens = self.norm(tokens)
                outputs.append(self._map(tokens, height, width))
                if layer.downsample is not None:
                    merged = layer.downsample(
                        tokens.reshape(tokens.shape[0], height, width, tokens.shape[-1])
                    )
                    height, width = merged.shape[1:3]
                    tokens = merged.reshape(
                        merged.shape[0], height * width, merged.shape[-1]
                    )
        if len(outputs) != 4:
            raise RuntimeError(f"DIME returned {len(outputs)} stages instead of four.")
        return outputs


def _load_farl_module(source: Path):
    model_file = source
    if source.is_dir():
        candidates = (
            source / "farl" / "network" / "farl" / "model.py",
            source / "model.py",
        )
        model_file = next((path for path in candidates if path.exists()), candidates[0])
    spec = importlib.util.spec_from_file_location(
        "_dime_landmark_farl_model", model_file
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import FaRL VisualTransformer from {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FaRLFeatureBackbone(FeatureBackbone):
    def __init__(self, visual: nn.Module, checkpoint_hash: str):
        super().__init__(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.visual = visual
        self.visual.transformer.use_checkpoint = True

        self.visual.ln_post.requires_grad_(False)
        if self.visual.proj is not None:
            self.visual.proj.requires_grad_(False)
        self.indices = (3, 5, 7, 11)
        self.out_channels = (768, 768, 768, 768)
        self.pyramid_type = "vit"
        self.checkpoint_sha256 = checkpoint_hash

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = self.visual.conv1(images)
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        cls = self.visual.class_embedding.to(tokens.dtype).view(1, 1, -1)
        tokens = torch.cat([cls.expand(batch, -1, -1), tokens], dim=1)
        position = _resize_position_embedding(
            self.visual.positional_embedding.unsqueeze(0),
            height,
            width,
            prefix_tokens=1,
        ).squeeze(0)
        tokens = self.visual.ln_pre(tokens + position.to(tokens.dtype))
        tokens = tokens.permute(1, 0, 2).contiguous()
        outputs: list[torch.Tensor] = []
        for index, block in enumerate(self.visual.transformer.resblocks):
            tokens = block(tokens)
            if index in self.indices:
                spatial = (
                    tokens[1:].permute(1, 2, 0).reshape(batch, channels, height, width)
                )
                outputs.append(spatial.float())
        return outputs


class TimmViTFeatureBackbone(FeatureBackbone):
    def __init__(self, model: nn.Module, checkpoint_hash: str):
        data_config = getattr(model, "pretrained_cfg", {})
        mean = tuple(data_config.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(data_config.get("std", (0.229, 0.224, 0.225)))
        super().__init__(mean, std)
        self.model = model

        for name in ("norm", "fc_norm", "head"):
            module = getattr(self.model, name, None)
            if isinstance(module, nn.Module):
                module.requires_grad_(False)
        self.indices = intermediate_indices(len(model.blocks))
        channels = int(model.num_features)
        self.out_channels = (channels, channels, channels, channels)
        self.pyramid_type = "vit"
        self.checkpoint_sha256 = checkpoint_hash

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        if getattr(self.model, "rope", None) is not None:
            return self.model.forward_intermediates(
                images,
                indices=list(self.indices),
                norm=False,
                intermediates_only=True,
            )
        tokens = self.model.patch_embed(images)
        height, width = self.model.patch_embed.grid_size
        tokens = self.model._pos_embed(tokens)
        tokens = self.model.patch_drop(tokens)
        tokens = self.model.norm_pre(tokens)
        outputs: list[torch.Tensor] = []
        prefix_tokens = int(self.model.num_prefix_tokens)
        for index, block in enumerate(self.model.blocks):
            tokens = block(tokens)
            if index in self.indices:
                spatial = (
                    tokens[:, prefix_tokens:]
                    .transpose(1, 2)
                    .reshape(tokens.shape[0], -1, height, width)
                )
                outputs.append(spatial)
        return outputs


def _load_timm_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    state = _strip_prefixes(_state_dict(_torch_load(checkpoint_path)))
    current = model.state_dict()
    if "pos_embed" in state and state["pos_embed"].shape != current["pos_embed"].shape:
        grid = model.patch_embed.grid_size
        state["pos_embed"] = _resize_position_embedding(
            state["pos_embed"], grid[0], grid[1], prefix_tokens=model.num_prefix_tokens
        )
    ignored = set(state).difference(current)
    invalid_ignored = {
        key for key in ignored if not key.startswith(("head.", "fc_norm."))
    }
    if invalid_ignored:
        raise RuntimeError(
            f"ViT checkpoint contains unsupported keys: {sorted(invalid_ignored)}"
        )
    filtered = {key: value for key, value in state.items() if key in current}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    allowed_missing = {key for key in current if key.startswith(("head.", "fc_norm."))}
    invalid_missing = set(missing).difference(allowed_missing)
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Strict ViT encoder loading failed; missing={sorted(invalid_missing)}, "
            f"unexpected={unexpected}"
        )


def build_backbone(
    config: dict[str, Any], *, initialize_pretrained: bool = True
) -> FeatureBackbone:
    backbone_config = config["backbone"]
    name = str(backbone_config["name"]).lower()
    if name == "dime":
        checkpoint = resolve_path(
            backbone_config.get("checkpoint"), must_exist=initialize_pretrained
        )
        if checkpoint is None:
            raise ValueError(
                "DIME requires backbone.checkpoint to infer and construct its architecture."
            )
        source = resolve_path(backbone_config["dime_source"], must_exist=True)
        assert source is not None
        backbone = DIMEFeatureBackbone.from_checkpoint(
            source,
            checkpoint,
            str(backbone_config.get("dime_model", "")),
            int(config["model"]["input_size"]),
        )
        backbone_config["dime_model"] = backbone.model_name
        backbone_config["checkpoint_sha256"] = backbone.checkpoint_sha256
        return backbone

    if name == "farl":
        source = resolve_path(backbone_config["farl_source"], must_exist=True)
        assert source is not None
        module = _load_farl_module(source)
        visual = module.VisualTransformer(
            input_resolution=224,
            patch_size=16,
            width=768,
            layers=12,
            heads=12,
            output_dim=512,
        )
        checkpoint_hash = "uninitialized"
        if initialize_pretrained:
            checkpoint_path = resolve_path(
                backbone_config.get("checkpoint"), must_exist=True
            )
            assert checkpoint_path is not None
            raw = _strip_prefixes(_state_dict(_torch_load(checkpoint_path)))
            state, ignored = _farl_visual_state_dict(raw)
            visual.load_state_dict(state, strict=True)
            checkpoint_hash = sha256_file(checkpoint_path)
            backbone_config["checkpoint_sha256"] = checkpoint_hash
            backbone_config["ignored_pretraining_tensors"] = len(ignored)
        input_size = int(config["model"]["input_size"])
        grid = input_size // 16
        visual.positional_embedding = nn.Parameter(
            _resize_position_embedding(
                visual.positional_embedding.unsqueeze(0),
                grid,
                grid,
                prefix_tokens=1,
            ).squeeze(0)
        )
        visual.input_resolution = input_size
        return FaRLFeatureBackbone(visual, checkpoint_hash)

    if name in {"dino", "mae"}:
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements-lumi.txt; timm is required."
            ) from exc
        checkpoint = resolve_path(backbone_config.get("checkpoint"))
        pretrained = (
            initialize_pretrained
            and checkpoint is None
            and bool(backbone_config.get("pretrained", True))
        )
        model = timm.create_model(
            str(backbone_config["timm_model"]),
            pretrained=pretrained,
            img_size=int(config["model"]["input_size"]),
            num_classes=0,
        )
        checkpoint_hash = "uninitialized"
        if initialize_pretrained and checkpoint is not None:
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            _load_timm_checkpoint(model, checkpoint)
            checkpoint_hash = sha256_file(checkpoint)
        elif pretrained:
            checkpoint_hash = (
                _module_sha256(model)
                if int(os.environ.get("RANK", "0")) == 0
                else "computed-on-rank-0"
            )
            backbone_config["checkpoint_sha256"] = checkpoint_hash
        return TimmViTFeatureBackbone(model, checkpoint_hash)
    raise ValueError(f"Unknown backbone.name: {name!r}")
