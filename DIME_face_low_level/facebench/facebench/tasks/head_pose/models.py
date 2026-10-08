from __future__ import annotations

import contextlib
import hashlib
import io
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import resolve_path
from .dime_encoder import DIMEEncoder
from .rotation import rotation_6d_to_matrix
from .utils import sha256_file


def _verify_expected_sha256(
    actual: str,
    expected: str | None,
    checkpoint_path: Path,
) -> None:
    expected_value = str(expected or "").strip().lower()
    if expected_value and actual.lower() != expected_value:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch for {checkpoint_path}: "
            f"expected {expected_value}, got {actual.lower()}."
        )


def _module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class HeadPoseModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        *,
        head_type: str = "linear",
        hidden_dim: int = 512,
        dropout: float = 0.0,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.head_type = head_type.lower()
        self.encoder_frozen = bool(freeze_encoder)
        if self.head_type == "linear":
            self.head = nn.Linear(feature_dim, 6)
            self._initialize_linear(self.head)
        elif self.head_type == "mlp":
            if hidden_dim <= 0:
                raise ValueError("head.hidden_dim must be positive.")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("head.dropout must be in [0, 1).")
            self.head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 6),
            )
            self._initialize_linear(self.head[0])
            self._initialize_linear(self.head[-1])
        else:
            raise ValueError(f"Unknown head.type: {head_type!r}")

        if self.encoder_frozen:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    @staticmethod
    def _initialize_linear(layer: nn.Linear) -> None:
        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(layer.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_frozen:

            self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.encoder_frozen:
            with torch.no_grad():
                features = self.encoder(images)
        else:
            features = self.encoder(images)

        return rotation_6d_to_matrix(self.head(features).float())


class TimmEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        pretrained: bool,
        checkpoint_path: Path | None,
        pooling: str,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt (timm is required).") from exc
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained and checkpoint_path is None,
            num_classes=0,
        )
        pretrained_feature_dim = int(self.model.num_features)
        self.num_lr_layers = len(getattr(self.model, "blocks", ())) + 2
        self.checkpoint_sha256 = "uninitialized"
        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            cleaned = {
                key.removeprefix("module.").removeprefix("model."): value
                for key, value in state.items()
            }
            self.model.load_state_dict(cleaned, strict=True)
            self.checkpoint_sha256 = sha256_file(checkpoint_path)
        elif pretrained:
            self.checkpoint_sha256 = (
                _module_sha256(self.model)
                if int(os.environ.get("RANK", "0")) == 0
                else "computed-on-rank-0"
            )
        self.pooling = pooling.strip().lower()
        if self.pooling == "cls":
            self.num_features = pretrained_feature_dim
        elif self.pooling == "cls_avg_concat":
            self.num_features = pretrained_feature_dim * 2
        elif self.pooling == "global_avg":
            self._enable_mae_global_average_pool(pretrained_feature_dim)
            self.num_features = pretrained_feature_dim
        else:
            raise ValueError(
                "model.pooling must be cls, cls_avg_concat, or global_avg; "
                f"got {pooling!r}."
            )

    def _enable_mae_global_average_pool(self, feature_dim: int) -> None:

        norm = getattr(self.model, "norm", None)
        if not isinstance(norm, nn.LayerNorm):
            raise TypeError(
                "MAE global_avg expects the pretrained ViT final norm to be LayerNorm."
            )
        self.model.fc_norm = nn.LayerNorm(
            feature_dim,
            eps=norm.eps,
            elementwise_affine=norm.elementwise_affine,
        )
        self.model.norm = nn.Identity()
        self.model.global_pool = "avg"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls_avg_concat":
            tokens = self.model.forward_features(images)
            if tokens.ndim != 3:
                raise RuntimeError(
                    "cls_avg_concat requires ViT token features shaped [B, N, C]."
                )
            prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
            if tokens.shape[1] <= prefix_tokens:
                raise RuntimeError(
                    "No patch tokens available for DINO average pooling."
                )
            return torch.cat(
                (
                    tokens[:, 0],
                    tokens[:, prefix_tokens:].mean(dim=1),
                ),
                dim=-1,
            )
        return self.model(images)

    def parameter_layer_id(self, name: str) -> int:
        name = name.removeprefix("model.")
        if name.startswith(
            ("patch_embed.", "cls_token", "pos_embed", "reg_token", "mask_token")
        ):
            return 0
        if name.startswith("blocks."):
            try:
                return min(int(name.split(".", 2)[1]) + 1, self.num_lr_layers - 1)
            except (IndexError, ValueError):
                return 0
        return self.num_lr_layers - 1


class RepVGGEncoder(nn.Module):

    def __init__(
        self,
        repo: Path,
        backbone_name: str,
        *,
        checkpoint_path: Path | None,
    ):
        super().__init__()
        package_dir = repo / "sixdrepnet"
        if str(package_dir) not in sys.path:
            sys.path.insert(0, str(package_dir))
        module_path = package_dir / "backbone" / "repvgg.py"
        spec = importlib.util.spec_from_file_location(
            "_dime_head_pose_repvgg", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import RepVGG from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:

            with contextlib.redirect_stdout(io.StringIO()):
                backbone = module.get_RepVGG_func_by_name(backbone_name)(deploy=False)
        except KeyError as exc:
            raise ValueError(f"Unknown RepVGG backbone {backbone_name!r}") from exc

        self.num_features = int(backbone.linear.in_features)
        self.num_lr_layers = 5
        self.checkpoint_sha256 = "uninitialized"
        if checkpoint_path is not None:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = saved.get("state_dict", saved.get("model", saved))
            if not isinstance(state, dict):
                raise TypeError("RepVGG checkpoint must contain a state dictionary.")
            cleaned = {
                key.removeprefix("module."): value for key, value in state.items()
            }
            backbone.load_state_dict(cleaned, strict=True)
            self.checkpoint_sha256 = sha256_file(checkpoint_path)

        backbone.linear = nn.Identity()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def parameter_layer_id(self, name: str) -> int:
        name = name.removeprefix("backbone.")
        if name.startswith("stage0."):
            return 0
        for index in range(1, 5):
            if name.startswith(f"stage{index}."):
                return index
        return self.num_lr_layers - 1


def _load_official_sixdrepnet(model_config: dict[str, Any]) -> nn.Module:
    repo = resolve_path(model_config["repo"], must_exist=True)
    checkpoint = resolve_path(model_config.get("official_checkpoint"), must_exist=True)
    package_dir = repo / "sixdrepnet"
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    spec = importlib.util.spec_from_file_location(
        "_dime_official_sixdrepnet_model", package_dir / "model.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import official 6DRepNet from {package_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SixDRepNet(
        backbone_name="RepVGG-B1g2",
        backbone_file="",
        deploy=bool(model_config.get("deploy", True)),
        pretrained=False,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("model_state_dict", saved.get("state_dict", saved))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.checkpoint_sha256 = sha256_file(checkpoint)
    return model


def build_model(
    config: dict[str, Any], *, initialize_encoder: bool = True
) -> nn.Module:
    model_config = config["model"]
    kind = str(model_config["kind"]).lower()
    head_config = config.get("head", {})
    finetuning = config.get("finetuning", {})
    encoder_mode = str(finetuning.get("encoder_mode", "full")).lower()
    if encoder_mode not in {"full", "frozen"}:
        raise ValueError("finetuning.encoder_mode must be either 'full' or 'frozen'.")

    def wrap(encoder: nn.Module, feature_dim: int) -> HeadPoseModel:
        return HeadPoseModel(
            encoder,
            feature_dim,
            head_type=str(head_config.get("type", "linear")),
            hidden_dim=int(head_config.get("hidden_dim", 512)),
            dropout=float(head_config.get("dropout", 0.0)),
            freeze_encoder=encoder_mode == "frozen",
        )

    if kind == "dime":
        checkpoint = resolve_path(
            model_config.get("pretrained_checkpoint"), must_exist=True
        )
        encoder = DIMEEncoder.from_checkpoint(
            resolve_path(model_config["dime_source"], must_exist=True),
            checkpoint,
            str(model_config.get("dime_model", "")),
        )
        model_config["dime_model"] = encoder.model_name
        model_config["pretrained_sha256"] = encoder.checkpoint_sha256
        return wrap(encoder, encoder.num_features)
    if kind in {"dino", "mae"}:
        optional_checkpoint = (
            resolve_path(model_config.get("pretrained_checkpoint"))
            if initialize_encoder
            else None
        )
        encoder = TimmEncoder(
            str(model_config["timm_model"]),
            pretrained=initialize_encoder
            and bool(model_config.get("pretrained", True)),
            checkpoint_path=optional_checkpoint,
            pooling=str(model_config.get("pooling", "cls")),
        )
        model_config["pretrained_sha256"] = encoder.checkpoint_sha256
        return wrap(encoder, encoder.num_features)
    if kind == "repvgg":
        checkpoint = (
            resolve_path(model_config.get("pretrained_checkpoint"), must_exist=True)
            if initialize_encoder
            else None
        )
        encoder = RepVGGEncoder(
            resolve_path(model_config["repo"], must_exist=True),
            str(model_config.get("backbone_name", "RepVGG-B1g2")),
            checkpoint_path=checkpoint,
        )
        if checkpoint is not None:
            _verify_expected_sha256(
                encoder.checkpoint_sha256,
                model_config.get("expected_pretrained_sha256"),
                checkpoint,
            )
        model_config["pretrained_sha256"] = encoder.checkpoint_sha256
        return wrap(encoder, encoder.num_features)
    if kind == "sixdrepnet_official":
        return _load_official_sixdrepnet(model_config)
    raise ValueError(f"Unknown model.kind: {kind}")


def load_finetuned_checkpoint(model: nn.Module, path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}
