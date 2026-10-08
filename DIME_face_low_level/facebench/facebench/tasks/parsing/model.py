from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import FeatureBackbone, build_backbone


class ConvModule(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        padding: int = 0,
    ):
        convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        normalization = nn.SyncBatchNorm(out_channels)
        activation = nn.ReLU(inplace=True)
        super().__init__(convolution, normalization, activation)
        nn.init.kaiming_normal_(convolution.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(normalization.weight)
        nn.init.zeros_(normalization.bias)


class ViTFeaturePyramid(nn.Module):

    def __init__(self, channels: int = 768, in_channels: int | None = None):
        super().__init__()
        in_channels = channels if in_channels is None else in_channels
        self.projections = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if in_channels == channels
                    else nn.Conv2d(in_channels, channels, 1)
                )
                for _ in range(4)
            ]
        )
        self.levels = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
                    nn.SyncBatchNorm(channels),
                    nn.GELU(),
                    nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
                ),
                nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
                nn.Identity(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != 4:
            raise ValueError("The ViT feature pyramid requires four feature maps.")
        return [
            level(projection(feature))
            for level, projection, feature in zip(
                self.levels, self.projections, features
            )
        ]


class NativeFeaturePyramid(nn.Module):

    def __init__(self, in_channels: Sequence[int]):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("The native feature pyramid requires four stages.")
        self.in_channels = tuple(int(value) for value in in_channels)

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.in_channels):
            raise ValueError("DIME did not return four native feature stages.")
        actual = tuple(int(feature.shape[1]) for feature in features)
        if actual != self.in_channels:
            raise ValueError(
                f"DIME feature channels are {actual}; expected {self.in_channels}."
            )
        return list(features)


class PyramidPoolingModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.scales = pool_scales
        self.convolutions = nn.ModuleList(
            [ConvModule(in_channels, channels, kernel_size=1) for _ in pool_scales]
        )

    def forward(self, feature: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for scale, convolution in zip(self.scales, self.convolutions):
            pooled = convolution(F.adaptive_avg_pool2d(feature, output_size=scale))
            outputs.append(
                F.interpolate(
                    pooled,
                    size=feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return outputs


class UPerHead(nn.Module):

    def __init__(
        self,
        in_channels: Sequence[int],
        *,
        channels: int,
        num_classes: int,
        dropout: float = 0.1,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("UPerHead requires exactly four feature levels.")
        self.ppm = PyramidPoolingModule(in_channels[-1], channels, pool_scales)
        self.ppm_bottleneck = ConvModule(
            in_channels[-1] + len(pool_scales) * channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.lateral_convs = nn.ModuleList(
            [ConvModule(value, channels, kernel_size=1) for value in in_channels[:-1]]
        )
        self.fpn_convs = nn.ModuleList(
            [
                ConvModule(channels, channels, kernel_size=3, padding=1)
                for _ in in_channels[:-1]
            ]
        )
        self.fpn_bottleneck = ConvModule(
            len(in_channels) * channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 4:
            raise ValueError("UPerHead requires exactly four feature levels.")
        laterals = [
            convolution(inputs[index])
            for index, convolution in enumerate(self.lateral_convs)
        ]
        top = torch.cat([inputs[-1], *self.ppm(inputs[-1])], dim=1)
        laterals.append(self.ppm_bottleneck(top))

        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        outputs = [
            convolution(laterals[index])
            for index, convolution in enumerate(self.fpn_convs)
        ]
        outputs.append(laterals[-1])
        target_size = outputs[0].shape[-2:]
        outputs = [
            (
                value
                if value.shape[-2:] == target_size
                else F.interpolate(
                    value,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            for value in outputs
        ]
        fused = self.fpn_bottleneck(torch.cat(outputs, dim=1))
        return self.classifier(self.dropout(fused))


class ParsingModel(nn.Module):
    def __init__(
        self,
        backbone: FeatureBackbone,
        *,
        input_size: int,
        output_size: int,
        head_channels: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        if self.input_size % 32 != 0:
            raise ValueError("model.input_size must be divisible by 32.")
        if self.output_size <= 0:
            raise ValueError("model.output_size must be positive.")
        if backbone.pyramid_type == "vit":
            if len(set(backbone.out_channels)) != 1:
                raise ValueError("ViT feature channels must be identical.")
            self.pyramid = ViTFeaturePyramid(head_channels, backbone.out_channels[0])
            head_in_channels = [head_channels] * 4
        elif backbone.pyramid_type == "native":
            self.pyramid = NativeFeaturePyramid(backbone.out_channels)
            head_in_channels = list(backbone.out_channels)
        else:
            raise ValueError(f"Unknown pyramid type {backbone.pyramid_type!r}.")
        self.head = UPerHead(
            head_in_channels,
            channels=head_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    @property
    def encoder(self) -> nn.Module:
        return self.backbone

    @property
    def decoder(self) -> tuple[nn.Module, nn.Module]:
        return self.pyramid, self.head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width].")
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        features = self.backbone(self.backbone.normalize(images))
        pyramid = self.pyramid(features)
        expected_sizes = (
            self.input_size // 4,
            self.input_size // 8,
            self.input_size // 16,
            self.input_size // 32,
        )
        actual_sizes = tuple(tuple(value.shape[-2:]) for value in pyramid)
        expected_shapes = tuple((size, size) for size in expected_sizes)
        if actual_sizes != expected_shapes:
            raise RuntimeError(
                f"Feature pyramid resolutions are {actual_sizes}; "
                f"expected {expected_shapes}."
            )
        logits = self.head(pyramid)
        return F.interpolate(
            logits,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )


def build_model(
    config: dict[str, Any],
    *,
    initialize_pretrained: bool = True,
) -> ParsingModel:
    model_config = config["model"]
    backbone = build_backbone(config, initialize_pretrained=initialize_pretrained)
    return ParsingModel(
        backbone,
        input_size=int(model_config.get("input_size", 448)),
        output_size=int(model_config.get("output_size", 512)),
        head_channels=int(model_config.get("head_channels", 768)),
        num_classes=int(config["dataset"]["num_classes"]),
        dropout=float(model_config.get("dropout", 0.1)),
    )


def load_finetuned_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    use_ema: bool = True,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Fine-tuned checkpoint must be a dictionary.")
    if use_ema and isinstance(checkpoint.get("ema"), dict):
        state = checkpoint["ema"]
    elif isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    elif all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        state = checkpoint
    else:
        requested = "raw model" if not use_ema else "EMA/model"
        raise KeyError(
            f"Checkpoint contains no {requested} state. `best.pt` is EMA-only; "
            "use EMA evaluation or evaluate raw weights from `last.pt`."
        )
    if not isinstance(state, dict):
        raise KeyError("Checkpoint contains neither an EMA nor model state.")
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=True)
    return checkpoint
