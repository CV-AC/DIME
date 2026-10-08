from __future__ import annotations


import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import FeatureBackbone, build_backbone


class ConvModule(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0
    ):
        convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        nn.init.kaiming_normal_(convolution.weight, mode="fan_out", nonlinearity="relu")
        super().__init__(
            convolution,
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )


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
        return [
            operation(projection(feature))
            for operation, projection, feature in zip(
                self.levels, self.projections, features
            )
        ]


class NativeFeaturePyramid(nn.Module):

    def __init__(self, in_channels: Sequence[int], channels: int = 768):
        super().__init__()
        self.projections = nn.ModuleList(
            [ConvModule(value, channels, kernel_size=1) for value in in_channels]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [
            projection(feature)
            for projection, feature in zip(self.projections, features)
        ]


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
            pooled = F.adaptive_avg_pool2d(feature, output_size=scale)
            pooled = convolution(pooled)
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
        channels: int,
        num_landmarks: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("UPerHead requires exactly four feature levels.")
        self.ppm = PyramidPoolingModule(in_channels[-1], channels)
        self.ppm_bottleneck = ConvModule(
            in_channels[-1] + 4 * channels, channels, kernel_size=3, padding=1
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
            len(in_channels) * channels, channels, kernel_size=3, padding=1
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(channels, num_landmarks, kernel_size=1)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward_features(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
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
                output
                if output.shape[-2:] == target_size
                else F.interpolate(
                    output, size=target_size, mode="bilinear", align_corners=False
                )
            )
            for output in outputs
        ]
        return self.dropout(self.fpn_bottleneck(torch.cat(outputs, dim=1)))

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


def heatmap_to_points(heatmap: torch.Tensor) -> torch.Tensor:

    if heatmap.ndim != 4:
        raise ValueError("heatmap must have shape [B,K,H,W].")
    _, _, height, width = heatmap.shape
    dtype, device = heatmap.dtype, heatmap.device
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    denominator = heatmap.sum(dim=(-2, -1)).clamp_min(1e-6)
    x = (heatmap * xx).sum(dim=(-2, -1)) / denominator
    y = (heatmap * yy).sum(dim=(-2, -1)) / denominator
    pixels = torch.stack((x, y), dim=-1)
    scale = torch.tensor((width, height), device=device, dtype=dtype)
    return (pixels + 0.5) / scale


def route_a_local_soft_argmax(
    heatmap: torch.Tensor,
    *,
    window_size: int = 5,
    temperature: float = 10.0,
) -> torch.Tensor:

    if heatmap.ndim != 4:
        raise ValueError("heatmap must have shape [B,K,H,W].")
    if window_size < 1 or window_size % 2 != 1:
        raise ValueError(
            "local soft-argmax window_size must be a positive odd integer."
        )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("local soft-argmax temperature must be finite and positive.")
    batch, landmarks, height, width = heatmap.shape
    flat = heatmap.float().reshape(batch, landmarks, height * width)
    peak = flat.argmax(dim=-1)
    peak_y = torch.div(peak, width, rounding_mode="floor")
    peak_x = peak.remainder(width)

    radius = window_size // 2
    offsets = torch.arange(-radius, radius + 1, device=heatmap.device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    dx = dx.reshape(1, 1, -1)
    dy = dy.reshape(1, 1, -1)
    x = peak_x[..., None] + dx
    y = peak_y[..., None] + dy
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    safe_x = x.clamp(0, width - 1)
    safe_y = y.clamp(0, height - 1)
    local_index = safe_y * width + safe_x
    values = flat.gather(-1, local_index)
    values = (float(temperature) * values).masked_fill(~valid, float("-inf"))
    weights = torch.softmax(values, dim=-1)
    x_value = (weights * x.to(weights.dtype)).sum(dim=-1)
    y_value = (weights * y.to(weights.dtype)).sum(dim=-1)
    pixels = torch.stack((x_value, y_value), dim=-1)
    scale = torch.tensor((width, height), device=heatmap.device, dtype=weights.dtype)
    return (pixels + 0.5) / scale


def points_to_heatmap(
    normalized_points: torch.Tensor,
    size: int = 128,
    radius: float = 5.0,
) -> torch.Tensor:

    points = normalized_points.float() * size - 0.5
    axis = torch.arange(size, device=points.device, dtype=points.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    dx = xx.view(1, 1, size, size) - points[..., 0, None, None]
    dy = yy.view(1, 1, size, size) - points[..., 1, None, None]
    squared_distance = dx.square() + dy.square()
    heatmap = torch.exp(-squared_distance * 16.0 / (2.0 * radius * radius))
    return heatmap * (squared_distance <= radius * radius)


def continuous_gaussian_heatmap(
    normalized_points: torch.Tensor,
    *,
    size: int,
    sigma: float = 1.0,
) -> torch.Tensor:

    if size <= 0:
        raise ValueError("Route A heatmap size must be positive.")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Route A Gaussian sigma must be finite and positive.")
    points = normalized_points.float() * size - 0.5
    axis = torch.arange(size, device=points.device, dtype=points.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    dx = xx.view(1, 1, size, size) - points[..., 0, None, None]
    dy = yy.view(1, 1, size, size) - points[..., 1, None, None]
    return torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma * sigma))


def foreground_weight_map(
    target: torch.Tensor,
    *,
    foreground_weight: float = 10.0,
    foreground_threshold: float = 0.2,
    dilation_kernel: int = 3,
) -> torch.Tensor:

    if foreground_weight < 0.0 or not math.isfinite(foreground_weight):
        raise ValueError("foreground_weight must be finite and non-negative.")
    if not 0.0 <= foreground_threshold <= 1.0:
        raise ValueError("foreground_threshold must be in [0, 1].")
    if dilation_kernel < 1 or dilation_kernel % 2 != 1:
        raise ValueError("dilation_kernel must be a positive odd integer.")
    dilated = F.max_pool2d(
        target,
        kernel_size=dilation_kernel,
        stride=1,
        padding=dilation_kernel // 2,
    )
    foreground = dilated >= float(foreground_threshold)
    return 1.0 + float(foreground_weight) * foreground.to(target.dtype)


def adaptive_wing_loss_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 2.1,
    omega: float = 14.0,
    epsilon: float = 1.0,
    theta: float = 0.5,
) -> torch.Tensor:

    values = (alpha, omega, epsilon, theta)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Adaptive Wing parameters must be finite.")
    if alpha <= 1.0 or omega <= 0.0 or epsilon <= 0.0 or theta <= 0.0:
        raise ValueError("Adaptive Wing requires alpha>1 and omega/epsilon/theta>0.")
    if prediction.shape != target.shape:
        raise ValueError(
            f"Heatmap prediction/target shapes differ: "
            f"{prediction.shape} versus {target.shape}."
        )

    delta = (target - prediction).abs()
    exponent = float(alpha) - target
    theta_over_epsilon = float(theta) / float(epsilon)
    power = torch.pow(
        torch.as_tensor(theta_over_epsilon, device=target.device, dtype=target.dtype),
        exponent,
    )
    coefficient = (
        float(omega)
        * (1.0 / (1.0 + power))
        * exponent
        * torch.pow(
            torch.as_tensor(
                theta_over_epsilon, device=target.device, dtype=target.dtype
            ),
            exponent - 1.0,
        )
        / float(epsilon)
    )
    constant = float(theta) * coefficient - float(omega) * torch.log1p(power)
    nonlinear = float(omega) * torch.log1p(torch.pow(delta / float(epsilon), exponent))
    linear = coefficient * delta - constant
    return torch.where(delta < float(theta), nonlinear, linear)


def foreground_balanced_heatmap_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    options: dict[str, Any] | None = None,
) -> torch.Tensor:

    options = dict(options or {})
    name = str(options.get("name", "adaptive_wing")).strip().lower()
    aliases = {
        "awing": "adaptive_wing",
        "bcewithlogits": "bce",
        "weighted_mse": "mse",
    }
    name = aliases.get(name, name)
    if name not in {"adaptive_wing", "bce", "mse"}:
        raise ValueError("Route A loss.name must be adaptive_wing, bce, or mse.")

    prediction = prediction.float()
    target = target.float()
    weights = foreground_weight_map(
        target,
        foreground_weight=float(options.get("foreground_weight", 10.0)),
        foreground_threshold=float(options.get("foreground_threshold", 0.2)),
        dilation_kernel=int(options.get("dilation_kernel", 3)),
    )
    if name == "adaptive_wing":
        loss_map = adaptive_wing_loss_map(
            prediction,
            target,
            alpha=float(options.get("alpha", 2.1)),
            omega=float(options.get("omega", 14.0)),
            epsilon=float(options.get("epsilon", 1.0)),
            theta=float(options.get("theta", 0.5)),
        )
    elif name == "bce":
        loss_map = F.binary_cross_entropy_with_logits(
            prediction, target, reduction="none"
        )
    else:
        loss_map = (prediction - target).square()
    return (loss_map * weights).mean()


@torch.no_grad()
def heatmap_diagnostics(
    prediction: torch.Tensor,
    target_points: torch.Tensor,
    *,
    collapse_min_std: float = 1.0e-4,
) -> dict[str, torch.Tensor]:

    if collapse_min_std < 0.0 or not math.isfinite(collapse_min_std):
        raise ValueError("collapse_min_std must be finite and non-negative.")
    prediction = prediction.detach().float()
    target_points = target_points.detach().float()
    if prediction.ndim != 4 or target_points.shape != (*prediction.shape[:2], 2):
        raise ValueError("Expected heatmaps [B,K,H,W] and matching points [B,K,2].")
    batch, landmarks, height, width = prediction.shape
    flat = prediction.reshape(batch, landmarks, height * width)
    per_heatmap_std = flat.std(dim=-1, unbiased=False)

    grids = (target_points * 2.0 - 1.0).reshape(batch * landmarks, 1, 1, 2)
    target_response = F.grid_sample(
        prediction.reshape(batch * landmarks, 1, height, width),
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(batch, landmarks)
    return {
        "heatmap_max": flat.amax(dim=-1).mean(),
        "heatmap_std": per_heatmap_std.mean(),
        "heatmap_peak_response": target_response.mean(),
        "heatmap_collapsed_fraction": (per_heatmap_std < float(collapse_min_std))
        .float()
        .mean(),
    }


class LandmarkModel(nn.Module):
    def __init__(
        self,
        backbone: FeatureBackbone,
        *,
        input_size: int = 448,
        pyramid_channels: int | None = None,
        head_channels: int = 768,
        num_landmarks: int = 98,
        auxiliary_training: dict[str, Any] | None = None,
        objective: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.input_size = int(input_size)
        self.objective = dict(objective or {"name": "farl"})
        self.objective_name = str(self.objective.get("name", "farl")).strip().lower()
        if self.objective_name not in {"farl", "route_a"}:
            raise ValueError("model.objective.name must be farl or route_a.")
        self.route_a_window_size = int(self.objective.get("window_size", 5))
        self.route_a_temperature = float(self.objective.get("temperature", 10.0))
        if self.objective_name == "route_a":
            native_size = self.input_size // 4
            if int(self.objective.get("heatmap_size", native_size)) != native_size:
                raise ValueError(
                    "Route A requires heatmap_size=input_size/4 so prediction, "
                    "target, loss, and decoding share one native grid."
                )
            if self.route_a_window_size < 1 or self.route_a_window_size % 2 != 1:
                raise ValueError("Route A window_size must be a positive odd integer.")
            if (
                not math.isfinite(self.route_a_temperature)
                or self.route_a_temperature <= 0.0
            ):
                raise ValueError("Route A temperature must be finite and positive.")
            sigma = float(self.objective.get("sigma", 1.0))
            if not math.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("Route A sigma must be finite and positive.")

            foreground_balanced_heatmap_loss(
                torch.zeros(1, 1, 1, 1),
                torch.zeros(1, 1, 1, 1),
                self.objective.get("loss"),
            )
            collapse = dict(self.objective.get("collapse_detection", {}))
            warmup_epochs = int(collapse.get("warmup_epochs", 2))
            patience = int(collapse.get("patience", 2))
            fraction_threshold = float(collapse.get("fraction_threshold", 0.95))
            if warmup_epochs < 0 or patience <= 0:
                raise ValueError(
                    "Route A collapse warmup_epochs must be >=0 and patience >0."
                )
            if not 0.0 <= fraction_threshold <= 1.0:
                raise ValueError(
                    "Route A collapse fraction_threshold must be in [0,1]."
                )
            min_std = float(collapse.get("min_heatmap_std", 1.0e-4))
            if min_std < 0.0 or not math.isfinite(min_std):
                raise ValueError(
                    "Route A collapse min_heatmap_std must be finite and non-negative."
                )
        head_channels = int(head_channels)

        pyramid_channels = int(
            head_channels if pyramid_channels is None else pyramid_channels
        )
        if pyramid_channels <= 0 or head_channels <= 0:
            raise ValueError("Pyramid and head channels must be positive.")
        if backbone.pyramid_type == "vit":
            if len(set(backbone.out_channels)) != 1:
                raise ValueError("ViT pyramid expects equal feature dimensions.")
            self.pyramid = ViTFeaturePyramid(pyramid_channels, backbone.out_channels[0])
        else:
            self.pyramid = NativeFeaturePyramid(backbone.out_channels, pyramid_channels)
        self.head = UPerHead(
            [pyramid_channels] * 4,
            channels=head_channels,
            num_landmarks=num_landmarks,
            dropout=0.1,
        )
        auxiliary_training = dict(auxiliary_training or {})
        self.auxiliary_enabled = bool(auxiliary_training.get("enabled", False))
        self.auxiliary_heads = nn.ModuleDict()
        if self.auxiliary_enabled:
            datasets = auxiliary_training.get("datasets", {})
            lapa = datasets.get("lapa", {})
            lp = datasets.get("300w_lp", {})
            if bool(lapa.get("enabled", True)):
                self.auxiliary_heads["lapa"] = self._make_classifier(
                    head_channels, int(lapa.get("num_landmarks", 106))
                )
                if bool(lapa.get("parsing_enabled", True)):
                    self.auxiliary_heads["lapa_parsing"] = self._make_classifier(
                        head_channels, int(lapa.get("num_parsing_classes", 11))
                    )
            if bool(lp.get("enabled", True)):
                self.auxiliary_heads["300w_lp"] = self._make_classifier(
                    head_channels, int(lp.get("num_landmarks", 68))
                )

    @staticmethod
    def _make_classifier(in_channels: int, out_channels: int) -> nn.Conv2d:
        classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.normal_(classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(classifier.bias)
        return classifier

    @property
    def encoder(self) -> nn.Module:
        return self.backbone

    @property
    def downstream_modules(self) -> tuple[nn.Module, ...]:
        if self.auxiliary_enabled:
            return self.pyramid, self.head, self.auxiliary_heads
        return self.pyramid, self.head

    def decode_heatmaps(self, logits: torch.Tensor) -> torch.Tensor:
        if self.objective_name == "farl":
            return heatmap_to_points(torch.sigmoid(logits.float()))
        return route_a_local_soft_argmax(
            logits.float(),
            window_size=self.route_a_window_size,
            temperature=self.route_a_temperature,
        )

    def _inactive_parameter_anchor(self, active: set[str]) -> torch.Tensor:

        anchor: torch.Tensor | None = None
        for name, module in self.auxiliary_heads.items():
            if name in active:
                continue
            for parameter in module.parameters():
                value = parameter.reshape(-1)[0] * 0.0
                anchor = value if anchor is None else anchor + value
        if "wflw" not in active:
            for parameter in self.head.classifier.parameters():
                value = parameter.reshape(-1)[0] * 0.0
                anchor = value if anchor is None else anchor + value
        if anchor is None:
            anchor = self.head.classifier.weight.new_zeros(())
        return anchor

    def forward(
        self, images: torch.Tensor, *, task: str = "wflw"
    ) -> dict[str, torch.Tensor]:
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
        actual_sizes = tuple(feature.shape[-1] for feature in pyramid)
        if actual_sizes != expected_sizes:
            raise RuntimeError(
                f"Feature pyramid resolutions are {actual_sizes}; expected {expected_sizes}."
            )
        decoded = self.head.forward_features(pyramid)
        if task == "wflw":
            classifier = self.head.classifier
        elif task in self.auxiliary_heads and task != "lapa_parsing":
            classifier = self.auxiliary_heads[task]
        else:
            raise ValueError(f"Unsupported landmark task {task!r}.")

        active = {task}
        if task == "lapa" and "lapa_parsing" in self.auxiliary_heads:
            active.add("lapa_parsing")
        anchor = self._inactive_parameter_anchor(active)
        logits = classifier(decoded) + anchor
        points = self.decode_heatmaps(logits)
        outputs = {"heatmap_logits": logits, "points": points}
        if task == "lapa" and "lapa_parsing" in self.auxiliary_heads:
            outputs["parsing_logits"] = self.auxiliary_heads["lapa_parsing"](decoded)
        return outputs


def landmark_losses(
    outputs: dict[str, torch.Tensor],
    landmarks_canvas: torch.Tensor,
    *,
    canvas_size: int = 512,
    objective: dict[str, Any] | None = None,
    heatmap_size: int = 128,
    heatmap_radius: float = 5.0,
) -> dict[str, torch.Tensor]:
    objective = dict(objective or {"name": "farl"})
    objective_name = str(objective.get("name", "farl")).strip().lower()
    target_points = (landmarks_canvas.float() + 0.5) / float(canvas_size)
    coordinate = (outputs["points"].float() - target_points).norm(dim=-1).mean()
    if objective_name == "route_a":
        native_size = int(objective.get("heatmap_size", 112))
        logits = outputs["heatmap_logits"].float()
        if logits.shape[-2:] != (native_size, native_size):
            raise ValueError(
                "Route A heatmap logits must already have the configured native "
                f"resolution {(native_size, native_size)}, got {logits.shape[-2:]}."
            )
        target_heatmap = continuous_gaussian_heatmap(
            target_points,
            size=native_size,
            sigma=float(objective.get("sigma", 1.0)),
        )
        heatmap = foreground_balanced_heatmap_loss(
            logits, target_heatmap, objective.get("loss")
        )
        collapse = dict(objective.get("collapse_detection", {}))
        diagnostics = heatmap_diagnostics(
            logits,
            target_points,
            collapse_min_std=float(collapse.get("min_heatmap_std", 1.0e-4)),
        )
        return {
            "loss": heatmap,
            "coordinate": coordinate,
            "heatmap": heatmap,
            **diagnostics,
        }
    if objective_name != "farl":
        raise ValueError("model.objective.name must be farl or route_a.")
    heatmap_size = int(objective.get("heatmap_size", heatmap_size))
    heatmap_radius = float(objective.get("heatmap_radius", heatmap_radius))
    if heatmap_size <= 0 or not math.isfinite(heatmap_radius) or heatmap_radius <= 0:
        raise ValueError("FaRL heatmap_size and heatmap_radius must be positive.")
    logits = F.interpolate(
        outputs["heatmap_logits"].float(),
        size=(heatmap_size, heatmap_size),
        mode="bilinear",
        align_corners=False,
    )
    target_heatmap = points_to_heatmap(target_points, heatmap_size, heatmap_radius)
    heatmap = F.binary_cross_entropy_with_logits(logits, target_heatmap)
    return {"loss": coordinate + heatmap, "coordinate": coordinate, "heatmap": heatmap}


def parsing_losses(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    dice_weight: float = 1.0,
    include_background: bool = False,
) -> dict[str, torch.Tensor]:

    target = F.interpolate(
        target[:, None].float(),
        size=logits.shape[-2:],
        mode="nearest",
    )[:, 0].long()
    if valid_mask is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = F.interpolate(
            valid_mask[:, None].float(),
            size=logits.shape[-2:],
            mode="nearest",
        )[:, 0].bool()
    valid_float = valid[:, None].to(dtype=torch.float32)
    cross_entropy_map = F.cross_entropy(logits.float(), target, reduction="none")
    cross_entropy = (
        cross_entropy_map * valid.to(dtype=cross_entropy_map.dtype)
    ).sum() / valid.sum().clamp_min(1)
    probabilities = logits.float().softmax(dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=probabilities.dtype)
    start = 0 if include_background else 1
    probabilities = probabilities[:, start:] * valid_float
    one_hot = one_hot[:, start:] * valid_float
    intersection = (probabilities * one_hot).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + one_hot.sum(dim=(-2, -1))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return {
        "loss": cross_entropy + float(dice_weight) * dice,
        "cross_entropy": cross_entropy,
        "dice": dice,
    }


def task_losses(
    outputs: dict[str, torch.Tensor],
    landmarks_canvas: torch.Tensor,
    *,
    task: str,
    auxiliary_training: dict[str, Any],
    objective: dict[str, Any] | None = None,
    parsing_target: torch.Tensor | None = None,
    parsing_valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    landmark = landmark_losses(outputs, landmarks_canvas, objective=objective)
    weights = auxiliary_training.get("loss_weights", {})
    landmark_weight = float(weights.get(task, 1.0))
    total = landmark["loss"] * landmark_weight
    result = {
        "loss": total,
        "coordinate": landmark["coordinate"],
        "heatmap": landmark["heatmap"],
    }
    for key in (
        "heatmap_max",
        "heatmap_std",
        "heatmap_peak_response",
        "heatmap_collapsed_fraction",
    ):
        if key in landmark:
            result[key] = landmark[key]
    if task == "lapa" and "parsing_logits" in outputs:
        if parsing_target is None:
            raise ValueError("LaPa parsing is enabled but parsing_mask is missing.")
        parsing = parsing_losses(
            outputs["parsing_logits"],
            parsing_target,
            valid_mask=parsing_valid_mask,
            dice_weight=float(weights.get("lapa_parsing_dice", 1.0)),
            include_background=bool(weights.get("parsing_include_background", False)),
        )
        result["loss"] = (
            result["loss"] + float(weights.get("lapa_parsing", 0.1)) * parsing["loss"]
        )
        result["parsing"] = parsing["loss"]
    return result


def build_model(
    config: dict[str, Any], *, initialize_pretrained: bool = True
) -> LandmarkModel:
    backbone = build_backbone(config, initialize_pretrained=initialize_pretrained)
    model_config = config["model"]
    return LandmarkModel(
        backbone,
        input_size=int(model_config.get("input_size", 448)),
        pyramid_channels=int(model_config.get("pyramid_channels", 768)),
        head_channels=int(model_config.get("head_channels", 768)),
        num_landmarks=int(model_config.get("num_landmarks", 98)),
        auxiliary_training=config.get("auxiliary_training"),
        objective=model_config.get("objective"),
    )


def load_finetuned_checkpoint(model: nn.Module, path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=True)
    return checkpoint
