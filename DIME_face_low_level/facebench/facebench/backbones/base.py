from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class FeatureBackbone(nn.Module, ABC):

    def __init__(
        self,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        checkpoint_hash: str = "",
    ) -> None:
        super().__init__()
        self.register_buffer(
            "_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False
        )

        self.checkpoint_hash = checkpoint_hash

    @property
    @abstractmethod
    def feature_dims(self) -> list[int]:
        pass

    @property
    def feature_strides(self) -> list[int]:

        return [16] * len(self.feature_dims)

    @property
    def pooled_dim(self) -> int:
        return self.feature_dims[-1]

    def normalize(self, images: torch.Tensor) -> torch.Tensor:

        return (images - self._mean) / self._std

    @abstractmethod
    def forward_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        pass

    def forward_pooled(self, images: torch.Tensor) -> torch.Tensor:

        return self.forward_features(images)[-1].mean(dim=(2, 3))

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_features(images)

    def parameter_layer_id(self, name: str) -> int:

        return 0

    @property
    def num_layers(self) -> int:
        return 1

    def freeze(self) -> "FeatureBackbone":

        self._frozen = True
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def train(self, mode: bool = True):

        if getattr(self, "_frozen", False):
            return super().train(False)
        return super().train(mode)
