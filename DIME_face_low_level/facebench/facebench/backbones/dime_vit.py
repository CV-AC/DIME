import sys
from pathlib import Path

import torch
from torch import nn


project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from DIME_VIT.encoder import load_encoder


class DIMEViTFeatureBackbone(nn.Module):
    def __init__(self, encoder, checkpoint_sha256):
        super().__init__()
        self.encoder = encoder
        self.model_name = encoder.model_name
        self.out_channels = (encoder.num_features,) * 4
        self.pyramid_type = "vit"
        self.checkpoint_sha256 = checkpoint_sha256
        self.register_buffer(
            "image_mean", torch.tensor(encoder.image_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(encoder.image_std).view(1, 3, 1, 1)
        )
        self.encoder.use_checkpoint = True

    def normalize(self, images):
        return (images - self.image_mean) / self.image_std

    def forward(self, images):
        return self.encoder.forward_intermediates(images)
