from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F


class FaceParsingTransformer(nn.Module):


    def __init__(self, backbone: nn.Module, head: nn.Module, out_size: Tuple[int, int]):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.out_size = out_size
        self.cuda().float()

    def forward(self, image):
        features, _ = self.backbone(image)
        logits = self.head(features)
        return F.interpolate(logits, size=self.out_size, mode='bilinear', align_corners=False), dict()
