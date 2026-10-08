from typing import List, Optional
import torch.nn as nn
from ...network import Activation, heatmap2points


class FaceAlignmentTransformer(nn.Module):


    def __init__(self, backbone: nn.Module, heatmap_head: nn.Module,
                 heatmap_act: Optional[str] = 'relu'):
        super().__init__()
        self.backbone = backbone
        self.heatmap_head = heatmap_head
        self.heatmap_act = Activation(heatmap_act)
        self.cuda().float()

    def forward(self, image):
        features, _ = self.backbone(image)
        heatmap = self.heatmap_head(features)
        heatmap_acted = self.heatmap_act(heatmap)
        landmark = heatmap2points(heatmap_acted)
        return landmark, {'heatmap': heatmap, 'heatmap_acted': heatmap_acted}
