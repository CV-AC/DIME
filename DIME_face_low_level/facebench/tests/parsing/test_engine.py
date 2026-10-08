from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from facebench.tasks.parsing.data import collate_evaluation
from facebench.tasks.parsing.engine import ModelEMA, evaluate_model, train_one_epoch
from facebench.tasks.parsing.labels import LAPA_LABELS


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Conv2d(3, 11, kernel_size=1)

    def forward(self, images):
        return self.classifier(images)


class TinyTrainDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "image": torch.full((3, 16, 16), index / 4.0),
            "label": torch.zeros((16, 16), dtype=torch.long),
            "sample_id": str(index),
        }


class TinyEvalDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 16, 16),
            "sample_id": str(index),
            "label_original": np.zeros((16, 16), dtype=np.uint8),
            "original_shape": (16, 16),
            "transform": np.eye(3),
        }


def test_training_and_inverse_evaluation(tmp_path: Path):
    device = torch.device("cpu")
    model = TinyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = ModelEMA(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    train_loader = DataLoader(TinyTrainDataset(), batch_size=2)
    values = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scaler,
        ema,
        device,
        amp_enabled=False,
        amp_dtype="fp16",
        gradient_clip_norm=0.0,
        log_interval=0,
    )
    assert values["loss"] > 0.0

    with torch.no_grad():
        ema.module.classifier.weight.zero_()
        ema.module.classifier.bias.zero_()
    eval_loader = DataLoader(
        TinyEvalDataset(), batch_size=2, collate_fn=collate_evaluation
    )
    metrics = evaluate_model(
        ema.module,
        eval_loader,
        LAPA_LABELS,
        device,
        amp_enabled=False,
        amp_dtype="fp16",
        canvas_size=16,
        warp_factor=0.0,
        prediction_dir=tmp_path,
    )
    assert metrics["pixel_accuracy"] == 1.0
    assert metrics["per_class"]["background"]["f1"] == 1.0
    assert (tmp_path / "0.png").is_file()
    assert (tmp_path / "1.png").is_file()
