from pathlib import Path

import numpy as np
import pytest
import torch

from facebench.tasks.parsing.data import (
    ParsingSample,
    load_sample_manifest,
    write_sample_manifest,
)
from facebench.tasks.parsing.model import load_finetuned_checkpoint


def test_lapa_manifest_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "LaPa"
    sample = ParsingSample(
        sample_id="train.face_001",
        image_path=root / "train/images/face_001.jpg",
        label_path=root / "train/labels/face_001.png",
        landmarks=np.arange(212, dtype=np.float32).reshape(106, 2),
    )
    path = tmp_path / "manifests/lapa_train.npz"
    write_sample_manifest(path, dataset="lapa", split="train", samples=[sample])
    restored = load_sample_manifest(path, dataset="lapa", split="train", root=root)
    assert len(restored) == 1
    assert restored[0].sample_id == sample.sample_id
    assert restored[0].image_path == sample.image_path
    np.testing.assert_array_equal(restored[0].landmarks, sample.landmarks)


def test_celeb_manifest_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "CelebAMask-HQ"
    cache = tmp_path / "masks"
    sample = ParsingSample(
        sample_id="val.42",
        image_path=root / "CelebA-HQ-img/42.jpg",
        label_path=cache / "00042.png",
        hq_id=42,
    )
    path = tmp_path / "manifests/celebamask_hq_val.npz"
    write_sample_manifest(path, dataset="celebamask_hq", split="val", samples=[sample])
    restored = load_sample_manifest(
        path,
        dataset="celebamask_hq",
        split="val",
        root=root,
        cache_root=cache,
    )
    assert restored == [sample]


def test_ema_only_checkpoint_rejects_raw_evaluation(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "format_version": 2,
            "checkpoint_type": "ema_inference",
            "ema": model.state_dict(),
        },
        checkpoint,
    )
    load_finetuned_checkpoint(model, checkpoint, use_ema=True)
    with pytest.raises(KeyError, match="EMA-only"):
        load_finetuned_checkpoint(model, checkpoint, use_ema=False)
