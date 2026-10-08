from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

from .config import resolve_path
from .lmdb_store import LmdbImageReader
from .rotation import euler_to_matrix
from .utils import read_jsonl, sha256_file

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _validate_manifest_summary(
    manifest: Path,
    records: list[dict[str, Any]],
    actual_hash: str,
) -> None:
    summary_path = manifest.with_suffix(".summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing manifest provenance summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_hash = str(summary.get("sha256", summary.get("manifest_sha256", "")))
    if expected_hash != actual_hash:
        raise RuntimeError(f"Manifest SHA256 mismatch: {manifest}")
    if int(summary.get("accepted", -1)) != len(records):
        raise RuntimeError(
            f"Manifest count mismatch: summary={summary.get('accepted')} "
            f"records={len(records)}"
        )


def train_transform(
    input_size: int = 224,
    *,
    scale: tuple[float, float] = (0.8, 1.0),
    ratio: tuple[float, float] = (0.75, 4.0 / 3.0),
):

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                input_size,
                scale=scale,
                ratio=ratio,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def test_transform(input_size: int = 224, resize_size: int = 256):
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _rotation_target(pitch: float, yaw: float, roll: float) -> torch.Tensor:
    return euler_to_matrix(
        torch.tensor(pitch, dtype=torch.float32),
        torch.tensor(yaw, dtype=torch.float32),
        torch.tensor(roll, dtype=torch.float32),
    )


class MatPoseDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        *,
        training: bool,
        input_size: int = 224,
        test_resize: int = 256,
        augmentation: dict[str, Any] | None = None,
        lmdb_path: str | Path | None = None,
    ):
        self.root = Path(root)
        self.manifest = Path(manifest)
        self.manifest_sha256 = sha256_file(self.manifest)
        self.records = read_jsonl(self.manifest)
        _validate_manifest_summary(self.manifest, self.records, self.manifest_sha256)
        self.image_reader = (
            LmdbImageReader(
                lmdb_path,
                manifest_sha256=self.manifest_sha256,
                record_count=len(self.records),
            )
            if lmdb_path is not None
            else None
        )
        self.storage_backend = "lmdb" if self.image_reader is not None else "files"
        self.storage_logical_content_sha256 = (
            self.image_reader.logical_content_sha256
            if self.image_reader is not None
            else None
        )
        self.training = training
        augmentation = dict(augmentation or {})
        scale = tuple(
            float(value)
            for value in augmentation.get("random_resized_crop_scale", (0.8, 1.0))
        )
        ratio = tuple(
            float(value)
            for value in augmentation.get(
                "random_resized_crop_ratio", (0.75, 4.0 / 3.0)
            )
        )
        if len(scale) != 2 or not 0.0 < scale[0] <= scale[1] <= 1.0:
            raise ValueError("augmentation.random_resized_crop_scale is invalid.")
        if len(ratio) != 2 or not 0.0 < ratio[0] <= ratio[1]:
            raise ValueError("augmentation.random_resized_crop_ratio is invalid.")
        self.horizontal_flip_probability = float(
            augmentation.get("horizontal_flip_probability", 0.5)
        )
        self.blur_probability = float(augmentation.get("blur_probability", 0.05))
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError(
                "augmentation.horizontal_flip_probability must be in [0,1]."
            )
        if not 0.0 <= self.blur_probability <= 1.0:
            raise ValueError("augmentation.blur_probability must be in [0,1].")
        self.transform = (
            train_transform(input_size, scale=scale, ratio=ratio)
            if training
            else test_transform(input_size, test_resize)
        )
        if not self.records:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def _crop(self, image: Image.Image, bbox: list[float]) -> Image.Image:
        x_min, y_min, x_max, y_max = bbox
        if self.training:
            k = float(np.random.random_sample() * 0.2 + 0.2)
            x_min -= 0.6 * k * abs(x_max - x_min)
            y_min -= 2.0 * k * abs(y_max - y_min)
            x_max += 0.6 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)
        else:
            k = 0.20
            x_min -= 2.0 * k * abs(x_max - x_min)
            y_min -= 2.0 * k * abs(y_max - y_min)
            x_max += 2.0 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)
        return image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self.image_reader is None:
            source_context = Image.open(self.root / record["image"])
        else:

            source_context = Image.open(
                io.BytesIO(self.image_reader.read(record["sample_id"]))
            )
        with source_context as source:
            image = source.convert("RGB")
        image = self._crop(image, record["landmark_bbox"])
        pitch, yaw, roll = record["pitch"], record["yaw"], record["roll"]
        if (
            self.training
            and np.random.random_sample() < self.horizontal_flip_probability
        ):
            yaw, roll = -yaw, -roll
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.training and np.random.random_sample() < self.blur_probability:
            image = image.filter(ImageFilter.BLUR)

        ypr = torch.tensor([yaw, pitch, roll], dtype=torch.float32)
        return {
            "image": self.transform(image),
            "rotation": _rotation_target(pitch, yaw, roll),
            "ypr": ypr,
            "sample_id": record["sample_id"],
        }


class BIWIDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        *,
        input_size: int = 224,
        test_resize: int = 256,
        lmdb_path: str | Path | None = None,
    ):
        self.root = Path(root)
        self.manifest = Path(manifest)
        self.manifest_sha256 = sha256_file(self.manifest)
        self.records = read_jsonl(self.manifest)
        _validate_manifest_summary(self.manifest, self.records, self.manifest_sha256)
        self.image_reader = (
            LmdbImageReader(
                lmdb_path,
                manifest_sha256=self.manifest_sha256,
                record_count=len(self.records),
            )
            if lmdb_path is not None
            else None
        )
        self.storage_backend = "lmdb" if self.image_reader is not None else "files"
        self.storage_logical_content_sha256 = (
            self.image_reader.logical_content_sha256
            if self.image_reader is not None
            else None
        )
        self.transform = test_transform(input_size, test_resize)
        if not self.records:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self.image_reader is None:
            source_context = Image.open(self.root / record["image"])
        else:
            source_context = Image.open(
                io.BytesIO(self.image_reader.read(record["sample_id"]))
            )
        with source_context as source:
            image = source.convert("RGB")
        yaw, pitch, roll = np.deg2rad(
            [record["yaw_deg"], record["pitch_deg"], record["roll_deg"]]
        ).tolist()
        return {
            "image": self.transform(image),
            "rotation": _rotation_target(pitch, yaw, roll),
            "ypr": torch.tensor([yaw, pitch, roll], dtype=torch.float32),
            "sample_id": record["sample_id"],
        }


_ROOT_KEYS = {
    "train": "train_root",
    "aflw2000": "aflw2000_root",
    "biwi": "biwi_processed_root",
}
_LMDB_KEYS = {
    "train": "train_lmdb",
    "aflw2000": "aflw2000_lmdb",
    "biwi": "biwi_lmdb",
}


def data_backend(config: dict[str, Any]) -> str:
    backend = str(config.get("data", {}).get("backend", "files")).strip().lower()
    if backend not in {"files", "lmdb"}:
        raise ValueError(f"data.backend must be 'files' or 'lmdb', got {backend!r}.")
    return backend


def validate_data_storage(
    config: dict[str, Any], splits: tuple[str, ...]
) -> dict[str, dict[str, Any]]:

    data = config["data"]
    backend = data_backend(config)
    validated: dict[str, dict[str, Any]] = {}
    for split in splits:
        if split not in _ROOT_KEYS:
            raise ValueError(f"Unknown split: {split}")
        if backend == "files":
            root = resolve_path(data[_ROOT_KEYS[split]], must_exist=True)
            validated[split] = {"backend": "files", "path": str(root)}
            continue
        manifest_key = {
            "train": "train_manifest",
            "aflw2000": "aflw2000_manifest",
            "biwi": "biwi_manifest",
        }[split]
        manifest = resolve_path(data[manifest_key], must_exist=True)
        database = resolve_path(data[_LMDB_KEYS[split]], must_exist=True)
        assert manifest is not None and database is not None
        records = read_jsonl(manifest)
        manifest_hash = sha256_file(manifest)
        _validate_manifest_summary(manifest, records, manifest_hash)

        reader = LmdbImageReader(
            database,
            manifest_sha256=manifest_hash,
            record_count=len(records),
        )
        validated[split] = {
            "backend": "lmdb",
            "path": str(reader.path),
            "record_count": len(records),
            "manifest_sha256": manifest_hash,
            "logical_content_sha256": reader.logical_content_sha256,
        }
        reader.close()
    return validated


def build_dataset(config: dict[str, Any], split: str) -> Dataset:
    data = config["data"]
    protocol = config["protocol"]
    input_size = int(protocol.get("input_size", 224))
    if split not in _ROOT_KEYS:
        raise ValueError(f"Unknown split: {split}")
    backend = data_backend(config)
    lmdb_path = (
        resolve_path(data[_LMDB_KEYS[split]], must_exist=True)
        if backend == "lmdb"
        else None
    )
    root = (
        resolve_path(data[_ROOT_KEYS[split]], must_exist=True)
        if backend == "files"
        else Path(".")
    )
    if split == "train":
        return MatPoseDataset(
            root,
            resolve_path(data["train_manifest"], must_exist=True),
            training=True,
            input_size=input_size,
            augmentation=config.get("augmentation"),
            lmdb_path=lmdb_path,
        )
    if split == "aflw2000":
        return MatPoseDataset(
            root,
            resolve_path(data["aflw2000_manifest"], must_exist=True),
            training=False,
            input_size=input_size,
            test_resize=int(protocol.get("test_resize", 256)),
            lmdb_path=lmdb_path,
        )
    if split == "biwi":
        return BIWIDataset(
            root,
            resolve_path(data["biwi_manifest"], must_exist=True),
            input_size=input_size,
            test_resize=int(protocol.get("test_resize", 256)),
            lmdb_path=lmdb_path,
        )
    raise ValueError(f"Unknown split: {split}")
