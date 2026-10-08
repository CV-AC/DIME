from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from .config import resolve_path
from .geometry import (
    face_align_matrix,
    forward_transform_map,
    random_update_matrix,
    remap,
)
from .labels import CELEBAMASK_HQ_LABELS, LAPA_LABELS, LabelSpace, label_space
from .utils import seed_worker


cv2.setNumThreads(0)

EXPECTED_COUNTS = {
    "lapa": {"train": 18176, "val": 2000, "test": 2000},
    "celebamask_hq": {"train": 24183, "val": 2993, "test": 2824},
}
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ParsingSample:
    sample_id: str
    image_path: Path
    label_path: Path
    landmarks: np.ndarray | None = None
    hq_id: int | None = None


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not decode image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"Could not decode label {path}")
    return label


def _validate_label(label: np.ndarray, space: LabelSpace, path: Path) -> None:
    if label.ndim != 2:
        raise ValueError(f"Expected a 2D label at {path}, got {label.shape}.")
    minimum, maximum = int(label.min()), int(label.max())
    if minimum < 0 or maximum >= space.num_classes:
        raise ValueError(
            f"{path} contains labels [{minimum},{maximum}], expected "
            f"[0,{space.num_classes - 1}]."
        )


def load_lapa_samples(root: Path, split: str) -> list[ParsingSample]:
    split_root = root / split
    image_root = split_root / "images"
    label_root = split_root / "labels"
    landmark_root = split_root / "landmarks"
    if not all(path.is_dir() for path in (image_root, label_root, landmark_root)):
        raise FileNotFoundError(
            f"LaPa split {split!r} must contain images/, labels/, landmarks/."
        )

    samples: list[ParsingSample] = []
    for image_path in sorted(image_root.glob("*.jpg")):
        stem = image_path.stem
        label_path = label_root / f"{stem}.png"
        landmark_path = landmark_root / f"{stem}.txt"
        if not label_path.is_file() or not landmark_path.is_file():
            raise FileNotFoundError(
                f"Incomplete LaPa sample {split}/{stem}: expected label and landmarks."
            )
        values = np.fromstring(
            landmark_path.read_text(encoding="utf-8"), sep=" ", dtype=np.float32
        )
        if values.size != 213 or values[0] != 106:
            raise ValueError(
                f"{landmark_path} must contain count 106 followed by 212 values."
            )
        samples.append(
            ParsingSample(
                sample_id=f"{split}.{stem}",
                image_path=image_path,
                label_path=label_path,
                landmarks=values[1:].reshape(106, 2),
            )
        )
    return samples


def _celeb_orig_to_hq(root: Path) -> dict[str, int]:
    path = root / "CelebA-HQ-to-CelebA-mapping.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) != 3 or not fields[-1].endswith(".jpg"):
                continue
            hq_id = int(fields[0])
            original_name = fields[2]
            if original_name in mapping:
                raise ValueError(f"Duplicate CelebA name at {path}:{line_number}")
            mapping[original_name] = hq_id
    if len(mapping) != 30000:
        raise ValueError(f"Expected 30,000 HQ mappings, found {len(mapping)}.")
    return mapping


def celeb_split_ids(root: Path) -> dict[str, list[int]]:
    mapping = _celeb_orig_to_hq(root)
    partition_path = root / "list_eval_partition.txt"
    if not partition_path.is_file():
        raise FileNotFoundError(
            f"{partition_path} is missing. Download CelebA's official "
            "Train/Val/Test Partitions file."
        )
    groups = {0: "train", 1: "val", 2: "test"}
    split_ids: dict[str, list[int]] = {value: [] for value in groups.values()}
    matched: set[int] = set()
    with partition_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"Malformed partition row {line_number}: {line!r}")
            original_name, raw_group = fields
            if original_name not in mapping:
                continue
            group = int(raw_group)
            if group not in groups:
                raise ValueError(
                    f"Invalid partition group {group} at row {line_number}."
                )
            hq_id = mapping[original_name]
            if hq_id in matched:
                raise ValueError(
                    f"HQ id {hq_id} occurs twice in the partition mapping."
                )
            matched.add(hq_id)
            split_ids[groups[group]].append(hq_id)
    if len(matched) != 30000:
        raise ValueError(f"Partition covers {len(matched)} of 30,000 HQ samples.")

    return split_ids


def celeb_cache_path(cache_root: Path, hq_id: int) -> Path:
    return cache_root / f"{hq_id:05d}.png"


@functools.lru_cache(maxsize=64)
def _cached_component_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask


def compose_celeb_label(root: Path, hq_id: int) -> np.ndarray:

    label = np.zeros((512, 512), dtype=np.uint8)
    prefix = root / "CelebAMask-HQ-mask-anno" / str(hq_id // 2000) / f"{hq_id:05d}"
    for value, suffix in enumerate(CELEBAMASK_HQ_LABELS.suffixes, start=1):
        path = Path(f"{prefix}_{suffix}.png")
        if not path.is_file():
            continue
        mask = _cached_component_mask(str(path))
        if mask.shape != (512, 512):
            raise ValueError(f"Expected a 512x512 component mask at {path}.")
        label[mask > 0] = value
    return label


def load_celeb_samples(root: Path, split: str, cache_root: Path) -> list[ParsingSample]:
    image_root = root / "CelebA-HQ-img"
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    split_ids = celeb_split_ids(root)[split]
    samples: list[ParsingSample] = []
    missing_cache: list[Path] = []
    for hq_id in split_ids:
        image_path = image_root / f"{hq_id}.jpg"
        label_path = celeb_cache_path(cache_root, hq_id)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not label_path.is_file():
            missing_cache.append(label_path)
        samples.append(
            ParsingSample(
                sample_id=f"{split}.{hq_id}",
                image_path=image_path,
                label_path=label_path,
                hq_id=hq_id,
            )
        )
    if missing_cache:
        raise FileNotFoundError(
            f"{len(missing_cache)} merged CelebAMask-HQ masks are missing under "
            f"{cache_root}. Run `python -m dime_parsing.prepare_data "
            f"--config <CONFIG>` first; first missing file: {missing_cache[0]}"
        )
    return samples


def manifest_path(manifest_root: Path, dataset: str, split: str) -> Path:
    return manifest_root / f"{dataset}_{split}.npz"


def write_sample_manifest(
    path: Path,
    *,
    dataset: str,
    split: str,
    samples: Sequence[ParsingSample],
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, np.ndarray] = {
        "version": np.asarray([MANIFEST_VERSION], dtype=np.int64),
        "dataset": np.asarray([dataset]),
        "split": np.asarray([split]),
        "sample_ids": np.asarray([sample.sample_id for sample in samples]),
    }
    if dataset == "lapa":
        if any(sample.landmarks is None for sample in samples):
            raise ValueError("Every LaPa sample must have landmarks.")
        payload["landmarks"] = np.stack(
            [np.asarray(sample.landmarks, dtype=np.float32) for sample in samples]
        )
    elif dataset == "celebamask_hq":
        if any(sample.hq_id is None for sample in samples):
            raise ValueError("Every CelebAMask-HQ sample must have an HQ id.")
        payload["hq_ids"] = np.asarray(
            [sample.hq_id for sample in samples], dtype=np.int64
        )
    else:
        raise ValueError(f"Unsupported manifest dataset {dataset!r}.")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def load_sample_manifest(
    path: Path,
    *,
    dataset: str,
    split: str,
    root: Path,
    cache_root: Path | None = None,
) -> list[ParsingSample]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared manifest is missing: {path}. Run "
            f"`python -m dime_parsing.prepare_data --config <CONFIG>` first."
        )
    with np.load(path, allow_pickle=False) as manifest:
        version = int(manifest["version"][0])
        stored_dataset = str(manifest["dataset"][0])
        stored_split = str(manifest["split"][0])
        sample_ids = [str(value) for value in manifest["sample_ids"]]
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"{path} uses manifest version {version}; expected {MANIFEST_VERSION}. "
                "Re-run prepare_data."
            )
        if (stored_dataset, stored_split) != (dataset, split):
            raise ValueError(
                f"{path} describes {stored_dataset}/{stored_split}, not "
                f"{dataset}/{split}."
            )
        samples: list[ParsingSample] = []
        if dataset == "lapa":
            landmarks = np.asarray(manifest["landmarks"], dtype=np.float32)
            if landmarks.shape != (len(sample_ids), 106, 2):
                raise ValueError(
                    f"Invalid LaPa landmark array in {path}: {landmarks.shape}"
                )
            split_root = root / split
            for sample_id, points in zip(sample_ids, landmarks):
                prefix = f"{split}."
                if not sample_id.startswith(prefix):
                    raise ValueError(f"Invalid LaPa sample id {sample_id!r} in {path}.")
                stem = sample_id[len(prefix) :]
                samples.append(
                    ParsingSample(
                        sample_id=sample_id,
                        image_path=split_root / "images" / f"{stem}.jpg",
                        label_path=split_root / "labels" / f"{stem}.png",
                        landmarks=points,
                    )
                )
        elif dataset == "celebamask_hq":
            if cache_root is None:
                raise ValueError("CelebAMask-HQ manifests require cache_root.")
            hq_ids = np.asarray(manifest["hq_ids"], dtype=np.int64)
            if hq_ids.shape != (len(sample_ids),):
                raise ValueError(f"Invalid HQ-id array in {path}: {hq_ids.shape}")
            image_root = root / "CelebA-HQ-img"
            for sample_id, raw_hq_id in zip(sample_ids, hq_ids):
                hq_id = int(raw_hq_id)
                if sample_id != f"{split}.{hq_id}":
                    raise ValueError(
                        f"Sample id {sample_id!r} does not match HQ id {hq_id}."
                    )
                samples.append(
                    ParsingSample(
                        sample_id=sample_id,
                        image_path=image_root / f"{hq_id}.jpg",
                        label_path=celeb_cache_path(cache_root, hq_id),
                        hq_id=hq_id,
                    )
                )
        else:
            raise ValueError(f"Unsupported manifest dataset {dataset!r}.")
    return samples


class FaRLParsingTransform:

    def __init__(
        self,
        *,
        dataset: str,
        training: bool,
        options: dict[str, Any],
    ):
        self.dataset = dataset
        self.training = bool(training)
        self.canvas_size = int(options.get("canvas_size", 512))
        self.shift_sigma = float(options.get("shift_sigma", 0.01))
        self.rotation_sigma = float(options.get("rotation_sigma", 0.314))
        self.scale_sigma = float(options.get("scale_sigma", 0.1))
        self.warp_factor = float(options.get("warp_factor", 0.0))
        self.gray_probability = float(options.get("gray_probability", 0.1))
        self.blur_scale = float(options.get("blur_scale", 0.01))
        if self.canvas_size != 512:
            raise ValueError("FaRL parsing requires a 512x512 supervision canvas.")
        if self.dataset == "lapa" and not 0.0 <= self.warp_factor <= 1.0:
            raise ValueError("LaPa warp_factor must be in [0,1].")
        if self.dataset == "celebamask_hq" and self.warp_factor != 0.0:
            raise ValueError("FaRL CelebAMask-HQ uses warp_factor=0.")
        if any(
            value < 0.0
            for value in (
                self.shift_sigma,
                self.rotation_sigma,
                self.scale_sigma,
                self.blur_scale,
            )
        ):
            raise ValueError("Augmentation sigmas/scales must be non-negative.")
        if not 0.0 <= self.gray_probability <= 1.0:
            raise ValueError("gray_probability must be in [0,1].")

    def base_matrix(self, sample: ParsingSample) -> np.ndarray:
        if self.dataset == "celebamask_hq":
            return np.eye(3, dtype=np.float32)
        if sample.landmarks is None:
            raise ValueError(f"LaPa sample {sample.sample_id} has no landmarks.")
        five_points = sample.landmarks[[104, 105, 54, 84, 90]]
        return face_align_matrix(five_points, self.canvas_size)

    def prepare_source(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.dataset == "celebamask_hq":
            image = cv2.resize(
                image,
                (self.canvas_size, self.canvas_size),
                interpolation=cv2.INTER_LINEAR,
            )
            if label.shape != (self.canvas_size, self.canvas_size):
                label = cv2.resize(
                    label,
                    (self.canvas_size, self.canvas_size),
                    interpolation=cv2.INTER_NEAREST,
                )
        if image.shape[:2] != label.shape:
            raise ValueError(
                f"Image/label shape mismatch: {image.shape[:2]} versus {label.shape}."
            )
        return image, label

    def _photometric(self, image: np.ndarray) -> np.ndarray:
        if np.random.uniform() < self.gray_probability:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.repeat(gray[..., None], 3, axis=-1)
        image = np.clip(image, 0.0, 1.0)
        gamma = int(np.random.choice([-1, 0, 1]))
        if gamma == -1:
            image = np.sqrt(image)
        elif gamma == 1:
            image = np.square(image)
        kernel_ratio = float(np.random.uniform(0.0, self.blur_scale))
        kernel_size = int((image.shape[0] + image.shape[1]) / 2 * kernel_ratio)
        if kernel_size > 1:
            image = cv2.blur(np.clip(image, 0.0, 1.0), (kernel_size, kernel_size))
        return image

    def __call__(
        self,
        image: np.ndarray,
        label: np.ndarray,
        sample: ParsingSample,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image, label = self.prepare_source(image, label)
        matrix = self.base_matrix(sample)
        if self.training:
            matrix = random_update_matrix(
                matrix,
                size=self.canvas_size,
                shift_sigma=self.shift_sigma,
                rotation_sigma=self.rotation_sigma,
                scale_sigma=self.scale_sigma,
            )
        transform_map = forward_transform_map(
            matrix,
            canvas_size=self.canvas_size,
            warp_factor=self.warp_factor,
        )
        image = remap(
            image.astype(np.float32) / 255.0,
            transform_map,
            interpolation=cv2.INTER_LINEAR,
        )
        label = remap(
            label,
            transform_map,
            interpolation=cv2.INTER_NEAREST,
        )
        if self.training:
            image = self._photometric(image)
        return (
            np.ascontiguousarray(image, dtype=np.float32),
            np.ascontiguousarray(label, dtype=np.uint8),
            np.asarray(matrix, dtype=np.float64),
        )


class FaceParsingDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ParsingSample],
        *,
        dataset: str,
        split: str,
        augmentation: dict[str, Any],
    ):
        self.samples = list(samples)
        self.dataset = dataset
        self.split = split
        self.space = label_space(dataset)
        self.training = split == "train"
        self.transform = FaRLParsingTransform(
            dataset=dataset,
            training=self.training,
            options=augmentation,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = _read_rgb(sample.image_path)
        label = _read_label(sample.label_path)
        _validate_label(label, self.space, sample.label_path)
        original_label = label
        original_shape = tuple(int(value) for value in label.shape)
        image, label, matrix = self.transform(image, label, sample)
        result: dict[str, object] = {
            "image": torch.from_numpy(image).permute(2, 0, 1),
            "sample_id": sample.sample_id,
        }
        if self.training:
            result["label"] = torch.from_numpy(label.astype(np.int64))
        else:
            result.update(
                {
                    "label_original": original_label,
                    "original_shape": original_shape,
                    "transform": matrix,
                }
            )
        return result


def collate_evaluation(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "image": torch.stack([row["image"] for row in rows]),
        "sample_id": [str(row["sample_id"]) for row in rows],
        "label_original": [np.asarray(row["label_original"]) for row in rows],
        "original_shape": [tuple(row["original_shape"]) for row in rows],
        "transform": [np.asarray(row["transform"]) for row in rows],
    }


class DistributedEvalSampler(Sampler[int]):

    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterable[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size


def build_dataset(config: dict[str, Any], split: str) -> FaceParsingDataset:
    dataset_config = config["dataset"]
    name = str(dataset_config["name"]).strip().lower()
    root = resolve_path(dataset_config["root"], must_exist=True)
    assert root is not None
    space = label_space(name)
    if int(dataset_config["num_classes"]) != space.num_classes:
        raise ValueError(
            f"dataset.num_classes={dataset_config['num_classes']} does not match "
            f"{name}'s label space ({space.num_classes})."
        )
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test.")
    use_manifests = bool(dataset_config.get("use_manifests", False))
    cache_root = None
    if name == "celebamask_hq":
        cache_root = resolve_path(dataset_config["mask_cache_dir"], must_exist=True)
        assert cache_root is not None
    if use_manifests:
        manifest_root = resolve_path(dataset_config["manifest_dir"], must_exist=True)
        assert manifest_root is not None
        samples = load_sample_manifest(
            manifest_path(manifest_root, name, split),
            dataset=name,
            split=split,
            root=root,
            cache_root=cache_root,
        )
    elif name == "lapa":
        samples = load_lapa_samples(root, split)
    elif name == "celebamask_hq":
        assert cache_root is not None
        samples = load_celeb_samples(root, split, cache_root)
    else:
        raise ValueError(f"Unsupported dataset {name!r}.")
    expected_counts = dataset_config.get("expected_counts", EXPECTED_COUNTS[name])
    expected = int(expected_counts[split])
    if len(samples) != expected:
        raise ValueError(
            f"{name}/{split} contains {len(samples)} samples, but the configured "
            f"protocol requires {expected}. Run prepare_data and inspect its audit."
        )
    return FaceParsingDataset(
        samples,
        dataset=name,
        split=split,
        augmentation=config["augmentation"],
    )


def build_loader(
    config: dict[str, Any],
    dataset: FaceParsingDataset,
    *,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, Sampler[int] | None]:
    training = dataset.training
    if training and world_size > 1:
        sampler: Sampler[int] | None = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(config["experiment"]["seed"]),
            drop_last=False,
        )
    elif not training and world_size > 1:
        sampler = DistributedEvalSampler(dataset, rank, world_size)
    else:
        sampler = None
    loader_config = config["loader"]
    workers = int(loader_config.get("workers_per_gpu", 4))
    batch_size = int(
        loader_config["batch_size_per_gpu" if training else "eval_batch_size_per_gpu"]
    )
    if workers < 0 or batch_size <= 0:
        raise ValueError("Loader workers must be non-negative and batch size positive.")
    generator = torch.Generator()
    generator.manual_seed(int(config["experiment"]["seed"]) + rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(loader_config.get("pin_memory", True)),
        persistent_workers=bool(
            loader_config.get("persistent_workers", True) and workers > 0
        ),
        prefetch_factor=(
            int(loader_config.get("prefetch_factor", 2)) if workers > 0 else None
        ),
        drop_last=False,
        collate_fn=None if training else collate_evaluation,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, sampler
