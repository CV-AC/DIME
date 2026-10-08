from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, Sampler

from .data import CANVAS_SIZE, FaRLAugment


LAPA_SPLITS = ("train", "val", "test")
LP_DIRECTORIES = (
    "AFW",
    "AFW_Flip",
    "HELEN",
    "HELEN_Flip",
    "IBUG",
    "IBUG_Flip",
    "LFPW",
    "LFPW_Flip",
)
AUXILIARY_INDEX_VERSION = 2


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _update_manifest(
    digest: Any,
    relative_path: str,
    points: np.ndarray,
    *file_sizes: int,
) -> None:
    digest.update(relative_path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(points, dtype="<f4").tobytes(order="C"))
    for size in file_sizes:
        digest.update(int(size).to_bytes(8, byteorder="little", signed=False))


def _parse_lapa_landmarks(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines or int(lines[0]) != 106:
        raise ValueError(f"{path}: expected a 106-point landmark header.")
    points = np.asarray(
        [[float(value) for value in line.split()] for line in lines[1:]],
        dtype=np.float32,
    )
    if points.shape != (106, 2) or not np.isfinite(points).all():
        raise ValueError(f"{path}: expected 106 finite x/y landmark pairs.")
    return points


def build_lapa_index(
    root: str | Path,
    output: str | Path,
    *,
    split: str = "train",
    exclude_eval_stem_overlap: bool = True,
    verify_decode: bool = True,
) -> dict[str, Any]:
    root, output = Path(root).resolve(), Path(output).resolve()
    if split not in LAPA_SPLITS:
        raise ValueError(f"Unknown LaPa split {split!r}.")
    split_root = root / split
    images = split_root / "images"
    labels = split_root / "labels"
    landmarks = split_root / "landmarks"
    for path in (images, labels, landmarks):
        if not path.is_dir():
            raise FileNotFoundError(path)

    stems = {path.stem for path in images.glob("*.jpg")}
    label_stems = {path.stem for path in labels.glob("*.png")}
    landmark_stems = {path.stem for path in landmarks.glob("*.txt")}
    if stems != label_stems or stems != landmark_stems:
        raise RuntimeError(
            "LaPa image, parsing, and landmark filenames are not one-to-one."
        )

    excluded: set[str] = set()
    if split == "train" and exclude_eval_stem_overlap:
        for name in ("val", "test"):
            excluded.update(
                path.stem for path in (root / name / "images").glob("*.jpg")
            )
    selected = sorted(stems - excluded)
    parsed_points = []
    manifest = hashlib.sha256()
    for index, stem in enumerate(selected, start=1):
        image_path = images / f"{stem}.jpg"
        parsing_path = labels / f"{stem}.png"
        landmark_path = landmarks / f"{stem}.txt"
        sample_points = _parse_lapa_landmarks(landmark_path)
        parsed_points.append(sample_points)
        _update_manifest(
            manifest,
            f"{split}/images/{stem}.jpg",
            sample_points,
            image_path.stat().st_size,
            parsing_path.stat().st_size,
            landmark_path.stat().st_size,
        )
        if verify_decode:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            parsing = cv2.imread(str(parsing_path), cv2.IMREAD_GRAYSCALE)
            if image is None or parsing is None:
                raise ValueError(f"Could not decode LaPa sample {stem}.")
            if image.shape[:2] != parsing.shape:
                raise ValueError(f"LaPa image/mask size mismatch for {stem}.")
            if int(parsing.max()) > 10:
                raise ValueError(f"LaPa mask {stem} contains a class above 10.")
        if index % 5_000 == 0 or index == len(selected):
            print(f"Index LaPa: {index}/{len(selected)}", flush=True)
    points = np.stack(parsed_points)
    relative_images = np.asarray([f"{split}/images/{stem}.jpg" for stem in selected])
    relative_parsing = np.asarray([f"{split}/labels/{stem}.png" for stem in selected])
    metadata = {
        "index_format_version": AUXILIARY_INDEX_VERSION,
        "dataset": "lapa",
        "source_root": str(root),
        "split": split,
        "samples": len(selected),
        "raw_samples": len(stems),
        "excluded_eval_stem_overlap": len(stems) - len(selected),
        "decoded_all_selected_samples": bool(verify_decode),
        "decoded_all_images": bool(verify_decode),
        "num_landmarks": 106,
        "num_parsing_classes": 11,
        "manifest_sha256": manifest.hexdigest(),
    }
    _atomic_savez(
        output,
        relative_images=relative_images,
        relative_parsing=relative_parsing,
        landmarks=points,
        metadata=np.asarray(json.dumps(metadata)),
    )
    return metadata


def _load_lp_points(path: Path) -> np.ndarray:
    matrix = loadmat(path)
    if "pt2d" not in matrix:
        raise ValueError(f"{path}: missing pt2d.")
    points = np.asarray(matrix["pt2d"], dtype=np.float32)
    if points.shape == (2, 68):
        points = points.T
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError(f"{path}: pt2d must contain 68 finite x/y pairs.")
    return points


def build_300w_lp_index(
    root: str | Path,
    output: str | Path,
    *,
    directories: Sequence[str] = LP_DIRECTORIES,
    verify_decode: bool = True,
) -> dict[str, Any]:
    root, output = Path(root).resolve(), Path(output).resolve()
    relative_images: list[str] = []
    points: list[np.ndarray] = []
    counts: dict[str, int] = {}
    manifest = hashlib.sha256()
    for name in directories:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        images = sorted(directory.glob("*.jpg"))
        counts[name] = len(images)
        for index, image_path in enumerate(images, start=1):
            if verify_decode and cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
                raise ValueError(f"Could not decode {image_path}.")
            annotation = image_path.with_suffix(".mat")
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            relative_image = image_path.relative_to(root).as_posix()
            sample_points = _load_lp_points(annotation)
            relative_images.append(relative_image)
            points.append(sample_points)
            _update_manifest(
                manifest,
                relative_image,
                sample_points,
                image_path.stat().st_size,
                annotation.stat().st_size,
            )
            if index % 5_000 == 0 or index == len(images):
                print(
                    f"Index 300W-LP/{name}: {index}/{len(images)}",
                    flush=True,
                )
    if not points:
        raise RuntimeError("The 300W-LP index would be empty.")
    metadata = {
        "index_format_version": AUXILIARY_INDEX_VERSION,
        "dataset": "300w_lp",
        "source_root": str(root),
        "samples": len(points),
        "directories": counts,
        "num_landmarks": 68,
        "decoded_all_images": bool(verify_decode),
        "manifest_sha256": manifest.hexdigest(),
    }
    _atomic_savez(
        output,
        relative_images=np.asarray(relative_images),
        landmarks=np.stack(points),
        metadata=np.asarray(json.dumps(metadata)),
    )
    return metadata


def _landmark_box(points: np.ndarray, expansion: float) -> np.ndarray:
    if expansion <= 0.0:
        raise ValueError("landmark_box_expansion must be positive.")
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    center = (minimum + maximum) * 0.5
    side = float(np.max(maximum - minimum)) * expansion
    if not math.isfinite(side) or side < 2.0:
        raise ValueError("Cannot construct a valid auxiliary landmark box.")
    half = side * 0.5
    return np.asarray(
        [center[0] - half, center[1] - half, center[0] + half, center[1] + half],
        dtype=np.float32,
    )


def _validate_relative_paths(paths: np.ndarray, path: Path) -> None:
    if paths.ndim != 1:
        raise ValueError(f"{path} relative_images must be one-dimensional.")
    for value in paths.astype(str):
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{path} contains an unsafe relative path: {value}")


def inspect_auxiliary_index(
    path: str | Path,
    *,
    expected_task: str,
    expected_root: str | Path | None = None,
    expected_num_landmarks: int | None = None,
    require_full_decode: bool = False,
    validate_arrays: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as index:
            required = {"relative_images", "landmarks", "metadata"}
            if not required.issubset(index.files):
                raise ValueError(
                    f"missing arrays {sorted(required - set(index.files))}"
                )
            metadata = json.loads(str(index["metadata"].item()))
            if validate_arrays:
                relative_images = index["relative_images"]
                landmarks = index["landmarks"]
                relative_parsing = (
                    index["relative_parsing"]
                    if "relative_parsing" in index.files
                    else None
                )
    except Exception as error:
        raise ValueError(f"Invalid auxiliary index {path}: {error}") from error
    if int(metadata.get("index_format_version", 0)) != AUXILIARY_INDEX_VERSION:
        raise ValueError(
            f"{path} uses an obsolete auxiliary index format. "
            "Run python -m dime_landmark.prepare_auxiliary_data again."
        )
    if metadata.get("dataset") != expected_task:
        raise ValueError(
            f"{path} belongs to {metadata.get('dataset')}, not {expected_task}."
        )
    if int(metadata.get("samples", 0)) <= 0:
        raise ValueError(f"{path} contains no indexed samples.")
    metadata_landmarks = int(metadata.get("num_landmarks", 0))
    if expected_num_landmarks is not None and metadata_landmarks != int(
        expected_num_landmarks
    ):
        raise ValueError(
            f"{path} contains {metadata_landmarks} landmarks, "
            f"expected {expected_num_landmarks}."
        )
    if validate_arrays:
        samples = int(metadata["samples"])
        _validate_relative_paths(relative_images, path)
        if len(relative_images) != samples or landmarks.shape[0] != samples:
            raise ValueError(f"{path} metadata and array sample counts disagree.")
        if (
            landmarks.ndim != 3
            or landmarks.shape[-1] != 2
            or not np.isfinite(landmarks).all()
        ):
            raise ValueError(f"{path} contains invalid landmark coordinates.")
        if landmarks.shape[1] != metadata_landmarks:
            raise ValueError(f"{path} landmark shape disagrees with its metadata.")
        if expected_task == "lapa":
            if relative_parsing is None or len(relative_parsing) != samples:
                raise ValueError(f"{path} has invalid LaPa parsing paths.")
            _validate_relative_paths(relative_parsing, path)
    manifest = str(metadata.get("manifest_sha256", ""))
    if len(manifest) != 64 or any(
        character not in "0123456789abcdef" for character in manifest
    ):
        raise ValueError(f"{path} has no valid source manifest.")
    if expected_root is not None:
        recorded_root = metadata.get("source_root")
        if (
            not recorded_root
            or Path(recorded_root).resolve() != Path(expected_root).resolve()
        ):
            raise ValueError(
                f"{path} was prepared for a different dataset root. "
                "Rebuild it from the root configured in base.yaml."
            )
    if require_full_decode and not bool(metadata.get("decoded_all_images", False)):
        raise ValueError(
            f"{path} was prepared without decoding every image. "
            "Rebuild it with verify_decode_during_prepare: true."
        )
    return metadata


class AuxiliaryLandmarkDataset(Dataset):

    def __init__(
        self,
        *,
        task: str,
        root: str | Path,
        index_file: str | Path,
        augmentation: dict[str, Any] | None,
        training: bool = True,
        landmark_box_expansion: float = 1.25,
        parsing_enabled: bool = False,
        parsing_ignore_artificial_occlusion: bool = True,
    ):
        if task not in {"lapa", "300w_lp"}:
            raise ValueError(f"Unsupported auxiliary task {task!r}.")
        self.task = task
        self.root = Path(root).resolve()
        self.index_file = Path(index_file).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if not self.index_file.is_file():
            raise FileNotFoundError(
                f"{self.index_file}. Build auxiliary indexes before training."
            )
        expected_points = 106 if task == "lapa" else 68
        inspect_auxiliary_index(
            self.index_file,
            expected_task=task,
            expected_root=self.root,
            expected_num_landmarks=expected_points,
            validate_arrays=False,
        )
        with np.load(self.index_file, allow_pickle=False) as index:
            self.relative_images = index["relative_images"].astype(str)
            self.landmarks = index["landmarks"].astype(np.float32)
            self.relative_parsing = (
                index["relative_parsing"].astype(str)
                if "relative_parsing" in index
                else None
            )
            self.metadata = json.loads(str(index["metadata"].item()))
        if self.metadata.get("dataset") != task:
            raise ValueError(
                f"{self.index_file} belongs to {self.metadata.get('dataset')}, "
                f"not {task}."
            )
        if self.landmarks.shape != (len(self.relative_images), expected_points, 2):
            raise ValueError(f"{self.index_file} has an invalid landmark shape.")
        if int(self.metadata.get("samples", -1)) != len(self.relative_images):
            raise ValueError(
                f"{self.index_file} metadata sample count is inconsistent."
            )
        if not np.isfinite(self.landmarks).all():
            raise ValueError(f"{self.index_file} contains non-finite landmarks.")
        _validate_relative_paths(self.relative_images, self.index_file)
        if self.relative_parsing is not None:
            if len(self.relative_parsing) != len(self.relative_images):
                raise ValueError(f"{self.index_file} has invalid parsing paths.")
            _validate_relative_paths(self.relative_parsing, self.index_file)
        self.parsing_enabled = bool(parsing_enabled)
        self.parsing_ignore_artificial_occlusion = bool(
            parsing_ignore_artificial_occlusion
        )
        if self.parsing_enabled and self.relative_parsing is None:
            raise ValueError("LaPa parsing is enabled but the index has no masks.")
        self.box_expansion = float(landmark_box_expansion)
        if not math.isfinite(self.box_expansion) or self.box_expansion <= 0.0:
            raise ValueError("landmark_box_expansion must be finite and positive.")
        self.training = bool(training)
        self.transform = FaRLAugment(training=self.training, options=augmentation)

    def __len__(self) -> int:
        return len(self.relative_images)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.root / self.relative_images[index]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not decode {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_points = self.landmarks[index]
        box = _landmark_box(original_points, self.box_expansion)
        transformed = self.transform(
            image_rgb,
            original_points,
            box,
            return_valid_mask=(
                self.parsing_enabled and self.parsing_ignore_artificial_occlusion
            ),
        )
        if len(transformed) == 4:
            image, points, matrix, parsing_valid_mask = transformed
        else:
            image, points, matrix = transformed
            parsing_valid_mask = None
        sample: dict[str, object] = {
            "image": torch.from_numpy(image).permute(2, 0, 1),
            "landmarks_canvas": torch.from_numpy(points.astype(np.float32)),
            "landmarks_original": torch.from_numpy(original_points.copy()),
            "transform": torch.from_numpy(matrix),
            "sample_id": f"{self.task}:{self.relative_images[index]}",
            "task": self.task,
        }
        if self.parsing_enabled:
            assert self.relative_parsing is not None
            parsing_path = self.root / self.relative_parsing[index]
            parsing = cv2.imread(str(parsing_path), cv2.IMREAD_GRAYSCALE)
            if parsing is None:
                raise FileNotFoundError(f"Could not decode {parsing_path}")
            if parsing.shape != image_rgb.shape[:2]:
                raise ValueError(
                    f"Image/mask size mismatch for {self.relative_images[index]}"
                )
            parsing = cv2.warpAffine(
                parsing,
                matrix[:2],
                (CANVAS_SIZE, CANVAS_SIZE),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            if int(parsing.max()) >= int(self.metadata.get("num_parsing_classes", 11)):
                raise ValueError(f"Invalid LaPa class in {parsing_path}")
            sample["parsing_mask"] = torch.from_numpy(
                np.ascontiguousarray(parsing, dtype=np.int64)
            )
            if parsing_valid_mask is not None:
                sample["parsing_valid_mask"] = torch.from_numpy(
                    np.ascontiguousarray(parsing_valid_mask)
                )
        return sample


def epoch_fractions(
    sampling: dict[str, Any], enabled_tasks: Sequence[str]
) -> dict[str, float]:

    values = sampling.get("epoch_fraction")
    if not isinstance(values, dict):
        raise ValueError(
            "auxiliary_training.sampling.epoch_fraction must be a mapping. "
            "Fractions describe coverage of each dataset, not batch ratios."
        )
    fractions = {
        task: float(values.get(task, 0.0)) for task in ("wflw", *enabled_tasks)
    }
    if fractions["wflw"] != 1.0:
        raise ValueError(
            "WFLW epoch_fraction must be exactly 1.0 so every epoch covers "
            "the complete WFLW training set."
        )
    if any(not 0.0 <= value <= 1.0 for value in fractions.values()):
        raise ValueError("Every epoch_fraction must be within [0, 1].")
    if not any(fractions[task] > 0.0 for task in enabled_tasks):
        raise ValueError(
            "At least one enabled auxiliary dataset needs a positive fraction."
        )
    return fractions


class DistributedEpochFractionSampler(Sampler[int]):

    def __init__(
        self,
        dataset_size: int,
        *,
        fraction: float,
        num_replicas: int,
        rank: int,
        seed: int,
    ):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be within [0, 1].")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid distributed sampler rank/world size.")
        self.dataset_size = int(dataset_size)
        self.fraction = float(fraction)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.selected_samples = (
            0
            if self.fraction == 0.0
            else max(1, int(math.floor(self.dataset_size * self.fraction + 0.5)))
        )
        self.samples_per_rank = (
            math.ceil(self.selected_samples / self.num_replicas)
            if self.selected_samples
            else 0
        )
        self.total_size = self.samples_per_rank * self.num_replicas
        self.padding_samples = self.total_size - self.selected_samples

    def __len__(self) -> int:
        return self.samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if self.selected_samples == 0:
            return iter(())
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        selected = torch.randperm(self.dataset_size, generator=generator)[
            : self.selected_samples
        ].tolist()
        if self.padding_samples:
            repeats = math.ceil(self.padding_samples / len(selected))
            selected.extend((selected * repeats)[: self.padding_samples])
        local = selected[self.rank : self.total_size : self.num_replicas]
        if len(local) != self.samples_per_rank:
            raise RuntimeError("Distributed fraction sampler produced an uneven shard.")
        return iter(local)


def auxiliary_task_schedule(
    counts: dict[str, int], *, seed: int, epoch: int
) -> list[str]:
    schedule = [task for task, count in counts.items() for _ in range(int(count))]
    random.Random(int(seed) + int(epoch) * 1_000_003).shuffle(schedule)
    return schedule
