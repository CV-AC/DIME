from __future__ import annotations


import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

cv2.setNumThreads(0)

NUM_LANDMARKS = 98
CANVAS_SIZE = 512
SUBSETS = ("largepose", "expression", "illumination", "makeup", "occlusion", "blur")
EXPECTED_COUNTS = {
    "train": 7500,
    "test": 2500,
    "largepose": 326,
    "expression": 314,
    "illumination": 698,
    "makeup": 206,
    "occlusion": 736,
    "blur": 773,
}


@dataclass(frozen=True)
class WFLWSample:
    image_path: Path
    relative_path: str
    landmarks: np.ndarray
    box_xyxy: np.ndarray
    attributes: tuple[int, ...]
    subsets: tuple[str, ...] = ()
    sample_id: str = ""


def annotation_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    annotations = root / "WFLW_annotations"
    main = annotations / "list_98pt_rect_attr_train_test"
    subset = annotations / "list_98pt_test"
    return {
        "train": main / "list_98pt_rect_attr_train.txt",
        "test": main / "list_98pt_rect_attr_test.txt",
        **{name: subset / f"list_98pt_test_{name}.txt" for name in SUBSETS},
    }


def _parse_full_line(line: str, image_root: Path, row_index: int) -> WFLWSample:
    fields = line.strip().split()
    expected = NUM_LANDMARKS * 2 + 4 + 6 + 1
    if len(fields) != expected:
        raise ValueError(f"Expected {expected} WFLW fields, got {len(fields)}")
    landmarks = np.asarray(fields[:196], dtype=np.float32).reshape(NUM_LANDMARKS, 2)
    box = np.asarray(fields[196:200], dtype=np.float32)
    attributes = tuple(int(value) for value in fields[200:206])
    relative_path = fields[-1].replace("\\", "/")
    return WFLWSample(
        image_path=image_root / Path(relative_path),
        relative_path=relative_path,
        landmarks=landmarks,
        box_xyxy=box,
        attributes=attributes,
        sample_id=f"row-{row_index:04d}:{relative_path}",
    )


def _subset_annotations(path: Path) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 197:
                raise ValueError(
                    f"{path}:{line_number}: expected 197 fields, got {len(fields)}"
                )
            relative_path = fields[-1].replace("\\", "/")
            rows.append(
                (
                    relative_path,
                    np.asarray(fields[:196], dtype=np.float32).reshape(98, 2),
                )
            )
    return rows


def load_wflw_records(root: str | Path) -> tuple[list[WFLWSample], list[WFLWSample]]:
    root = Path(root).expanduser().resolve()
    paths = annotation_paths(root)
    image_root = root / "WFLW_images"
    records: dict[str, list[WFLWSample]] = {}
    for split in ("train", "test"):
        rows: list[WFLWSample] = []
        with paths[split].open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(_parse_full_line(line, image_root, len(rows)))
                except Exception as exc:
                    raise ValueError(f"{paths[split]}:{line_number}: {exc}") from exc
        records[split] = rows

    subset_members: dict[str, set[str]] = {}
    test_by_path: dict[str, list[WFLWSample]] = defaultdict(list)
    for sample in records["test"]:
        test_by_path[sample.relative_path].append(sample)
    for subset in SUBSETS:
        subset_rows = _subset_annotations(paths[subset])
        unknown = {path for path, _ in subset_rows}.difference(test_by_path)
        if unknown:
            raise ValueError(
                f"{subset} contains {len(unknown)} samples outside official test."
            )
        members: set[str] = set()
        for relative_path, points in subset_rows:
            candidates = [
                sample
                for sample in test_by_path[relative_path]
                if sample.sample_id not in members
                and np.allclose(points, sample.landmarks, atol=1e-5)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"{subset}: expected one unused landmark match for {relative_path}, "
                    f"found {len(candidates)}"
                )
            members.add(candidates[0].sample_id)
        subset_members[subset] = members

    test = [
        WFLWSample(
            image_path=sample.image_path,
            relative_path=sample.relative_path,
            landmarks=sample.landmarks,
            box_xyxy=sample.box_xyxy,
            attributes=sample.attributes,
            subsets=tuple(
                name for name in SUBSETS if sample.sample_id in subset_members[name]
            ),
            sample_id=sample.sample_id,
        )
        for sample in records["test"]
    ]
    return records["train"], test


def development_split(
    samples: Sequence[WFLWSample], validation_size: int = 750
) -> tuple[list[WFLWSample], list[WFLWSample]]:
    if validation_size <= 0 or validation_size >= len(samples):
        raise ValueError("validation_size must be between zero and the dataset size.")

    groups: dict[str, list[WFLWSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.relative_path].append(sample)
    ranked_groups = sorted(
        groups.items(),
        key=lambda item: (
            hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
            item[0],
        ),
    )
    validation_ids: set[str] = set()
    remaining = validation_size
    for _, group in ranked_groups:
        if len(group) <= remaining:
            validation_ids.update(sample.sample_id for sample in group)
            remaining -= len(group)
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError(
            f"Could not construct a source-image-disjoint validation set of "
            f"exactly {validation_size} samples."
        )
    train = [sample for sample in samples if sample.sample_id not in validation_ids]
    validation = [sample for sample in samples if sample.sample_id in validation_ids]
    return train, validation


def _crop_matrix(box_xyxy: np.ndarray, size: int = CANVAS_SIZE) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    width, height = x2 - x1, y2 - y1
    side = max(width, height)
    if not math.isfinite(side) or side <= 0:
        raise ValueError(f"Invalid WFLW bounding box: {box_xyxy.tolist()}")
    center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    square = np.asarray(
        [
            center_x - side * 0.5,
            center_y - side * 0.5,
            center_x + side * 0.5,
            center_y + side * 0.5,
        ],
        dtype=np.float32,
    )
    x1, y1, x2, y2 = (float(value) for value in square)
    ax = size / (x2 - x1)
    ay = size / (y2 - y1)
    return np.asarray(
        [[ax, 0.0, -x1 * ax - 0.5], [0.0, ay, -y1 * ay - 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _rotate_scale_matrix(
    angle: float,
    scale: float,
    shift_xy: tuple[float, float],
    size: int = CANVAS_SIZE,
) -> np.ndarray:
    cosv, sinv = math.cos(angle), math.sin(angle)
    center = (size - 1) / 2.0
    acos, asin = scale * cosv, scale * sinv
    return np.asarray(
        [
            [acos, -asin, center - acos * center + asin * center + shift_xy[0]],
            [asin, acos, center - asin * center - acos * center + shift_xy[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    return (homogeneous @ matrix.T)[:, :2]


class FaRLAugment:

    def __init__(
        self,
        training: bool,
        size: int = CANVAS_SIZE,
        options: dict[str, Any] | None = None,
    ):
        self.training = training
        self.size = size
        self.options = dict(options or {})
        legacy_robust = bool(self.options.get("enabled", False))
        self.preset = (
            str(
                self.options.get(
                    "preset", "custom_robust" if legacy_robust else "farl_wflw"
                )
            )
            .strip()
            .lower()
        )
        if self.preset not in {"farl_wflw", "custom_robust"}:
            raise ValueError("augmentation.preset must be farl_wflw or custom_robust.")
        self.custom_robust = self.preset == "custom_robust"
        if self.preset == "farl_wflw":
            shift_sigma = float(self.options.get("shift_sigma", 0.05))
            rot_sigma = float(self.options.get("rot_sigma", 0.174))
            scale_mu = float(self.options.get("scale_mu", 0.8))
            scale_sigma = float(self.options.get("scale_sigma", 0.1))
            warp_factor = float(self.options.get("warp_factor", 0.0))
            noise_probability = float(self.options.get("noise_fusion_probability", 0.5))
            if shift_sigma < 0.0 or rot_sigma < 0.0 or scale_sigma < 0.0:
                raise ValueError("FaRL augmentation sigmas must be non-negative.")
            if scale_mu - scale_sigma <= 0.0:
                raise ValueError("FaRL scale_mu - scale_sigma must be positive.")
            if warp_factor != 0.0:
                raise ValueError(
                    "Only FaRL's WFLW warp_factor=0.0 protocol is supported."
                )
            if not 0.0 <= noise_probability <= 1.0:
                raise ValueError(
                    "augmentation.noise_fusion_probability must be in [0,1]."
                )
        if self.custom_robust:
            for name in (
                "hard_probability",
                "occlusion_probability",
                "noise_probability",
                "jpeg_probability",
                "low_resolution_probability",
            ):
                value = float(self.options.get(name, 0.0))
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"augmentation.{name} must be in [0,1].")
            for prefix in ("mild", "hard"):
                if float(self.options.get(f"{prefix}_shift", 0.0)) < 0.0:
                    raise ValueError(
                        f"augmentation.{prefix}_shift must be non-negative."
                    )
                if float(self.options.get(f"{prefix}_rotation_degrees", 0.0)) < 0.0:
                    raise ValueError(
                        f"augmentation.{prefix}_rotation_degrees must be non-negative."
                    )
                scale = self.options.get(f"{prefix}_scale", [0.7, 0.9])
                if (
                    len(scale) != 2
                    or float(scale[0]) <= 0.0
                    or float(scale[1]) < float(scale[0])
                ):
                    raise ValueError(
                        f"augmentation.{prefix}_scale must be increasing and positive."
                    )
            jpeg = self.options.get("jpeg_quality", [30, 90])
            low_resolution = self.options.get("low_resolution_side", [32, 160])
            if len(jpeg) != 2 or not 1 <= int(jpeg[0]) <= int(jpeg[1]) <= 100:
                raise ValueError("augmentation.jpeg_quality must be within [1,100].")
            if (
                len(low_resolution) != 2
                or not 2
                <= int(low_resolution[0])
                <= int(low_resolution[1])
                <= self.size
            ):
                raise ValueError(
                    "augmentation.low_resolution_side must be within the canvas."
                )

    def geometry_matrix(self, box_xyxy: np.ndarray) -> np.ndarray:
        matrix = _crop_matrix(box_xyxy, self.size)
        if self.training:
            robust = self.custom_robust
            hard = robust and (
                np.random.uniform() < float(self.options.get("hard_probability", 0.3))
            )
            if robust:
                prefix = "hard" if hard else "mild"
                shift_limit = float(self.options.get(f"{prefix}_shift", 0.05))
                angle_limit = math.radians(
                    float(self.options.get(f"{prefix}_rotation_degrees", 10.0))
                )
                scale_range = self.options.get(f"{prefix}_scale", [0.7, 0.9])
                if len(scale_range) != 2 or float(scale_range[0]) <= 0:
                    raise ValueError(
                        f"{prefix}_scale must contain two positive values."
                    )
            else:
                shift_limit = float(self.options.get("shift_sigma", 0.05))
                angle_limit = float(self.options.get("rot_sigma", 0.174))
                scale_mu = float(self.options.get("scale_mu", 0.8))
                scale_sigma = float(self.options.get("scale_sigma", 0.1))
                scale_range = [
                    scale_mu - scale_sigma,
                    scale_mu + scale_sigma,
                ]
            shift = tuple(
                float(value) * self.size
                for value in np.random.uniform(-shift_limit, shift_limit, size=2)
            )
            angle = float(np.random.uniform(-angle_limit, angle_limit))
            scale = float(
                np.random.uniform(float(scale_range[0]), float(scale_range[1]))
            )
        else:
            shift, angle, scale = (0.0, 0.0), 0.0, 0.8
        return _rotate_scale_matrix(angle, scale, shift, self.size) @ matrix

    @staticmethod
    def _random_occlusion(
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width, _ = image.shape
        relative_h = random.random() * 0.6 + 0.2
        relative_w = relative_h - 0.2 + 0.4 * random.random()
        center_x = int((height - 1) * random.random())
        center_y = int((width - 1) * random.random())
        delta_h = int(height / 2 * relative_h)
        delta_w = int(width / 2 * relative_w)
        x0 = max(0, center_x - delta_w // 2)
        y0 = max(0, center_y - delta_h // 2)
        x1 = min(width - 1, center_x + delta_w // 2)
        y1 = min(height - 1, center_y + delta_h // 2)
        output = np.array(image)
        output[y0 : y1 + 1, x0 : x1 + 1, :] = 0
        valid = np.ones((height, width), dtype=np.bool_)
        valid[y0 : y1 + 1, x0 : x1 + 1] = False
        return output, valid

    @staticmethod
    def _noise_fusion(image: np.ndarray) -> np.ndarray:
        noise = np.random.rand(*image.shape)
        alpha = 0.5 * random.random()
        return ((1.0 - alpha) * image + alpha * noise).astype(image.dtype)

    @staticmethod
    def _random_gray(image: np.ndarray) -> np.ndarray:
        if np.random.uniform() < 0.1:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return np.repeat(gray[..., None], 3, axis=-1)
        return image

    @staticmethod
    def _random_gamma(image: np.ndarray) -> np.ndarray:
        image = np.clip(image, 0.0, 1.0)
        gamma = int(np.random.choice([-1, 0, 1]))
        if gamma == -1:
            return np.sqrt(image)
        if gamma == 1:
            return np.square(image)
        return image

    @staticmethod
    def _random_blur(image: np.ndarray) -> np.ndarray:
        kernel_ratio = float(np.random.uniform(0.0, 0.01))
        kernel_size = int((image.shape[0] + image.shape[1]) / 2 * kernel_ratio)
        if kernel_size > 1:
            return cv2.blur(np.clip(image, 0.0, 1.0), (kernel_size, kernel_size))
        return image

    @staticmethod
    def _jpeg_compression(
        image: np.ndarray, quality_range: Sequence[int]
    ) -> np.ndarray:
        quality = int(
            np.random.randint(int(quality_range[0]), int(quality_range[1]) + 1)
        )
        bgr = cv2.cvtColor(
            np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        success, encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not success:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    @staticmethod
    def _low_resolution(image: np.ndarray, side_range: Sequence[int]) -> np.ndarray:
        side = int(np.random.randint(int(side_range[0]), int(side_range[1]) + 1))
        small = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
        return cv2.resize(
            small,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    def __call__(
        self,
        image_rgb: np.ndarray,
        landmarks: np.ndarray,
        box_xyxy: np.ndarray,
        *,
        return_valid_mask: bool = False,
    ) -> tuple[Any, ...]:
        matrix = self.geometry_matrix(box_xyxy)
        image = cv2.warpAffine(
            image_rgb.astype(np.float32) / 255.0,
            matrix[:2],
            (self.size, self.size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        points = transform_points(landmarks, matrix)
        valid_mask = np.ones((self.size, self.size), dtype=np.bool_)
        if self.training:
            robust = self.custom_robust
            occlusion_probability = float(
                self.options.get("occlusion_probability", 1.0) if robust else 1.0
            )
            if np.random.uniform() <= occlusion_probability:
                image, valid_mask = self._random_occlusion(image)
            if np.random.uniform() <= float(
                self.options.get("noise_probability", 0.5)
                if robust
                else self.options.get("noise_fusion_probability", 0.5)
            ):
                image = self._noise_fusion(image)
            image = self._random_gray(image)
            image = self._random_gamma(image)
            image = self._random_blur(image)
            if self.custom_robust and np.random.uniform() <= float(
                self.options.get("jpeg_probability", 0.2)
            ):
                image = self._jpeg_compression(
                    image, self.options.get("jpeg_quality", [30, 90])
                )
            if self.custom_robust and np.random.uniform() <= float(
                self.options.get("low_resolution_probability", 0.2)
            ):
                image = self._low_resolution(
                    image, self.options.get("low_resolution_side", [32, 160])
                )
        output: tuple[Any, ...] = (
            np.ascontiguousarray(image, dtype=np.float32),
            points,
            matrix,
        )
        if return_valid_mask:
            output = (*output, np.ascontiguousarray(valid_mask))
        return output


class WFLWDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[WFLWSample],
        *,
        training: bool,
        augmentation: dict[str, Any] | None = None,
    ):
        self.samples = list(samples)
        self.transform = FaRLAugment(training=training, options=augmentation)
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not decode {sample.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image, points, matrix = self.transform(
            image_rgb, sample.landmarks, sample.box_xyxy
        )
        flags = np.asarray([name in sample.subsets for name in SUBSETS], dtype=np.bool_)
        result: dict[str, object] = {
            "image": torch.from_numpy(image).permute(2, 0, 1),
            "landmarks_canvas": torch.from_numpy(points.astype(np.float32)),
            "landmarks_original": torch.from_numpy(sample.landmarks.copy()),
            "transform": torch.from_numpy(matrix),
            "subset_flags": torch.from_numpy(flags),
            "sample_id": sample.sample_id,
            "task": "wflw",
        }
        return result


class DistributedEvalSampler(Sampler[int]):

    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterable[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size


def build_dataset(
    root: str | Path,
    split: str,
    *,
    augmentation: dict[str, Any] | None = None,
) -> WFLWDataset:
    train, test = load_wflw_records(root)
    dev_train, dev_validation = development_split(train)
    choices = {
        "dev_train": (dev_train, True),
        "dev_validation": (dev_validation, False),
        "full_train": (train, True),
        "test": (test, False),
    }
    try:
        samples, training = choices[split]
    except KeyError as exc:
        raise ValueError(f"Unknown WFLW split: {split}") from exc
    return WFLWDataset(
        samples,
        training=training,
        augmentation=augmentation,
    )
