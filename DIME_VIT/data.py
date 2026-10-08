from __future__ import annotations

import io
import logging
import math
import os
import pickle
import random
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ImageSize = int | tuple[int, int]


def _to_hw(size: ImageSize) -> tuple[int, int]:
    if isinstance(size, int):
        size = (size, size)
    if len(size) != 2 or min(size) <= 0:
        raise ValueError(
            f"input_size must be a positive int or (height, width), got {size}"
        )
    return int(size[0]), int(size[1])


def _open_lmdb(path: str, readonly: bool) -> lmdb.Environment:
    return lmdb.open(
        path,
        subdir=Path(path).is_dir(),
        readonly=readonly,
        lock=not readonly,
        readahead=False,
        meminit=False,
        max_readers=512,
    )


class FaceLMDBDataset(Dataset):

    def __init__(
        self,
        lmdb_path: str | Path,
        transform: Callable | None = None,
        subset_ratio: float = 1.0,
        keep_env_open: bool = True,
    ) -> None:
        if not 0.0 < subset_ratio <= 1.0:
            raise ValueError(f"subset_ratio must be in (0, 1], got {subset_ratio}")

        self.path = str(Path(lmdb_path).expanduser().resolve())
        if not Path(self.path).exists():
            raise FileNotFoundError(f"LMDB does not exist: {self.path}")
        self.transform = transform
        self.keep_env_open = bool(keep_env_open)
        self._env: lmdb.Environment | None = None
        self._env_pid: int | None = None
        self._identity_index: dict[int, list[int]] | None = None

        env = _open_lmdb(self.path, readonly=True)
        try:
            with env.begin(write=False) as txn:
                meta_buf = txn.get(b"__meta__")
                valid_keys_buf = txn.get(b"__valid_keys__")
        finally:
            env.close()

        if meta_buf is None:
            raise ValueError(
                f"LMDB is missing the required __meta__ entry: {self.path}"
            )
        meta = pickle.loads(meta_buf)
        if not isinstance(meta, dict) or "num_samples" not in meta:
            raise ValueError("__meta__ must be a dict containing 'num_samples'")

        self.meta: dict[str, Any] = meta
        self.jpeg_encoded = bool(meta.get("jpeg_encoded", True))
        total = int(meta["num_samples"])
        if total <= 0:
            raise ValueError(f"LMDB contains no samples: {self.path}")

        self._raw_keys: list[int] | None = None
        if valid_keys_buf is not None:
            valid_keys = pickle.loads(valid_keys_buf)
            self._raw_keys = [int(key) for key in valid_keys]
            total = len(self._raw_keys)

        self.total_length = total
        self.length = max(1, int(total * subset_ratio))
        if self._raw_keys is not None:
            self._raw_keys = self._raw_keys[: self.length]

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        state["_env_pid"] = None
        return state

    def _get_env(self) -> lmdb.Environment:
        pid = os.getpid()
        if self._env is not None and self._env_pid != pid:

            self._env.close()
            self._env = None
        if self._env is None:
            self._env = _open_lmdb(self.path, readonly=True)
            self._env_pid = pid
        return self._env

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_pid = None

    def _raw_key(self, index: int) -> int:
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(f"sample index {index} is outside [0, {self.length})")
        return self._raw_keys[index] if self._raw_keys is not None else index

    def _read_record(self, index: int) -> dict[str, Any]:
        raw_key = self._raw_key(index)
        key = f"{raw_key:08d}".encode("ascii")
        try:
            with self._get_env().begin(write=False) as txn:
                value = txn.get(key)
        finally:
            if not self.keep_env_open:
                self.close()
        if value is None:
            raise KeyError(f"missing sample key {key!r} in {self.path}")
        record = pickle.loads(value)
        if (
            not isinstance(record, dict)
            or "image" not in record
            or "label" not in record
        ):
            raise ValueError(f"invalid sample record at key {key!r}")
        return record

    def __getitem__(self, index: int) -> tuple[Any, int]:
        record = self._read_record(index)
        image_data = record["image"]
        if self.jpeg_encoded:
            with Image.open(io.BytesIO(image_data)) as image:
                image = image.convert("RGB")
        else:
            array = (
                pickle.loads(image_data)
                if isinstance(image_data, bytes)
                else image_data
            )
            image = Image.fromarray(np.asarray(array)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, int(record["label"])

    def get_sample_info(self, index: int) -> dict[str, Any]:
        record = self._read_record(index)
        fields = ("label", "path", "width", "height", "size")
        return {name: record[name] for name in fields if name in record}

    def load_identity_index(self) -> dict[int, list[int]]:

        if self._identity_index is not None:
            return self._identity_index

        try:
            with self._get_env().begin(write=False) as txn:
                cached = txn.get(b"__identity_index__")
        finally:
            if not self.keep_env_open:
                self.close()

        if cached is None:
            LOGGER.warning(
                "LMDB has no __identity_index__; scanning %d samples once. "
                "Use build_identity_index_cache() to persist it.",
                self.length,
            )
            identity_index: dict[int, list[int]] = defaultdict(list)
            try:
                with self._get_env().begin(write=False) as txn:
                    for position in range(self.length):
                        raw_key = self._raw_key(position)
                        value = txn.get(f"{raw_key:08d}".encode("ascii"))
                        if value is None:
                            raise KeyError(
                                f"missing sample key {raw_key:08d} in {self.path}"
                            )
                        identity_index[int(pickle.loads(value)["label"])].append(
                            position
                        )
            finally:
                if not self.keep_env_open:
                    self.close()
            self._identity_index = dict(identity_index)
            return self._identity_index

        raw_index = pickle.loads(cached)
        if not isinstance(raw_index, dict):
            raise TypeError("__identity_index__ must be a dict[label, list[index]]")

        identity_index = {}
        for label, cached_positions in raw_index.items():
            positions = [
                int(position)
                for position in cached_positions
                if 0 <= int(position) < self.length
            ]
            if positions:
                identity_index[int(label)] = positions
        self._identity_index = identity_index
        return self._identity_index

    build_identity_index = load_identity_index


def build_identity_index_cache(
    path: str | Path, overwrite: bool = False
) -> dict[int, list[int]]:

    path = str(Path(path).expanduser().resolve())
    if not Path(path).exists():
        raise FileNotFoundError(f"LMDB does not exist: {path}")
    env = _open_lmdb(path, readonly=False)
    try:
        with env.begin(write=False) as txn:
            existing = txn.get(b"__identity_index__")
            meta_buf = txn.get(b"__meta__")
            valid_keys_buf = txn.get(b"__valid_keys__")
        if existing is not None and not overwrite:
            return pickle.loads(existing)
        if meta_buf is None:
            raise ValueError(f"LMDB is missing __meta__: {path}")
        meta = pickle.loads(meta_buf)
        raw_keys: Sequence[int]
        if valid_keys_buf is not None:
            raw_keys = [int(key) for key in pickle.loads(valid_keys_buf)]
        else:
            raw_keys = range(int(meta["num_samples"]))

        identity_index: dict[int, list[int]] = defaultdict(list)
        with env.begin(write=False) as txn:
            for position, raw_key in enumerate(raw_keys):
                value = txn.get(f"{raw_key:08d}".encode("ascii"))
                if value is None:
                    raise KeyError(f"missing sample key {raw_key:08d} in {path}")
                identity_index[int(pickle.loads(value)["label"])].append(position)
        result = dict(identity_index)
        with env.begin(write=True) as txn:
            txn.put(
                b"__identity_index__", pickle.dumps(result, pickle.HIGHEST_PROTOCOL)
            )
        return result
    finally:
        env.close()


def build_transform(
    input_size: ImageSize,
    is_train: bool,
    crop_scale: tuple[float, float] = (0.2, 1.0),
    crop_ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    hflip_prob: float = 0.5,
    crop_pct: float = 0.875,
    interpolation: str = "bicubic",
    antialias: bool = True,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:

    height, width = _to_hw(input_size)
    interpolation_name = str(interpolation).lower()
    interpolation_modes = {
        "nearest": transforms.InterpolationMode.NEAREST,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
    }
    if interpolation_name not in interpolation_modes:
        raise ValueError(
            "interpolation must be nearest, bilinear, or bicubic, "
            f"got {interpolation!r}"
        )
    interpolation_mode = interpolation_modes[interpolation_name]
    normalize = transforms.Normalize(tuple(mean), tuple(std))
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=crop_scale,
                    ratio=crop_ratio,
                    interpolation=interpolation_mode,
                    antialias=antialias,
                ),
                transforms.RandomHorizontalFlip(hflip_prob),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if not 0.0 < crop_pct <= 1.0:
        raise ValueError(f"crop_pct must be in (0, 1], got {crop_pct}")
    resize_size = (round(height / crop_pct), round(width / crop_pct))
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                interpolation=interpolation_mode,
                antialias=antialias,
            ),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class DistributedIdentityPairBatchSampler(Sampler[list[int]]):

    def __init__(
        self,
        identity_to_indices: dict[int, Sequence[int]],
        batch_size: int,
        dataset_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        num_batches: int | None = None,
    ) -> None:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError(
                f"batch_size must be a positive even number, got {batch_size}"
            )
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(
                f"invalid distributed context: rank={rank}, world_size={world_size}"
            )

        valid = [
            tuple(int(index) for index in identity_to_indices[label])
            for label in sorted(identity_to_indices)
            if len(identity_to_indices[label]) >= 2
        ]
        if not valid:
            raise ValueError(
                "same-identity pairing requires at least one identity with two images"
            )

        self.identities = valid
        self.batch_size = batch_size
        self.pairs_per_batch = batch_size // 2
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.num_batches = (
            int(num_batches)
            if num_batches is not None
            else math.ceil(dataset_size / (world_size * batch_size))
        )
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive")
        if len(valid) < world_size:
            LOGGER.warning(
                "Only %d valid identities for %d ranks; rank identity overlap is unavoidable.",
                len(valid),
                world_size,
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_identity_pool(self, generator: torch.Generator) -> list[int]:
        permutation = torch.randperm(len(self.identities), generator=generator).tolist()
        pool = permutation[self.rank :: self.world_size]
        return pool if pool else permutation

    def __iter__(self) -> Iterator[list[int]]:
        identity_generator = torch.Generator()
        identity_generator.manual_seed(self.seed + 1_000_003 * self.epoch)
        pool = self._rank_identity_pool(identity_generator)
        slots = self.num_batches * self.pairs_per_batch
        schedule: list[int] = []
        while len(schedule) < slots:
            order = torch.randperm(len(pool), generator=identity_generator).tolist()
            schedule.extend(pool[position] for position in order)
        schedule = schedule[:slots]
        pair_rng = random.Random(self.seed + 2_000_003 * self.epoch + self.rank)
        for start in range(0, slots, self.pairs_per_batch):
            first: list[int] = []
            second: list[int] = []
            for identity_id in schedule[start : start + self.pairs_per_batch]:
                image_a, image_b = pair_rng.sample(self.identities[identity_id], 2)
                first.append(image_a)
                second.append(image_b)
            yield first + second[::-1]

    def __len__(self) -> int:
        return self.num_batches


class DistributedDisjointPairBatchSampler(Sampler[list[int]]):

    def __init__(
        self,
        identity_to_indices: dict[int, Sequence[int]],
        batch_size: int,
        dataset_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        num_batches: int | None = None,
    ) -> None:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError(
                f"batch_size must be a positive even number, got {batch_size}"
            )
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(
                f"invalid distributed context: rank={rank}, world_size={world_size}"
            )
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")

        seen = np.zeros(dataset_size, dtype=np.bool_)
        self.identities = []
        self.singleton_images = 0
        for label in sorted(identity_to_indices):
            indices = tuple(int(index) for index in identity_to_indices[label])
            if not indices:
                continue
            if min(indices) < 0 or max(indices) >= dataset_size:
                raise ValueError("Identity index contains an out-of-range image")
            if len(set(indices)) != len(indices) or seen[list(indices)].any():
                raise ValueError("Identity index assigns an image more than once")
            seen[list(indices)] = True
            if len(indices) == 1:
                self.singleton_images += 1
            else:
                self.identities.append(indices)
        self.unindexed_images = dataset_size - int(seen.sum())
        self.odd_identity_images = sum(len(indices) % 2 for indices in self.identities)
        self.total_pairs = sum(len(indices) // 2 for indices in self.identities)
        if self.total_pairs < world_size:
            raise ValueError("Not enough distinct same-identity pairs for every rank")

        self.batch_size = batch_size
        self.pairs_per_batch = batch_size // 2
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        pairs_per_rank = self.total_pairs // world_size
        available_batches = math.ceil(pairs_per_rank / self.pairs_per_batch)
        self.num_batches = (
            available_batches if num_batches is None else int(num_batches)
        )
        if not 1 <= self.num_batches <= available_batches:
            raise ValueError("num_batches must not exceed the non-repeating epoch")
        self.pairs_per_rank = min(
            pairs_per_rank, self.num_batches * self.pairs_per_batch
        )
        self.num_samples = 2 * self.pairs_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        pairs = np.empty((self.total_pairs, 2), dtype=np.int64)
        offset = 0
        for indices in self.identities:
            shuffled = generator.permutation(indices)
            count = len(shuffled) // 2
            pairs[offset : offset + count] = shuffled[: 2 * count].reshape(count, 2)
            offset += count
        usable_pairs = self.pairs_per_rank * self.world_size
        order = generator.permutation(self.total_pairs)[:usable_pairs]
        rank_order = order[self.rank :: self.world_size]
        if self.rank == 0:
            LOGGER.info(
                "pairing epoch %d | pairs %d | singleton images %d | odd-identity "
                "leftover images %d | distributed-tail pairs %d | unindexed images %d "
                "| batch-limit pairs %d",
                self.epoch + 1,
                usable_pairs,
                self.singleton_images,
                self.odd_identity_images,
                self.total_pairs % self.world_size,
                self.unindexed_images,
                self.total_pairs - self.total_pairs % self.world_size - usable_pairs,
            )
        for start in range(0, self.pairs_per_rank, self.pairs_per_batch):
            batch = pairs[rank_order[start : start + self.pairs_per_batch]]
            yield batch[:, 0].tolist() + batch[::-1, 1].tolist()

    def __len__(self) -> int:
        return self.num_batches


def build_eval_pairs(
    dataset: FaceLMDBDataset, num_pairs: int, seed: int = 0
) -> list[tuple[int, int]]:

    if num_pairs <= 0:
        raise ValueError(f"num_pairs must be positive, got {num_pairs}")
    identity_index = dataset.load_identity_index()
    identities = [
        tuple(sorted(identity_index[label]))
        for label in sorted(identity_index)
        if len(identity_index[label]) >= 2
    ]
    if not identities:
        raise ValueError("evaluation requires at least one identity with two images")

    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    while len(pairs) < num_pairs:
        order = list(range(len(identities)))
        rng.shuffle(order)
        for identity_id in order:
            pairs.append(tuple(rng.sample(identities[identity_id], 2)))
            if len(pairs) == num_pairs:
                break
    return pairs


class FixedIdentityPairBatchSampler(Sampler[list[int]]):

    def __init__(self, pairs: Sequence[tuple[int, int]], batch_size: int) -> None:
        if batch_size <= 0 or batch_size % 2:
            raise ValueError(
                f"batch_size must be a positive even number, got {batch_size}"
            )
        self.pairs = list(pairs)
        self.pairs_per_batch = batch_size // 2

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(0, len(self.pairs), self.pairs_per_batch):
            chunk = self.pairs[start : start + self.pairs_per_batch]
            first = [pair[0] for pair in chunk]
            second = [pair[1] for pair in chunk]
            yield first + second[::-1]

    def __len__(self) -> int:
        return math.ceil(len(self.pairs) / self.pairs_per_batch)


def _distributed_context(rank: int | None, world_size: int | None) -> tuple[int, int]:
    if rank is None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if world_size is None:
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
    return rank, world_size


def _loader_kwargs(num_workers: int, prefetch_factor: int) -> dict[str, Any]:
    if num_workers <= 0:
        return {}
    return {
        "persistent_workers": True,
        "prefetch_factor": prefetch_factor,
    }


def build_train_loader(
    data_path: str | Path,
    input_size: ImageSize,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    seed: int = 42,
    rank: int | None = None,
    world_size: int | None = None,
    subset_ratio: float = 1.0,
    crop_scale: tuple[float, float] = (0.2, 1.0),
    crop_ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    hflip_prob: float = 0.5,
    interpolation: str = "bicubic",
    antialias: bool = True,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    prefetch_factor: int = 4,
    pair_sampling: str = "identity_uniform",
) -> tuple[DataLoader, Sampler]:

    rank, world_size = _distributed_context(rank, world_size)
    dataset = FaceLMDBDataset(
        data_path,
        transform=build_transform(
            input_size,
            is_train=True,
            crop_scale=crop_scale,
            crop_ratio=crop_ratio,
            hflip_prob=hflip_prob,
            interpolation=interpolation,
            antialias=antialias,
            mean=mean,
            std=std,
        ),
        subset_ratio=subset_ratio,
        keep_env_open=num_workers > 0,
    )
    try:
        identity_index = dataset.load_identity_index()
    finally:

        dataset.close()
    samplers = {
        "identity_uniform": DistributedIdentityPairBatchSampler,
        "image_disjoint": DistributedDisjointPairBatchSampler,
    }
    if pair_sampling not in samplers:
        raise ValueError("pair_sampling must be identity_uniform or image_disjoint")
    sampler = samplers[pair_sampling](
        identity_index,
        batch_size=batch_size,
        dataset_size=len(dataset),
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed + rank)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        **_loader_kwargs(num_workers, prefetch_factor),
    )
    return loader, sampler


def build_eval_loader(
    data_path: str | Path,
    input_size: ImageSize,
    num_pairs: int,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 0,
    subset_ratio: float = 1.0,
    crop_pct: float = 0.875,
    interpolation: str = "bicubic",
    antialias: bool = True,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    prefetch_factor: int = 2,
) -> DataLoader:

    dataset = FaceLMDBDataset(
        data_path,
        transform=build_transform(
            input_size,
            is_train=False,
            crop_pct=crop_pct,
            interpolation=interpolation,
            antialias=antialias,
            mean=mean,
            std=std,
        ),
        subset_ratio=subset_ratio,
        keep_env_open=num_workers > 0,
    )
    try:
        pairs = build_eval_pairs(dataset, num_pairs=num_pairs, seed=seed)
    finally:

        dataset.close()
    sampler = FixedIdentityPairBatchSampler(pairs, batch_size=batch_size)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **_loader_kwargs(num_workers, prefetch_factor),
    )


def denormalize(
    images: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:

    shape = (1, 3, 1, 1) if images.ndim == 4 else (3, 1, 1)
    mean_tensor = images.new_tensor(tuple(mean)).view(shape)
    std_tensor = images.new_tensor(tuple(std)).view(shape)
    return (images * std_tensor + mean_tensor).clamp_(0.0, 1.0)
