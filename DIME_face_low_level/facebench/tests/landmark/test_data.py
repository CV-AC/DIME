from pathlib import Path

import numpy as np

from facebench.tasks.landmark.data import (
    DistributedEvalSampler,
    FaRLAugment,
    WFLWSample,
    development_split,
    transform_points,
)


def _sample(index: int) -> WFLWSample:
    return WFLWSample(
        image_path=Path(f"{index}.jpg"),
        relative_path=f"faces/{index:04d}.jpg",
        landmarks=np.zeros((98, 2), dtype=np.float32),
        box_xyxy=np.asarray([10, 20, 110, 120], dtype=np.float32),
        attributes=(0, 0, 0, 0, 0, 0),
        sample_id=f"row-{index:04d}:faces/{index:04d}.jpg",
    )


def test_eval_affine_round_trip_is_subpixel_exact() -> None:
    points = np.asarray(
        [[10.25, 20.5], [70.0, 45.0], [109.75, 119.25]], dtype=np.float32
    )
    matrix = FaRLAugment(training=False).geometry_matrix(
        np.asarray([10, 20, 110, 120], dtype=np.float32)
    )
    restored = transform_points(
        transform_points(points, matrix), np.linalg.inv(matrix).astype(np.float32)
    )
    assert np.max(np.abs(points - restored)) < 1e-4


def test_eval_crop_is_bbox_centered_with_1p25_context() -> None:
    box = np.asarray([10, 20, 110, 120], dtype=np.float32)
    matrix = FaRLAugment(training=False).geometry_matrix(box)
    center = transform_points(np.asarray([[60.0, 70.0]], dtype=np.float32), matrix)[0]
    horizontal = transform_points(
        np.asarray([[10.0, 70.0], [110.0, 70.0]], dtype=np.float32), matrix
    )
    assert np.max(np.abs(center - np.asarray([255.5, 255.5]))) < 1e-4
    assert abs(float(horizontal[1, 0] - horizontal[0, 0]) - 409.6) < 1e-4


def test_development_split_is_stable_and_disjoint() -> None:
    samples = [_sample(index) for index in range(100)]
    train_a, validation_a = development_split(samples, validation_size=10)
    train_b, validation_b = development_split(
        list(reversed(samples)), validation_size=10
    )
    ids = lambda rows: {row.sample_id for row in rows}
    assert ids(validation_a) == ids(validation_b)
    assert ids(train_a) == ids(train_b)
    assert not ids(train_a).intersection(ids(validation_a))
    assert len(train_a) == 90
    assert len(validation_a) == 10


def test_development_split_keeps_faces_from_one_image_together() -> None:
    samples = [_sample(index) for index in range(20)]
    duplicate = _sample(100)
    duplicate = WFLWSample(
        image_path=samples[0].image_path,
        relative_path=samples[0].relative_path,
        landmarks=duplicate.landmarks,
        box_xyxy=duplicate.box_xyxy,
        attributes=duplicate.attributes,
        sample_id="row-0100:faces/0000.jpg",
    )
    train, validation = development_split([*samples, duplicate], validation_size=5)
    train_paths = {sample.relative_path for sample in train}
    validation_paths = {sample.relative_path for sample in validation}
    assert not train_paths.intersection(validation_paths)
    assert len(validation) == 5


def test_distributed_eval_sampler_never_pads_or_duplicates() -> None:
    dataset = list(range(10))
    shards = [
        list(DistributedEvalSampler(dataset, rank=rank, world_size=3))
        for rank in range(3)
    ]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))
