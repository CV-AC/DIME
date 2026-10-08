import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from facebench.tasks.head_pose.data import BIWIDataset, MatPoseDataset, build_dataset
from facebench.tasks.head_pose.lmdb_store import LmdbImageReader, validate_lmdb_archive
from facebench.tasks.head_pose.pack_lmdb import pack_dataset
from facebench.tasks.head_pose.utils import sha256_file


def _write_manifest(path: Path, records: list[dict], **summary_fields) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    summary = {
        "dataset": summary_fields.pop("dataset", "test"),
        "accepted": len(records),
        "sha256": sha256_file(path),
        **summary_fields,
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _make_rgb_image(path: Path, offset: int) -> None:
    image = Image.new("RGB", (80, 72))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (
                (x * 3 + offset) % 256,
                (y * 5 + offset) % 256,
                (x + y + offset) % 256,
            )
    image.save(path, quality=93, subsampling=0)


def test_mat_pose_lmdb_is_pixel_and_target_equivalent(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _make_rgb_image(source / "one.jpg", 7)
    _make_rgb_image(source / "two.jpg", 29)
    records = [
        {
            "sample_id": "set/one",
            "image": "one.jpg",
            "pitch": 0.1,
            "yaw": -0.2,
            "roll": 0.3,
            "landmark_bbox": [8.0, 7.0, 70.0, 66.0],
        },
        {
            "sample_id": "set/two",
            "image": "two.jpg",
            "pitch": -0.2,
            "yaw": 0.15,
            "roll": -0.05,
            "landmark_bbox": [10.0, 9.0, 68.0, 64.0],
        },
    ]
    manifest = tmp_path / "mat.jsonl"
    _write_manifest(manifest, records, dataset="300W-LP")
    database = tmp_path / "mat.lmdb"
    pack_dataset(
        dataset_name="test 300W-LP",
        root=source,
        manifest=manifest,
        output=database,
        workers=2,
        commit_every=1,
    )

    file_dataset = MatPoseDataset(
        source,
        manifest,
        training=False,
        input_size=48,
        test_resize=56,
    )
    lmdb_dataset = MatPoseDataset(
        source,
        manifest,
        training=False,
        input_size=48,
        test_resize=56,
        lmdb_path=database,
    )
    for index in range(len(records)):
        file_sample = file_dataset[index]
        lmdb_sample = lmdb_dataset[index]
        assert torch.equal(file_sample["image"], lmdb_sample["image"])
        assert torch.equal(file_sample["rotation"], lmdb_sample["rotation"])
        assert torch.equal(file_sample["ypr"], lmdb_sample["ypr"])
        assert file_sample["sample_id"] == lmdb_sample["sample_id"]

    lmdb_dataset.image_reader.close()
    file_train = MatPoseDataset(
        source,
        manifest,
        training=True,
        input_size=48,
        augmentation={
            "horizontal_flip_probability": 0.5,
            "blur_probability": 0.05,
        },
    )
    lmdb_train = MatPoseDataset(
        source,
        manifest,
        training=True,
        input_size=48,
        augmentation={
            "horizontal_flip_probability": 0.5,
            "blur_probability": 0.05,
        },
        lmdb_path=database,
    )
    for index in range(len(records)):
        np.random.seed(1234 + index)
        torch.manual_seed(5678 + index)
        file_sample = file_train[index]
        np.random.seed(1234 + index)
        torch.manual_seed(5678 + index)
        lmdb_sample = lmdb_train[index]
        assert torch.equal(file_sample["image"], lmdb_sample["image"])
        assert torch.equal(file_sample["rotation"], lmdb_sample["rotation"])
        assert torch.equal(file_sample["ypr"], lmdb_sample["ypr"])


def test_biwi_lmdb_is_pixel_and_target_equivalent(tmp_path: Path):
    source = tmp_path / "biwi"
    source.mkdir()
    _make_rgb_image(source / "frame.jpg", 13)
    records = [
        {
            "sample_id": "01/frame",
            "image": "frame.jpg",
            "yaw_deg": 4.0,
            "pitch_deg": -3.0,
            "roll_deg": 2.0,
        }
    ]
    manifest = tmp_path / "biwi.jsonl"
    _write_manifest(manifest, records, dataset="BIWI")
    database = tmp_path / "biwi.lmdb"
    pack_dataset(
        dataset_name="test BIWI",
        root=source,
        manifest=manifest,
        output=database,
        workers=1,
        commit_every=1,
    )

    file_sample = BIWIDataset(source, manifest, input_size=48, test_resize=56)[0]
    lmdb_sample = BIWIDataset(
        source,
        manifest,
        input_size=48,
        test_resize=56,
        lmdb_path=database,
    )[0]
    assert torch.equal(file_sample["image"], lmdb_sample["image"])
    assert torch.equal(file_sample["rotation"], lmdb_sample["rotation"])
    assert torch.equal(file_sample["ypr"], lmdb_sample["ypr"])


def test_lmdb_rejects_manifest_mismatch_and_missing_keys(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _make_rgb_image(source / "one.jpg", 3)
    records = [{"sample_id": "one", "image": "one.jpg"}]
    manifest = tmp_path / "data.jsonl"
    _write_manifest(manifest, records)
    database = tmp_path / "data.lmdb"
    pack_dataset(
        dataset_name="test",
        root=source,
        manifest=manifest,
        output=database,
        workers=1,
    )
    with pytest.raises(RuntimeError, match="manifest SHA256"):
        validate_lmdb_archive(
            database,
            manifest_sha256="0" * 64,
            record_count=1,
        )
    reader = LmdbImageReader(
        database,
        manifest_sha256=sha256_file(manifest),
        record_count=1,
    )
    with pytest.raises(KeyError, match="missing"):
        reader.read("missing")


def test_build_dataset_uses_lmdb_without_source_root(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _make_rgb_image(source / "one.jpg", 5)
    records = [
        {
            "sample_id": "one",
            "image": "one.jpg",
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "landmark_bbox": [8.0, 8.0, 70.0, 64.0],
        }
    ]
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, records)
    database = tmp_path / "train.lmdb"
    pack_dataset(
        dataset_name="train",
        root=source,
        manifest=manifest,
        output=database,
        workers=1,
    )
    config = {
        "data": {
            "backend": "lmdb",
            "train_root": str(tmp_path / "intentionally-absent"),
            "train_manifest": str(manifest),
            "train_lmdb": str(database),
        },
        "protocol": {"input_size": 48},
        "augmentation": {
            "random_resized_crop_scale": [1.0, 1.0],
            "random_resized_crop_ratio": [1.0, 1.0],
            "horizontal_flip_probability": 0.0,
            "blur_probability": 0.0,
        },
    }
    dataset = build_dataset(config, "train")
    assert len(dataset) == 1
    assert dataset[0]["image"].shape == (3, 48, 48)


def test_pack_failure_never_publishes_partial_archive(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "corrupt.jpg").write_bytes(b"not an encoded image")
    manifest = tmp_path / "corrupt.jsonl"
    _write_manifest(
        manifest,
        [{"sample_id": "corrupt", "image": "corrupt.jpg"}],
    )
    database = tmp_path / "corrupt.lmdb"
    with pytest.raises(RuntimeError, match="decode failed"):
        pack_dataset(
            dataset_name="corrupt",
            root=source,
            manifest=manifest,
            output=database,
            workers=1,
        )
    assert not database.exists()
    assert not Path(f"{database}.summary.json").exists()
    assert not list(tmp_path.glob(".corrupt.lmdb.tmp-*"))
