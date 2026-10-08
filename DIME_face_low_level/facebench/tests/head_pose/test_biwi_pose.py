from pathlib import Path

import cv2
import numpy as np
import pytest

from facebench.tasks.head_pose import promote_existing_biwi
from facebench.tasks.head_pose.prepare_biwi import (
    _create_detector,
    _process_sequence,
    _select_face,
    discover_sequences,
    read_biwi_pose,
    sample_ids_sha256,
)
from facebench.tasks.head_pose.utils import write_json, write_jsonl


def test_identity_biwi_pose(tmp_path: Path):
    pose = tmp_path / "frame_pose.txt"
    matrix_and_translation = np.vstack((np.eye(3), np.zeros((1, 3))))
    np.savetxt(pose, matrix_and_translation)
    yaw, pitch, roll = read_biwi_pose(pose)
    assert abs(yaw) < 1e-8
    assert abs(pitch) < 1e-8
    assert abs(roll) < 1e-8


def test_discovers_direct_fsa_net_layout(tmp_path: Path):
    for index in range(1, 25):
        (tmp_path / f"{index:02d}").mkdir()
    assert [path.name for path in discover_sequences(tmp_path)] == [
        f"{index:02d}" for index in range(1, 25)
    ]


def test_discovers_download_archive_layout(tmp_path: Path):
    for index in range(1, 25):
        archive = "BK-1" if index <= 10 else "BK-2"
        (tmp_path / archive / f"{index:02d}").mkdir(parents=True)
    assert [path.name for path in discover_sequences(tmp_path)] == [
        f"{index:02d}" for index in range(1, 25)
    ]


def test_fsa_net_face_selection_rule_and_strict_confidence():
    detections = [
        {"confidence": 0.90, "box": [10, 10, 20, 20]},
        {"confidence": 0.95, "box": [50, 50, 20, 20]},
        {"confidence": 0.99, "box": [75, 75, 20, 20]},
    ]
    selected = _select_face(
        detections,
        image_width=120,
        image_height=120,
        margin=0.4,
        confidence_threshold=0.90,
    )
    assert selected is not None
    crop, confidence = selected
    assert crop == [67, 103, 67, 103]
    assert confidence == 0.99


def test_detector_reports_removed_pkg_resources_before_data_audit(monkeypatch):
    monkeypatch.setattr(
        "facebench.tasks.head_pose.prepare_biwi.importlib.util.find_spec",
        lambda name: None if name == "pkg_resources" else object(),
    )
    with pytest.raises(RuntimeError, match="setuptools==80.9.0"):
        _create_detector()


def test_sequence_worker_preserves_temporal_order(tmp_path: Path):
    root = tmp_path / "BIWI"
    sequence = root / "01"
    output = tmp_path / "processed"
    sequence.mkdir(parents=True)
    pose = np.vstack((np.eye(3), np.zeros((1, 3))))
    for frame in (3, 4):
        cv2.imwrite(
            str(sequence / f"frame_{frame:05d}_rgb.png"),
            np.zeros((140, 140, 3), dtype=np.uint8),
        )
        np.savetxt(sequence / f"frame_{frame:05d}_pose.txt", pose)

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect_faces(self, _image):
            x1 = 100 if self.calls == 0 else 0
            self.calls += 1
            return [{"confidence": 0.99, "box": [x1, 50, 20, 20]}]

    records, skipped, processed = _process_sequence(
        root,
        sequence,
        output,
        image_size=64,
        margin=0.4,
        confidence_threshold=0.90,
        continuity_threshold=80,
        max_angle=99,
        limit=None,
        detector=FakeDetector(),
        show_progress=False,
    )
    assert processed == 2
    assert len(records) == 1
    assert skipped[0]["reason"] == "temporal_discontinuity"


def test_sequence_reuses_previous_bbox_for_nonempty_low_confidence_detection(
    tmp_path: Path,
):
    root = tmp_path / "BIWI"
    sequence = root / "01"
    output = tmp_path / "processed"
    sequence.mkdir(parents=True)
    pose = np.vstack((np.eye(3), np.zeros((1, 3))))
    for frame in (1, 2):
        image = np.zeros((140, 140, 3), dtype=np.uint8)
        image[:, :, frame] = 255
        cv2.imwrite(str(sequence / f"frame_{frame:05d}_rgb.png"), image)
        np.savetxt(sequence / f"frame_{frame:05d}_pose.txt", pose)

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect_faces(self, _image):
            self.calls += 1
            if self.calls == 1:
                return [{"confidence": 0.99, "box": [50, 50, 20, 20]}]
            return [{"confidence": 0.50, "box": [5, 5, 20, 20]}]

    records, skipped, processed = _process_sequence(
        root,
        sequence,
        output,
        image_size=64,
        margin=0.4,
        confidence_threshold=0.90,
        continuity_threshold=80,
        max_angle=99,
        limit=None,
        detector=FakeDetector(),
        show_progress=False,
    )
    assert processed == 2
    assert not skipped
    assert len(records) == 2
    assert records[0]["bbox_xyxy"] == records[1]["bbox_xyxy"]
    assert records[1]["bbox_source"] == "previous_confident_detection"
    assert records[1]["bbox_source_sample_id"] == "01/frame_00001"
    assert records[1]["current_max_detector_confidence"] == 0.50


def test_sequence_does_not_reuse_bbox_when_detector_returns_no_faces(tmp_path: Path):
    root = tmp_path / "BIWI"
    sequence = root / "01"
    output = tmp_path / "processed"
    sequence.mkdir(parents=True)
    pose = np.vstack((np.eye(3), np.zeros((1, 3))))
    for frame in (1, 2):
        cv2.imwrite(
            str(sequence / f"frame_{frame:05d}_rgb.png"),
            np.zeros((140, 140, 3), dtype=np.uint8),
        )
        np.savetxt(sequence / f"frame_{frame:05d}_pose.txt", pose)

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def detect_faces(self, _image):
            self.calls += 1
            if self.calls == 1:
                return [{"confidence": 0.99, "box": [50, 50, 20, 20]}]
            return []

    records, skipped, processed = _process_sequence(
        root,
        sequence,
        output,
        image_size=64,
        margin=0.4,
        confidence_threshold=0.90,
        continuity_threshold=80,
        max_angle=99,
        limit=None,
        detector=FakeDetector(),
        show_progress=False,
    )
    assert processed == 2
    assert len(records) == 1
    assert skipped == [{"sample_id": "01/frame_00002", "reason": "no_face_detected"}]


def test_promotes_only_a_complete_canonical_existing_run(tmp_path: Path, monkeypatch):
    raw_root = tmp_path / "BIWI"
    output_root = tmp_path / "processed"
    manifest = output_root / "biwi_test.jsonl"
    for index in range(1, 25):
        (raw_root / f"{index:02d}").mkdir(parents=True)

    pose = np.vstack((np.eye(3), np.zeros((1, 3))))
    for frame in (1, 2, 3):
        (raw_root / "01" / f"frame_{frame:05d}_rgb.png").touch()
        np.savetxt(raw_root / "01" / f"frame_{frame:05d}_pose.txt", pose)

    records = []
    for frame in (1, 2):
        relative_image = Path("crops") / "01" / f"frame_{frame:05d}_rgb.png"
        crop = output_root / relative_image
        crop.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop), np.zeros((256, 256, 3), dtype=np.uint8))
        records.append(
            {
                "sample_id": f"01/frame_{frame:05d}",
                "image": relative_image.as_posix(),
                "source_pose": f"01/frame_{frame:05d}_pose.txt",
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
            }
        )
    skipped = [{"sample_id": "01/frame_00003", "reason": "no_face_detected"}]
    write_jsonl(manifest, records)
    write_jsonl(output_root / "biwi_test_skipped.jsonl", skipped)
    write_json(
        output_root / "biwi_test.summary.json",
        {
            "protocol": "old_failed_guard",
            "image_size": 256,
            "raw_frames_seen": 3,
        },
    )

    monkeypatch.setattr(promote_existing_biwi, "EXPECTED_ACCEPTED_FRAMES", 2)
    monkeypatch.setattr(promote_existing_biwi, "EXPECTED_RAW_FRAMES", 3)
    monkeypatch.setattr(
        promote_existing_biwi,
        "EXPECTED_ACCEPTED_SAMPLE_IDS_SHA256",
        sample_ids_sha256(records),
    )
    summary = promote_existing_biwi.validate_and_promote(
        manifest, raw_root, output_root
    )
    assert summary["accepted"] == 2
    assert summary["skipped"] == 1
    assert summary["validation"]["status"] == "passed"
    assert summary["validation"]["raw_partition_complete"] is True
