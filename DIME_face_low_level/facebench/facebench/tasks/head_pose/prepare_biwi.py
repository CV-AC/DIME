from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import math
import multiprocessing
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .utils import sha256_file, write_json, write_jsonl


FSA_NET_BIWI_SOURCE = (
    "https://github.com/shamangary/FSA-Net/blob/master/data/" "TYY_create_db_biwi.py"
)
FSA_NET_OFFICIAL_DATA_ZIP = (
    "https://drive.google.com/file/d/" "1j6GMx33DCcbUOS8J3NHZ-BMHgk7H-oC_/view"
)
FSA_NET_OFFICIAL_NPZ_SHA256 = (
    "83480287422e49493fbb9dff88a227485e436cc1af50324def805583ad03b68a"
)
FSA_NET_BIWI_PROTOCOL = "fsa_net_tyy_create_db_biwi_2019_canonical_v2"
EXPECTED_SEQUENCES = tuple(f"{index:02d}" for index in range(1, 25))
EXPECTED_RAW_FRAMES = 15_678
EXPECTED_ACCEPTED_FRAMES = 13_219
EXPECTED_ACCEPTED_SAMPLE_IDS_SHA256 = (
    "4b44a0b49a835a8101c66cbe5a0d950c56cef03125ec96a080a2d9ef84795b2d"
)


def discover_sequences(root: Path) -> list[Path]:

    direct = [root / name for name in EXPECTED_SEQUENCES]
    if all(path.is_dir() for path in direct):
        return direct

    candidates = [
        path
        for archive_dir in sorted(root.glob("BK-*"))
        if archive_dir.is_dir()
        for path in archive_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    sequences = sorted(candidates, key=lambda value: int(value.name))
    expected = list(EXPECTED_SEQUENCES)
    observed = [path.name for path in sequences]
    if observed != expected:
        raise RuntimeError(
            "Expected BIWI sequences 01..24 either directly below "
            f"{root} or below BK-* archive directories; found {observed}."
        )
    return sequences


def read_biwi_pose(path: Path) -> tuple[float, float, float]:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ")
    if values.size != 12 or not np.isfinite(values).all():
        raise ValueError(f"Invalid BIWI pose file: {path}")
    rotation = values[:9].reshape(3, 3).T
    roll = -math.atan2(rotation[1, 0], rotation[0, 0]) * 180.0 / math.pi
    yaw = (
        -math.atan2(
            -rotation[2, 0],
            math.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2),
        )
        * 180.0
        / math.pi
    )
    pitch = math.atan2(rotation[2, 1], rotation[2, 2]) * 180.0 / math.pi
    return yaw, pitch, roll


def sample_ids_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["sample_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit(root: Path) -> dict[str, Any]:
    sequences = discover_sequences(root)
    frame_count = 0
    max_det_error = 0.0
    max_orthogonality_error = 0.0
    missing: list[str] = []
    for sequence in tqdm(sequences, desc="Audit BIWI", unit="sequence"):
        for image_path in sorted(sequence.glob("frame_*_rgb.png")):
            pose_path = image_path.with_name(
                image_path.name.replace("_rgb.png", "_pose.txt")
            )
            depth_path = image_path.with_name(
                image_path.name.replace("_rgb.png", "_depth.bin")
            )
            if not pose_path.is_file() or not depth_path.is_file():
                missing.append(str(image_path))
                continue
            values = np.fromstring(pose_path.read_text(encoding="utf-8"), sep=" ")
            if values.size != 12:
                raise ValueError(f"Invalid pose: {pose_path}")
            rotation = values[:9].reshape(3, 3)
            max_det_error = max(max_det_error, abs(np.linalg.det(rotation) - 1.0))
            orthogonality = rotation @ rotation.T - np.eye(3)
            max_orthogonality_error = max(
                max_orthogonality_error, float(np.abs(orthogonality).max())
            )
            frame_count += 1
    return {
        "sequence_count": len(sequences),
        "frame_count": frame_count,
        "missing_triplets": missing,
        "max_determinant_error": max_det_error,
        "max_orthogonality_error": max_orthogonality_error,
    }


def _detector_class():
    if importlib.util.find_spec("pkg_resources") is None:
        raise RuntimeError(
            "mtcnn==0.1.1 requires pkg_resources, which was removed from "
            "setuptools>=82. Install the pinned compatibility version with: "
            "python -m pip install --force-reinstall setuptools==80.9.0"
        )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            from mtcnn.mtcnn import MTCNN
    except ImportError as exc:
        raise RuntimeError(
            "Official BIWI preprocessing could not import mtcnn/TensorFlow. "
            "Install every pinned dependency from requirements-biwi.txt in the "
            "BIWI_PYTHON environment."
        ) from exc
    return MTCNN


def _create_detector():
    return _detector_class()()


def _detect_faces(detector: Any, image: np.ndarray) -> list[dict[str, Any]]:

    with contextlib.redirect_stdout(io.StringIO()):
        return detector.detect_faces(image)


def _installed_version(*distribution_names: str) -> str:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _select_face(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    margin: float,
    confidence_threshold: float,
) -> tuple[list[int], float] | None:
    candidates: list[tuple[float, list[int], float]] = []
    for detection in detections:
        confidence = float(detection.get("confidence", 0.0))
        if confidence <= confidence_threshold:
            continue
        x1, y1, width, height = [float(value) for value in detection["box"]]
        x2, y2 = x1 + width, y1 + height
        crop = [
            max(int(x1 - margin * width), 0),
            min(int(x2 + margin * width), image_width - 1),
            max(int(y1 - margin * height), 0),
            min(int(y2 + margin * height), image_height - 1),
        ]
        distance = abs(crop[0] - image_width * 2.0 / 3.0) + abs(
            crop[2] - image_height * 2.0 / 3.0
        )
        candidates.append((distance, crop, confidence))
    if not candidates:
        return None
    _, crop, confidence = min(candidates, key=lambda item: item[0])
    return crop, confidence


def _process_sequence(
    root: Path,
    sequence: Path,
    output_root: Path,
    *,
    image_size: int,
    margin: float,
    confidence_threshold: float,
    continuity_threshold: float,
    max_angle: float,
    limit: int | None,
    detector: Any,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    processed = 0
    previous_accepted_x1: int | None = None
    active_crop: list[int] | None = None
    active_confidence: float | None = None
    active_source_sample_id: str | None = None
    images = sorted(sequence.glob("frame_*_rgb.png"))
    iterator = tqdm(
        images,
        desc=f"BIWI {sequence.name}",
        unit="frame",
        disable=not show_progress,
    )
    for image_path in iterator:
        if limit is not None and processed >= limit:
            break
        processed += 1
        pose_path = image_path.with_name(
            image_path.name.replace("_rgb.png", "_pose.txt")
        )
        sample_id = f"{sequence.name}/{image_path.stem.replace('_rgb', '')}"
        yaw, pitch, roll = read_biwi_pose(pose_path)
        if max(abs(yaw), abs(pitch), abs(roll)) > max_angle:
            skipped.append({"sample_id": sample_id, "reason": "out_of_range"})
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append({"sample_id": sample_id, "reason": "image_read_failed"})
            continue
        height, width = image.shape[:2]
        detections = _detect_faces(detector, image)
        if not detections:

            skipped.append({"sample_id": sample_id, "reason": "no_face_detected"})
            continue
        selected = _select_face(
            detections,
            width,
            height,
            margin,
            confidence_threshold,
        )
        bbox_source = "current_confident_detection"
        if selected is not None:
            active_crop, active_confidence = selected
            active_source_sample_id = sample_id
        elif active_crop is None:

            skipped.append(
                {
                    "sample_id": sample_id,
                    "reason": "no_confident_face_without_history",
                    "detection_count": len(detections),
                    "max_detector_confidence": max(
                        float(item.get("confidence", 0.0)) for item in detections
                    ),
                }
            )
            continue
        else:

            bbox_source = "previous_confident_detection"

        assert active_crop is not None
        assert active_confidence is not None
        assert active_source_sample_id is not None
        x1, x2, y1, y2 = active_crop
        if (
            previous_accepted_x1 is not None
            and abs(x1 - previous_accepted_x1) >= continuity_threshold
        ):
            skipped.append(
                {
                    "sample_id": sample_id,
                    "bbox": [x1, y1, x2, y2],
                    "reason": "temporal_discontinuity",
                    "bbox_source": bbox_source,
                    "bbox_source_sample_id": active_source_sample_id,
                }
            )
            continue
        previous_accepted_x1 = x1
        crop = image[y1 : y2 + 1, x1 : x2 + 1]
        if crop.size == 0:
            skipped.append({"sample_id": sample_id, "reason": "empty_crop"})
            continue
        crop = cv2.resize(
            crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR
        )
        output_relative = Path("crops") / sequence.name / f"{image_path.stem}.png"
        output_path = output_root / output_relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), crop):
            raise OSError(f"Could not write {output_path}")
        records.append(
            {
                "sample_id": sample_id,
                "image": output_relative.as_posix(),
                "source_image": image_path.relative_to(root).as_posix(),
                "source_pose": pose_path.relative_to(root).as_posix(),
                "sequence": sequence.name,
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "roll_deg": roll,
                "bbox_xyxy": [x1, y1, x2, y2],
                "detector_confidence": active_confidence,
                "bbox_source": bbox_source,
                "bbox_source_sample_id": active_source_sample_id,
                "current_detection_count": len(detections),
                "current_max_detector_confidence": max(
                    float(item.get("confidence", 0.0)) for item in detections
                ),
            }
        )
    return records, skipped, processed


def _process_sequence_worker(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    sequence = Path(payload["sequence"])
    started = time.time()
    print(f"[BIWI worker] start sequence={sequence.name}", flush=True)
    result = _process_sequence(
        Path(payload["root"]),
        sequence,
        Path(payload["output_root"]),
        image_size=int(payload["image_size"]),
        margin=float(payload["margin"]),
        confidence_threshold=float(payload["confidence_threshold"]),
        continuity_threshold=float(payload["continuity_threshold"]),
        max_angle=float(payload["max_angle"]),
        limit=None,
        detector=_create_detector(),
        show_progress=False,
    )
    reason_counts = dict(
        sorted(Counter(str(item["reason"]) for item in result[1]).items())
    )
    reused = sum(
        item.get("bbox_source") == "previous_confident_detection" for item in result[0]
    )
    print(
        f"[BIWI worker] done sequence={sequence.name} "
        f"frames={result[2]} accepted={len(result[0])} "
        f"skipped={len(result[1])} reused_previous_bbox={reused} "
        f"skip_reasons={reason_counts} seconds={time.time() - started:.1f}",
        flush=True,
    )
    return result


def process(
    root: Path,
    output_root: Path,
    manifest_path: Path,
    *,
    image_size: int,
    margin: float,
    confidence_threshold: float,
    continuity_threshold: float,
    max_angle: float,
    limit: int | None,
    workers: int,
) -> dict[str, Any]:
    sequences = discover_sequences(root)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    processed = 0
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if limit is not None and workers != 1:
        raise ValueError("--limit is debug-only and requires --workers 1.")

    if workers == 1:
        detector = _create_detector()
        for sequence in sequences:
            remaining = None if limit is None else max(limit - processed, 0)
            if remaining == 0:
                break
            sequence_records, sequence_skipped, sequence_processed = _process_sequence(
                root,
                sequence,
                output_root,
                image_size=image_size,
                margin=margin,
                confidence_threshold=confidence_threshold,
                continuity_threshold=continuity_threshold,
                max_angle=max_angle,
                limit=remaining,
                detector=detector,
                show_progress=True,
            )
            records.extend(sequence_records)
            skipped.extend(sequence_skipped)
            processed += sequence_processed
    else:
        payloads = [
            {
                "root": str(root),
                "sequence": str(sequence),
                "output_root": str(output_root),
                "image_size": image_size,
                "margin": margin,
                "confidence_threshold": confidence_threshold,
                "continuity_threshold": continuity_threshold,
                "max_angle": max_angle,
            }
            for sequence in sequences
        ]
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(sequences)),
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(_process_sequence_worker, payload)
                for payload in payloads
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"BIWI parallel ({workers} workers)",
                unit="sequence",
            ):
                sequence_records, sequence_skipped, sequence_processed = future.result()
                records.extend(sequence_records)
                skipped.extend(sequence_skipped)
                processed += sequence_processed

    records.sort(key=lambda item: item["sample_id"])
    skipped.sort(key=lambda item: item["sample_id"])
    write_jsonl(manifest_path, records)
    skipped_path = manifest_path.with_name(manifest_path.stem + "_skipped.jsonl")
    write_jsonl(skipped_path, skipped)
    summary = {
        "protocol": FSA_NET_BIWI_PROTOCOL,
        "source_reference": FSA_NET_BIWI_SOURCE,
        "canonical_reference_artifact": FSA_NET_OFFICIAL_DATA_ZIP,
        "canonical_reference_npz": "data/BIWI_noTrack.npz",
        "canonical_reference_npz_sha256": FSA_NET_OFFICIAL_NPZ_SHA256,
        "canonical_reference_image_size": 64,
        "canonical_reference_accepted": EXPECTED_ACCEPTED_FRAMES,
        "implementation_notes": (
            "Server-safe, traceable implementation of the FSA-Net crop/pose "
            "semantics, including its stateful reuse of the preceding confident "
            "bbox when detections exist but none exceed 0.90. The accepted raw-frame "
            "membership is pinned to the authors' official BIWI_noTrack.npz; crops "
            "are emitted at the 256px size stated by 6DRepNet instead of the 64px "
            "size stored in FSA-Net's downloadable artifact."
        ),
        "source_root": str(root),
        "output_root": str(output_root),
        "raw_frames_seen": processed,
        "accepted": len(records),
        "accepted_sample_ids_sha256": sample_ids_sha256(records),
        "skipped": len(skipped),
        "image_size": image_size,
        "margin": margin,
        "confidence_threshold": confidence_threshold,
        "continuity_threshold_pixels": continuity_threshold,
        "face_selection": (
            "minimum abs(x1-2W/3)+abs(y1-2H/3), matching " "TYY_create_db_biwi.py"
        ),
        "low_confidence_behavior": (
            "reuse preceding confident bbox only when detections are non-empty, "
            "matching TYY_create_db_biwi.py function-scoped xw1..yw2 state"
        ),
        "previous_bbox_reuse_count": sum(
            item.get("bbox_source") == "previous_confident_detection"
            for item in records
        ),
        "skip_reasons": dict(
            sorted(Counter(str(item["reason"]) for item in skipped).items())
        ),
        "max_abs_angle_degrees": max_angle,
        "detector": "mtcnn.mtcnn.MTCNN",
        "detector_version": _installed_version("mtcnn"),
        "tensorflow_version": _installed_version("tensorflow-cpu", "tensorflow"),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "detector_stdout_suppressed": True,
        "workers": workers,
        "multiprocessing_start_method": "spawn" if workers > 1 else "none",
        "manifest_sha256": sha256_file(manifest_path),
    }
    write_json(manifest_path.with_suffix(".summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare traceable BIWI crops with the FSA-Net "
            "TYY_create_db_biwi.py protocol used by 6DRepNet."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("processed/BIWI"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--continuity-threshold", type=float, default=80.0)
    parser.add_argument("--max-angle", type=float, default=99.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Independent sequence workers; use 1 for sequential processing."
        ),
    )
    parser.add_argument(
        "--expected-accepted",
        type=int,
        default=EXPECTED_ACCEPTED_FRAMES,
        help=(
            "Fail if the full detector run does not reproduce the official "
            "FSA-Net BIWI_noTrack.npz membership count. "
            "Set to 0 to disable the count check."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Debug only; never use for results."
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not args.audit_only:
        _detector_class()
    audit_summary = audit(root)
    print(audit_summary)
    if audit_summary["missing_triplets"]:
        raise RuntimeError(
            f"BIWI has {len(audit_summary['missing_triplets'])} incomplete frame triplets."
        )
    if audit_summary["frame_count"] != EXPECTED_RAW_FRAMES:
        raise RuntimeError(
            f"Expected {EXPECTED_RAW_FRAMES} complete BIWI frames, got "
            f"{audit_summary['frame_count']}."
        )
    if args.audit_only:
        return
    output_root = args.output_root.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output_root / "biwi_test.jsonl"
    )
    summary = process(
        root,
        output_root,
        manifest,
        image_size=args.image_size,
        margin=args.margin,
        confidence_threshold=args.confidence,
        continuity_threshold=args.continuity_threshold,
        max_angle=args.max_angle,
        limit=args.limit,
        workers=args.workers,
    )
    print(summary)
    if (
        args.limit is None
        and args.expected_accepted > 0
        and (
            summary["accepted"] != args.expected_accepted
            or summary["accepted_sample_ids_sha256"]
            != EXPECTED_ACCEPTED_SAMPLE_IDS_SHA256
        )
    ):
        raise RuntimeError(
            "FSA-Net official BIWI_noTrack.npz membership mismatch: "
            f"expected count/hash {args.expected_accepted}/"
            f"{EXPECTED_ACCEPTED_SAMPLE_IDS_SHA256}, got "
            f"{summary['accepted']}/{summary['accepted_sample_ids_sha256']}. "
            "Do not use this manifest for the controlled table."
        )


if __name__ == "__main__":
    main()
