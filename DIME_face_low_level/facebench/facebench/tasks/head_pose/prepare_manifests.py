from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
from tqdm import tqdm

from .utils import sha256_file, write_json, write_jsonl


TRAIN_DIRS = (
    "AFW",
    "AFW_Flip",
    "HELEN",
    "HELEN_Flip",
    "IBUG",
    "IBUG_Flip",
    "LFPW",
    "LFPW_Flip",
)


def _mat_record(
    root: Path, image_path: Path, max_angle: float
) -> tuple[dict, str | None]:
    mat_path = image_path.with_suffix(".mat")
    if not mat_path.is_file():
        return {}, "missing_mat"
    annotation = sio.loadmat(mat_path, variable_names=("Pose_Para", "pt2d"))
    pose = np.asarray(annotation["Pose_Para"]).reshape(-1)
    points = np.asarray(annotation["pt2d"], dtype=np.float64)

    if pose.size < 3 or points.ndim != 2 or points.shape[0] != 2 or points.shape[1] < 3:
        return {}, "invalid_annotation"
    pitch, yaw, roll = (float(pose[0]), float(pose[1]), float(pose[2]))
    if not np.isfinite([pitch, yaw, roll]).all() or not np.isfinite(points).all():
        return {}, "non_finite"
    degrees = np.abs(np.rad2deg([pitch, yaw, roll]))
    if np.any(degrees > max_angle):
        return {}, "out_of_range"
    bbox = [
        float(points[0].min()),
        float(points[1].min()),
        float(points[0].max()),
        float(points[1].max()),
    ]
    return {
        "sample_id": image_path.relative_to(root).with_suffix("").as_posix(),
        "image": image_path.relative_to(root).as_posix(),
        "annotation": mat_path.relative_to(root).as_posix(),
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "landmark_bbox": bbox,
    }, None


def _mat_record_worker(arguments: tuple[str, str, float]) -> tuple[dict, str | None]:
    root, image_path, max_angle = arguments
    return _mat_record(Path(root), Path(image_path), max_angle)


def _records(
    root: Path,
    images: Iterable[Path],
    max_angle: float,
    description: str,
    workers: int,
) -> tuple[list[dict], dict[str, int]]:
    image_list = sorted(images, key=lambda value: value.as_posix())
    records: list[dict] = []
    rejected: dict[str, int] = {}
    arguments = ((str(root), str(image), max_angle) for image in image_list)
    if workers > 1:
        executor_context = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        executor = executor_context.__enter__()
        results = executor.map(_mat_record_worker, arguments, chunksize=128)
    else:
        executor_context = None
        results = map(_mat_record_worker, arguments)
    try:
        iterator = tqdm(results, total=len(image_list), desc=description, unit="image")
        for record, reason in iterator:
            if reason is None:
                records.append(record)
            else:
                rejected[reason] = rejected.get(reason, 0) + 1
    finally:
        if executor_context is not None:
            executor_context.__exit__(None, None, None)
    return records, rejected


def build_300wlp(root: Path, max_angle: float, workers: int) -> tuple[list[dict], dict]:
    missing_dirs = [name for name in TRAIN_DIRS if not (root / name).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing 300W-LP directories: {missing_dirs}")
    images = (path for name in TRAIN_DIRS for path in (root / name).glob("*.jpg"))
    records, rejected = _records(root, images, max_angle, "300W-LP", workers)
    return records, {
        "dataset": "300W-LP",
        "raw_pairs_expected": 122450,
        "accepted": len(records),
        "rejected": rejected,
        "max_abs_angle_degrees": max_angle,
        "contains_pregenerated_flip_directories": True,
    }


def build_aflw2000(
    root: Path, max_angle: float, workers: int
) -> tuple[list[dict], dict]:
    images = root.glob("*.jpg")
    records, rejected = _records(root, images, max_angle, "AFLW2000", workers)
    return records, {
        "dataset": "AFLW2000",
        "raw_pairs_expected": 2000,
        "accepted": len(records),
        "rejected": rejected,
        "max_abs_angle_degrees": max_angle,
    }


def save_manifest(output: Path, records: list[dict], summary: dict) -> None:
    count = write_jsonl(output, records)
    if count != len(records):
        raise RuntimeError("Manifest write count mismatch.")
    summary = dict(summary)
    summary["manifest"] = output.name
    summary["sha256"] = sha256_file(output)
    write_json(output.with_suffix(".summary.json"), summary)
    print(f"{output}: {count} samples, sha256={summary['sha256']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic, ±99° 6DRepNet manifests."
    )
    parser.add_argument("--300wlp", dest="root_300wlp", type=Path, required=True)
    parser.add_argument("--aflw2000", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--max-angle", type=float, default=99.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel MATLAB readers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_300wlp = args.root_300wlp.expanduser().resolve()
    root_aflw = args.aflw2000.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not math.isfinite(args.max_angle) or args.max_angle <= 0:
        raise ValueError("--max-angle must be finite and positive.")

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    train_records, train_summary = build_300wlp(
        root_300wlp, args.max_angle, args.workers
    )
    test_records, test_summary = build_aflw2000(root_aflw, args.max_angle, args.workers)
    save_manifest(output_dir / "300wlp_train.jsonl", train_records, train_summary)
    save_manifest(output_dir / "aflw2000_test.jsonl", test_records, test_summary)


if __name__ == "__main__":
    main()
