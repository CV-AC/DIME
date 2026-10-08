from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from tqdm import tqdm

from .config import LANDMARK_ROOT
from .data import (
    EXPECTED_COUNTS,
    SUBSETS,
    annotation_paths,
    development_split,
    load_wflw_records,
)
from .utils import sha256_file, sha256_text, write_json


def _validate_hrnet_csv(csv_path: Path, records: Sequence) -> dict[str, float | int]:
    records_by_path = defaultdict(list)
    for sample in records:
        records_by_path[sample.relative_path].append(sample)
    checked = 0
    max_center_delta = 0.0
    max_scale_delta = 0.0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if not row:
                continue
            relative_path = row[0].replace("\\", "/")
            candidates = records_by_path.get(relative_path)
            if not candidates:

                relative_path = relative_path.split("WFLW_images/")[-1]
                candidates = records_by_path.get(relative_path)
            if not candidates:
                raise ValueError(f"HRNet CSV sample not found in WFLW: {row[0]}")
            scale, center_x, center_y = map(float, row[1:4])
            requested_center = np.asarray((center_x, center_y))

            def discrepancy(candidate) -> float:
                x1, y1, x2, y2 = candidate.box_xyxy
                center = np.asarray(((x1 + x2) / 2, (y1 + y2) / 2))
                candidate_scale = max(x2 - x1, y2 - y1) / 200.0
                return float(np.abs(center - requested_center).max()) + abs(
                    candidate_scale - scale
                )

            sample = min(candidates, key=discrepancy)
            x1, y1, x2, y2 = sample.box_xyxy
            expected_center = np.asarray(((x1 + x2) / 2, (y1 + y2) / 2))
            expected_scale = max(x2 - x1, y2 - y1) / 200.0
            max_center_delta = max(
                max_center_delta,
                float(np.abs(np.asarray((center_x, center_y)) - expected_center).max()),
            )
            max_scale_delta = max(max_scale_delta, abs(scale - expected_scale))
            checked += 1
    return {
        "checked": checked,
        "max_center_delta": max_center_delta,
        "max_scale_delta": max_scale_delta,
    }


def audit(
    data_root: str | Path,
    *,
    output: str | Path | None = None,
    hrnet_csv_root: str | Path | None = None,
    decode_images: bool = True,
) -> dict:
    root = Path(data_root).expanduser().resolve()
    paths = annotation_paths(root)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    train, test = load_wflw_records(root)
    counts = Counter()
    for sample in test:
        counts.update(sample.subsets)
    observed = {"train": len(train), "test": len(test), **dict(counts)}
    for name, expected in EXPECTED_COUNTS.items():
        if observed.get(name, 0) != expected:
            raise ValueError(
                f"{name}: expected {expected}, found {observed.get(name, 0)}"
            )

    train_paths = [sample.relative_path for sample in train]
    test_paths = [sample.relative_path for sample in test]
    train_ids = [sample.sample_id for sample in train]
    test_ids = [sample.sample_id for sample in test]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Duplicate annotation-level IDs in WFLW train.")
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Duplicate annotation-level IDs in WFLW test.")

    def annotation_signature(sample) -> str:
        digest = hashlib.sha256(sample.relative_path.encode("utf-8"))
        digest.update(sample.landmarks.astype(np.float32).tobytes())
        return digest.hexdigest()

    train_signatures = {annotation_signature(sample) for sample in train}
    test_signatures = {annotation_signature(sample) for sample in test}
    annotation_overlap = train_signatures.intersection(test_signatures)
    if annotation_overlap:
        raise ValueError(
            f"WFLW train/test contain {len(annotation_overlap)} identical face annotations."
        )
    shared_source_images = set(train_paths).intersection(test_paths)

    bad_images: list[str] = []
    bad_values: list[str] = []
    for sample in tqdm([*train, *test], desc="Audit WFLW", unit="image"):
        if (
            not np.isfinite(sample.landmarks).all()
            or not np.isfinite(sample.box_xyxy).all()
        ):
            bad_values.append(sample.relative_path)
        x1, y1, x2, y2 = sample.box_xyxy
        if x2 <= x1 or y2 <= y1:
            bad_values.append(sample.relative_path)
        if not sample.image_path.is_file():
            bad_images.append(sample.relative_path)
        elif (
            decode_images
            and cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR) is None
        ):
            bad_images.append(sample.relative_path)
    if bad_images:
        raise ValueError(
            f"{len(bad_images)} missing/undecodable images; first={bad_images[0]}"
        )
    if bad_values:
        raise ValueError(
            f"{len(set(bad_values))} invalid annotations; first={bad_values[0]}"
        )

    dev_train, dev_validation = development_split(train)
    dev_path_overlap = {sample.relative_path for sample in dev_train}.intersection(
        sample.relative_path for sample in dev_validation
    )
    if dev_path_overlap:
        raise ValueError(
            f"Development train/validation share {len(dev_path_overlap)} source images."
        )
    annotation_hashes = {name: sha256_file(path) for name, path in paths.items()}
    report: dict = {
        "dataset": "WFLW",
        "root": str(root),
        "counts": observed,
        "development_split": {
            "train": len(dev_train),
            "validation": len(dev_validation),
            "shared_source_images": 0,
            "validation_ids_sha256": sha256_text(
                sorted(sample.sample_id for sample in dev_validation)
            ),
        },
        "unique_image_paths": {
            "train": len(set(train_paths)),
            "test": len(set(test_paths)),
            "shared_between_splits": len(shared_source_images),
        },
        "annotation_sha256": annotation_hashes,
        "manifest_sha256": sha256_text(
            [
                *(f"train:{value}" for value in train_ids),
                *(f"test:{value}" for value in test_ids),
            ]
        ),
        "decoded_all_images": decode_images,
        "status": "ok",
    }
    if hrnet_csv_root:
        csv_root = Path(hrnet_csv_root).expanduser().resolve()
        report["hrnet_csv_parity"] = {
            "train": _validate_hrnet_csv(
                csv_root / "face_landmarks_wflw_train.csv",
                train,
            ),
            "test": _validate_hrnet_csv(
                csv_root / "face_landmarks_wflw_test.csv",
                test,
            ),
        }
    output_path = (
        Path(output).expanduser().resolve()
        if output
        else LANDMARK_ROOT / "data_audit" / "wflw_audit.json"
    )
    write_json(output_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the complete official WFLW layout."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--hrnet-csv-root", default="")
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Only check image paths; full decode is the formal preflight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit(
        args.data_root,
        output=args.output or None,
        hrnet_csv_root=args.hrnet_csv_root or None,
        decode_images=not args.skip_decode,
    )


if __name__ == "__main__":
    main()
