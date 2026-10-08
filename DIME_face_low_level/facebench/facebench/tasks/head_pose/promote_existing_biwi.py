from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from .prepare_biwi import (
    EXPECTED_ACCEPTED_FRAMES,
    EXPECTED_ACCEPTED_SAMPLE_IDS_SHA256,
    EXPECTED_RAW_FRAMES,
    FSA_NET_BIWI_PROTOCOL,
    FSA_NET_OFFICIAL_DATA_ZIP,
    FSA_NET_OFFICIAL_NPZ_SHA256,
    discover_sequences,
    read_biwi_pose,
    sample_ids_sha256,
)
from .utils import read_jsonl, sha256_file, write_json


def _raw_sample_ids(root: Path) -> set[str]:
    return {
        f"{sequence.name}/{image.stem.replace('_rgb', '')}"
        for sequence in discover_sequences(root)
        for image in sequence.glob("frame_*_rgb.png")
    }


def validate_and_promote(
    manifest: Path,
    raw_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    summary_path = manifest.with_suffix(".summary.json")
    skipped_path = manifest.with_name(manifest.stem + "_skipped.jsonl")
    for path in (manifest, summary_path, skipped_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed BIWI output: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = sorted(read_jsonl(manifest), key=lambda item: item["sample_id"])
    skipped = sorted(read_jsonl(skipped_path), key=lambda item: item["sample_id"])
    if len(records) != EXPECTED_ACCEPTED_FRAMES:
        raise RuntimeError(
            f"Expected {EXPECTED_ACCEPTED_FRAMES} accepted records, got {len(records)}."
        )
    accepted_hash = sample_ids_sha256(records)
    if accepted_hash != EXPECTED_ACCEPTED_SAMPLE_IDS_SHA256:
        raise RuntimeError(
            "Existing run does not have the same accepted raw-frame membership as "
            "FSA-Net's official BIWI_noTrack.npz: expected "
            f"{EXPECTED_ACCEPTED_SAMPLE_IDS_SHA256}, got {accepted_hash}."
        )

    accepted_ids = [str(item["sample_id"]) for item in records]
    skipped_ids = [str(item["sample_id"]) for item in skipped]
    if len(set(accepted_ids)) != len(accepted_ids):
        raise RuntimeError("Duplicate sample_id values in the BIWI manifest.")
    if set(accepted_ids).intersection(skipped_ids):
        raise RuntimeError(
            "A BIWI sample appears in both accepted and skipped outputs."
        )
    raw_ids = _raw_sample_ids(raw_root)
    if len(raw_ids) != EXPECTED_RAW_FRAMES:
        raise RuntimeError(
            f"Expected {EXPECTED_RAW_FRAMES} raw BIWI frames, found {len(raw_ids)}."
        )
    if set(accepted_ids).union(skipped_ids) != raw_ids:
        raise RuntimeError(
            "Accepted and skipped manifests do not form an exact partition of raw BIWI."
        )

    for record in records:
        crop = output_root / str(record["image"])
        pose_path = raw_root / str(record["source_pose"])
        if not crop.is_file():
            raise FileNotFoundError(f"Missing accepted crop: {crop}")
        crop_image = cv2.imread(str(crop), cv2.IMREAD_COLOR)
        if crop_image is None or crop_image.shape != (256, 256, 3):
            raise RuntimeError(
                f"Accepted crop is not a readable 256x256 RGB image: {crop}"
            )
        if not pose_path.is_file():
            raise FileNotFoundError(f"Missing source pose: {pose_path}")
        yaw, pitch, roll = read_biwi_pose(pose_path)
        observed = (
            float(record["yaw_deg"]),
            float(record["pitch_deg"]),
            float(record["roll_deg"]),
        )
        expected = (yaw, pitch, roll)
        if max(abs(a - b) for a, b in zip(observed, expected)) > 1e-8:
            raise RuntimeError(
                f"Pose provenance mismatch for {record['sample_id']}: "
                f"manifest={observed}, raw={expected}."
            )

    if int(summary.get("image_size", -1)) != 256:
        raise RuntimeError(
            "This promotion entry is only for the completed 256px controlled run; "
            f"summary reports image_size={summary.get('image_size')}."
        )
    if int(summary.get("raw_frames_seen", -1)) != EXPECTED_RAW_FRAMES:
        raise RuntimeError("Existing summary did not process every raw BIWI frame.")

    previous_protocol = summary.get("protocol")
    summary.update(
        {
            "protocol": FSA_NET_BIWI_PROTOCOL,
            "accepted": len(records),
            "skipped": len(skipped),
            "accepted_sample_ids_sha256": accepted_hash,
            "manifest_sha256": sha256_file(manifest),
            "canonical_reference_artifact": FSA_NET_OFFICIAL_DATA_ZIP,
            "canonical_reference_npz": "data/BIWI_noTrack.npz",
            "canonical_reference_npz_sha256": FSA_NET_OFFICIAL_NPZ_SHA256,
            "canonical_reference_image_size": 64,
            "canonical_reference_accepted": EXPECTED_ACCEPTED_FRAMES,
            "skip_reasons": dict(
                sorted(Counter(str(item["reason"]) for item in skipped).items())
            ),
            "validation": {
                "status": "passed",
                "previous_protocol": previous_protocol,
                "accepted_membership": "official_fsa_net_npz_pose_order_match",
                "raw_partition_complete": True,
                "all_crops_readable_256px": True,
                "all_pose_labels_match_raw": True,
            },
        }
    )
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and promote the already completed 13,219-crop BIWI run. "
            "This never reruns MTCNN and never changes crops or the manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("processed/BIWI/biwi_test.jsonl"),
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed/BIWI"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_and_promote(
        args.manifest.expanduser().resolve(),
        args.raw_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
    )
    print(
        {
            "status": summary["validation"]["status"],
            "protocol": summary["protocol"],
            "accepted": summary["accepted"],
            "skipped": summary["skipped"],
            "accepted_sample_ids_sha256": summary["accepted_sample_ids_sha256"],
            "manifest_sha256": summary["manifest_sha256"],
        }
    )


if __name__ == "__main__":
    main()
