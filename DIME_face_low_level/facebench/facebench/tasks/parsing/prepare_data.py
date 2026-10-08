from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .config import apply_overrides, load_config, resolve_path
from .data import (
    EXPECTED_COUNTS,
    celeb_cache_path,
    celeb_split_ids,
    compose_celeb_label,
    load_celeb_samples,
    load_lapa_samples,
    manifest_path,
    write_sample_manifest,
)
from .labels import label_space
from .utils import sha256_file, sha256_text, write_json


def _write_mask_atomic(path: Path, label: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(str(temporary), label):
        raise OSError(f"Could not write {temporary}.")
    os.replace(temporary, path)


def _prepare_one_celeb_mask(
    root: Path,
    cache_root: Path,
    hq_id: int,
    *,
    force: bool,
) -> tuple[int, np.ndarray, bool]:
    path = celeb_cache_path(cache_root, hq_id)
    created = force or not path.is_file()
    if created:
        label = compose_celeb_label(root, hq_id)
        if int(label.max()) == 0:
            raise RuntimeError(
                f"No component annotation was found for CelebAMask-HQ id {hq_id}."
            )
        _write_mask_atomic(path, label)
    else:
        label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Could not decode cached mask {path}.")
    if label.shape != (512, 512):
        raise ValueError(f"Cached mask {path} has shape {label.shape}, not 512x512.")
    if int(label.min()) < 0 or int(label.max()) >= 19:
        raise ValueError(f"Cached mask {path} has invalid class indices.")
    return hq_id, np.bincount(label.reshape(-1), minlength=19), created


def prepare_celeb_cache(
    root: Path,
    cache_root: Path,
    ids: Iterable[int],
    *,
    workers: int,
    force: bool,
    limit: int | None = None,
) -> dict[str, Any]:
    all_ids = list(ids)
    selected = all_ids if limit is None else all_ids[:limit]
    cache_root.mkdir(parents=True, exist_ok=True)
    histogram = np.zeros(19, dtype=np.int64)
    created = 0

    def process(hq_id: int) -> tuple[int, np.ndarray, bool]:
        return _prepare_one_celeb_mask(root, cache_root, hq_id, force=force)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, (_, counts, was_created) in enumerate(
            executor.map(process, selected), start=1
        ):
            histogram += counts
            created += int(was_created)
            if index % 500 == 0 or index == len(selected):
                print(
                    f"CelebAMask-HQ masks: {index}/{len(selected)} "
                    f"(created {created})",
                    flush=True,
                )
    return {
        "requested": len(selected),
        "created": created,
        "reused": len(selected) - created,
        "complete": len(selected) == len(all_ids),
        "class_pixels": histogram.tolist(),
    }


def _audit_samples(
    samples,
    *,
    num_classes: int,
    decode: bool,
) -> dict[str, Any]:
    manifest_rows: list[str] = []
    histogram = np.zeros(num_classes, dtype=np.int64)
    for index, sample in enumerate(samples, start=1):

        manifest_rows.append(
            f"{sample.sample_id}\t{sample.image_path.name}\t{sample.label_path.name}"
        )
        if decode:
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            label = cv2.imread(str(sample.label_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not decode image {sample.image_path}.")
            if label is None:
                raise FileNotFoundError(f"Could not decode label {sample.label_path}.")
            if int(label.min()) < 0 or int(label.max()) >= num_classes:
                raise ValueError(
                    f"{sample.label_path} contains class indices outside "
                    f"[0,{num_classes - 1}]."
                )
            histogram += np.bincount(label.reshape(-1), minlength=num_classes)
        if index % 2000 == 0:
            print(f"Audited {index}/{len(samples)} samples", flush=True)
    return {
        "count": len(samples),
        "manifest_sha256": sha256_text(manifest_rows),
        "class_pixels": histogram.tolist() if decode else None,
    }


def prepare(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset_config = config["dataset"]
    dataset_name = str(dataset_config["name"]).strip().lower()
    root = resolve_path(dataset_config["root"], must_exist=True)
    assert root is not None
    space = label_space(dataset_name)
    expected = dict(
        dataset_config.get("expected_counts", EXPECTED_COUNTS[dataset_name])
    )
    report: dict[str, Any] = {
        "dataset": dataset_name,
        "root": str(root),
        "num_classes": space.num_classes,
        "class_names": list(space.names),
        "splits": {},
    }

    if dataset_name == "celebamask_hq":
        cache_root = resolve_path(dataset_config["mask_cache_dir"])
        assert cache_root is not None
        split_ids = celeb_split_ids(root)
        ids = sorted(value for values in split_ids.values() for value in values)
        report["mask_cache"] = prepare_celeb_cache(
            root,
            cache_root,
            ids,
            workers=args.workers,
            force=args.force,
            limit=args.limit,
        )
        report["mask_cache_dir"] = str(cache_root)
        if args.limit is not None:
            report["partial"] = True
            return report

    for split in ("train", "val", "test"):
        if dataset_name == "lapa":
            samples = load_lapa_samples(root, split)
        else:
            cache_root = resolve_path(dataset_config["mask_cache_dir"], must_exist=True)
            assert cache_root is not None
            samples = load_celeb_samples(root, split, cache_root)
        actual = len(samples)
        expected_count = int(expected[split])
        if actual != expected_count and not args.allow_nonstandard_counts:
            raise ValueError(
                f"{dataset_name}/{split} has {actual} samples; expected "
                f"{expected_count}. Use --allow-nonstandard-counts only for "
                "an intentionally modified dataset."
            )
        report["splits"][split] = _audit_samples(
            samples,
            num_classes=space.num_classes,
            decode=not args.skip_decode,
        )
        manifest_root = resolve_path(
            dataset_config.get("manifest_dir", "data_cache/manifests")
        )
        assert manifest_root is not None
        output_manifest = manifest_path(manifest_root, dataset_name, split)
        write_sample_manifest(
            output_manifest,
            dataset=dataset_name,
            split=split,
            samples=samples,
        )
        report["splits"][split]["prepared_manifest"] = str(output_manifest)
        report["splits"][split]["prepared_manifest_sha256"] = sha256_file(
            output_manifest
        )

    partition = root / "list_eval_partition.txt"
    if partition.is_file():
        report["list_eval_partition_sha256"] = sha256_file(partition)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate LaPa or prepare and validate CelebAMask-HQ."
    )
    parser.add_argument("--config", required=True, help="Experiment YAML file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted YAML value; may be repeated.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel mask preparation workers.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing CelebAMask-HQ merged masks.",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Check paths and manifests without decoding every sample.",
    )
    parser.add_argument(
        "--allow-nonstandard-counts",
        action="store_true",
        help="Accept split sizes that differ from the official protocol.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    config = apply_overrides(load_config(args.config), args.set)
    report = prepare(config, args)
    audit_path = resolve_path(
        config["dataset"].get(
            "audit_file",
            f"data_cache/{config['dataset']['name']}_audit.json",
        )
    )
    assert audit_path is not None
    write_json(audit_path, report)
    print(f"Data preparation complete: {audit_path}")


if __name__ == "__main__":
    main()
