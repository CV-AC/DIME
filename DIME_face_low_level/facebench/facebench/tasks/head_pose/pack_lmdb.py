from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
import uuid
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile
from tqdm import tqdm

from .lmdb_store import (
    LMDB_FORMAT,
    META_KEY,
    _import_lmdb,
    image_key,
    lmdb_summary_path,
    validate_lmdb_archive,
)
from .utils import read_jsonl, sha256_file

ImageFile.LOAD_TRUNCATED_IMAGES = False
GIB = 1024**3


@dataclass(frozen=True)
class SourceItem:
    index: int
    sample_id: str
    key: bytes
    path: Path
    expected_size: int


@dataclass(frozen=True)
class LoadedItem:
    index: int
    sample_id: str
    key: bytes
    data: bytes
    width: int
    height: int
    image_format: str


def _stable_content_update(digest, key: bytes, data) -> None:
    digest.update(len(key).to_bytes(4, byteorder="big", signed=False))
    digest.update(key)
    digest.update(len(data).to_bytes(8, byteorder="big", signed=False))
    digest.update(data)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_and_validate_image(item: SourceItem) -> LoadedItem:
    try:
        data = item.path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read sample {item.sample_id!r} from {item.path}."
        ) from exc
    if not data:
        raise RuntimeError(f"Empty image for sample {item.sample_id!r}: {item.path}")
    if len(data) != item.expected_size:
        raise RuntimeError(
            f"Image changed while packing sample {item.sample_id!r}: "
            f"stat={item.expected_size} bytes, read={len(data)} bytes."
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = str(image.format or "unknown")
            width, height = image.size
            image.load()

            rgb = image.convert("RGB")
            rgb.load()
    except Exception as exc:
        raise RuntimeError(
            f"Image decode failed for sample {item.sample_id!r}: {item.path}"
        ) from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid dimensions for sample {item.sample_id!r}: {width}x{height}."
        )
    return LoadedItem(
        index=item.index,
        sample_id=item.sample_id,
        key=item.key,
        data=data,
        width=width,
        height=height,
        image_format=image_format,
    )


def _bounded_ordered_map(
    function,
    items: Iterable[SourceItem],
    *,
    workers: int,
    pending_limit: int,
) -> Iterator[LoadedItem]:
    iterator = iter(items)
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="lmdb-pack"
    ) as pool:
        pending: deque[Future] = deque()
        for _ in range(pending_limit):
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:

            result = pending.popleft().result()
            yield result
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                pass


def _manifest_records(
    manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    records = read_jsonl(manifest)
    if not records:
        raise RuntimeError(f"Manifest is empty: {manifest}")
    manifest_hash = sha256_file(manifest)
    summary_path = manifest.with_suffix(".summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing manifest summary: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid manifest summary: {summary_path}") from exc
    if not isinstance(summary, dict):
        raise TypeError(f"Expected a JSON object in {summary_path}.")
    expected_hash = str(summary.get("sha256", summary.get("manifest_sha256", "")))
    if expected_hash != manifest_hash:
        raise RuntimeError(f"Manifest SHA256 does not match {summary_path}.")
    if int(summary.get("accepted", -1)) != len(records):
        raise RuntimeError(
            f"Manifest count does not match {summary_path}: "
            f"{len(records)} != {summary.get('accepted')}."
        )
    return records, summary, manifest_hash


def _source_items(
    root: Path,
    records: list[dict[str, Any]],
) -> tuple[list[SourceItem], int]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    items: list[SourceItem] = []
    keys: set[bytes] = set()
    source_bytes = 0
    for index, record in enumerate(
        tqdm(records, desc="Preflight files", unit="image", dynamic_ncols=True)
    ):
        try:
            sample_id = str(record["sample_id"])
            relative_image = Path(str(record["image"]))
        except KeyError as exc:
            raise RuntimeError(
                f"Manifest record {index} lacks sample_id or image."
            ) from exc
        key = image_key(sample_id)
        if key in keys:
            raise RuntimeError(f"Duplicate sample_id in manifest: {sample_id!r}")
        keys.add(key)
        path = (root / relative_image).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"Image path escapes dataset root for sample {sample_id!r}: {path}"
            ) from exc
        try:
            stat = path.stat()
        except OSError as exc:
            raise FileNotFoundError(
                f"Missing source image for sample {sample_id!r}: {path}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(
                f"Source image is not a regular file for {sample_id!r}: {path}"
            )
        if stat.st_size <= 0:
            raise RuntimeError(f"Empty source image for {sample_id!r}: {path}")
        source_bytes += int(stat.st_size)
        items.append(
            SourceItem(
                index=index,
                sample_id=sample_id,
                key=key,
                path=path,
                expected_size=int(stat.st_size),
            )
        )
    return items, source_bytes


def _initial_map_size(source_bytes: int, map_size_factor: float) -> int:

    requested = int(source_bytes * map_size_factor) + 512 * 1024**2
    return max(GIB, int(math.ceil(requested / GIB) * GIB))


def _commit_batch(environment, batch: list[tuple[bytes, bytes]]) -> None:
    lmdb = _import_lmdb()
    while True:
        transaction = environment.begin(write=True)
        try:
            for key, value in batch:
                if not transaction.put(key, value, overwrite=False):
                    raise RuntimeError(f"Duplicate LMDB key while writing: {key!r}")
            transaction.commit()
            return
        except lmdb.MapFullError:
            try:
                transaction.abort()
            except lmdb.Error:
                pass
            current = int(environment.info()["map_size"])
            environment.set_mapsize(current * 2)
            print(f"LMDB map full; safely resized map from {current} to {current * 2}.")
        except Exception:
            try:
                transaction.abort()
            except lmdb.Error:
                pass
            raise


def _verify_published_content(
    database_path: Path,
    records: list[dict[str, Any]],
    *,
    expected_hash: str,
) -> None:
    lmdb = _import_lmdb()
    digest = hashlib.sha256()
    environment = lmdb.open(
        str(database_path),
        subdir=False,
        readonly=True,
        create=False,
        lock=False,
        readahead=False,
        max_spare_txns=1,
    )
    try:
        with environment.begin(write=False, buffers=True) as transaction:
            if int(transaction.stat()["entries"]) != len(records) + 1:
                raise RuntimeError(
                    "Post-write verification found an unexpected LMDB entry count."
                )
            for record in tqdm(
                records,
                desc="Verify LMDB",
                unit="image",
                dynamic_ncols=True,
            ):
                key = image_key(str(record["sample_id"]))
                value = transaction.get(key)
                if value is None:
                    raise RuntimeError(
                        f"Post-write verification cannot find {record['sample_id']!r}."
                    )
                _stable_content_update(digest, key, value)
            if transaction.get(META_KEY) is None:
                raise RuntimeError(
                    "Post-write verification found no internal metadata."
                )
    finally:
        environment.close()
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Post-write logical content SHA256 mismatch: "
            f"{actual_hash} != {expected_hash}."
        )


def pack_dataset(
    *,
    dataset_name: str,
    root: str | Path,
    manifest: str | Path,
    output: str | Path,
    workers: int = 16,
    pending_per_worker: int = 2,
    commit_every: int = 512,
    map_size_factor: float = 1.35,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:

    if workers < 1:
        raise ValueError("workers must be >= 1.")
    if pending_per_worker < 1:
        raise ValueError("pending_per_worker must be >= 1.")
    if commit_every < 1:
        raise ValueError("commit_every must be >= 1.")
    if map_size_factor < 1.05:
        raise ValueError("map_size_factor must be >= 1.05.")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive.")

    lmdb = _import_lmdb()
    started = time.monotonic()
    root_path = Path(root).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = lmdb_summary_path(output_path)

    records, manifest_summary, manifest_hash = _manifest_records(manifest_path)
    if output_path.exists() or summary_path.exists():
        if resume and output_path.is_file() and summary_path.is_file():
            metadata = validate_lmdb_archive(
                output_path,
                manifest_sha256=manifest_hash,
                record_count=len(records),
            )
            print(
                f"[resume] verified and skipped {dataset_name}: {metadata.path} "
                f"({metadata.record_count} images)"
            )
            return json.loads(summary_path.read_text(encoding="utf-8"))
        if not overwrite:
            raise FileExistsError(
                f"Refusing to replace existing LMDB output: {output_path} or "
                f"{summary_path}. Use --resume for a matching completed archive, "
                "or --overwrite only when no training job is reading it."
            )

    items, stat_source_bytes = _source_items(root_path, records)
    map_size = _initial_map_size(stat_source_bytes, map_size_factor)
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temporary_lock_path = Path(f"{temporary_path}-lock")
    if temporary_path.exists():
        temporary_path.unlink()
    if temporary_lock_path.exists():
        temporary_lock_path.unlink()

    print(
        f"Packing {dataset_name}: records={len(records)} "
        f"source_bytes={stat_source_bytes} map_size={map_size} workers={workers}"
    )
    environment = None
    image_bytes = 0
    width_min = height_min = 2**63 - 1
    width_max = height_max = 0
    formats: dict[str, int] = {}
    digest = hashlib.sha256()
    try:
        environment = lmdb.open(
            str(temporary_path),
            subdir=False,
            map_size=map_size,
            readonly=False,
            create=True,
            lock=True,
            readahead=False,
            max_readers=64,
        )
        batch: list[tuple[bytes, bytes]] = []
        loaded_items = _bounded_ordered_map(
            _read_and_validate_image,
            items,
            workers=workers,
            pending_limit=workers * pending_per_worker,
        )
        for expected_index, item in enumerate(
            tqdm(
                loaded_items,
                total=len(items),
                desc=f"Pack {dataset_name}",
                unit="image",
                dynamic_ncols=True,
            )
        ):
            if item.index != expected_index:
                raise RuntimeError(
                    f"Internal ordering error: {item.index} != {expected_index}."
                )
            batch.append((item.key, item.data))
            _stable_content_update(digest, item.key, item.data)
            image_bytes += len(item.data)
            width_min = min(width_min, item.width)
            width_max = max(width_max, item.width)
            height_min = min(height_min, item.height)
            height_max = max(height_max, item.height)
            formats[item.image_format] = formats.get(item.image_format, 0) + 1
            if len(batch) >= commit_every:
                _commit_batch(environment, batch)
                batch.clear()
        if batch:
            _commit_batch(environment, batch)

        logical_hash = digest.hexdigest()
        internal_metadata = {
            "format": LMDB_FORMAT,
            "complete": True,
            "dataset": dataset_name,
            "record_count": len(records),
            "manifest_sha256": manifest_hash,
            "logical_content_sha256": logical_hash,
            "key_schema": "image/<utf8 sample_id>",
            "value_schema": "original encoded image bytes; no re-encoding or crop",
        }
        _commit_batch(
            environment,
            [
                (
                    META_KEY,
                    json.dumps(
                        internal_metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                )
            ],
        )
        environment.sync()
        final_map_size = int(environment.info()["map_size"])
        environment.close()
        environment = None

        if temporary_lock_path.exists():
            temporary_lock_path.unlink()

        _verify_published_content(
            temporary_path,
            records,
            expected_hash=logical_hash,
        )

        os.replace(temporary_path, output_path)

        elapsed = time.monotonic() - started
        summary = {
            **internal_metadata,
            "database": str(output_path),
            "source_root": str(root_path),
            "manifest": str(manifest_path),
            "manifest_dataset": manifest_summary.get("dataset"),
            "manifest_protocol": manifest_summary.get("protocol"),
            "source_image_bytes": image_bytes,
            "lmdb_file_bytes": output_path.stat().st_size,
            "lmdb_map_size": final_map_size,
            "image_width_min": width_min,
            "image_width_max": width_max,
            "image_height_min": height_min,
            "image_height_max": height_max,
            "encoded_formats": dict(sorted(formats.items())),
            "workers": workers,
            "commit_every": commit_every,
            "seconds": elapsed,
            "post_write_full_scan_verified": True,
            "lmdb_python_version": str(lmdb.__version__),
        }
        _atomic_write_json(summary_path, summary)
        validate_lmdb_archive(
            output_path,
            manifest_sha256=manifest_hash,
            record_count=len(records),
        )
        print(
            f"Published {dataset_name}: {output_path} "
            f"({len(records)} images, {image_bytes} source bytes, {elapsed:.1f}s)"
        )
        return summary
    except Exception:
        if environment is not None:
            environment.close()
        if temporary_path.exists():
            temporary_path.unlink()
        if temporary_lock_path.exists():
            temporary_lock_path.unlink()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pack 300W-LP, AFLW2000, and processed BIWI into three independent, "
            "byte-exact LMDB archives."
        )
    )
    parser.add_argument("--300wlp-root", dest="wlp300_root", type=Path, required=True)
    parser.add_argument(
        "--300wlp-manifest",
        dest="wlp300_manifest",
        type=Path,
        default=Path("manifests/300wlp_train.jsonl"),
    )
    parser.add_argument("--aflw2000-root", type=Path, required=True)
    parser.add_argument(
        "--aflw2000-manifest",
        type=Path,
        default=Path("manifests/aflw2000_test.jsonl"),
    )
    parser.add_argument("--biwi-root", type=Path, required=True)
    parser.add_argument(
        "--biwi-manifest",
        type=Path,
        default=Path("processed/BIWI/biwi_test.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("packed"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pending-per-worker", type=int, default=2)
    parser.add_argument("--commit-every", type=int, default=512)
    parser.add_argument("--map-size-factor", type=float, default=1.35)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip an existing archive only after its manifest/count metadata matches.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing archives. Never use while training reads them.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    jobs = (
        (
            "300W-LP train",
            args.wlp300_root,
            args.wlp300_manifest,
            output_dir / "300wlp_train.lmdb",
        ),
        (
            "AFLW2000 test",
            args.aflw2000_root,
            args.aflw2000_manifest,
            output_dir / "aflw2000_test.lmdb",
        ),
        (
            "BIWI test",
            args.biwi_root,
            args.biwi_manifest,
            output_dir / "biwi_test.lmdb",
        ),
    )
    summaries = []
    for dataset_name, root, manifest, output in jobs:
        summaries.append(
            pack_dataset(
                dataset_name=dataset_name,
                root=root,
                manifest=manifest,
                output=output,
                workers=args.workers,
                pending_per_worker=args.pending_per_worker,
                commit_every=args.commit_every,
                map_size_factor=args.map_size_factor,
                resume=args.resume,
                overwrite=args.overwrite,
            )
        )
    aggregate = {
        "format": LMDB_FORMAT,
        "complete": True,
        "archives": [
            {
                "dataset": summary["dataset"],
                "database": summary["database"],
                "record_count": summary["record_count"],
                "manifest_sha256": summary["manifest_sha256"],
                "logical_content_sha256": summary["logical_content_sha256"],
            }
            for summary in summaries
        ],
    }
    _atomic_write_json(output_dir / "PACKED_DATASETS.complete.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
