from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LMDB_FORMAT = "dime_head_pose_lmdb_v1"
META_KEY = b"__dime_head_pose_metadata__"
IMAGE_KEY_PREFIX = b"image/"


def _import_lmdb():
    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError(
            "The LMDB data backend requires the 'lmdb' package. Install "
            "head_pose/requirements-lumi.txt before packing or training."
        ) from exc
    return lmdb


def lmdb_summary_path(database_path: str | Path) -> Path:
    path = Path(database_path)
    return Path(f"{path}.summary.json")


def image_key(sample_id: str) -> bytes:
    normalized = str(sample_id).strip()
    if not normalized:
        raise ValueError("LMDB image keys require a non-empty sample_id.")
    key = IMAGE_KEY_PREFIX + normalized.encode("utf-8")

    if len(key) > 511:
        raise ValueError(
            f"LMDB key for sample_id={normalized!r} is {len(key)} bytes; "
            "the maximum supported size is 511."
        )
    return key


@dataclass(frozen=True)
class LmdbMetadata:
    path: Path
    record_count: int
    manifest_sha256: str
    logical_content_sha256: str


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read valid LMDB metadata from {path}.") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _validate_metadata_fields(
    metadata: dict[str, Any],
    *,
    source: Path,
    manifest_sha256: str,
    record_count: int,
) -> None:
    if metadata.get("format") != LMDB_FORMAT:
        raise RuntimeError(
            f"Unsupported LMDB format in {source}: {metadata.get('format')!r}; "
            f"expected {LMDB_FORMAT!r}."
        )
    if metadata.get("complete") is not True:
        raise RuntimeError(f"LMDB archive is not marked complete: {source}")
    if int(metadata.get("record_count", -1)) != int(record_count):
        raise RuntimeError(
            f"LMDB record-count mismatch in {source}: "
            f"{metadata.get('record_count')} != {record_count}."
        )
    if str(metadata.get("manifest_sha256", "")) != manifest_sha256:
        raise RuntimeError(
            f"LMDB/manifest SHA256 mismatch in {source}. Repack this dataset "
            "from the current manifest."
        )
    content_hash = str(metadata.get("logical_content_sha256", ""))
    if len(content_hash) != 64:
        raise RuntimeError(f"Missing logical content hash in {source}.")


def validate_lmdb_archive(
    database_path: str | Path,
    *,
    manifest_sha256: str,
    record_count: int,
) -> LmdbMetadata:

    lmdb = _import_lmdb()
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    summary_path = lmdb_summary_path(path)
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing LMDB completion metadata: {summary_path}. "
            "Do not train from an incompletely published database."
        )

    summary = _read_json_mapping(summary_path)
    _validate_metadata_fields(
        summary,
        source=summary_path,
        manifest_sha256=manifest_sha256,
        record_count=record_count,
    )

    try:
        environment = lmdb.open(
            str(path),
            subdir=False,
            readonly=True,
            create=False,
            lock=False,
            readahead=False,
            max_spare_txns=1,
        )
        try:
            with environment.begin(write=False) as transaction:
                raw_metadata = transaction.get(META_KEY)
                entries = int(transaction.stat()["entries"])
        finally:
            environment.close()
    except lmdb.Error as exc:
        raise RuntimeError(f"Cannot open LMDB archive {path}: {exc}") from exc

    if raw_metadata is None:
        raise RuntimeError(f"LMDB archive has no internal metadata: {path}")
    try:
        internal = json.loads(bytes(raw_metadata).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid internal LMDB metadata in {path}.") from exc
    if not isinstance(internal, dict):
        raise TypeError(f"Invalid internal LMDB metadata in {path}.")
    _validate_metadata_fields(
        internal,
        source=path,
        manifest_sha256=manifest_sha256,
        record_count=record_count,
    )
    if entries != record_count + 1:
        raise RuntimeError(
            f"LMDB entry-count mismatch in {path}: {entries} entries for "
            f"{record_count} images plus one metadata record."
        )
    if internal["logical_content_sha256"] != summary["logical_content_sha256"]:
        raise RuntimeError(f"Internal/sidecar content hash mismatch for {path}.")

    return LmdbMetadata(
        path=path,
        record_count=record_count,
        manifest_sha256=manifest_sha256,
        logical_content_sha256=str(summary["logical_content_sha256"]),
    )


class LmdbImageReader:

    def __init__(
        self,
        database_path: str | Path,
        *,
        manifest_sha256: str,
        record_count: int,
    ) -> None:
        metadata = validate_lmdb_archive(
            database_path,
            manifest_sha256=manifest_sha256,
            record_count=record_count,
        )
        self.path = metadata.path
        self.logical_content_sha256 = metadata.logical_content_sha256
        self._environment = None
        self._pid: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()

        state["_environment"] = None
        state["_pid"] = None
        return state

    def _get_environment(self):
        current_pid = os.getpid()
        if self._environment is not None and self._pid != current_pid:

            self._environment = None
            self._pid = None
        if self._environment is None:
            lmdb = _import_lmdb()
            self._environment = lmdb.open(
                str(self.path),
                subdir=False,
                readonly=True,
                create=False,
                lock=False,
                readahead=False,
                max_spare_txns=1,
            )
            self._pid = current_pid
        return self._environment

    def read(self, sample_id: str) -> bytes:
        environment = self._get_environment()
        with environment.begin(write=False, buffers=True) as transaction:
            value = transaction.get(image_key(sample_id))
            if value is None:
                raise KeyError(
                    f"Sample {sample_id!r} is missing from LMDB archive {self.path}."
                )
            return bytes(value)

    def close(self) -> None:
        if self._environment is not None and self._pid == os.getpid():
            self._environment.close()
        self._environment = None
        self._pid = None
