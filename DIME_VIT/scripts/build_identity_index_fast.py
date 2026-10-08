from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle
import struct
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import lmdb


META_KEY = b"__meta__"
VALID_KEYS_KEY = b"__valid_keys__"
IDENTITY_KEY = b"__identity_index__"
LABEL_MARKER = b"\x8c\x05label"


def _skip_pickle_memo(data: bytes, offset: int) -> int:

    while offset < len(data):
        opcode = data[offset]
        if opcode == 0x94:
            offset += 1
        elif opcode == ord("q"):
            offset += 2
        elif opcode == ord("r"):
            offset += 5
        elif opcode in (ord("p"), ord("g")):
            newline = data.find(b"\n", offset + 1)
            if newline < 0:
                raise ValueError("unterminated pickle memo opcode")
            offset = newline + 1
        else:
            return offset
    raise ValueError("pickle ended before label value")


def _decode_pickle_integer(data: bytes, offset: int) -> int:
    offset = _skip_pickle_memo(data, offset)
    opcode = data[offset]
    offset += 1
    if opcode == ord("K"):
        return data[offset]
    if opcode == ord("M"):
        return struct.unpack_from("<H", data, offset)[0]
    if opcode == ord("J"):
        return struct.unpack_from("<i", data, offset)[0]
    if opcode == 0x8A:
        size = data[offset]
        start = offset + 1
        return int.from_bytes(data[start : start + size], "little", signed=True)
    if opcode == 0x8B:
        size = struct.unpack_from("<I", data, offset)[0]
        start = offset + 4
        return int.from_bytes(data[start : start + size], "little", signed=True)
    if opcode in (ord("I"), ord("L")):
        newline = data.find(b"\n", offset)
        if newline < 0:
            raise ValueError("unterminated textual pickle integer")
        raw = data[offset:newline].rstrip(b"L")
        return int(raw)
    raise ValueError(f"unsupported pickle integer opcode 0x{opcode:02x}")


def extract_label(record: memoryview | bytes) -> tuple[int, bool]:

    tail_size = min(len(record), 4096)
    tail = bytes(record[-tail_size:])
    marker = tail.rfind(LABEL_MARKER)
    if marker >= 0:
        try:
            offset = marker + len(LABEL_MARKER)
            return _decode_pickle_integer(tail, offset), False
        except (IndexError, struct.error, ValueError):
            pass

    return int(pickle.loads(bytes(record))["label"]), True


def _open_readonly(path: str, readahead: bool) -> lmdb.Environment:
    return lmdb.open(
        path,
        subdir=Path(path).is_dir(),
        readonly=True,
        lock=False,
        readahead=readahead,
        meminit=False,
        max_readers=512,
    )


def _scan_chunk(task):
    path, start, end, raw_keys, readahead = task
    index: dict[int, list[int]] = defaultdict(list)
    fallback_count = 0
    env = _open_readonly(path, readahead)
    try:
        with env.begin(write=False, buffers=True) as txn:
            for position in range(start, end):
                raw_key = (
                    raw_keys[position - start] if raw_keys is not None else position
                )
                key = f"{int(raw_key):08d}".encode("ascii")
                record = txn.get(key)
                if record is None:
                    raise KeyError(f"missing LMDB sample key {key!r}")
                label, used_fallback = extract_label(record)
                index[label].append(position)
                fallback_count += int(used_fallback)
    finally:
        env.close()
    return start, end, dict(index), fallback_count


def _partition_tasks(
    path: str,
    total: int,
    workers: int,
    chunks_per_worker: int,
    valid_keys: Sequence[int] | None,
    readahead: bool,
):
    task_count = min(total, workers * chunks_per_worker)
    chunk_size = math.ceil(total / task_count)
    tasks = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        keys = None if valid_keys is None else tuple(valid_keys[start:end])
        tasks.append((path, start, end, keys, readahead))
    return tasks


def _write_index(
    path: Path,
    payload: bytes,
    current_map_size: int,
    overwrite: bool,
) -> bool:
    data_file = path / "data.mdb" if path.is_dir() else path
    reserve = max(1 << 30, len(payload) * 2)
    map_size = max(current_map_size, data_file.stat().st_size + reserve)
    page_size = 4096
    map_size = ((map_size + page_size - 1) // page_size) * page_size

    env = lmdb.open(
        str(path),
        subdir=path.is_dir(),
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_size=map_size,
        max_readers=512,
    )
    try:
        with env.begin(write=True) as txn:
            if txn.get(IDENTITY_KEY) is not None and not overwrite:
                print(
                    "Another process created __identity_index__; keeping it.",
                    flush=True,
                )
                return False
            txn.put(IDENTITY_KEY, payload, overwrite=True)

        env.sync()
        with env.begin(write=False, buffers=True) as txn:
            stored = txn.get(IDENTITY_KEY)
            if stored is None or len(stored) != len(payload):
                raise RuntimeError("identity-index verification failed after commit")
    finally:
        env.close()
    return True


def build(args: argparse.Namespace) -> None:
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LMDB does not exist: {path}")

    env = _open_readonly(str(path), args.readahead)
    try:
        with env.begin(write=False) as txn:
            meta_raw = txn.get(META_KEY)
            valid_raw = txn.get(VALID_KEYS_KEY)
            existing = txn.get(IDENTITY_KEY)
        current_map_size = int(env.info()["map_size"])
    finally:
        env.close()

    if meta_raw is None:
        raise ValueError("LMDB is missing the required __meta__ entry")
    meta = pickle.loads(meta_raw)
    if not isinstance(meta, dict) or "num_samples" not in meta:
        raise ValueError("__meta__ must contain num_samples")
    if existing is not None and not args.overwrite:
        print(
            f"__identity_index__ already exists ({len(existing):,} bytes); nothing to do.",
            flush=True,
        )
        return

    valid_keys = (
        None if valid_raw is None else [int(key) for key in pickle.loads(valid_raw)]
    )
    total = len(valid_keys) if valid_keys is not None else int(meta["num_samples"])
    if total <= 0:
        raise ValueError("LMDB contains no samples")
    workers = max(1, min(int(args.workers), total))
    tasks = _partition_tasks(
        str(path), total, workers, args.chunks_per_worker, valid_keys, args.readahead
    )
    print(
        f"Scanning {total:,} samples with {workers} workers in {len(tasks)} contiguous chunks",
        flush=True,
    )
    print(f"LMDB: {path}", flush=True)

    identity_index: dict[int, list[int]] = defaultdict(list)
    processed = 0
    fallback_total = 0
    started = time.perf_counter()
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:

        for start, end, partial, fallbacks in executor.map(_scan_chunk, tasks):
            for label, positions in partial.items():
                identity_index[label].extend(positions)
            processed += end - start
            fallback_total += fallbacks
            elapsed = max(time.perf_counter() - started, 1.0e-6)
            print(
                f"progress={processed:,}/{total:,} ({processed / total:.1%}) "
                f"rate={processed / elapsed:,.0f} records/s",
                flush=True,
            )

    indexed = sum(len(positions) for positions in identity_index.values())
    if indexed != total:
        raise RuntimeError(f"indexed {indexed:,} positions, expected {total:,}")
    result = dict(identity_index)
    payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - started
    print(
        f"Scan complete: identities={len(result):,}, fallback_decodes={fallback_total:,}, "
        f"index_size={len(payload) / 2**20:.1f} MiB, elapsed={elapsed / 60:.1f} min",
        flush=True,
    )
    wrote = _write_index(path, payload, current_map_size, args.overwrite)
    if wrote:
        print("Committed and verified LMDB key __identity_index__.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="LMDB directory or file")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(
            os.environ.get("SLURM_CPUS_PER_TASK", min(os.cpu_count() or 1, 64))
        ),
    )
    parser.add_argument("--chunks-per-worker", type=int, default=2)
    parser.add_argument("--readahead", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.chunks_per_worker < 1:
        parser.error("--workers and --chunks-per-worker must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())
