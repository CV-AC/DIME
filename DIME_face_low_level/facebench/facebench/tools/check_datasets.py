from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


DEFAULT_ROOT = Path("DIME_face_low_level")

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class Result:
    task: str
    dataset: str
    status: str
    detail: str
    counts: dict = field(default_factory=dict)


def _lmdb_entries(path: Path) -> int:

    import lmdb

    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        subdir=path.is_dir(),
        max_readers=1,
    )
    try:
        return env.stat()["entries"]
    finally:
        env.close()


def _lmdb_sample_decodes(path: Path, n: int = 3) -> str:

    import lmdb
    from PIL import Image

    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        subdir=path.is_dir(),
        max_readers=1,
    )
    try:
        with env.begin() as txn:
            cur = txn.cursor()
            if not cur.first():
                return "empty"
            seen = []
            for _ in range(n):
                key, value = cur.item()
                try:
                    im = Image.open(io.BytesIO(value))
                    seen.append(f"{im.format} {im.size[0]}x{im.size[1]}")
                except Exception:
                    seen.append(f"<{len(value)}B non-image>")
                if not cur.next():
                    break
            return ", ".join(dict.fromkeys(seen))
    finally:
        env.close()


def _lmdb_image_entries(path: Path) -> int:

    import lmdb

    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        subdir=path.is_dir(),
        max_readers=1,
    )
    try:
        with env.begin() as txn:
            cur = txn.cursor()
            if not cur.first():
                return 0
            n = 0
            while True:
                if not cur.key().startswith(b"__"):
                    n += 1
                if not cur.next():
                    return n
    finally:
        env.close()


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def _check(
    task: str,
    dataset: str,
    expected: Optional[int],
    measure: Callable[[], int],
    note: str = "",
    tolerance: int = 0,
) -> Result:
    try:
        actual = measure()
    except FileNotFoundError as exc:
        return Result(task, dataset, FAIL, f"missing: {exc}")
    except Exception as exc:
        return Result(task, dataset, FAIL, f"{type(exc).__name__}: {exc}")
    counts = {"actual": actual, "expected": expected}
    if expected is None:
        return Result(
            task, dataset, OK, f"{actual:,} items{note and '  ' + note}", counts
        )
    if actual == expected:
        return Result(
            task, dataset, OK, f"{actual:,} == expected{note and '  ' + note}", counts
        )
    if abs(actual - expected) <= tolerance:
        return Result(
            task,
            dataset,
            WARN,
            f"{actual:,} vs expected {expected:,} (within tolerance {tolerance})",
            counts,
        )
    return Result(task, dataset, FAIL, f"{actual:,} but expected {expected:,}", counts)


def check_head_pose(root: Path) -> list[Result]:

    packed = root / "datasets/head_pose/packed"
    out = []
    complete = packed / "PACKED_DATASETS.complete.json"
    if not complete.exists():
        return [
            Result(
                "head_pose",
                "packed",
                FAIL,
                "PACKED_DATASETS.complete.json missing: the pack never finished",
            )
        ]
    archives = json.loads(complete.read_text())["archives"]
    for arc in archives:

        name = Path(arc["database"]).name
        expected = arc["record_count"]
        out.append(
            _check(
                "head_pose",
                f"packed/{name}",
                expected,
                lambda p=packed / name: _lmdb_image_entries(p),
                _lmdb_sample_decodes(packed / name) if (packed / name).exists() else "",
            )
        )
    for manifest in sorted((root / "datasets/head_pose/manifests").glob("*.jsonl")):
        out.append(
            _check(
                "head_pose",
                f"manifests/{manifest.name}",
                None,
                lambda m=manifest: _count_lines(m),
            )
        )
    return out


def check_landmark(root: Path) -> list[Result]:

    ann = root / "landmark_dataset/WFLW/WFLW_annotations"
    out = []
    pairs = [("train", 7_500), ("test", 2_500)]
    for split, expected in pairs:
        f = ann / f"list_98pt_rect_attr_train_test/list_98pt_rect_attr_{split}.txt"
        out.append(
            _check(
                "landmark", f"WFLW {split} list", expected, lambda p=f: _count_lines(p)
            )
        )

    subsets = sorted(
        f
        for f in (ann / "list_98pt_test").glob("*.txt")
        if f.name != "list_98pt_test.txt"
    )
    out.append(_check("landmark", "WFLW test subsets", 6, lambda: len(subsets)))
    out.append(
        _check(
            "landmark",
            "WFLW image dirs",
            None,
            lambda: sum(
                1 for _ in (root / "landmark_dataset/WFLW/WFLW_images").iterdir()
            ),
        )
    )
    return out


def check_parsing(root: Path) -> list[Result]:

    out = []
    lapa = root / "parsing_dataset/LaPa"
    for split, expected in (("train", 18_176), ("val", 2_000), ("test", 2_000)):
        for sub in ("images", "labels"):
            out.append(
                _check(
                    "parsing",
                    f"LaPa {split}/{sub}",
                    expected,
                    lambda p=lapa / split / sub: sum(1 for _ in p.iterdir()),
                )
            )
    celeba = root / "parsing_dataset/CelebAMask-HQ"
    out.append(
        _check(
            "parsing",
            "CelebAMask-HQ images",
            30_000,
            lambda: sum(1 for _ in (celeba / "CelebA-HQ-img").iterdir()),
        )
    )
    out.append(
        _check(
            "parsing",
            "CelebAMask-HQ mask groups",
            15,
            lambda: sum(
                1 for _ in (celeba / "CelebAMask-HQ-mask-anno").iterdir() if _.is_dir()
            ),
        )
    )
    return out


CHECKS = {
    "head_pose": check_head_pose,
    "landmark": check_landmark,
    "parsing": check_parsing,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument(
        "--task",
        choices=sorted(CHECKS),
        action="append",
        help="repeatable; default is every task",
    )
    ap.add_argument("--json", type=Path, help="also write the report here")
    ap.add_argument(
        "--strict", action="store_true", help="exit non-zero if anything is not OK"
    )
    args = ap.parse_args()

    tasks = args.task or sorted(CHECKS)
    results: list[Result] = []
    for task in tasks:
        results.extend(CHECKS[task](args.root))

    width = max(len(f"{r.task}/{r.dataset}") for r in results) + 2
    current = None
    for r in results:
        if r.task != current:
            current = r.task
            print(f"\n── {r.task} " + "─" * (60 - len(r.task)))
        mark = {OK: "ok  ", WARN: "warn", FAIL: "FAIL"}[r.status]
        print(f"  [{mark}] {r.dataset:<{width}} {r.detail}")

    counts = {s: sum(1 for r in results if r.status == s) for s in (OK, WARN, FAIL)}
    print(
        f"\n{counts[OK]} ok, {counts[WARN]} warn, {counts[FAIL]} fail "
        f"({len(results)} checks over {len(tasks)} task(s))"
    )

    if args.json:
        args.json.write_text(
            json.dumps([r.__dict__ for r in results], indent=2) + "\n", encoding="utf-8"
        )
        print(f"report written to {args.json}")

    if args.strict and (counts[FAIL] or counts[WARN]):
        return 1
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
