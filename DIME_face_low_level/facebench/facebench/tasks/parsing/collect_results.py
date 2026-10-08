from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def _last_jsonl_row(path: Path) -> dict[str, Any]:
    last: dict[str, Any] = {}
    if not path.is_file():
        return last
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path} contains a non-object JSON row.")
                last = value
    return last


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect parsing test_metrics.json files into one CSV."
    )
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output", default="outputs/results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    rows = []
    for metrics_path in sorted(root.rglob("test_metrics.json")):
        metrics = _read_json(metrics_path)
        run_dir = metrics_path.parent
        metadata_path = run_dir / "run_metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        training = _last_jsonl_row(run_dir / "metrics.jsonl")
        per_class = metrics.get("per_class", {})
        rows.append(
            {
                "dataset": metadata.get("dataset", run_dir.parent.name),
                "backbone": metadata.get("backbone", run_dir.name),
                "checkpoint_sha256": metadata.get("backbone_checkpoint_sha256", ""),
                "best_epoch": training.get("best_epoch", ""),
                "best_selection_foreground_mean_f1": training.get(
                    "best_foreground_mean_f1", ""
                ),
                "foreground_mean_f1": metrics.get("foreground_mean_f1"),
                "foreground_mean_iou": metrics.get("foreground_mean_iou"),
                "pixel_accuracy": metrics.get("pixel_accuracy"),
                "per_class_f1": json.dumps(
                    {name: values.get("f1") for name, values in per_class.items()},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "metrics_file": str(metrics_path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No test_metrics.json files found under {root}.")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    print(f"Wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
