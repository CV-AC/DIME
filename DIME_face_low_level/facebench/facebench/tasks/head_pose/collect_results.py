from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .utils import write_json


METHOD_ORDER = {
    "repvgg_b1g2_head_pose": 0,
    "dino_vitb16_head_pose": 1,
    "mae_vitb16_head_pose": 2,
    "dime_full_head_pose": 3,
}


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _format(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    mean, std = _mean_std(values)
    return f"{mean:.3f} ± {std:.3f}"


def collect(input_root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(input_root).expanduser().resolve()
    rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in root.glob("*/seed_*/evaluation/*_metrics.json"):
        metric = json.loads(path.read_text(encoding="utf-8"))
        rows[str(metric["method"])][str(metric["dataset"])].append(metric)
    for method in rows:
        for split in rows[method]:
            rows[method][split].sort(key=lambda value: int(value["seed"]))

    summary: dict[str, Any] = {"input_root": str(root), "methods": {}}
    selections = sorted(
        {
            str(metric.get("selection", "unspecified"))
            for splits in rows.values()
            for metrics in splits.values()
            for metric in metrics
        }
    )
    summary["checkpoint_selection"] = selections
    lines = [
        "# Head-pose results",
        "",
        "Checkpoint selection: " + ", ".join(selections),
        "",
        "| Encoder run | Seeds | AFLW Yaw ↓ | Pitch ↓ | Roll ↓ | Mean ↓ | "
        "BIWI Yaw ↓ | Pitch ↓ | Roll ↓ | Mean ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(rows, key=lambda name: (METHOD_ORDER.get(name, 99), name))
    for method in ordered:
        aflw = rows[method].get("aflw2000", [])
        biwi = rows[method].get("biwi", [])
        seeds = sorted(
            set(int(value["seed"]) for value in aflw)
            & set(int(value["seed"]) for value in biwi)
        )
        aflw = [value for value in aflw if int(value["seed"]) in seeds]
        biwi = [value for value in biwi if int(value["seed"]) in seeds]
        summary["methods"][method] = {
            "seeds": seeds,
            "aflw2000": aflw,
            "biwi": biwi,
        }
        if not seeds:
            lines.append(f"| {method} | 0 | _incomplete_ | | | | | | | |")
            continue
        cells = []
        for values in (aflw, biwi):
            for key in ("yaw_mae", "pitch_mae", "roll_mae", "mean_mae"):
                cells.append(_format([float(value[key]) for value in values]))
        lines.append(
            f"| {method} | {','.join(str(seed) for seed in seeds)} | "
            + " | ".join(cells)
            + " |"
        )
    if not ordered:
        lines.append("| _No completed runs found_ | 0 | | | | | | | | |")

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_path.with_suffix(".json"), summary)
    print(output_path.read_text(encoding="utf-8"))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect controlled head-pose results; aggregate explicitly repeated seeds."
    )
    parser.add_argument("--input-root", default="outputs/head_pose")
    parser.add_argument("--output", default="outputs/head_pose/results.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(args.input_root, args.output)


if __name__ == "__main__":
    main()
