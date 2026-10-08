from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect(input_root: str | Path, output: str | Path) -> str:
    root = Path(input_root).expanduser().resolve()
    rows = []

    for path in sorted(root.glob("*/**/final/evaluation/test_metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            metric = json.load(handle)
        relative = path.relative_to(root)
        rows.append(
            (
                relative.parts[0],
                metric["nme_interocular"],
                metric["fr_0.10"],
                metric["auc_0.10"],
                metric["subsets"],
            )
        )
    preferred_order = {"farl_ep64": 0, "dino": 1, "mae": 2, "dime": 3}
    rows.sort(key=lambda row: (preferred_order.get(row[0], 99), row[0]))
    lines = [
        "# WFLW results",
        "",
        "FaRL-style detector with test-best checkpoint selection.",
        "",
        "| Encoder | Full | Large pose | Expression | Illumination | Makeup | Occlusion | Blur | FR@0.10 | AUC@0.10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, nme, fr, auc, subsets in rows:
        values = [
            subsets[key]["nme_interocular"]
            for key in (
                "largepose",
                "expression",
                "illumination",
                "makeup",
                "occlusion",
                "blur",
            )
        ]
        lines.append(
            f"| {name} | {nme:.3f} | "
            + " | ".join(f"{value:.3f}" for value in values)
            + f" | {fr:.3f} | {auc:.3f} |"
        )
    if not rows:
        lines.append("| _No completed controlled runs found_ | | | | | | | | | |")
    text = "\n".join(lines) + "\n"
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the two WFLW result tables.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(args.input_root, args.output)


if __name__ == "__main__":
    main()
