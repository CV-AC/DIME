from __future__ import annotations

import argparse
import json

from .auxiliary_data import build_300w_lp_index, build_lapa_index
from .config import load_config, resolve_path


def prepare(config: dict) -> dict:
    auxiliary = config.get("auxiliary_training", {})
    datasets = auxiliary.get("datasets", {})
    verify_decode = bool(auxiliary.get("verify_decode_during_prepare", True))
    results: dict[str, dict] = {}

    lapa = datasets.get("lapa", {})
    if bool(lapa.get("enabled", True)):
        root = resolve_path(lapa.get("root"), must_exist=True)
        index = resolve_path(lapa.get("index_file"))
        assert root is not None and index is not None
        results["lapa"] = build_lapa_index(
            root,
            index,
            split=str(lapa.get("split", "train")),
            exclude_eval_stem_overlap=bool(lapa.get("exclude_eval_stem_overlap", True)),
            verify_decode=verify_decode,
        )

    lp = datasets.get("300w_lp", {})
    if bool(lp.get("enabled", True)):
        root = resolve_path(lp.get("root"), must_exist=True)
        index = resolve_path(lp.get("index_file"))
        assert root is not None and index is not None
        results["300w_lp"] = build_300w_lp_index(
            root, index, verify_decode=verify_decode
        )

    if not results:
        raise ValueError("No auxiliary dataset is enabled in the config.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate LaPa/300W-LP and build compact training indexes."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
