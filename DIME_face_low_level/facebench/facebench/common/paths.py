from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parents[3])


def data_root() -> Path:
    return Path(os.environ.get("FACEBENCH_DATA_ROOT", DEFAULT_DATA_ROOT))


def dataset_path(*parts: str) -> str:

    return str(data_root().joinpath(*parts))


def resolve_data_path(path: Path) -> Path:
    parts = path.parts
    if (
        len(parts) > 1
        and parts[0] == ".."
        and parts[1] in {"datasets", "landmark_dataset", "parsing_dataset"}
    ):
        return data_root().joinpath(*parts[1:])
    return path


def bench_root() -> Path:

    return Path(os.environ.get("FACEBENCH_ROOT", Path(__file__).resolve().parents[2]))
