from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from facebench.common.paths import resolve_data_path


HEAD_POSE_ROOT = Path(
    os.environ.get("FACEBENCH_ROOT", Path(__file__).resolve().parents[3])
)


CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    parent = config.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        config = _deep_merge(load_config(parent_path), config)
    config["_config_path"] = str(config_path)
    config["_head_pose_root"] = str(HEAD_POSE_ROOT)
    return config


def resolve_path(value: str | Path | None, *, must_exist: bool = False) -> Path | None:
    if value is None or str(value).strip() == "":
        if must_exist:
            raise ValueError(
                "A required path is empty. Fill the corresponding YAML field."
            )
        return None
    raw_value = str(value).strip()
    expanded = os.path.expandvars(raw_value)
    unresolved = re.findall(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*", expanded)
    if unresolved:
        raise ValueError(
            "Unresolved environment variable(s) in path "
            f"{raw_value!r}: {', '.join(unresolved)}"
        )
    path = resolve_data_path(Path(expanded).expanduser())
    if not path.is_absolute():

        if path.parts and path.parts[0] in ("configs", "scripts"):
            path = CONFIG_ROOT.parent / path
        else:
            path = HEAD_POSE_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(config)
    for key in list(output):
        if key.startswith("_"):
            output.pop(key)
    wandb = output.get("wandb")
    if isinstance(wandb, dict) and str(wandb.get("api_key", "")).strip():
        wandb["api_key"] = "<redacted>"
    return output


def experiment_output_dir(config: dict[str, Any], seed: int | None = None) -> Path:

    base = resolve_path(config["experiment"]["output_dir"])
    assert base is not None
    resolved_seed = int(config.get("seed", 0) if seed is None else seed)
    return base / f"seed_{resolved_seed:03d}"
