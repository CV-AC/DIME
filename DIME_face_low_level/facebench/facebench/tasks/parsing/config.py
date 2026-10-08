from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

from facebench.common.paths import resolve_data_path


PARSING_ROOT = Path(
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
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    parent = config.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        config = _deep_merge(load_config(parent_path), config)
    config["_config_path"] = str(config_path)
    config["_parsing_root"] = str(PARSING_ROOT)
    return config


def apply_overrides(
    config: dict[str, Any], overrides: Iterable[str] | None
) -> dict[str, Any]:

    output = deepcopy(config)
    for expression in overrides or ():
        if "=" not in expression:
            raise ValueError(f"Override must be KEY=VALUE, got {expression!r}")
        dotted_key, raw_value = expression.split("=", 1)
        parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
        if not parts:
            raise ValueError(f"Override has an empty key: {expression!r}")
        value = yaml.safe_load(raw_value)
        cursor: dict[str, Any] = output
        for part in parts[:-1]:
            current = cursor.get(part)
            if current is None:
                current = {}
                cursor[part] = current
            if not isinstance(current, dict):
                raise ValueError(
                    f"Cannot set {dotted_key!r}: {part!r} is not a mapping."
                )
            cursor = current
        cursor[parts[-1]] = value
    return output


def resolve_path(value: str | Path | None, *, must_exist: bool = False) -> Path | None:
    if value is None or not str(value).strip():
        if must_exist:
            raise ValueError("A required path is empty in the YAML configuration.")
        return None
    path = resolve_data_path(Path(os.path.expandvars(str(value))).expanduser())
    if not path.is_absolute():

        if path.parts and path.parts[0] == "configs":
            path = CONFIG_ROOT.parent / path
        else:
            path = PARSING_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def experiment_output_dir(config: dict[str, Any]) -> Path:
    path = resolve_path(config["experiment"]["output_dir"])
    assert path is not None
    return path


def public_config(config: dict[str, Any]) -> dict[str, Any]:

    output = deepcopy(config)
    for key in tuple(output):
        if key.startswith("_"):
            output.pop(key)
    wandb_config = output.get("wandb")
    if isinstance(wandb_config, dict) and wandb_config.get("api_key"):
        wandb_config["api_key"] = "***REDACTED***"
    return output
