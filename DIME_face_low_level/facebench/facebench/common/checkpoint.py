from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


_WRAPPER_PREFIXES = ("module.", "_orig_mod.", "student.", "model.")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_module(module: nn.Module) -> str:

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:

    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(checkpoint).__name__}")
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return value
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint
    raise KeyError("no model / model_state_dict / state_dict found in checkpoint")


def strip_prefixes(
    state: dict[str, torch.Tensor], prefixes: Iterable[str] = _WRAPPER_PREFIXES
) -> dict[str, torch.Tensor]:

    prefixes = tuple(prefixes)
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        if new_key in out:
            raise KeyError(
                f"key collision after prefix cleanup: {new_key}. Two "
                f"different tensors would map to one name; strip fewer "
                f"prefixes rather than losing one of them"
            )
        out[new_key] = value
    return out


def checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:

    value = checkpoint.get("config", {})
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return {}


def load_strict(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    allow_missing: Iterable[str] = (),
    allow_unexpected: Iterable[str] = (),
) -> None:

    allow_missing, allow_unexpected = tuple(allow_missing), tuple(allow_unexpected)
    result = module.load_state_dict(state, strict=False)
    missing = [
        k
        for k in result.missing_keys
        if not k.startswith(allow_missing) and k not in allow_missing
    ]
    unexpected = [
        k
        for k in result.unexpected_keys
        if not k.startswith(allow_unexpected) and k not in allow_unexpected
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model.\n"
            f"  missing ({len(missing)}): {missing[:10]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:10]}\n"
            f"If these are genuinely expected, list them in allow_missing / "
            f"allow_unexpected so the exception becomes a documented decision."
        )


def resize_position_embedding(
    position: torch.Tensor, height: int, width: int, *, prefix_tokens: int
) -> torch.Tensor:

    prefix = position[:, :prefix_tokens]
    spatial = position[:, prefix_tokens:]
    old = int(math.sqrt(spatial.shape[1]))
    if old * old != spatial.shape[1]:
        raise ValueError(
            f"position embedding token count {spatial.shape[1]} is not square"
        )
    if (old, old) == (height, width):
        return position
    spatial = spatial.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    spatial = F.interpolate(
        spatial.float(), size=(height, width), mode="bicubic", align_corners=False
    ).to(position.dtype)
    spatial = spatial.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    return torch.cat([prefix, spatial], dim=1)
