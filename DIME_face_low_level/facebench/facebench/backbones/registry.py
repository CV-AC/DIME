from __future__ import annotations

from typing import Callable

from .base import FeatureBackbone

_REGISTRY: dict[str, Callable[..., FeatureBackbone]] = {}


def register(name: str) -> Callable:

    def wrap(builder: Callable[..., FeatureBackbone]) -> Callable[..., FeatureBackbone]:
        key = name.lower()
        if key in _REGISTRY:
            raise KeyError(
                f"backbone '{name}' is already registered by "
                f"{_REGISTRY[key]!r}; pick a distinct name rather than "
                f"shadowing it, or two configs will silently differ"
            )
        _REGISTRY[key] = builder
        return builder

    return wrap


def build_backbone(name: str, **kwargs) -> FeatureBackbone:

    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown backbone '{name}'. Registered: "
            f"{sorted(_REGISTRY)}. Implement facebench.backbones.base."
            f"FeatureBackbone and decorate it with @register to add one."
        )
    backbone = _REGISTRY[key](**kwargs)
    if not isinstance(backbone, FeatureBackbone):
        raise TypeError(
            f"backbone '{name}' returned {type(backbone).__name__}, "
            f"which does not implement FeatureBackbone"
        )
    return backbone


def list_backbones() -> list[str]:
    return sorted(_REGISTRY)
