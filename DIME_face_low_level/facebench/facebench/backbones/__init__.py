from __future__ import annotations

import importlib
import logging

from .base import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, FeatureBackbone
from . import registry as _registry
from .registry import register

logger = logging.getLogger(__name__)


_ADAPTER_MODULES = ("timm_vit", "farl", "dime")

_loaded = False


def _load_adapters() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for name in _ADAPTER_MODULES:
        try:
            importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:

            logger.warning(
                "backbone adapter %r unavailable: %s: %s", name, type(exc).__name__, exc
            )


def build_backbone(name: str, **kwargs) -> FeatureBackbone:
    _load_adapters()
    return _registry.build_backbone(name, **kwargs)


def list_backbones() -> list[str]:
    _load_adapters()
    return _registry.list_backbones()


__all__ = [
    "FeatureBackbone",
    "build_backbone",
    "list_backbones",
    "register",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "CLIP_MEAN",
    "CLIP_STD",
]
