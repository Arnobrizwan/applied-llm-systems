"""Adapter discovery for the hosted demos.

Adapters are plain modules so that adding a system to the site is one file, and
so a broken adapter degrades to one broken page instead of a 500 on every route.
"""
from __future__ import annotations

import importlib
import os
import pkgutil
from typing import Any, Dict, List, Optional

_ADAPTER_PKG = "web.adapters"


def _load() -> List[Any]:
    mods = []
    pkg = importlib.import_module(_ADAPTER_PKG)
    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.name.startswith("a"):
            continue
        try:
            mods.append(importlib.import_module(f"{_ADAPTER_PKG}.{info.name}"))
        except Exception as exc:  # a broken adapter must not take the site down
            print(f"adapter {info.name} failed to import: {exc}")
    mods.sort(key=lambda m: getattr(m, "NUMBER", 99))
    return mods


_CACHE: Optional[List[Any]] = None


def all_systems() -> List[Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load()
    return _CACHE


def by_slug(slug: str) -> Optional[Any]:
    for m in all_systems():
        if getattr(m, "SLUG", None) == slug:
            return m
    return None


def meta(mod: Any) -> Dict[str, Any]:
    return {
        "number": getattr(mod, "NUMBER", 0),
        "slug": getattr(mod, "SLUG", ""),
        "title": getattr(mod, "TITLE", ""),
        "tagline": getattr(mod, "TAGLINE", ""),
        "what": getattr(mod, "WHAT_IT_DOES", ""),
        "input_label": getattr(mod, "INPUT_LABEL", None),
        "placeholder": getattr(mod, "PLACEHOLDER", ""),
        "examples": list(getattr(mod, "EXAMPLES", []) or []),
        "source": getattr(mod, "SOURCE", ""),
    }
