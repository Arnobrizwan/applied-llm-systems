"""Tool discovery and versioning.

A registry rather than a module-level dict because three things need to be true
at once and a dict gives you none of them: the agent needs to filter the catalog
down to a task-relevant subset, operations needs to run two versions of a tool
side by side during a migration, and the prompt needs a rendering of the schemas
that is stable across runs.

Versioning is deliberately explicit. `get("search_docs")` returns the highest
version, `get("search_docs", "1.0.0")` pins. The alternative, a single mutable
entry per name, means a tool change is instantly global, which is exactly the
change you want to be able to roll out to one agent at a time.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import ToolSpec

__all__ = ["ToolRegistry", "ToolNotFound"]


class ToolNotFound(KeyError):
    """Raised on an explicit lookup miss. The agent loop never sees this: an
    unknown tool name coming from a model is data, not an exception, and the
    sandbox turns it into a structured error instead."""


def _version_key(version: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for piece in version.split("."):
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


class ToolRegistry:
    """Name plus version to ToolSpec, with tag filtering and prompt rendering."""

    def __init__(self) -> None:
        self._by_name: Dict[str, Dict[str, ToolSpec]] = {}

    # -- registration ----------------------------------------------------
    def register(self, func_or_spec: Any) -> ToolSpec:
        """Register a @tool-decorated function or a ToolSpec directly."""
        spec = func_or_spec if isinstance(func_or_spec, ToolSpec) else getattr(func_or_spec, "tool_spec", None)
        if spec is None:
            raise TypeError(f"{func_or_spec!r} is not decorated with @tool")
        versions = self._by_name.setdefault(spec.name, {})
        if spec.version in versions:
            raise ValueError(f"{spec.qualified_name} is already registered")
        versions[spec.version] = spec
        return spec

    def register_all(self, funcs: Iterable[Any]) -> List[ToolSpec]:
        return [self.register(f) for f in funcs]

    # -- lookup ----------------------------------------------------------
    def get(self, name: str, version: Optional[str] = None) -> ToolSpec:
        versions = self._by_name.get(name)
        if not versions:
            raise ToolNotFound(name)
        if version is None:
            latest = max(versions, key=_version_key)
            return versions[latest]
        if version not in versions:
            raise ToolNotFound(f"{name}@{version}")
        return versions[version]

    def find(self, name: str, version: Optional[str] = None) -> Optional[ToolSpec]:
        """Lookup that returns None instead of raising, for the agent path."""
        try:
            return self.get(name, version)
        except ToolNotFound:
            return None

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_name.values())

    # -- discovery -------------------------------------------------------
    def list(self, *, tags: Sequence[str] = (), all_versions: bool = False) -> List[ToolSpec]:
        """Catalog, newest version per name unless `all_versions`.

        Sorted by name so the rendered prompt is byte-identical between runs.
        An unstable tool order silently invalidates prompt caching and makes two
        otherwise identical traces diff against each other.
        """
        specs: List[ToolSpec] = []
        for name, versions in self._by_name.items():
            if all_versions:
                specs.extend(versions[v] for v in sorted(versions, key=_version_key))
            else:
                specs.append(self.get(name))
        if tags:
            wanted = set(tags)
            specs = [s for s in specs if wanted & set(s.tags)]
        return sorted(specs, key=lambda s: (s.name, _version_key(s.version)))

    def tags(self) -> List[str]:
        return sorted({t for spec in self.list() for t in spec.tags})

    def names(self, *, tags: Sequence[str] = ()) -> List[str]:
        return [s.name for s in self.list(tags=tags)]

    # -- prompt rendering -------------------------------------------------
    def render_prompt(self, *, tags: Sequence[str] = (), indent: Optional[int] = None) -> str:
        """Render the catalog as the tool block that goes into the system prompt.

        JSON rather than prose: the schema is what the model has to satisfy, and
        restating it in English adds tokens and a second source of truth that can
        contradict the first.
        """
        specs = self.list(tags=tags)
        lines = [f"You can call {len(specs)} tool(s). Each call must match its parameter schema."]
        for spec in specs:
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  parameters: {json.dumps(spec.parameters, sort_keys=True, indent=indent)}")
        return "\n".join(lines)
