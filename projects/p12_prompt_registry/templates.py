"""Prompt templates whose version id is a hash of their own content.

Two rules here, and both exist because of the same production failure: nobody can
tell which prompt produced a bad answer.

  1. The version id is the content hash. An edited prompt cannot keep its old id,
     because the id is derived from the bytes rather than assigned alongside them.
     A hand-assigned "v3" that someone tweaked in place is worse than no version
     at all: it makes the log look trustworthy while it lies.
  2. Rendering fails loudly. A missing variable and an unexpected variable are
     both errors, not warnings and not silent no-ops.

The second rule is the one people push back on. `str.format` raising on a missing
key is fine; the argument is about *extra* variables. Passing a variable a
template does not use is almost always a rename that only got applied on one side,
and the symptom is a prompt that silently stops including the customer's name.
Failing at render time turns a quality regression into a stack trace.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from string import Formatter
from typing import Any, Dict, List, Mapping, Optional, Sequence

VERSION_PREFIX = "v"
VERSION_LENGTH = 12


class TemplateError(ValueError):
    """Base class for template problems that must not be swallowed."""


class MissingVariable(TemplateError):
    pass


class UnexpectedVariable(TemplateError):
    pass


def content_hash(name: str, template: str, variables: Sequence[str], config: Mapping[str, Any]) -> str:
    """Stable id over everything that changes behaviour.

    The config is inside the hash on purpose. A prompt at temperature 0.2 and the
    same prompt at temperature 0.9 are different systems, and an experiment that
    treats them as one version cannot explain its own results.
    """
    payload = json.dumps(
        {
            "name": name,
            "template": template,
            "variables": sorted(variables),
            "config": {k: config[k] for k in sorted(config)},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:VERSION_LENGTH]
    return f"{VERSION_PREFIX}{digest}"


def declared_placeholders(template: str) -> List[str]:
    """Every `{name}` the template actually uses, in first-seen order."""
    seen: List[str] = []
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        root = re.split(r"[.\[]", field_name)[0]
        if root and root not in seen:
            seen.append(root)
    return seen


@dataclass(frozen=True)
class PromptVersion:
    """An immutable prompt plus its model config.

    Frozen because the id is a hash of the contents: a mutable version object
    would let a caller change the text and keep the id, which is exactly the
    failure the hash exists to prevent.
    """

    name: str
    template: str
    variables: tuple
    config: tuple  # sorted (key, value) pairs, so the dataclass stays hashable
    author: str = "unknown"
    notes: str = ""

    @property
    def version_id(self) -> str:
        return content_hash(self.name, self.template, self.variables, dict(self.config))

    @property
    def config_dict(self) -> Dict[str, Any]:
        return dict(self.config)

    def render(self, variables: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> str:
        """Fill the template. Raises on a missing or unexpected variable."""
        supplied: Dict[str, Any] = dict(variables or {})
        supplied.update(kwargs)
        declared = set(self.variables)
        given = set(supplied)

        missing = declared - given
        if missing:
            raise MissingVariable(
                f"{self.name} {self.version_id} needs {sorted(missing)}; got {sorted(given) or 'nothing'}"
            )
        unexpected = given - declared
        if unexpected:
            raise UnexpectedVariable(
                f"{self.name} {self.version_id} does not use {sorted(unexpected)}; "
                f"it declares {sorted(declared)}"
            )
        try:
            return self.template.format(**supplied)
        except (KeyError, IndexError) as exc:  # pragma: no cover - guarded by the checks above
            raise MissingVariable(f"{self.name} {self.version_id}: {exc}") from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "name": self.name,
            "template": self.template,
            "variables": list(self.variables),
            "config": self.config_dict,
            "author": self.author,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromptVersion":
        return cls(
            name=payload["name"],
            template=payload["template"],
            variables=tuple(payload["variables"]),
            config=tuple(sorted(dict(payload.get("config") or {}).items())),
            author=payload.get("author", "unknown"),
            notes=payload.get("notes", ""),
        )


def build_version(
    name: str,
    template: str,
    variables: Optional[Sequence[str]] = None,
    config: Optional[Mapping[str, Any]] = None,
    author: str = "unknown",
    notes: str = "",
) -> PromptVersion:
    """Create a version, checking the declared variables against the template.

    Declaring variables explicitly rather than only inferring them is deliberate.
    Inference alone cannot catch the case that actually bites: a template that
    stopped using `{customer_name}` after an edit, while every caller still passes
    it. Declared-and-checked turns that into an error at registration time.
    """
    used = declared_placeholders(template)
    declared = list(variables) if variables is not None else used
    missing_in_template = [v for v in declared if v not in used]
    if missing_in_template:
        raise TemplateError(
            f"{name}: declared variables {missing_in_template} never appear in the template"
        )
    undeclared = [v for v in used if v not in declared]
    if undeclared:
        raise TemplateError(f"{name}: template uses undeclared variables {undeclared}")
    return PromptVersion(
        name=name,
        template=template,
        variables=tuple(declared),
        config=tuple(sorted(dict(config or {}).items())),
        author=author,
        notes=notes,
    )
