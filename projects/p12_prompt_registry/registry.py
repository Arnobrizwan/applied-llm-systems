"""The registry: versions, labels, an audit trail and one-call rollback.

A label (dev, staging, prod) is a pointer to a version id. That single choice is
what makes promotion and rollback cheap: promoting staging to prod moves a
pointer, and rolling back moves it to the id it had before. Neither operation
copies a prompt, so the two environments cannot drift into near-identical text
that differs by a trailing space nobody can see.

Every label move is appended to an audit trail with who, when, from and to.
Rollback reads that trail rather than a separately maintained "previous" field,
because a previous-pointer is a second source of truth that goes stale the moment
someone moves a label twice.

Storage is a JSON file. Not because JSON is the right database, but because the
interface is small and explicit: `load`, `save`, and an on-disk format a human
can diff during an incident.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .templates import PromptVersion, build_version

DEFAULT_ENVIRONMENTS = ("dev", "staging", "prod")


class RegistryError(RuntimeError):
    pass


class UnknownVersion(RegistryError):
    pass


class NothingToRollBack(RegistryError):
    pass


@dataclass
class AuditEntry:
    """One label move. The only record of intent this system keeps."""

    prompt: str
    label: str
    from_version: Optional[str]
    to_version: str
    actor: str
    reason: str
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def row(self) -> str:
        when = time.strftime("%H:%M:%S", time.localtime(self.at))
        origin = self.from_version or "(unset)"
        return f"  {when}  {self.prompt}/{self.label:<8} {origin:<15} -> {self.to_version:<15} by {self.actor:<10} {self.reason}"


class PromptRegistry:
    """Content-addressed prompt versions with environment labels."""

    def __init__(self, path: Optional[str] = None, environments: Sequence[str] = DEFAULT_ENVIRONMENTS):
        self.path = path
        self.environments = tuple(environments)
        self.versions: Dict[str, PromptVersion] = {}
        self.labels: Dict[str, Dict[str, str]] = {}  # prompt -> label -> version id
        self.audit: List[AuditEntry] = []

    # -- versions --------------------------------------------------------
    def register(
        self,
        name: str,
        template: str,
        variables: Optional[Sequence[str]] = None,
        config: Optional[Mapping[str, Any]] = None,
        author: str = "unknown",
        notes: str = "",
    ) -> PromptVersion:
        """Register a prompt. Registering identical content twice is a no-op.

        Idempotent because the id is the content hash: re-registering the same
        text cannot create a second version, which means a deploy that runs the
        registration code on every boot does not fill the registry with
        duplicates of the same prompt.
        """
        version = build_version(name, template, variables, config, author=author, notes=notes)
        existing = self.versions.get(version.version_id)
        if existing is not None:
            return existing
        self.versions[version.version_id] = version
        self.labels.setdefault(name, {})
        return version

    def get(self, version_id: str) -> PromptVersion:
        try:
            return self.versions[version_id]
        except KeyError as exc:
            raise UnknownVersion(f"no version {version_id!r} in the registry") from exc

    def versions_of(self, name: str) -> List[PromptVersion]:
        return [v for v in self.versions.values() if v.name == name]

    # -- labels ----------------------------------------------------------
    def set_label(self, name: str, label: str, version_id: str, actor: str, reason: str = "") -> AuditEntry:
        """Point a label at a version and record who did it and why."""
        if label not in self.environments:
            raise RegistryError(f"unknown environment {label!r}; known: {list(self.environments)}")
        version = self.get(version_id)
        if version.name != name:
            raise RegistryError(f"version {version_id} belongs to prompt {version.name!r}, not {name!r}")
        current = self.labels.setdefault(name, {}).get(label)
        self.labels[name][label] = version_id
        entry = AuditEntry(
            prompt=name, label=label, from_version=current, to_version=version_id, actor=actor, reason=reason
        )
        self.audit.append(entry)
        return entry

    def resolve(self, name: str, label: str) -> PromptVersion:
        """The version a label currently points at."""
        version_id = self.labels.get(name, {}).get(label)
        if version_id is None:
            raise RegistryError(f"{name!r} has no version labelled {label!r}")
        return self.get(version_id)

    def label_of(self, name: str, label: str) -> Optional[str]:
        return self.labels.get(name, {}).get(label)

    def history(self, name: str, label: Optional[str] = None) -> List[AuditEntry]:
        return [e for e in self.audit if e.prompt == name and (label is None or e.label == label)]

    def rollback(self, name: str, label: str, actor: str, reason: str = "rollback") -> AuditEntry:
        """Move a label back to the version it pointed at before the last move.

        One call, because the moment you need it is the moment nobody wants to
        look up a version id. It reads the audit trail, so rolling back twice
        walks back two steps rather than oscillating between two versions.
        """
        entries = self.history(name, label)
        if not entries:
            raise NothingToRollBack(f"{name}/{label} has never been set")
        previous = entries[-1].from_version
        if previous is None:
            raise NothingToRollBack(
                f"{name}/{label} has only ever pointed at one version; there is nothing behind it"
            )
        return self.set_label(name, label, previous, actor=actor, reason=reason)

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "environments": list(self.environments),
            "versions": [v.to_dict() for v in self.versions.values()],
            "labels": self.labels,
            "audit": [e.to_dict() for e in self.audit],
        }

    def save(self, path: Optional[str] = None) -> str:
        target = path or self.path
        if not target:
            raise RegistryError("no path given and the registry was created without one")
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return target

    @classmethod
    def load(cls, path: str) -> "PromptRegistry":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        registry = cls(path=path, environments=payload.get("environments", DEFAULT_ENVIRONMENTS))
        for row in payload.get("versions", []):
            version = PromptVersion.from_dict(row)
            stored_id = row.get("version_id")
            if stored_id and stored_id != version.version_id:
                # The file was edited by hand and the text no longer matches the
                # id it claims. Refusing to load is the point of content
                # addressing: a registry that silently accepts this is a registry
                # whose ids mean nothing.
                raise RegistryError(
                    f"tampered registry: version {stored_id} hashes to {version.version_id}"
                )
            registry.versions[version.version_id] = version
        registry.labels = {k: dict(v) for k, v in payload.get("labels", {}).items()}
        registry.audit = [AuditEntry(**row) for row in payload.get("audit", [])]
        return registry
