"""Core data types shared by all 15 systems.

Deliberately dependency-free dataclasses: every project in this repo has to run
on a stock Python install with no wheels to build and no API key to buy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """One turn in a chat exchange."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


def assistant(content: str) -> Message:
    return Message("assistant", content)


@dataclass
class LLMResponse:
    """A single completion plus the accounting metadata production needs."""

    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    finish_reason: str = "stop"
    raw: Optional[Dict[str, Any]] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Document:
    """A source document before chunking."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit of text with a stable, citable id."""

    id: str
    doc_id: str
    text: str
    ordinal: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        title = self.metadata.get("title") or self.doc_id
        return f"{title}#{self.ordinal}"


@dataclass
class ScoredChunk:
    """A chunk with the score and the retriever that produced it."""

    chunk: Chunk
    score: float
    source: str = "vector"
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class Usage:
    """Rolled-up usage for a request, tenant or run."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, resp: LLMResponse, cost_usd: float = 0.0) -> "Usage":
        self.calls += 1
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens
        self.cost_usd += cost_usd
        return self

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
