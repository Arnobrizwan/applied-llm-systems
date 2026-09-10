"""The three things this system stores, and the accounting fields they carry.

Every record carries `access_count` and `last_used_turn` because eviction is not
a size problem, it is a value problem: the cheapest thing to throw away is the
record nobody has looked at in twenty turns, not the oldest one. Every record
also carries provenance back to the turn it came from, because a memory system
that cannot answer "where did you get that" is a memory system nobody will trust
with anything that matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from llmkit import count_tokens


@dataclass
class Turn:
    """One conversational turn, numbered from 1 the way a human counts them."""

    index: int
    role: str
    text: str

    @property
    def tokens(self) -> int:
        return count_tokens(self.rendered)

    @property
    def rendered(self) -> str:
        return f"[turn {self.index}] {self.role}: {self.text}"


@dataclass
class Episode:
    """A compressed summary of a run of turns that left working memory."""

    id: str
    text: str
    first_turn: int
    last_turn: int
    source_turns: List[int]
    original_tokens: int
    importance: float = 0.5
    access_count: int = 0
    last_used_turn: int = 0

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.tokens / self.original_tokens

    def touch(self, turn: int) -> None:
        self.access_count += 1
        self.last_used_turn = turn

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "turns": [self.first_turn, self.last_turn],
            "source_turns": list(self.source_turns),
            "tokens": self.tokens,
            "original_tokens": self.original_tokens,
            "compression_ratio": round(self.compression_ratio, 3),
            "importance": self.importance,
            "access_count": self.access_count,
        }


@dataclass
class Fact:
    """A subject/predicate/object assertion with provenance.

    The triple shape is what makes contradiction handling possible at all. Two
    free-text memories saying different things about the same database version
    look like two unrelated strings; two facts sharing a (subject, predicate) key
    are visibly a conflict, and the newer one can supersede the older one instead
    of both sitting in the prompt daring the model to pick.
    """

    id: str
    subject: str
    predicate: str
    object: str
    turn_index: int
    confidence: float = 0.7
    pinned: bool = False
    superseded_by: Optional[str] = None
    supersedes: Optional[str] = None
    access_count: int = 0
    last_used_turn: int = 0
    evidence: str = ""

    @property
    def key(self) -> Tuple[str, str]:
        return (self.subject, self.predicate)

    @property
    def active(self) -> bool:
        return self.superseded_by is None

    @property
    def text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    @property
    def rendered(self) -> str:
        pin = " [pinned]" if self.pinned else ""
        return f"- {self.text} (from turn {self.turn_index}){pin}"

    @property
    def tokens(self) -> int:
        return count_tokens(self.rendered)

    def touch(self, turn: int) -> None:
        self.access_count += 1
        self.last_used_turn = turn

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "turn_index": self.turn_index,
            "confidence": self.confidence,
            "pinned": self.pinned,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "access_count": self.access_count,
            "evidence": self.evidence,
        }


@dataclass
class MemoryEvent:
    """One thing the memory system did, for the demo timeline and the tests."""

    turn: int
    kind: str  # fact | supersede | episode | evict
    detail: str
    tokens: int = 0


@dataclass
class RecallResult:
    """What `recall` assembled, and the parts it was assembled from."""

    text: str
    facts: List[Fact] = field(default_factory=list)
    episodes: List[Episode] = field(default_factory=list)
    turns: List[Turn] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)

    def contains(self, needle: str) -> bool:
        return needle.lower() in self.text.lower()
