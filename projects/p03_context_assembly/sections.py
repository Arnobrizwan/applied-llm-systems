"""What a caller declares before any budgeting happens.

A request does not hand the assembler a string. It hands it *sections*: named
groups of items with a priority, a guaranteed floor, a ceiling expressed as a
share of the window, and the compression strategy that is acceptable for that
kind of content.

The reason for making this declarative rather than letting each caller trim its
own strings: trimming at the call site means every source is optimising in
isolation, and the only place the total is known is the provider's 400 response.
Declaring intent and centralising the arithmetic is what makes the overflow
impossible rather than unlikely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from llmkit import count_tokens


class Compression(str, Enum):
    """What may be done to a section when it does not fit.

    The strategy is a property of the *content*, not of the budget. A system
    prompt and a legal disclaimer must survive whole or not at all; a chat
    history can lose its oldest turns; retrieved documents can be reduced to the
    sentences that actually bear on the question. Encoding that at declaration
    time is what stops the allocator from silently truncating an instruction
    block and turning a formatting bug into a behaviour bug.
    """

    KEEP_WHOLE = "keep_whole"
    TRUNCATE_TAIL = "truncate_tail"  # keep the head, cut the end
    TRUNCATE_HEAD = "truncate_head"  # keep the end, cut the head (recent history)
    EXTRACTIVE = "extractive"  # keep the sentences that match the query
    SUMMARIZE = "summarize"  # hand to the model, then hard-cap the result
    DROP = "drop"  # never compress, remove instead


@dataclass
class ContextItem:
    """One indivisible candidate piece of context.

    `value` is the caller's estimate of how much this item is worth for this
    request: a retrieval score, a memory confidence, a recency weight. The
    assembler does not invent it, because only the caller knows what its
    retriever's scores mean.
    """

    id: str
    text: str
    source: str
    value: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    _tokens: Optional[int] = field(default=None, repr=False, compare=False)

    @property
    def tokens(self) -> int:
        # Cached because the allocator asks for the same size several times per
        # request and token counting is the hot loop in this service.
        if self._tokens is None:
            self._tokens = count_tokens(self.text)
        return self._tokens


@dataclass
class SectionSpec:
    """Policy for one named region of the prompt."""

    name: str
    priority: float
    min_tokens: int = 0
    max_share: float = 1.0
    strategy: Compression = Compression.EXTRACTIVE
    position: str = "flow"  # "first" | "flow" | "last"
    header: str = ""

    def __post_init__(self) -> None:
        if self.priority <= 0:
            raise ValueError(f"{self.name}: priority must be positive")
        if not 0.0 < self.max_share <= 1.0:
            raise ValueError(f"{self.name}: max_share must be in (0, 1]")
        if self.min_tokens < 0:
            raise ValueError(f"{self.name}: min_tokens cannot be negative")
        if self.position not in ("first", "flow", "last"):
            raise ValueError(f"{self.name}: position must be first, flow or last")

    @property
    def header_tokens(self) -> int:
        return count_tokens(self.header) if self.header else 0


@dataclass
class Section:
    """A spec bound to the items a caller actually produced this request."""

    spec: SectionSpec
    items: List[ContextItem] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def item_tokens(self) -> int:
        return sum(i.tokens for i in self.items)

    @property
    def demand(self) -> int:
        """Tokens this section would consume if nothing were cut."""
        if not self.items:
            return 0
        return self.item_tokens + self.spec.header_tokens

    @property
    def value_density(self) -> float:
        """Token-weighted mean item value, used as the marginal-value signal.

        Token-weighted rather than a plain mean so a section of one short
        high-scoring snippet is not treated as more valuable per token than a
        section of ten equally-scoring ones.
        """
        total = self.item_tokens
        if not total:
            return 0.0
        return sum(i.value * i.tokens for i in self.items) / total


@dataclass
class ContextRequest:
    """Everything the assembler needs for one call."""

    query: str
    sections: List[Section]
    model_window: int
    reserve_tokens: int
    request_id: str = "req-1"

    def __post_init__(self) -> None:
        if self.model_window <= 0:
            raise ValueError("model_window must be positive")
        if self.reserve_tokens < 0:
            raise ValueError("reserve_tokens cannot be negative")
        if self.reserve_tokens >= self.model_window:
            raise ValueError(
                "reserve_tokens must leave room for a prompt: "
                f"{self.reserve_tokens} >= {self.model_window}"
            )
        names = [s.name for s in self.sections]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate section names: {names}")

    @property
    def available_tokens(self) -> int:
        """The real budget: the window minus the completion reserve.

        This subtraction is the whole point of the service. A model window is
        shared between the prompt and the completion, and a prompt that fills
        the window leaves the model no room to answer. The failure is not a
        clean error either: providers either reject the call or truncate the
        completion mid-sentence, which downstream JSON parsing then reports as a
        model quality problem.
        """
        return self.model_window - self.reserve_tokens

    def section(self, name: str) -> Section:
        for s in self.sections:
            if s.name == name:
                return s
        raise KeyError(name)
