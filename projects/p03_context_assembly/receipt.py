"""The receipt: a structured record of every decision the assembler made.

This exists because of a specific on-call experience shape. A user reports that
the assistant "forgot" something it was told. The prompt is 8000 tokens of
concatenated string by the time it reaches the provider, the retriever logs say
the right document was retrieved, and nothing in between recorded that the
document was truncated to its first two sentences to make room for a chat
history that nobody needed. Without a receipt the only way to find that out is
to reconstruct the request by hand.

Every item that entered the assembler leaves it with a status, a token count
before and after, and a reason. The receipt is JSON-serialisable so it can be
attached to a trace span or written next to the request log.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

STATUS_INCLUDED = "included"
STATUS_COMPRESSED = "compressed"
STATUS_DROPPED = "dropped"
STATUS_DEDUPED = "deduplicated"


@dataclass
class ItemRecord:
    item_id: str
    section: str
    source: str
    status: str
    before_tokens: int
    after_tokens: int
    method: str = ""
    reason: str = ""

    @property
    def saved_tokens(self) -> int:
        return self.before_tokens - self.after_tokens


@dataclass
class SectionRecord:
    name: str
    priority: float
    strategy: str
    demand_tokens: int
    granted_tokens: int
    used_tokens: int
    items: List[ItemRecord] = field(default_factory=list)
    note: str = ""

    def count(self, status: str) -> int:
        return sum(1 for i in self.items if i.status == status)

    @property
    def retention(self) -> float:
        """Share of this section's offered tokens that survived into the prompt."""
        if self.demand_tokens == 0:
            return 1.0
        return self.used_tokens / self.demand_tokens


@dataclass
class Receipt:
    request_id: str
    query: str
    model_window: int
    reserve_tokens: int
    available_tokens: int
    prompt_tokens: int
    sections: List[SectionRecord] = field(default_factory=list)
    duplicates: List[Dict[str, Any]] = field(default_factory=list)
    order: List[str] = field(default_factory=list)

    @property
    def headroom(self) -> int:
        """Tokens left inside the budget. Never negative in a valid assembly."""
        return self.available_tokens - self.prompt_tokens

    @property
    def offered_tokens(self) -> int:
        return sum(s.demand_tokens for s in self.sections)

    def items(self, status: str = "") -> List[ItemRecord]:
        out: List[ItemRecord] = []
        for s in self.sections:
            out.extend(i for i in s.items if not status or i.status == status)
        return out

    def dedup_tokens_saved(self) -> int:
        return sum(int(d["tokens_saved"]) for d in self.duplicates)

    def compression_tokens_saved(self) -> int:
        return sum(i.saved_tokens for i in self.items(STATUS_COMPRESSED))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "model_window": self.model_window,
            "reserve_tokens": self.reserve_tokens,
            "available_tokens": self.available_tokens,
            "prompt_tokens": self.prompt_tokens,
            "headroom": self.headroom,
            "section_order": self.order,
            "sections": [asdict(s) for s in self.sections],
            "duplicates": self.duplicates,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def render(self) -> str:
        """Human-readable receipt. This is what gets pasted into an incident."""
        lines: List[str] = []
        lines.append(f"CONTEXT RECEIPT  request={self.request_id}")
        lines.append(f"  query: {self.query}")
        lines.append(
            f"  window={self.model_window}  reserve={self.reserve_tokens}  "
            f"budget={self.available_tokens}  used={self.prompt_tokens}  headroom={self.headroom}"
        )
        lines.append(f"  section order: {' -> '.join(self.order) if self.order else '(empty)'}")
        lines.append("")
        head = f"  {'section':<22}{'prio':>5}{'offered':>9}{'granted':>9}{'used':>7}{'keep':>7}  {'in/cmp/drop':<13} note"
        lines.append(head)
        lines.append("  " + "-" * (len(head) - 2))
        for s in self.sections:
            mix = f"{s.count(STATUS_INCLUDED)}/{s.count(STATUS_COMPRESSED)}/{s.count(STATUS_DROPPED)}"
            lines.append(
                f"  {s.name:<22}{s.priority:>5.1f}{s.demand_tokens:>9}{s.granted_tokens:>9}"
                f"{s.used_tokens:>7}{s.retention * 100:>6.0f}%  {mix:<13} {s.note}"
            )

        compressed = self.items(STATUS_COMPRESSED)
        if compressed:
            lines.append("")
            lines.append("  compressed:")
            for i in compressed:
                lines.append(
                    f"    {i.item_id:<26} {i.method:<13} {i.before_tokens:>5} -> {i.after_tokens:<5}"
                    f" ({i.after_tokens / max(1, i.before_tokens) * 100:.0f}% kept)"
                )

        dropped = self.items(STATUS_DROPPED)
        if dropped:
            lines.append("")
            lines.append("  dropped:")
            for i in dropped:
                lines.append(f"    {i.item_id:<26} {i.before_tokens:>5} tokens  reason: {i.reason}")

        if self.duplicates:
            lines.append("")
            lines.append("  deduplicated across sources:")
            for d in self.duplicates:
                lines.append(
                    f"    {d['dropped_item_id']} ({d['dropped_section']}) == "
                    f"{d['kept_item_id']} ({d['kept_section']})  containment={d['similarity']}"
                    f"  saved {d['tokens_saved']} tokens"
                )
        lines.append("")
        lines.append(
            f"  savings: {self.compression_tokens_saved()} tokens from compression, "
            f"{self.dedup_tokens_saved()} tokens from deduplication"
        )
        return "\n".join(lines)
