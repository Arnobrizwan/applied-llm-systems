"""The thing this service replaces, measured honestly.

The naive baseline is what almost every first version of an LLM feature does:
format each source into a string, join them in the order the code happens to run
in, send it. It has two failure modes and both are represented here.

  1. It overflows. The prompt plus the completion exceeds the window and the
     provider either rejects the call or truncates the answer.
  2. It gets prefix-truncated. The usual quick fix is to cut the joined string
     down to the window, which keeps whatever ran first and destroys whatever ran
     last, with no relationship to what the request needed.

Both are simulated exactly rather than estimated, so the numbers in the README
come from the same counter the assembler uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from llmkit import count_tokens

from .receipt import STATUS_COMPRESSED, STATUS_DEDUPED, STATUS_INCLUDED, Receipt
from .sections import ContextRequest, Section

HIGH_PRIORITY = 3.0
# An item counts as represented in the prompt once this much of it survives.
MIN_REPRESENTED_TOKENS = 12


def naive_prompt(request: ContextRequest) -> str:
    """Every item, declaration order, no budget awareness."""
    blocks: List[str] = []
    for section in request.sections:
        if not section.items:
            continue
        body = "\n\n".join(i.text for i in section.items)
        blocks.append(f"{section.spec.header}\n{body}" if section.spec.header else body)
    return "\n\n".join(blocks)


def _high_priority_sections(request: ContextRequest, threshold: float) -> List[Section]:
    return [s for s in request.sections if s.spec.priority >= threshold and s.items]


def prefix_truncation_metrics(
    request: ContextRequest, threshold: float = HIGH_PRIORITY
) -> Tuple[float, float]:
    """Token retention and source coverage after a cut-to-fit of the naive prompt.

    Walks the naive prompt in declaration order, spending the budget item by
    item. An item that fits entirely counts fully, an item that straddles the cut
    counts for the part that fits, and everything after the cut counts zero.

    Coverage is the second number because retention alone flatters prefix
    truncation: keeping the first three documents whole scores well on tokens
    while silently deleting the tool output and the user's actual question, which
    are at the end of the string.
    """
    budget = request.available_tokens
    high = {s.name for s in _high_priority_sections(request, threshold)}
    total = kept = 0
    items_total = items_represented = 0
    for section in request.sections:
        if not section.items:
            continue
        budget -= section.spec.header_tokens
        for item in section.items:
            if section.name in high:
                total += item.tokens
                items_total += 1
            take = min(item.tokens, budget) if budget > 0 else 0
            budget -= take
            if section.name in high:
                kept += take
                if take >= min(item.tokens, MIN_REPRESENTED_TOKENS):
                    items_represented += 1
    retention = kept / total if total else 1.0
    coverage = items_represented / items_total if items_total else 1.0
    return retention, coverage


def assembled_metrics(
    request: ContextRequest, receipt: Receipt, threshold: float = HIGH_PRIORITY
) -> Tuple[float, float]:
    """Share of high-priority tokens the assembler kept.

    A deduplicated item counts as retained when the twin that replaced it made
    it into the prompt: the fact is present, it was simply paid for once. A
    compressed item counts for the tokens that survived compression.
    """
    high = {s.name for s in _high_priority_sections(request, threshold)}
    if not high:
        return 1.0, 1.0
    status_by_id: Dict[str, str] = {}
    for section in receipt.sections:
        for item in section.items:
            status_by_id[item.item_id] = item.status
    kept_by_id: Dict[str, int] = {}
    for section in receipt.sections:
        for item in section.items:
            if item.status in (STATUS_INCLUDED, STATUS_COMPRESSED):
                kept_by_id[item.item_id] = item.after_tokens
            elif item.status == STATUS_DEDUPED:
                twin = item.reason.split()[-3] if item.reason else ""
                survived = status_by_id.get(twin) in (STATUS_INCLUDED, STATUS_COMPRESSED)
                kept_by_id[item.item_id] = item.before_tokens if survived else 0

    total = kept = 0
    items_total = items_represented = 0
    for section in request.sections:
        if section.name not in high:
            continue
        for item in section.items:
            total += item.tokens
            items_total += 1
            survived = min(item.tokens, kept_by_id.get(item.id, 0))
            kept += survived
            if survived >= min(item.tokens, MIN_REPRESENTED_TOKENS):
                items_represented += 1
    retention = kept / total if total else 1.0
    coverage = items_represented / items_total if items_total else 1.0
    return retention, coverage


@dataclass
class Comparison:
    request_id: str
    window: int
    reserve: int
    available: int
    naive_tokens: int
    assembled_tokens: int
    naive_overflow_tokens: int
    tokens_saved: int
    naive_high_priority_retention: float
    assembled_high_priority_retention: float
    naive_high_priority_coverage: float
    assembled_high_priority_coverage: float

    @property
    def naive_overflows(self) -> bool:
        return self.naive_overflow_tokens > 0

    def row(self) -> str:
        overflow = f"+{self.naive_overflow_tokens}" if self.naive_overflows else "fits"
        return (
            f"  {self.request_id:<14}{self.window:>7}{self.naive_tokens:>8}{overflow:>10}"
            f"{self.assembled_tokens:>11}{self.tokens_saved:>8}"
            f"{self.naive_high_priority_retention * 100:>9.0f}%{self.assembled_high_priority_retention * 100:>8.0f}%"
            f"{self.naive_high_priority_coverage * 100:>10.0f}%{self.assembled_high_priority_coverage * 100:>8.0f}%"
        )

    @staticmethod
    def header() -> str:
        return (
            f"  {'request':<14}{'window':>7}{'naive':>8}{'overflow':>10}{'assembled':>11}{'saved':>8}"
            f"{'hp tok n':>10}{'ours':>8}{'hp src n':>10}{'ours':>8}"
        )


def compare(request: ContextRequest, receipt: Receipt, threshold: float = HIGH_PRIORITY) -> Comparison:
    naive_tokens = count_tokens(naive_prompt(request))
    naive_retention, naive_coverage = prefix_truncation_metrics(request, threshold)
    ours_retention, ours_coverage = assembled_metrics(request, receipt, threshold)
    return Comparison(
        request_id=request.request_id,
        window=request.model_window,
        reserve=request.reserve_tokens,
        available=request.available_tokens,
        naive_tokens=naive_tokens,
        assembled_tokens=receipt.prompt_tokens,
        naive_overflow_tokens=max(0, naive_tokens - request.available_tokens),
        tokens_saved=naive_tokens - receipt.prompt_tokens,
        naive_high_priority_retention=naive_retention,
        assembled_high_priority_retention=ours_retention,
        naive_high_priority_coverage=naive_coverage,
        assembled_high_priority_coverage=ours_coverage,
    )


def summarise(comparisons: Sequence[Comparison]) -> Dict[str, float]:
    if not comparisons:
        return {}
    overflowed = sum(1 for c in comparisons if c.naive_overflows)
    return {
        "requests": len(comparisons),
        "naive_overflows": overflowed,
        "assembled_overflows": sum(1 for c in comparisons if c.assembled_tokens > c.available),
        "tokens_saved": sum(c.tokens_saved for c in comparisons),
        "naive_tokens": sum(c.naive_tokens for c in comparisons),
        "assembled_tokens": sum(c.assembled_tokens for c in comparisons),
        "naive_high_priority_retention": sum(c.naive_high_priority_retention for c in comparisons) / len(comparisons),
        "assembled_high_priority_retention": sum(c.assembled_high_priority_retention for c in comparisons)
        / len(comparisons),
        "naive_high_priority_coverage": sum(c.naive_high_priority_coverage for c in comparisons) / len(comparisons),
        "assembled_high_priority_coverage": sum(c.assembled_high_priority_coverage for c in comparisons)
        / len(comparisons),
    }
