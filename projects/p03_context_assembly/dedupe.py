"""Cross-source deduplication.

The same fact reaches the prompt twice all the time: the retriever finds the
release note and the memory store also remembers it, or two chunks overlap
because the chunker used a stride. Paying for it twice costs tokens that a
lower-priority section then does not get, and repetition measurably biases a
model toward the repeated claim.

Matching is lexical containment over token shingles, not embeddings. An
embedding threshold loose enough to catch a restatement is also loose enough to
merge "tokens expire after 90 days" with "tokens expire after 30 days", and
silently dropping the correct half of a contradiction is a far worse failure than
paying for a duplicate. Containment over shingles only fires when one text really
does repeat almost all of the other's wording.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from llmkit import tokenize

from .sections import ContextItem, Section

SHINGLE = 4
# A candidate this much longer than its twin is treated as the superset copy and
# wins, even from a lower-priority section: keeping the shorter restatement would
# throw away the sentences the longer version adds.
SUPERSET_RATIO = 1.25


@dataclass
class DuplicateRecord:
    """One removal, kept so the receipt can explain a missing item."""

    dropped_item_id: str
    dropped_section: str
    kept_item_id: str
    kept_section: str
    similarity: float
    tokens_saved: int


def _shingles(text: str, n: int = SHINGLE) -> Set[Tuple[str, ...]]:
    toks = tokenize(text)
    if len(toks) <= n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def containment(a: str, b: str, n: int = SHINGLE) -> float:
    """Overlap divided by the size of the *smaller* shingle set.

    Jaccard was rejected: a one-line memory fact restated inside a 200-token
    retrieved paragraph has a tiny Jaccard score but is completely redundant,
    which is exactly the case worth catching.
    """
    sa, sb = _shingles(a, n), _shingles(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def deduplicate(sections: Sequence[Section], threshold: float = 0.8) -> List[DuplicateRecord]:
    """Remove redundant items in place and return what was removed.

    Survivor rule, in order: the copy that carries strictly more text wins, and
    when two copies are about the same length the one in the higher-priority
    section wins. The second half matters because the high-priority section is
    the one with a guaranteed floor, so the surviving copy is the one most likely
    to still be in the prompt after the budget tightens.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    ranked = sorted(sections, key=lambda s: (-s.spec.priority, s.name))
    kept: List[Tuple[Section, ContextItem]] = []
    removals: List[DuplicateRecord] = []

    for section in ranked:
        for item in sorted(section.items, key=lambda i: (-i.value, i.id)):
            twin_idx = -1
            best = 0.0
            for idx, (_, keep_item) in enumerate(kept):
                sim = containment(item.text, keep_item.text)
                if sim >= threshold and sim > best:
                    twin_idx, best = idx, sim
            if twin_idx < 0:
                kept.append((section, item))
                continue
            twin_section, twin_item = kept[twin_idx]
            if item.tokens > twin_item.tokens * SUPERSET_RATIO:
                removals.append(
                    DuplicateRecord(
                        dropped_item_id=twin_item.id,
                        dropped_section=twin_section.name,
                        kept_item_id=item.id,
                        kept_section=section.name,
                        similarity=round(best, 3),
                        tokens_saved=twin_item.tokens,
                    )
                )
                kept[twin_idx] = (section, item)
            else:
                removals.append(
                    DuplicateRecord(
                        dropped_item_id=item.id,
                        dropped_section=section.name,
                        kept_item_id=twin_item.id,
                        kept_section=twin_section.name,
                        similarity=round(best, 3),
                        tokens_saved=item.tokens,
                    )
                )

    survivors: Dict[str, Set[str]] = {s.name: set() for s in sections}
    for keep_section, keep_item in kept:
        survivors[keep_section.name].add(keep_item.id)
    for section in sections:
        # Original ordering is preserved; final ordering is decided later by the
        # lost-in-the-middle arrangement, not here.
        section.items = [i for i in section.items if i.id in survivors[section.name]]

    return removals


def duplicate_tokens(records: Sequence[DuplicateRecord]) -> int:
    return sum(r.tokens_saved for r in records)


def by_section(records: Sequence[DuplicateRecord]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        out[r.dropped_section] = out.get(r.dropped_section, 0) + 1
    return out
