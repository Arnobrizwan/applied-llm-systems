"""Text splitting utilities used by the retrieval and context projects."""
from __future__ import annotations

import re
from typing import List

from .tokens import count_tokens

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WS_RE = re.compile(r"[ \t]+")


def normalize_ws(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def sentence_split(text: str) -> List[str]:
    """Cheap sentence splitter. Good enough for chunk boundaries, not for NLP."""
    parts = [p.strip() for p in _SENT_RE.split(normalize_ws(text)) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_text(text: str, target_tokens: int = 220, overlap_tokens: int = 40) -> List[str]:
    """Sentence-aware chunker with token overlap.

    Splitting on sentences rather than characters keeps citations readable; the
    overlap stops an answer-bearing sentence from being orphaned at a boundary,
    which is the single most common cause of "the answer was in the corpus but
    retrieval never saw it".
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    sentences = sentence_split(text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        st = count_tokens(sent)
        if current and current_tokens + st > target_tokens:
            chunks.append(" ".join(current))
            # carry the tail of the previous chunk forward as overlap
            carry: List[str] = []
            carry_tokens = 0
            for prev in reversed(current):
                pt = count_tokens(prev)
                if carry_tokens + pt > overlap_tokens:
                    break
                carry.insert(0, prev)
                carry_tokens += pt
            current = carry
            current_tokens = carry_tokens
        current.append(sent)
        current_tokens += st

    if current:
        chunks.append(" ".join(current))
    return chunks


def keyword_overlap(a: str, b: str) -> float:
    """Jaccard overlap of lowercased word sets. Used for cheap heuristics only."""
    sa = {w for w in re.findall(r"\w+", a.lower()) if len(w) > 2}
    sb = {w for w in re.findall(r"\w+", b.lower()) if len(w) > 2}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
