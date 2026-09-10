"""Making a piece of text fit a token budget, five different ways.

Every function here is measured on the way out: the caller gets back the text
*and* the before/after token counts, because a compression step that silently
produced nothing useful is one of the harder context bugs to see. A section that
looks present in the prompt but has been reduced to a sentence fragment reads,
from the outside, as a model that ignored its instructions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from llmkit import count_tokens, keyword_overlap, sentence_split, truncate_to_tokens

from .sections import Compression

# Below this, a fragment carries no usable information and only costs tokens and
# invites the model to answer from a half sentence. Dropping is more honest.
MIN_USEFUL_TOKENS = 12


@dataclass
class CompressionResult:
    text: str
    method: str
    before_tokens: int
    after_tokens: int

    @property
    def ratio(self) -> float:
        """Fraction of the original tokens kept. 1.0 means untouched."""
        if self.before_tokens == 0:
            return 1.0
        return self.after_tokens / self.before_tokens


def truncate_head(text: str, max_tokens: int) -> str:
    """Keep the tail. Used for chat history, where recent turns matter most."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[-mid:])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[-lo:])


def extractive(text: str, max_tokens: int, query: str) -> str:
    """Keep the sentences that bear on the query, in their original order.

    Greedy selection by query overlap with a small lead bias, then re-sorted back
    into document order. Re-sorting matters: a set of sentences shuffled into
    relevance order reads as disconnected assertions and loses the pronoun and
    ordering cues the model uses to attribute claims.

    Rejected alternative: an LLM call per document. It is better at this, and it
    costs a round trip per document at assembly time, on the latency path of
    every single request. Extraction is free and deterministic, so it is the
    default and the model summary is opt-in per section.
    """
    if max_tokens <= 0:
        return ""
    sentences = sentence_split(text)
    if len(sentences) <= 1:
        return truncate_to_tokens(text, max_tokens)

    scored = []
    for idx, sent in enumerate(sentences):
        lead_bias = 0.05 if idx == 0 else 0.0  # first sentences carry topic
        scored.append((keyword_overlap(query, sent) + lead_bias, idx, sent))
    scored.sort(key=lambda x: (-x[0], x[1]))

    chosen: List[int] = []
    used = 0
    for _, idx, sent in scored:
        cost = count_tokens(sent)
        if used + cost > max_tokens:
            continue
        chosen.append(idx)
        used += cost
    if not chosen:
        return truncate_to_tokens(text, max_tokens)
    chosen.sort()
    return " ".join(sentences[i] for i in chosen)


def summarize(text: str, max_tokens: int, query: str, llm) -> str:
    """Ask the model for a summary, then hard-cap the result.

    The cap is not defensive decoration. Models routinely overshoot a stated
    length, and an over-long summary here would be the exact overflow this
    service exists to prevent, arriving from inside the service itself.
    """
    if max_tokens <= 0:
        return ""
    if llm is None:
        return extractive(text, max_tokens, query)
    resp = llm.complete(
        [
            {
                "role": "system",
                "content": (
                    "Compress the evidence below for answering the user question. "
                    "Keep concrete facts, numbers and names. Drop everything else.\n"
                    f"[S1] {text}"
                ),
            },
            {"role": "user", "content": query},
        ]
    )
    return truncate_to_tokens(resp.text.strip(), max_tokens)


def compress(
    text: str,
    max_tokens: int,
    strategy: Compression,
    query: str = "",
    llm=None,
    min_useful_tokens: int = MIN_USEFUL_TOKENS,
) -> Optional[CompressionResult]:
    """Fit `text` into `max_tokens` under `strategy`. None means "drop it".

    None rather than an empty string so the caller has to make a decision and
    record it, instead of appending a blank section to the prompt.
    """
    before = count_tokens(text)
    if max_tokens >= before and strategy is not Compression.DROP:
        return CompressionResult(text=text, method="whole", before_tokens=before, after_tokens=before)
    if strategy is Compression.DROP or strategy is Compression.KEEP_WHOLE:
        return None
    if max_tokens < min_useful_tokens:
        return None

    if strategy is Compression.TRUNCATE_TAIL:
        out, method = truncate_to_tokens(text, max_tokens), "truncate_tail"
    elif strategy is Compression.TRUNCATE_HEAD:
        out, method = truncate_head(text, max_tokens), "truncate_head"
    elif strategy is Compression.EXTRACTIVE:
        out, method = extractive(text, max_tokens, query), "extractive"
    elif strategy is Compression.SUMMARIZE:
        out, method = summarize(text, max_tokens, query, llm), "summarize"
    else:  # pragma: no cover - Compression is a closed enum
        raise ValueError(f"unknown strategy {strategy!r}")

    out = out.strip()
    after = count_tokens(out)
    if not out or after < min_useful_tokens:
        return None
    if after > max_tokens:  # summarize path can overshoot before the cap lands
        out = truncate_to_tokens(out, max_tokens)
        after = count_tokens(out)
    return CompressionResult(text=out, method=method, before_tokens=before, after_tokens=after)


def arrange_hourglass(values: Sequence[float]) -> List[int]:
    """Index order that puts the best items at the two ends and filler between.

    This is an empirical heuristic, not a proof. It follows the "lost in the
    middle" finding (Liu et al., 2023) that retrieval-augmented models attend
    more reliably to evidence at the start and end of a long context than to
    evidence buried in the middle. The effect size depends on the model, the
    window and the prompt format, so this is a defensible default rather than a
    guarantee, and it is worth re-measuring per model rather than trusting.

    Highest value goes first, second-highest goes last, then third first, fourth
    last, and so on inward.
    """
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    front: List[int] = []
    back: List[int] = []
    for rank, idx in enumerate(order):
        (front if rank % 2 == 0 else back).append(idx)
    back.reverse()
    return front + back
