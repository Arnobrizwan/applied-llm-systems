"""Token accounting without a paid tokenizer download.

`tiktoken` is a native wheel and a network fetch on first use, so the default
here is a deterministic word/punctuation counter calibrated against BPE ratios
(~1.3 tokens per word for English prose). Every project treats this as an
*estimate* and says so; swap in a real tokenizer by setting LLMKIT_TOKENIZER=tiktoken
if the environment allows it.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, List

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_TOKENS_PER_WORD = 1.3

_real_encoder = None
if os.environ.get("LLMKIT_TOKENIZER") == "tiktoken":  # pragma: no cover - optional path
    try:
        import tiktoken

        _real_encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _real_encoder = None


def count_tokens(text: str) -> int:
    """Estimated token count for `text`. Never returns a negative number."""
    if not text:
        return 0
    if _real_encoder is not None:  # pragma: no cover - optional path
        return len(_real_encoder.encode(text))
    pieces = _WORD_RE.findall(text)
    words = sum(1 for p in pieces if p.isalnum())
    punct = len(pieces) - words
    return max(1, int(round(words * _TOKENS_PER_WORD)) + punct)


def count_message_tokens(messages: Iterable) -> int:
    """Token estimate for a chat payload, including per-message overhead."""
    total = 0
    for m in messages:
        content = m.content if hasattr(m, "content") else m.get("content", "")
        total += count_tokens(content) + 4  # role + delimiters
    return total + 2


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Trim `text` so its estimate fits `max_tokens`, on a word boundary."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    words: List[str] = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


# Public, editable price book. Zero for every provider this repo defaults to,
# because they are all free; the routing and observability projects read these
# numbers so cost maths stays real when a paid model is plugged in later.
PRICE_PER_1K_USD = {
    "echo": (0.0, 0.0),
    "ollama": (0.0, 0.0),
    "small": (0.00015, 0.0006),
    "medium": (0.0006, 0.0024),
    "large": (0.003, 0.015),
}


def estimate_cost(model_tier: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost for a call at a named tier. Unknown tiers cost nothing."""
    inp, out = PRICE_PER_1K_USD.get(model_tier, (0.0, 0.0))
    return (prompt_tokens / 1000.0) * inp + (completion_tokens / 1000.0) * out
