"""Ablation helper: rescore a text with one rule category switched off.

Kept separate from `injection.scan` because it exists for the measurement, not
for the runtime path. A published precision and recall pair means very little
on its own if one rule is doing all the work, since that rule is then the whole
system and its blind spots are the system's blind spots. Recomputing recall with
each category disabled shows how much of the result survives losing any one
family of rules.
"""
from __future__ import annotations

import re

from .injection import RULES, _noisy_or
from .pii import decode_base64_payload

__all__ = ["ablate_category"]


def ablate_category(text: str, disabled: str) -> float:
    """The injection score for `text` with every rule in `disabled` removed."""
    weights = []
    for _rule_id, category, weight, pattern in RULES:
        if category == disabled:
            continue
        if pattern.search(text):
            weights.append(weight)
    if disabled != "encoded_payload":
        for token in re.findall(r"\b[A-Za-z0-9+/]{20,}={0,2}\b", text):
            decoded = decode_base64_payload(token)
            if decoded is None:
                continue
            inner = any(p.search(decoded) for _i, c, _w, p in RULES if c != disabled)
            weights.append(0.6 if inner else 0.25)
    return _noisy_or(weights)
