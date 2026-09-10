"""Output-stage checks: what the model says, not what the user asked.

The input stage cannot catch these. A prompt can be entirely benign and the
response can still leak a credential the model was shown in its context, echo
the system prompt because the user asked a question that happened to resemble
it, contain a phone number that was never in the conversation, or wander into a
topic this deployment does not allow. Output filtering is the only place those
are visible.

System-prompt echo is detected with word shingles rather than a substring
search, because a model that paraphrases the system prompt has leaked it just as
effectively as one that quotes it, and an exact-match check finds neither.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .pii import Vault, detect

__all__ = ["OutputFinding", "shingle_overlap", "scan_output"]

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class OutputFinding:
    kind: str  # secret_leak | system_prompt_echo | new_pii | banned_topic
    detail: str
    evidence: str = ""


def shingle_overlap(candidate: str, reference: str, n: int = 6) -> float:
    """Fraction of the reference's n-word shingles that appear in the candidate.

    Measured against the reference, not the union: the question is "how much of
    the system prompt came back", and a long answer that quotes the whole prompt
    would score low on a symmetric measure purely because it is long.
    """
    def shingles(text: str) -> set:
        words = _WORD.findall(text.lower())
        return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}

    reference_shingles = shingles(reference)
    if not reference_shingles:
        return 0.0
    return len(reference_shingles & shingles(candidate)) / len(reference_shingles)


def scan_output(
    text: str,
    *,
    system_prompt: str = "",
    input_text: str = "",
    secrets: Sequence[str] = (),
    banned_topics: Optional[Dict[str, Tuple[str, ...]]] = None,
    echo_threshold: float = 0.25,
    vault: Optional[Vault] = None,
) -> List[OutputFinding]:
    """Every output-stage problem found in `text`."""
    findings: List[OutputFinding] = []

    for secret in secrets:
        if secret and secret in text:
            # Only the shape is reported, never the value: an audit log that
            # quotes the leaked secret has leaked it a second time.
            findings.append(OutputFinding("secret_leak", "a configured secret appears verbatim",
                                          f"{secret[:4]}... ({len(secret)} chars)"))

    output_pii = detect(text)
    for match in output_pii:
        if match.kind == "api_key":
            findings.append(OutputFinding("secret_leak", "the response contains a credential-shaped string",
                                          f"{match.value[:6]}... ({match.kind})"))

    if system_prompt:
        overlap = shingle_overlap(text, system_prompt)
        if overlap >= echo_threshold:
            findings.append(OutputFinding("system_prompt_echo",
                                          f"{overlap:.0%} of the system prompt's 6-word shingles came back",
                                          f"overlap={overlap:.2f}"))

    known = set(vault.values()) if vault else set()
    for match in output_pii:
        if match.kind == "api_key":
            continue  # already reported as a secret leak
        if match.value in input_text or match.value in known:
            continue  # the user supplied it; returning it is not a leak
        findings.append(OutputFinding("new_pii",
                                      f"{match.kind} in the response was not in the user message",
                                      match.value[:6] + "..."))

    lowered = text.lower()
    for topic, phrases in (banned_topics or {}).items():
        for phrase in phrases:
            if phrase in lowered:
                findings.append(OutputFinding("banned_topic", f"matched the {topic} policy", phrase))
                break
    return findings
