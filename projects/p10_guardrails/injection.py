"""Prompt-injection detection: weighted rules, not one regex.

A single "ignore previous instructions" regex is the standard implementation and
it is bad in both directions. It misses everything phrased differently, and it
fires on "please ignore the noise in the previous column", which is a real
sentence a real user sends.

The design here is a small set of independent rules, each with a category, a
weight and the exact evidence it matched. Weights combine with noisy-OR:

    score = 1 - product(1 - weight_i)

Noisy-OR was chosen over a plain sum for a specific reason. A sum has to be
clamped, and once two or three rules fire everything clamps to 1.0, so a prompt
with three weak signals ranks the same as one with a decisive signal. Noisy-OR
saturates smoothly, so more independent evidence always raises the score without
any single weak rule being able to reach the block threshold alone. It also
gives a defensible reading: each weight is roughly "probability this rule alone
means an attack", and the combination assumes the rules are independent, which
they are not quite, so the score is a ranking signal and not a calibrated
probability. That caveat is stated rather than hidden.

Base64 payloads get a second pass: the blob is decoded and re-scanned, and if
the decoded text trips a rule the encoded-payload weight is raised. Encoding is
not itself an attack, but encoding an instruction override is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

from .pii import decode_base64_payload

__all__ = ["InjectionSignal", "InjectionReport", "scan", "RULES"]


@dataclass(frozen=True)
class InjectionSignal:
    rule_id: str
    category: str
    weight: float
    evidence: str


@dataclass
class InjectionReport:
    score: float
    signals: List[InjectionSignal] = field(default_factory=list)

    @property
    def categories(self) -> List[str]:
        return sorted({s.category for s in self.signals})

    @property
    def rule_ids(self) -> List[str]:
        return [s.rule_id for s in self.signals]


# rule_id, category, weight, pattern
RULES: Sequence[Tuple[str, str, float, "re.Pattern[str]"]] = (
    ("override.ignore_previous", "instruction_override", 0.55,
     re.compile(r"(?i)\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all)\b"
                # 15 characters, not 30: a wider window matched "disregard the
                # previous estimate because the rules changed", which is a
                # sentence a real user writes.
                r"[^.\n]{0,15}\b(instruction|instructions|prompt|prompts|rule|rules|context)\b")),
    ("override.new_instructions", "instruction_override", 0.45,
     # Anchored to a sentence boundary rather than a line start: the payload is
     # usually the second sentence of an otherwise ordinary message.
     re.compile(r"(?i)(?:^|\n|(?<=[.!?])\s)\s*(new|updated|revised)\s+(instructions?|rules?|system prompt)\s*[:\-]")),
    ("override.forget_everything", "instruction_override", 0.5,
     re.compile(r"(?i)\b(forget|ignore|disregard)\b[^.\n]{0,30}\b(everything|all)\b"
                r"[^.\n]{0,30}\b(above|before|earlier|you were told|that came before)\b")),
    ("override.follow_external", "instruction_override", 0.4,
     # Instructing the model to obey content it is about to fetch is the whole
     # of indirect prompt injection, independent of what the content turns out
     # to be, so it is scored on its own.
     re.compile(r"(?i)\b(follow|do what|obey|comply with)\b[^.\n]{0,25}"
                r"\b(it says|the instructions|these instructions|them|below|at (this|that) (link|url))\b")),
    ("override.nothing_above", "instruction_override", 0.4,
     re.compile(r"(?i)\b(nothing|none of (the|what))\b[^.\n]{0,20}\babove\b[^.\n]{0,20}\b(applies|matters|counts)\b")),
    ("hijack.you_are_now", "role_hijack", 0.45,
     # "You are now" needs a persona or capability word after it. Without that
     # constraint the rule fires on "you are now looking at version two", which
     # is a sentence a colleague writes.
     re.compile(r"(?i)\byou\s+are\s+(now|no longer)\b[^.\n]{0,30}"
                r"\b(dan|unrestricted|unfiltered|uncensored|jailbroken|free|bound|restricted|"
                r"allowed|able|mode|assistant|model|ai|admin|root|developer|policy)\b"
                r"|\bfrom now on,? you\b[^.\n]{0,40}\b(will|must|are|shall|should)\b")),
    ("hijack.pretend", "role_hijack", 0.35,
     re.compile(r"(?i)\b(pretend|act) (that )?(to be|you are|as if|as though)\b")),
    ("hijack.jailbreak_persona", "role_hijack", 0.5,
     re.compile(r"(?i)\b(developer mode|dan mode|do anything now|jailbreak|unfiltered mode)\b")),
    ("exfil.reveal_system_prompt", "system_exfiltration", 0.55,
     re.compile(r"(?i)\b(reveal|show|print|repeat|output|display|reproduce)\b[^.\n]{0,30}"
                r"\b(system prompt|initial instructions|your instructions|your rules|the prompt above)\b")),
    ("exfil.what_are_your_instructions", "system_exfiltration", 0.45,
     # The qualifier is mandatory. Without it the rule fired on "what are your
     # instructions for a severity 1 incident", which is someone asking the
     # assistant about a runbook, not about its own prompt.
     re.compile(r"(?i)\bwhat\s+(are|were)\s+(your|the)\s+(original|initial|system|exact|first|"
                r"real|underlying)\s+(instructions|rules|prompt|directives|guidelines)\b")),
    ("exfil.repeat_text_above", "system_exfiltration", 0.4,
     re.compile(r"(?i)\brepeat\b[^.\n]{0,20}\b(everything|the text|all text)\b[^.\n]{0,20}\babove\b")),
    ("delimiter.fake_tags", "delimiter_escape", 0.4,
     re.compile(r"(?i)</?\s*(system|assistant|instructions?|im_start|im_end)\s*>|\[/?INST\]|<\|[a-z_]+\|>")),
    ("delimiter.section_break", "delimiter_escape", 0.4,
     re.compile(r"(?i)(^|\n)\s*(#{2,}|-{3,})\s*(end of (prompt|instructions)|system|admin)\b")),
    ("exfil.send_data_out", "data_exfiltration", 0.5,
     re.compile(r"(?i)\b(send|post|upload|forward|email|leak)\b[^.\n]{0,40}"
                r"\b(to)\b\s*(https?://|www\.|[A-Za-z0-9._%+\-]+@)")),
    ("url.raw_ip", "suspicious_url", 0.35,
     re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}")),
    ("url.shortener", "suspicious_url", 0.3,
     re.compile(r"(?i)https?://(bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|rb\.gy)/\S+")),
    ("url.data_uri", "suspicious_url", 0.35,
     re.compile(r"(?i)data:text/(html|javascript)[;,]")),
    ("shell.pipe_to_shell", "suspicious_url", 0.45,
     re.compile(r"(?i)\b(curl|wget)\b[^\n|]{0,60}\|\s*(ba)?sh\b")),
)

_B64_TOKEN = re.compile(r"\b[A-Za-z0-9+/]{20,}={0,2}\b")

BLOCK_THRESHOLD = 0.70
FLAG_THRESHOLD = 0.35


def _apply_rules(text: str) -> List[InjectionSignal]:
    signals: List[InjectionSignal] = []
    for rule_id, category, weight, pattern in RULES:
        match = pattern.search(text)
        if match:
            evidence = " ".join(match.group().split())[:80]
            signals.append(InjectionSignal(rule_id, category, weight, evidence))
    return signals


def _noisy_or(weights: Iterable[float]) -> float:
    product = 1.0
    for weight in weights:
        product *= max(0.0, 1.0 - weight)
    return round(1.0 - product, 4)


def scan(text: str) -> InjectionReport:
    """Score `text` for prompt injection and return the evidence."""
    signals = _apply_rules(text)

    for token in _B64_TOKEN.findall(text):
        decoded = decode_base64_payload(token)
        if not decoded:
            continue
        inner = _apply_rules(decoded)
        # A base64 blob on its own is weak evidence; a base64 blob that decodes
        # to an instruction override is strong evidence, so the weight is raised
        # rather than the inner signals being reported as if they were plain text.
        weight = 0.6 if inner else 0.25
        preview = " ".join(decoded.split())[:60]
        signals.append(InjectionSignal("encoded.base64_payload", "encoded_payload", weight,
                                       f"{token[:16]}... decodes to: {preview}"))
    return InjectionReport(score=_noisy_or(s.weight for s in signals), signals=signals)
