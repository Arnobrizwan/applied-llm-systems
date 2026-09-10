"""PII detection and reversible redaction.

Two decisions shape this module.

**Detection is regex plus a validator, never regex alone.** A sixteen-digit
regex matches order numbers, invoice references and tracking codes as happily as
it matches a card, and a guardrail that blocks a support ticket because it
contains an order number gets switched off within a week. Card candidates are
therefore Luhn-checked, IP octets are range-checked, and phone candidates are
normalised to digits and checked against the actual numbering plans for
Bangladesh and Malaysia rather than a shape.

**Redaction is reversible.** The placeholder is stable within a request
(`[[EMAIL_1]]` is the same address everywhere it appears), so a model can still
reason about "the same person wrote both messages" without ever seeing the
address, and the real value can be put back on the way out when policy allows.
The rejected alternative, replacing with a constant `[REDACTED]`, destroys
co-reference: two different emails become the same token, and the model starts
answering about the wrong person.

The vault is per-request and in memory. Persisting it would turn a redaction
layer into a PII database, which is the opposite of the point.
"""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = ["PIIMatch", "Vault", "detect", "redact", "luhn_ok", "PII_KINDS"]

PII_KINDS = ("api_key", "credit_card", "nid_bd", "nric_my", "email", "ip_address", "phone")

# Lower number wins when two detectors claim overlapping spans. An API key that
# happens to contain a digit run must not be reported as a card number.
_PRIORITY = {kind: i for i, kind in enumerate(PII_KINDS)}


@dataclass(frozen=True)
class PIIMatch:
    kind: str
    value: str
    start: int
    end: int

    @property
    def span(self) -> Tuple[int, int]:
        return self.start, self.end


def luhn_ok(digits: str) -> bool:
    """Luhn checksum. Rejects the roughly 90 percent of digit runs that are not cards."""
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, char in enumerate(reversed(digits)):
        value = int(char)
        if i % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


# -- patterns --------------------------------------------------------------

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_IPV4 = re.compile(
    r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b"
)
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ \-]?){12,18}\d\b")
_PHONE_CANDIDATE = re.compile(r"\+?\d[\d\s().\-]{7,17}\d")
_NRIC_MY = re.compile(r"\b(\d{2})(\d{2})(\d{2})-\d{2}-\d{4}\b")
# A bare ten-digit run is not evidence of a national ID, so the Bangladeshi NID
# detector requires the label next to it. Precision over recall here: a false
# positive on this rule redacts an order number and confuses the model.
_NID_BD = re.compile(r"(?i)\bN\.?I\.?D\.?(?:\s*(?:no\.?|number|card))?\s*[:#\-]?\s*(\d{10}|\d{13}|\d{17})\b")

_API_KEY_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{12,20}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),  # JWT
    re.compile(r"(?i)\b(?:api[_\- ]?key|secret|token|password)\b\s*[:=]\s*[\"']?([A-Za-z0-9_\-]{16,})[\"']?"),
]


def _valid_phone(raw: str) -> Optional[str]:
    """Classify a phone candidate, or return None.

    Numbering plans, not shapes: Bangladeshi mobiles are 880 plus a ten-digit
    number starting 1 and then 3 to 9; Malaysian mobiles are 60 plus a nine or
    ten digit number starting 1. A bare local number is only accepted for
    Bangladesh (01XXXXXXXXX, eleven digits), because the Malaysian local form is
    ambiguous with it and guessing wrong is worse than not matching.
    """
    digits = re.sub(r"\D", "", raw)
    plus = raw.strip().startswith("+")
    if digits.startswith("880") and len(digits) == 13 and digits[3] == "1" and digits[4] in "3456789":
        return "phone"
    if digits.startswith("60") and len(digits) in (11, 12) and digits[2] == "1":
        return "phone"
    if digits.startswith("01") and len(digits) == 11 and digits[2] in "3456789":
        return "phone"
    if plus and 10 <= len(digits) <= 15:
        return "phone"
    return None


def _raw_matches(text: str) -> List[PIIMatch]:
    found: List[PIIMatch] = []

    for pattern in _API_KEY_PATTERNS:
        for m in pattern.finditer(text):
            # Group 1 exists on the labelled "api_key: value" form, where only
            # the value should be redacted, not the label.
            start, end = (m.span(1) if m.lastindex else m.span())
            found.append(PIIMatch("api_key", text[start:end], start, end))

    for m in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if luhn_ok(digits):
            found.append(PIIMatch("credit_card", m.group(), *m.span()))

    for m in _NID_BD.finditer(text):
        found.append(PIIMatch("nid_bd", m.group(1), *m.span(1)))

    for m in _NRIC_MY.finditer(text):
        month, day = int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:  # the first six digits are a birth date
            found.append(PIIMatch("nric_my", m.group(), *m.span()))

    for m in _EMAIL.finditer(text):
        found.append(PIIMatch("email", m.group(), *m.span()))

    for m in _IPV4.finditer(text):
        found.append(PIIMatch("ip_address", m.group(), *m.span()))

    for m in _PHONE_CANDIDATE.finditer(text):
        kind = _valid_phone(m.group())
        if kind:
            found.append(PIIMatch(kind, m.group().strip(), *m.span()))

    return found


def detect(text: str) -> List[PIIMatch]:
    """All PII in `text`, with overlaps resolved, ordered by position.

    Overlap resolution is by detector priority first and match length second.
    Without it a card number is reported twice, once as a card and once as a
    phone number, and the redacted text ends up with nested placeholders.
    """
    candidates = sorted(
        _raw_matches(text),
        key=lambda m: (_PRIORITY.get(m.kind, 99), -(m.end - m.start), m.start),
    )
    accepted: List[PIIMatch] = []
    for candidate in candidates:
        if any(candidate.start < a.end and a.start < candidate.end for a in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda m: m.start)


class Vault:
    """Placeholder to original value, for one request.

    Deliberately not persisted anywhere. A redaction layer that keeps a durable
    map of placeholders to real values has quietly become the PII store it was
    supposed to remove the need for.
    """

    def __init__(self) -> None:
        self._to_value: Dict[str, str] = {}
        self._to_placeholder: Dict[Tuple[str, str], str] = {}
        self._counts: Dict[str, int] = {}

    def placeholder_for(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key not in self._to_placeholder:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            placeholder = f"[[{kind.upper()}_{self._counts[kind]}]]"
            self._to_placeholder[key] = placeholder
            self._to_value[placeholder] = value
        return self._to_placeholder[key]

    def restore(self, text: str) -> str:
        """Put the real values back. Longest placeholder first so [[EMAIL_1]]
        cannot be partially matched inside [[EMAIL_10]]."""
        for placeholder in sorted(self._to_value, key=len, reverse=True):
            text = text.replace(placeholder, self._to_value[placeholder])
        return text

    def values(self) -> List[str]:
        return list(self._to_value.values())

    def placeholders(self) -> List[str]:
        return list(self._to_value)

    def __len__(self) -> int:
        return len(self._to_value)


def redact(text: str, vault: Optional[Vault] = None) -> Tuple[str, List[PIIMatch], Vault]:
    """Replace detected PII with stable placeholders.

    Placeholders are numbered left to right so [[PHONE_1]] is the first phone
    number a reader sees, then the substitution runs right to left so earlier
    spans keep their offsets while later ones are being rewritten.
    """
    vault = vault or Vault()
    matches = detect(text)
    for match in matches:
        vault.placeholder_for(match.kind, match.value)
    redacted = text
    for match in sorted(matches, key=lambda m: m.start, reverse=True):
        placeholder = vault.placeholder_for(match.kind, match.value)
        redacted = redacted[: match.start] + placeholder + redacted[match.end :]
    return redacted, matches, vault


def decode_base64_payload(token: str) -> Optional[str]:
    """Decode a base64 blob to text, or None. Used by the injection scanner."""
    if len(token) < 16 or len(token) % 4 not in (0, 2, 3):
        return None
    try:
        raw = base64.b64decode(token + "=" * (-len(token) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    printable = sum(1 for c in decoded if c.isprintable() or c in "\n\t")
    return decoded if decoded and printable / len(decoded) > 0.9 else None
