"""Pull a JSON value out of realistically-broken model output.

Every failure mode handled here was observed in the wild, not invented:

* a fenced code block with a language tag,
* a prose preamble ("Sure! Here is the JSON you asked for:"),
* trailing commentary after the closing brace,
* single-quoted keys and values, which is Python's repr leaking through,
* trailing commas, which JavaScript allows and JSON does not,
* an object truncated mid-token because the response hit the token cap.

The repairs run as an ordered pipeline and each one that fires is recorded.
That record matters: "we recovered 41 percent of malformed responses" is a much
weaker claim than "unfencing recovered 24 percent and truncation repair
recovered 17 percent", and only the second tells you what to fix upstream.

A deliberate non-goal: this module never guesses at missing *values*. It repairs
syntax so the text can be parsed, then hands the result to the validator. If a
required field is absent, that is a semantic problem and the retry loop is the
right place to fix it, because only the model knows what the value should be.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

__all__ = ["Extraction", "extract_json", "REPAIR_NAMES"]

_OPENERS = {"{": "}", "[": "]"}
_CLOSERS = {"}", "]"}


@dataclass
class Extraction:
    """Result of trying to get a JSON value out of raw model text."""

    ok: bool
    value: Any = None
    repairs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    text: str = ""
    raw: str = ""

    @property
    def needed_repair(self) -> bool:
        return bool(self.repairs)


def _walk(text: str):
    """Yield (index, char, in_string, depth) with string and escape awareness.

    Every repair below needs to distinguish a brace inside a string value from a
    structural brace. Writing that scan once and reusing it is the difference
    between repairs that work on real payloads and regexes that corrupt any
    document containing a comma inside a sentence.
    """
    in_string = False
    quote = ""
    escape = False
    depth = 0
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            yield i, ch, True, depth
            continue
        if ch in ('"', "'"):
            in_string = True
            quote = ch
            yield i, ch, True, depth
            continue
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
        yield i, ch, False, depth


def _scan_span(text: str, start: int) -> Tuple[Optional[int], List[str], bool, str]:
    """Scan a bracketed span. Returns (end_exclusive, open_stack, in_string, quote)."""
    stack: List[str] = []
    in_string = False
    quote = ""
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string, quote = True, ch
        elif ch in _OPENERS:
            stack.append(_OPENERS[ch])
        elif ch in _CLOSERS:
            if stack and stack[-1] == ch:
                stack.pop()
                if not stack:
                    return i + 1, [], False, ""
            else:
                return i, stack, False, ""  # mismatched closer, stop here
    return None, stack, in_string, quote


def _try_parse(text: str) -> Tuple[bool, Any, Optional[str]]:
    try:
        return True, json.loads(text), None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


# -- individual repairs ----------------------------------------------------

_FENCE_RE = re.compile(r"```[ \t]*([a-zA-Z0-9_+-]*)[ \t]*\r?\n([\s\S]*?)(?:```|\Z)")


def _unfence(text: str) -> Optional[str]:
    """Take the body of the first fenced block, closed or not.

    The open-ended `\\Z` alternative matters: a response truncated inside a code
    block never emits the closing fence, and that is exactly the case where the
    truncation repair below needs the body.
    """
    match = _FENCE_RE.search(text)
    if not match:
        return None
    return match.group(2).strip()


def _slice_to_json(text: str) -> Optional[str]:
    """Drop the prose either side of the first balanced JSON value."""
    start = next((i for i, ch in enumerate(text) if ch in _OPENERS), None)
    if start is None:
        return None
    end, _stack, _in_string, _quote = _scan_span(text, start)
    return text[start:end] if end is not None else text[start:]


_PY_LITERALS = {"True": "true", "False": "false", "None": "null"}
_PY_LITERAL_RE = re.compile(r"\b(True|False|None)\b")


def _python_literals(text: str) -> Optional[str]:
    """Rewrite Python's True/False/None outside strings. A repr leak, not JSON."""
    out: List[str] = []
    consumed_until = 0
    for i, _ch, in_string, _depth in _walk(text):
        if in_string or i < consumed_until:
            continue
        match = _PY_LITERAL_RE.match(text, i)
        if match:
            out.append((text[consumed_until:i], _PY_LITERALS[match.group(1)]))
            consumed_until = match.end()
    if not out:
        return None
    rebuilt: List[str] = []
    cursor = 0
    for prefix, replacement in out:
        rebuilt.append(prefix)
        rebuilt.append(replacement)
        cursor += len(prefix) + len(replacement)
    rebuilt.append(text[consumed_until:])
    return "".join(rebuilt)


def _single_quotes(text: str) -> Optional[str]:
    """Convert single-quoted strings to double-quoted ones.

    Guarded by a count check rather than applied unconditionally: on a payload
    that already uses double quotes correctly, a blanket swap would turn every
    apostrophe in a sentence into a string delimiter. Only run this when single
    quotes clearly outnumber double quotes, which is the repr-leak signature.
    This is a heuristic and it is allowed to fail; the pipeline records that it
    fired and moves on.
    """
    if text.count("'") <= text.count('"'):
        return None
    out: List[str] = []
    in_string = False
    escape = False
    quote = ""
    for ch in text:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
            elif ch == "\\":
                out.append(ch)
                escape = True
            elif ch == quote:
                out.append('"')
                in_string = False
            elif ch == '"':
                out.append('\\"')
            else:
                out.append(ch)
            continue
        if ch in ('"', "'"):
            in_string, quote = True, ch
            out.append('"')
        else:
            out.append(ch)
    return "".join(out)


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _trailing_commas(text: str) -> Optional[str]:
    """Remove `,}` and `,]`, applied only to structural positions."""
    protected = {i for i, _ch, in_string, _d in _walk(text) if in_string}

    def repl(match: "re.Match[str]") -> str:
        if match.start() in protected:
            return match.group(0)
        return match.group(1)

    fixed = _TRAILING_COMMA_RE.sub(repl, text)
    return fixed if fixed != text else None


def _drop_last_member(text: str) -> Optional[str]:
    """Cut the text back to the last structural comma.

    Used when a truncated payload ends mid-member ({"a": 1, "b": ). Closing the
    brackets alone leaves a dangling key, so the incomplete member is discarded
    and the resulting object is handed to the validator, which will report the
    missing field and let the retry loop ask for it properly.
    """
    last = None
    for i, ch, in_string, _depth in _walk(text):
        if ch == "," and not in_string:
            last = i
    if last is None:
        return None
    return text[:last]


def _close_truncated(text: str) -> Optional[str]:
    """Close an unterminated string and any open brackets.

    Tries the cheap repair first (just close what is open), and only if that
    still does not parse does it start discarding trailing members. Discarding
    is lossy, so it is the last thing attempted, not the first.
    """
    start = next((i for i, ch in enumerate(text) if ch in _OPENERS), None)
    if start is None:
        return None
    end, stack, in_string, _quote = _scan_span(text, start)
    if end is not None and not stack:
        return None  # already balanced, nothing to close

    body = text[start:]
    for _ in range(4):
        candidate = body.rstrip()
        if in_string:
            candidate += '"'
        candidate = re.sub(r"[,:]\s*$", "", candidate)
        _e, open_stack, _s, _q = _scan_span(candidate, 0)
        candidate += "".join(reversed(open_stack))
        ok, _value, _err = _try_parse(candidate)
        if ok:
            return candidate
        shorter = _drop_last_member(body)
        if shorter is None:
            return candidate
        body, in_string = shorter, False
    return body


_STRATEGIES: List[Tuple[str, Callable[[str], Optional[str]]]] = [
    ("unfence", _unfence),
    ("slice_to_json", _slice_to_json),
    ("python_literals", _python_literals),
    ("single_quotes", _single_quotes),
    ("trailing_commas", _trailing_commas),
    ("close_truncated", _close_truncated),
]

REPAIR_NAMES = [name for name, _fn in _STRATEGIES]


def extract_json(raw: str) -> Extraction:
    """Parse `raw`, applying repairs in order until something parses.

    Repairs are cumulative: unfencing then slicing then quote-normalising all
    operate on the output of the previous step, because real broken output
    usually has more than one thing wrong with it at once (the echo provider's
    "single quotes plus trailing comma" corruption is a faithful example).
    """
    text = (raw or "").strip()
    ok, value, error = _try_parse(text)
    if ok:
        return Extraction(ok=True, value=value, text=text, raw=raw)

    repairs: List[str] = []
    for name, strategy in _STRATEGIES:
        try:
            candidate = strategy(text)
        except Exception as exc:  # a repair must never take the caller down
            error = f"repair {name} failed: {exc}"
            continue
        if not candidate or candidate == text:
            continue
        text, repairs = candidate, repairs + [name]
        ok, value, error = _try_parse(text)
        if ok:
            return Extraction(ok=True, value=value, repairs=repairs, text=text, raw=raw)

    return Extraction(ok=False, repairs=repairs, error=error, text=text, raw=raw)
