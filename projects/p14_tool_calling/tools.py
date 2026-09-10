"""Four real tools. Nothing here is a stub that returns a canned string.

Chosen to cover the four shapes a tool can have, because each fails differently:

* `calculate` takes free text and must refuse most of it (untrusted input),
* `search_docs` reads a corpus and returns a variable-size result (context cost),
* `convert_units` is pure and total except for one semantic error case
  (a well-formed call that is still wrong),
* `current_time` reads ambient state, which is what makes agent runs
  irreproducible unless you control it.

Parameter types are picked so the schema itself does most of the validation.
`Literal` for unit names rather than `str` means an unknown unit is caught by
argument coercion with a list of the valid options, before the function is
called, rather than by a KeyError inside it.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Literal

from llmkit import BM25
from llmkit.corpus import chunks

from .safe_math import UnsafeExpression, safe_eval
from .schema import tool

__all__ = ["calculate", "search_docs", "convert_units", "current_time", "ALL_TOOLS", "set_clock"]

# --------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------


@tool(tags=("math",), constraints={"precision": {"minimum": 0, "maximum": 10}, "expression": {"maxLength": 200}})
def calculate(expression: str, precision: int = 4) -> Dict[str, Any]:
    """Evaluate an arithmetic expression safely, without eval.

    Args:
        expression: An arithmetic expression such as "600 * 60 / 1000" using
            + - * / // % ** and the functions abs, round, min, max, sqrt,
            floor, ceil, log, log10 and exp.
        precision: Decimal places to round the result to.

    Returns:
        The expression and its numeric result.
    """
    try:
        value = safe_eval(expression)
    except UnsafeExpression as exc:
        # Re-raised as a ValueError so the sandbox reports it as a permanent
        # tool error rather than something worth retrying. Retrying a rejected
        # expression produces the identical rejection.
        raise ValueError(f"rejected expression: {exc}") from exc
    return {"expression": expression, "result": round(value, precision)}


# --------------------------------------------------------------------------
# corpus search
# --------------------------------------------------------------------------

_INDEX: BM25 = BM25()
_CHUNKS = {c.doc_id: c for c in chunks()}
for _chunk in _CHUNKS.values():
    _INDEX.add(_chunk.doc_id, _chunk.text)


@tool(tags=("search", "read"), constraints={"k": {"minimum": 1, "maximum": 5}, "query": {"minLength": 2}})
def search_docs(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Search the Meridian product documentation and return the best passages.

    Args:
        query: What to look for, in natural language.
        k: How many passages to return.

    Returns:
        A list of matches, each with the document id, title, score and an excerpt.
    """
    results = []
    for doc_id, score in _INDEX.search(query, k=k):
        chunk = _CHUNKS[doc_id]
        results.append(
            {
                "doc_id": doc_id,
                "title": chunk.metadata.get("title", doc_id),
                "score": round(score, 3),
                # Excerpt rather than the full passage: the sandbox caps output
                # size, and a tool that returns everything trains the agent to
                # spend its whole context window on one call.
                "excerpt": chunk.text[:220].rsplit(" ", 1)[0] + "...",
            }
        )
    return results


# --------------------------------------------------------------------------
# unit conversion
# --------------------------------------------------------------------------

Unit = Literal["m", "km", "cm", "mm", "mi", "ft", "in", "kg", "g", "lb", "oz", "c", "f", "k"]

# Metres, kilograms. Temperature is handled separately because it is affine,
# not linear, and pretending otherwise is the classic unit-conversion bug.
_LENGTH = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mi": 1609.344, "ft": 0.3048, "in": 0.0254}
_MASS = {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523125}
_TEMPERATURE = {"c", "f", "k"}


def _family(unit: str) -> str:
    if unit in _LENGTH:
        return "length"
    if unit in _MASS:
        return "mass"
    return "temperature"


def _to_celsius(value: float, unit: str) -> float:
    return {"c": value, "f": (value - 32.0) * 5.0 / 9.0, "k": value - 273.15}[unit]


def _from_celsius(value: float, unit: str) -> float:
    return {"c": value, "f": value * 9.0 / 5.0 + 32.0, "k": value + 273.15}[unit]


@tool(tags=("math", "convert"))
def convert_units(value: float, from_unit: Unit, to_unit: Unit) -> Dict[str, Any]:
    """Convert a quantity between units of length, mass or temperature.

    Args:
        value: The quantity to convert.
        from_unit: The unit the value is currently in.
        to_unit: The unit to convert to.

    Returns:
        The converted value and the units either side of the conversion.
    """
    source, target = _family(from_unit), _family(to_unit)
    if source != target:
        # A well-formed call that is still wrong. The schema cannot express
        # "these two enums must agree", so this check lives in the tool and the
        # message names both families so the model can pick a valid pair.
        raise ValueError(f"cannot convert {source} ({from_unit}) to {target} ({to_unit})")
    if source == "temperature":
        result = _from_celsius(_to_celsius(value, from_unit), to_unit)
    else:
        table = _LENGTH if source == "length" else _MASS
        result = value * table[from_unit] / table[to_unit]
    return {"value": value, "from": from_unit, "to": to_unit, "result": round(result, 6)}


# --------------------------------------------------------------------------
# clock
# --------------------------------------------------------------------------

# Pinned instant: 2023-11-14T22:13:20Z. The clock is injectable and defaults to
# a fixed value so demo output and test assertions are byte-stable. An agent
# that reads the real wall clock cannot be replayed, and replay is the only way
# to debug a multi-step failure after the fact. Call set_clock(time.time) in
# production, where you want the real thing.
FIXED_EPOCH = 1_700_000_000.0
_clock: Callable[[], float] = lambda: FIXED_EPOCH


def set_clock(clock: Callable[[], float]) -> None:
    """Replace the clock the `current_time` tool reads."""
    global _clock
    _clock = clock


@tool(name="current_time", tags=("time",), constraints={"offset_hours": {"minimum": -12, "maximum": 14}})
def current_time(offset_hours: int = 0) -> Dict[str, Any]:
    """Report the current UTC date and time, optionally offset by whole hours.

    Args:
        offset_hours: Hours to add to UTC, for example 6 for Asia/Dhaka.

    Returns:
        The epoch seconds and an ISO-8601 style timestamp for the offset zone.
    """
    epoch = _clock() + offset_hours * 3600
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch))
    sign = "+" if offset_hours >= 0 else "-"
    return {"epoch": int(_clock()), "timestamp": f"{stamp}{sign}{abs(offset_hours):02d}:00",
            "offset_hours": offset_hours}


ALL_TOOLS = [calculate, search_docs, convert_units, current_time]
