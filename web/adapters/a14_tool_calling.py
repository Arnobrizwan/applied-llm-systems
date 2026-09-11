"""Web adapter for project 14, the tool-calling framework.

The visitor types a request. A small deterministic router in this file picks
the tool and drafts the arguments, standing in for the model's decision, since
this page has no API key behind it. Everything after that point is the
project's own code: the generated schema, argument coercion, and the sandbox
that decides whether the call is allowed to happen at all.

The sandbox refusals at the bottom run on every request, so a visitor who never
types anything adversarial still gets to watch a shell injection, a runaway
exponent and an attribute walk get turned down.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from projects.p14_tool_calling.registry import ToolRegistry
from projects.p14_tool_calling.sandbox import ToolSandbox
from projects.p14_tool_calling.schema import coerce_arguments
from projects.p14_tool_calling.tools import ALL_TOOLS

NUMBER = 14
SLUG = "tool-calling"
TITLE = "Tool-Calling Framework"
TAGLINE = "Ask for a calculation or a lookup and watch the request turn into a checked tool call, including the ones the sandbox refuses to run."

WHAT_IT_DOES = """When an assistant can run code on your behalf, the dangerous part is not the answer, it is everything between the request and the call. The arguments arrive as text and may be the wrong type. The tool name may not exist. The expression may be an attempt to open a file, import a module or run an exponent big enough to pin a processor for a week.

Type a request and this page shows the whole path. Which of the four tools was picked, the exact description and parameter list that tool publishes, the arguments as they were drafted, what they looked like after being checked and converted, and then the sandbox: allowlist, time budget, retries, output size cap, and the one line of result the assistant would actually see.

At the bottom, a fixed set of hostile inputs runs on every request so you can watch them get turned down and read the reason for each one."""

INPUT_LABEL = "Ask for a calculation, a lookup or a conversion"
PLACEHOLDER = "what is 600*60/1000"

EXAMPLES = [
    "what is 600*60/1000",
    "search the docs for rate limits",
    "convert 5 miles to km",
    '__import__("os").system("echo pwned")',
]

SOURCE = "projects/p14_tool_calling"

REGISTRY = ToolRegistry()
REGISTRY.register_all(ALL_TOOLS)
TOOL_NAMES = REGISTRY.names()

# Fixed inputs that must be refused, run on every request.
REFUSALS: List[Tuple[str, str]] = [
    ('__import__("os").system("echo pwned")', "shell command through an import"),
    ("().__class__.__base__.__subclasses__()", "attribute walk from a literal to every loaded class"),
    ("2 ** 10000000", "arithmetic that never returns"),
    ("open('/etc/passwd').read()", "reading a file off the host"),
    ("eval('1+1')", "nested evaluation"),
    ("1/0", "ordinary division by zero"),
]

_UNITS = {
    "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "km": "km", "kilometre": "km", "kilometres": "km", "kilometer": "km", "kilometers": "km",
    "cm": "cm", "centimetre": "cm", "centimetres": "cm", "centimeters": "cm",
    "mm": "mm", "millimetre": "mm", "millimetres": "mm",
    "mi": "mi", "mile": "mi", "miles": "mi",
    "ft": "ft", "foot": "ft", "feet": "ft",
    "in": "in", "inch": "in", "inches": "in",
    "kg": "kg", "kilo": "kg", "kilos": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gram": "g", "grams": "g",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "c": "c", "celsius": "c", "centigrade": "c",
    "f": "f", "fahrenheit": "f",
    "k": "k", "kelvin": "k",
}

_CONVERT_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:degrees\s+)?([a-zA-Z]+)\s+(?:to|in|into|as)\s+([a-zA-Z]+)")
_CODEY = re.compile(r"__|\bimport\b|\blambda\b|\beval\b|\bexec\b|\bglobals\b|\bopen\s*\(|\)\s*\.")
_ARITHMETIC = re.compile(r"\d\s*[-+*/%^]|\bsqrt\b|\blog\b|\bround\b|\babs\b")
_LEAD_IN = re.compile(
    r"(?i)^\s*(please\s+)?(what(?:'s| is| are)?|whats|calculate|compute|evaluate|work out|tell me)\s+")
_SEARCHY = re.compile(r"(?i)\b(search|docs?|documentation|look ?up|find|how|why|when|where|which|explain)\b")

_LINE = "-" * 74


def _cell(value: Any, width: int) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) > width - 2:
        text = text[: width - 5] + "..."
    return text.ljust(width)


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + " ... [clipped]"


def _expression_of(text: str) -> str:
    expression = _LEAD_IN.sub("", text).strip()
    return expression.rstrip("?.").strip() or text.strip()


def route(text: str) -> Tuple[str, Dict[str, Any], str]:
    """Pick a tool and draft its arguments. This stands in for the model.

    Arguments are drafted as JSON strings where a model would plausibly emit
    strings, because that is the case argument coercion exists to handle.
    """
    lowered = text.lower()

    named = re.search(r"(?i)\b(?:use|call|run) (?:the )?([a-z_][a-z0-9_]*)\b", text)
    if named and named.group(1) not in TOOL_NAMES and named.group(1) not in ("it", "this", "that"):
        return named.group(1), {"query": text[:80]}, "you named a tool directly, so it is looked up by that name"

    if _CODEY.search(text):
        return "calculate", {"expression": text.strip(), "precision": "4"}, \
            "the text looks like code rather than a question, so it goes to the calculator, which refuses anything that is not arithmetic"

    match = _CONVERT_RE.search(text)
    if match and (match.group(2).lower() in _UNITS or match.group(3).lower() in _UNITS):
        return "convert_units", {
            "value": match.group(1),
            "from_unit": _UNITS.get(match.group(2).lower(), match.group(2).lower()),
            "to_unit": _UNITS.get(match.group(3).lower(), match.group(3).lower()),
        }, f"'{match.group(1)} {match.group(2)} to {match.group(3)}' is a unit conversion"

    if re.search(r"(?i)\b(time|clock|what hour)\b", lowered):
        offset = re.search(r"(?i)utc\s*([+-]\d{1,2})", text)
        return "current_time", {"offset_hours": offset.group(1) if offset else "0"}, \
            "the request asks about the time"

    if _ARITHMETIC.search(text) or re.search(r"(?i)^\s*(calculate|compute|evaluate|work out)\b", text):
        return "calculate", {"expression": _expression_of(text), "precision": "4"}, \
            "the request contains an arithmetic expression"

    # Anything left over is treated as a documentation question, which is the
    # least destructive default of the four.
    why = ("it reads as a question about the product, so the documentation is searched"
           if _SEARCHY.search(text) or text.strip().endswith("?")
           else "nothing else matched, so it falls back to searching the documentation")
    return "search_docs", {"query": text.strip()[:200], "k": "3"}, why


def _schema_lines(spec) -> List[str]:
    out: List[str] = []
    properties = spec.parameters.get("properties") or {}
    required = spec.parameters.get("required") or []
    for name, prop in properties.items():
        declared = prop.get("type", "string")
        declared = "/".join(declared) if isinstance(declared, list) else str(declared)
        bits = []
        if name in required:
            bits.append("required")
        else:
            bits.append(f"default={prop.get('default')!r}")
        for key in ("minimum", "maximum", "minLength", "maxLength"):
            if key in prop:
                bits.append(f"{key}={prop[key]}")
        if "enum" in prop:
            options = prop["enum"]
            shown = "|".join(str(v) for v in options[:8])
            bits.append(f"one of {shown}" + ("|..." if len(options) > 8 else ""))
        out.append("  " + _cell(name, 14) + _cell(declared, 10) + ", ".join(bits))
        description = (prop.get("description") or "").strip().replace("\n", " ")
        if description:
            out.append("      " + _clip(re.sub(r"\s+", " ", description), 200))
    return out


def _refusal_table() -> List[str]:
    # Its own sandbox, with the failure budget lifted, so each row shows the
    # reason it was refused rather than the circuit breaker that the earlier
    # rows would otherwise have tripped. The breaker gets its own row below.
    sandbox = ToolSandbox(REGISTRY, allowlist=TOOL_NAMES, breaker_threshold=99)
    out = [f"{'input':<44}{'outcome':<15}why"]
    for expression, _note in REFUSALS:
        result = sandbox.call("calculate", {"expression": expression})
        reason = (result.error or {}).get("message", "") if not result.ok else "allowed"
        reason = reason.replace("ValueError: rejected expression: ", "").replace("ValueError: ", "")
        out.append(_cell(expression, 44) + _cell(result.audit.outcome if result.audit else "?", 15)
                   + _clip(reason, 68))

    unknown = sandbox.call("delete_database", {"table": "customers"})
    out.append(_cell("delete_database(table=customers)", 44)
               + _cell(unknown.audit.outcome if unknown.audit else "?", 15)
               + "there is no tool by that name")

    locked = ToolSandbox(REGISTRY, allowlist=["search_docs"])
    denied = locked.call("calculate", {"expression": "1+1"})
    out.append(_cell("calculate, on a read-only agent", 44)
               + _cell(denied.audit.outcome if denied.audit else "?", 15)
               + "this agent's allowlist holds search_docs only")

    bad_args = sandbox.call("calculate", {"expression": "1+1", "precision": 99})
    out.append(_cell("calculate(precision=99)", 44)
               + _cell(bad_args.audit.outcome if bad_args.audit else "?", 15)
               + _clip((bad_args.error or {}).get("message", ""), 60))

    # Three refusals in a row on one tool, on a sandbox with the normal budget.
    tripping = ToolSandbox(REGISTRY, allowlist=TOOL_NAMES, breaker_threshold=3)
    for _ in range(3):
        tripping.call("calculate", {"expression": "open('/etc/passwd')"})
    tripped = tripping.call("calculate", {"expression": "1+1"})
    out.append(_cell("1+1, after three refusals in a row", 44)
               + _cell(tripped.audit.outcome if tripped.audit else "?", 15)
               + "the tool is taken out of service for a cooldown")
    return out


def run(user_input: str) -> str:
    try:
        text = (user_input or "").strip() or EXAMPLES[0]
        text = text[:400]
        sandbox = ToolSandbox(REGISTRY, allowlist=TOOL_NAMES, max_output_bytes=4096)
        out: List[str] = []

        out.append("THE REQUEST")
        out.append("  " + _clip(text, 300))
        out.append("")

        # -- step 1: which tool ------------------------------------------
        name, drafted, reason = route(text)
        out.append("STEP 1  WHICH TOOL")
        out.append(_LINE)
        out.append("  available: " + ", ".join(TOOL_NAMES))
        out.append(f"  picked:    {name}")
        out.append("  because:   " + reason)
        out.append("  This page has no API key, so the choice above is made by a small router")
        out.append("  in the page rather than by a model. Everything below it is the real thing.")
        out.append("")

        spec = REGISTRY.find(name)

        # -- step 2: the published schema --------------------------------
        out.append("STEP 2  WHAT THAT TOOL PUBLISHES (generated from its signature and docstring)")
        out.append(_LINE)
        if spec is None:
            out.append(f"  Nothing. No tool called {name!r} is registered, so there is no schema")
            out.append("  and no arguments to check. The sandbox refuses it below.")
        else:
            out.append(f"  name:        {spec.qualified_name}")
            out.append(f"  description: {_clip(spec.description, 160)}")
            out.append(f"  tags:        {', '.join(spec.tags) or 'none'}   time budget: {spec.timeout_s:g}s")
            out.append("  parameters:")
            out.extend(_schema_lines(spec))
        out.append("")

        # -- step 3: arguments -------------------------------------------
        out.append("STEP 3  ARGUMENTS")
        out.append(_LINE)
        out.append("  drafted:  " + _clip(json.dumps(drafted), 300))
        if spec is None:
            out.append("  Skipped. An unknown tool name never reaches the argument checker,")
            out.append("  which is deliberate: attacker-shaped input should be turned away first.")
        else:
            coerced, arg_errors = coerce_arguments(spec.parameters, drafted)
            out.append("  checked:  " + _clip(json.dumps(coerced, default=str), 300))
            changes = [f"{k}: {drafted[k]!r} -> {coerced[k]!r}"
                       for k in coerced if k in drafted and coerced[k] != drafted[k]]
            defaults = [f"{k}: filled in as {coerced[k]!r}" for k in coerced if k not in drafted]
            for line in changes + defaults:
                out.append("            " + line)
            if arg_errors:
                out.append("  problems, written back for the caller to fix:")
                for err in arg_errors:
                    out.append(f"    {_cell(err.path, 16)}{_cell(err.rule, 16)}{_clip(err.message, 70)}")
            else:
                out.append("  problems: none")
        out.append("")

        # -- step 4: the sandbox -----------------------------------------
        out.append("STEP 4  THE SANDBOX")
        out.append(_LINE)
        result = sandbox.call(name, drafted)
        audit = result.audit
        allowed = "yes" if name in TOOL_NAMES else "no, not in the allowlist"
        out.append(f"  on the allowlist:  {allowed}")
        out.append("  circuit breaker:   closed, this sandbox is fresh for your request")
        out.append(f"  outcome:           {audit.outcome if audit else 'unknown'}")
        if audit:
            out.append(f"  cost:              {audit.duration_ms:.2f} ms, {audit.attempts} attempt(s), "
                       f"{audit.result_bytes} bytes returned")
        out.append("")
        if result.ok:
            out.append("  result:")
            out.append(_clip(json.dumps(result.value, indent=2, default=str), 1800))
        else:
            error = result.error or {}
            out.append(f"  refused with code {error.get('code')}:")
            out.append("    " + _clip(error.get("message", ""), 400))
            details = error.get("details")
            if details:
                out.append("    details: " + _clip(json.dumps(details, default=str), 260))
        out.append("")
        out.append("  the single line the assistant is given back:")
        out.append("    " + _clip(result.observation(), 300))
        out.append("")

        # -- step 5: the standing refusals -------------------------------
        out.append("WHAT THE SANDBOX TURNS DOWN (run fresh on every request)")
        out.append(_LINE)
        out.extend(_refusal_table())
        out.append(_LINE)
        out.append("None of these are matched against a list of banned words. The calculator")
        out.append("parses the expression and allows a short list of arithmetic node types,")
        out.append("so anything it has not been taught about is refused by default.")

        return "\n".join(out)
    except Exception as exc:  # an adapter must never take the page down
        return f"This demo could not run: {type(exc).__name__}: {exc}"
