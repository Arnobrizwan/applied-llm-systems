"""Web adapter for project 02, the structured output engine.

The visitor pastes model output that is broken in one of the ways real model
output is broken, and the page runs the project's own repair pipeline, its
validator and its bounded retry loop over it.

The stand-in model here is `ScriptedLLM`, replaying whatever the visitor pasted
on every attempt. That is deliberate: it makes the retry loop show its real
worst case (a model that never corrects itself), so the attempt budget running
out and the typed fallback coming back are genuine, not narrated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from llmkit import ScriptedLLM

from projects.p02_structured_output.engine import StructuredOutputEngine
from projects.p02_structured_output.extraction import REPAIR_NAMES, extract_json
from projects.p02_structured_output.schema_builder import schema_from_dataclass
from projects.p02_structured_output.validator import describe_errors, validate

NUMBER = 2
SLUG = "structured-output"
TITLE = "Structured Output Engine"
TAGLINE = "Paste model output that is broken in one of the usual ways and watch it get repaired, checked, and either fixed or safely replaced."

WHAT_IT_DOES = """A model asked for JSON returns JSON most of the time. The rest of the time it comes back wrapped in a code fence, with a chatty sentence in front of it, with the wrong kind of quotes, or cut off halfway because the reply ran out of room. Code that simply tries to read it falls over.

Paste any of that here. The page runs six repair steps in order and tells you which ones fired and what the text looked like afterwards. Then it checks the result against the exact shape a support ticket triage is supposed to have and lists every problem it finds, with the field each one is in.

Finally it runs the retry loop. On this page the stand-in model just repeats what you pasted, so if the text can be repaired you see the finished object, and if it cannot you see the attempt budget run out and the safe default that the rest of the application receives instead of a crash."""

INPUT_LABEL = "Paste broken model output"
PLACEHOLDER = 'Sure! Here is the JSON:\n```json\n{"ticket_id": "TKT-9"}\n```'

EXAMPLES = [
    'Sure! Here is the JSON you asked for:\n```json\n{"ticket_id": "TKT-4417", "answer": "The invoice retry failed because the card on file expired.", '
    '"findings": [{"summary": "Card expired on file", "severity": "high", "confidence": 0.91}], "owner": "billing"}\n```\nHope that helps!',

    "{'ticket_id': 'TKT-8802', 'answer': 'Rate limit hit on the search endpoint.', "
    "'findings': [{'summary': 'Burst above 60 requests per minute', 'severity': 'medium', 'confidence': 0.74},], 'owner': None}",

    '{"ticket_id": "TKT-1290", "answer": "Webhook retries stop after six attempts.", '
    '"findings": [{"summary": "Endpoint returned 500 four times", "severity": "medi',

    '{"ticket_id": "T9", "answer": "", "findings": [{"summary": "x", "severity": "critical", "confidence": 1.4}], "sentiment": "angry"}',
]

SOURCE = "projects/p02_structured_output"


# -- the shape the output has to have -------------------------------------
# Declared once, as a dataclass. The schema in the prompt and the object the
# application consumes both come from this, so they cannot drift apart.

@dataclass
class Finding:
    """One issue found in a support ticket."""

    summary: str = field(metadata={"minLength": 4, "maxLength": 200})
    severity: Literal["low", "medium", "high"] = field(
        metadata={"description": "How badly the customer is affected"})
    confidence: float = field(metadata={"minimum": 0.0, "maximum": 1.0})


@dataclass
class TicketTriage:
    """Structured triage of one inbound support ticket."""

    ticket_id: str = field(metadata={"minLength": 3, "maxLength": 24})
    answer: str = field(metadata={"minLength": 1, "maxLength": 400})
    findings: List[Finding] = field(metadata={"minItems": 1, "maxItems": 5})
    owner: Optional[str] = None


SCHEMA: Dict[str, Any] = schema_from_dataclass(TicketTriage)

# What the caller gets when the model will not comply. Deep-copied on the way
# out by the engine, so one caller cannot mutate the next caller's default.
FALLBACK: Dict[str, Any] = {
    "ticket_id": "TKT-000",
    "answer": "Automatic triage was not possible. A human has been paged.",
    "findings": [{"summary": "Triage failed, needs a human", "severity": "medium", "confidence": 0.0}],
    "owner": None,
}

PROMPT = "Triage this inbound support ticket and return the result as JSON."

_WHAT_REPAIRS_DO = {
    "unfence": "take the body of the code fence",
    "slice_to_json": "drop the prose either side of the JSON",
    "python_literals": "rewrite True / False / None as true / false / null",
    "single_quotes": "turn single-quoted strings into double-quoted ones",
    "trailing_commas": "remove a comma before a closing brace or bracket",
    "close_truncated": "close an unterminated string and any open brackets",
}

_LINE = "-" * 74


def _clip(text: str, limit: int) -> str:
    text = text if isinstance(text, str) else str(text)
    return text if len(text) <= limit else text[:limit] + " ... [clipped]"


def _cell(text: Any, width: int) -> str:
    """One column of a fixed-width table: truncated, then padded."""
    text = text if isinstance(text, str) else str(text)
    if len(text) > width - 2:
        text = text[: width - 5] + "..."
    return text.ljust(width)


def _type_of(prop: Dict[str, Any]) -> str:
    declared = prop.get("type", "string")
    return "/".join(declared) if isinstance(declared, list) else str(declared)


def _rules_of(prop: Dict[str, Any]) -> str:
    bits = []
    for key in ("minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems"):
        if key in prop:
            bits.append(f"{key}={prop[key]}")
    if "enum" in prop:
        bits.append("one of " + "|".join(str(v) for v in prop["enum"]))
    return ", ".join(bits)


def _schema_lines(schema: Dict[str, Any], indent: int = 2) -> List[str]:
    out: List[str] = []
    for name, prop in (schema.get("properties") or {}).items():
        pad = " " * indent
        out.append(f"{pad}{name:<12}{_type_of(prop):<14}{_rules_of(prop)}".rstrip())
        if prop.get("type") == "array" and isinstance(prop.get("items"), dict):
            out.extend(_schema_lines(prop["items"], indent + 4))
        elif prop.get("type") == "object" and prop.get("properties"):
            out.extend(_schema_lines(prop, indent + 4))
    return out


def run(user_input: str) -> str:
    try:
        raw = (user_input or "").strip() or EXAMPLES[0]
        raw = raw[:4000]
        out: List[str] = []

        out.append("WHAT YOU PASTED")
        out.append(_clip(raw, 700))
        out.append("")

        # -- step 1: the repair pipeline ---------------------------------
        extraction = extract_json(raw)
        fired = list(extraction.repairs)
        out.append("STEP 1  REPAIR PIPELINE, IN ORDER")
        out.append(_LINE)
        out.append(f"{'step':<18}{'fired':<8}what it does")
        for name in REPAIR_NAMES:
            out.append(f"{name:<18}{('yes' if name in fired else 'no'):<8}{_WHAT_REPAIRS_DO[name]}")
        out.append(_LINE)
        if extraction.ok and not fired:
            out.append("It parsed as it was. No repair needed.")
        elif extraction.ok:
            out.append(f"Parsed after {len(fired)} repair(s): " + ", ".join(fired))
            out.append("Text after repair: " + _clip(extraction.text, 400))
        else:
            out.append(f"Could not be parsed. Repairs tried: {', '.join(fired) or 'none applied'}")
            out.append(f"Last parser error: {extraction.error}")
            out.append("Nothing here invents a missing value. If there is no JSON in the text,")
            out.append("failing is the correct answer and the retry loop below takes over.")
        out.append("")

        # -- step 2: the schema ------------------------------------------
        out.append("STEP 2  THE SHAPE IT HAS TO HAVE (built from the TicketTriage dataclass)")
        out.append(_LINE)
        out.extend(_schema_lines(SCHEMA))
        out.append(f"  required: {', '.join(SCHEMA.get('required', []))}")
        out.append("  extra keys the model invents are reported, not silently dropped")
        out.append("")

        # -- step 3: validation ------------------------------------------
        out.append("STEP 3  VALIDATION")
        out.append(_LINE)
        if not extraction.ok:
            out.append("Skipped, because there was nothing parseable to check.")
        else:
            errors = validate(extraction.value, SCHEMA)
            if not errors:
                out.append("No problems. Every rule above is satisfied.")
            else:
                out.append(f"{'where':<28}{'rule':<20}{'got':<14}expected")
                for err in errors[:10]:
                    out.append(_cell(err.path, 28) + _cell(err.rule, 20)
                               + _cell(repr(err.got), 14) + _clip(repr(err.expected), 28))
                if len(errors) > 10:
                    out.append(f"... and {len(errors) - 10} more")
                out.append("")
                out.append("Sent back to the model verbatim, which is the whole point of keeping")
                out.append("the errors structured rather than returning true or false:")
                out.append(describe_errors(errors))
        out.append("")

        # -- step 4: the retry loop --------------------------------------
        engine = StructuredOutputEngine(ScriptedLLM([raw]), max_attempts=3)
        result = engine.generate(PROMPT, SCHEMA, fallback=FALLBACK)

        out.append("STEP 4  THE RETRY LOOP (stand-in model replays your text every time)")
        out.append(_LINE)
        out.append(f"{'attempt':<10}{'parsed':<9}{'outcome':<14}{'repairs':<30}problems")
        for attempt in result.attempts:
            out.append(
                _cell(attempt.index + 1, 10) + _cell("yes" if attempt.parsed else "no", 9)
                + _cell(attempt.outcome, 14) + _cell(", ".join(attempt.repairs) or "-", 30)
                + str(len(attempt.errors))
            )
        out.append(_LINE)

        if result.ok:
            out.append(f"Valid on attempt {result.n_attempts}, source: {result.source}")
            out.append("The caller receives this object:")
            out.append(_clip(json.dumps(result.value, indent=2), 1400))
        else:
            out.append(f"Every attempt failed, so the loop gave up after {result.n_attempts} tries.")
            out.append("Nothing raised. The caller receives the typed fallback, flagged as not ok,")
            out.append("and decides for itself whether that is worth an error page:")
            out.append(json.dumps(FALLBACK, indent=2))

        return "\n".join(out)
    except Exception as exc:  # an adapter must never take the page down
        return f"This demo could not run: {type(exc).__name__}: {exc}"
