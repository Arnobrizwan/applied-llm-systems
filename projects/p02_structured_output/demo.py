"""End-to-end demo for the Structured Output Engine.

Runs four things: the schema built from a dataclass, the repair pipeline
against hand-written broken payloads, a measured success table across several
provider fault rates, and the fallback path with its attempt trace.

Everything here is offline and deterministic. The numbers printed are the
numbers quoted in the README.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from llmkit import EchoLLM

from projects.p02_structured_output.engine import StructuredOutputEngine
from projects.p02_structured_output.extraction import extract_json
from projects.p02_structured_output.schema_builder import schema_from_dataclass
from projects.p02_structured_output.validator import describe_errors, validate

SAMPLES = 40
MAX_ATTEMPTS = 3
FAULT_RATES = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass
class Finding:
    """A single issue pulled out of a support ticket."""

    summary: str = field(metadata={"minLength": 4, "maxLength": 200})
    severity: Literal["low", "medium", "high"] = field(metadata={"description": "Impact on the customer"})
    confidence: float = field(metadata={"minimum": 0.0, "maximum": 1.0})


@dataclass
class TicketTriage:
    """Structured triage of one inbound support ticket."""

    ticket_id: str = field(metadata={"pattern": r"^TKT-\d+$"})
    answer: str = field(metadata={"minLength": 1, "maxLength": 400})
    findings: List[Finding] = field(metadata={"minItems": 1, "maxItems": 5})
    owner: Optional[str] = None


@dataclass
class TicketSummary:
    """The shape used for the measured runs below.

    Same engine, same validator, but no `pattern` rule. EchoLLM synthesises
    placeholder strings such as "ticket_id-652", so a regex constraint fails one
    hundred percent of the time no matter how good the repair and retry paths
    are. Measuring against a constraint the provider structurally cannot satisfy
    would measure the provider, not the engine, so the measured schema states
    only rules a well-behaved model can meet.
    """

    answer: str = field(metadata={"minLength": 1, "maxLength": 400})
    findings: List[Finding] = field(metadata={"minItems": 1, "maxItems": 5})
    owner: Optional[str] = None


BROKEN_PAYLOADS = [
    ("fenced block with a preamble", 'Sure! Here is the JSON you asked for:\n```json\n{"ticket_id": "TKT-9"}\n```'),
    ("trailing commentary", '{"ticket_id": "TKT-9"}\n\nLet me know if you need anything else.'),
    ("single quotes and a trailing comma", "{'ticket_id': 'TKT-9', 'answer': 'ok',}"),
    ("python repr literals", '{"ticket_id": "TKT-9", "resolved": True, "owner": None}'),
    ("truncated mid string", '{\n  "ticket_id": "TKT-9",\n  "answ'),
    ("truncated mid array", '{"ticket_id": "TKT-9", "findings": [{"summary": "disk full"'),
    ("no JSON at all", "I am not able to help with that request."),
]


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def section_schema(schema):
    rule("1. Schema derived from the TicketTriage dataclass")
    print(json.dumps(schema, indent=2))
    print("\nThe prompt and the parsed object come from this one declaration, so a")
    print("renamed field cannot silently drift out of the prompt.")


def section_repairs():
    rule("2. Repair pipeline against hand-written broken output")
    print(f"{'payload':<36} {'parsed':<7} repairs applied")
    print("-" * 78)
    recovered = 0
    for label, payload in BROKEN_PAYLOADS:
        result = extract_json(payload)
        recovered += 1 if result.ok else 0
        repairs = ", ".join(result.repairs) or ("none needed" if result.ok else "-")
        print(f"{label:<36} {str(result.ok):<7} {repairs}")
    print("-" * 78)
    print(f"recovered {recovered}/{len(BROKEN_PAYLOADS)}; the last case is unrecoverable by design,")
    print("because there is no JSON in it to repair and inventing one would be worse.")


def section_validation(schema):
    rule("3. Structured validation errors, the exact text fed back to the model")
    bad = {
        "ticket_id": "9",
        "answer": "",
        "findings": [{"summary": "x", "severity": "critical", "confidence": 1.4}],
        "extra": "invented by the model",
    }
    errors = validate(bad, schema)
    for err in errors:
        print(f"  {err.path:<28} {err.rule:<20} got={err.got!r} expected={err.expected!r}")
    print("\nRendered for the retry prompt:")
    print(describe_errors(errors))


def measure(schema, fallback):
    """Run SAMPLES prompts at each fault rate and count how each one resolved."""
    rows = []
    for fault_rate in FAULT_RATES:
        engine = StructuredOutputEngine(EchoLLM(fault_rate=fault_rate), max_attempts=MAX_ATTEMPTS)
        results = [
            engine.generate(f"Triage support ticket TKT-{100 + i} about a failed deploy.", schema, fallback=fallback)
            for i in range(SAMPLES)
        ]
        first = Counter(r.attempts[0].outcome for r in results)
        source = Counter(r.source for r in results)
        calls = sum(r.n_attempts for r in results)
        rows.append(
            {
                "fault_rate": fault_rate,
                "clean_at_1": first["clean"],
                "repaired_at_1": first["repaired"],
                "success_at_1": first["clean"] + first["repaired"],
                "success_at_k": sum(1 for r in results if r.ok),
                "via_reprompt": source["reprompted"],
                "fallbacks": source["fallback"],
                "calls": calls,
                "repairs": Counter(r for res in results for r in res.repairs_used),
            }
        )
    return rows


def section_table(rows):
    rule(f"4. Success at 1 and at k={MAX_ATTEMPTS}, {SAMPLES} prompts per fault rate")
    print("measured against the TicketSummary schema (see its docstring for why the")
    print("regex rule is excluded from the measurement).\n")
    header = f"{'fault':>6} {'clean@1':>8} {'repair@1':>9} {'succ@1':>8} {'succ@k':>8} {'reprompt':>9} {'fallback':>9} {'calls':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['fault_rate']:>6.2f} "
            f"{row['clean_at_1'] / SAMPLES:>7.0%} "
            f"{row['repaired_at_1'] / SAMPLES:>8.0%} "
            f"{row['success_at_1'] / SAMPLES:>7.0%} "
            f"{row['success_at_k'] / SAMPLES:>7.0%} "
            f"{row['via_reprompt']:>9} "
            f"{row['fallbacks']:>9} "
            f"{row['calls']:>6}"
        )

    print("\nWhere the recovery came from (responses that were not clean on the first call):")
    print(f"{'fault':>6} {'broken':>7} {'fixed by repair':>16} {'fixed by re-prompt':>19} {'unrecovered':>12}")
    for row in rows:
        broken = SAMPLES - row["clean_at_1"]
        if broken == 0:
            print(f"{row['fault_rate']:>6.2f} {broken:>7} {'-':>16} {'-':>19} {'-':>12}")
            continue
        print(
            f"{row['fault_rate']:>6.2f} {broken:>7} "
            f"{row['repaired_at_1'] / broken:>15.0%} "
            f"{row['via_reprompt'] / broken:>18.0%} "
            f"{row['fallbacks'] / broken:>11.0%}"
        )

    merged = Counter()
    for row in rows:
        merged.update(row["repairs"])
    print("\nRepair strategies that fired, all fault rates combined:")
    for name, count in merged.most_common():
        print(f"  {name:<20} {count}")


def section_fallback(schema, fallback):
    rule("5. The fallback path: the caller never sees an exception")
    engine = StructuredOutputEngine(EchoLLM(fault_rate=1.0), max_attempts=1)
    result = engine.generate("Summarise support ticket TKT-500 about a billing error.", schema, fallback=fallback)
    print(f"ok={result.ok} source={result.source} attempts={result.n_attempts}")
    for attempt in result.attempts:
        preview = attempt.raw.replace("\n", " ")[:64]
        print(f"  attempt {attempt.index}: {attempt.outcome:<12} repairs={attempt.repairs} raw={preview!r}")
    if result.errors:
        print("  errors that would have gone back to the model:")
        print("  " + describe_errors(result.errors).replace("\n", "\n  "))
    print(f"  value returned to the caller: {json.dumps(result.value)}")


def main():
    declared_schema = schema_from_dataclass(TicketTriage)
    measured_schema = schema_from_dataclass(TicketSummary)
    fallback = {"answer": "unavailable", "findings": [], "owner": None}

    print("Structured Output Engine")
    print("provider: EchoLLM (offline, deterministic), fault_rate varied per run")

    section_schema(declared_schema)
    section_repairs()
    section_validation(declared_schema)
    section_table(measure(measured_schema, fallback))
    section_fallback(measured_schema, fallback)
    print()


if __name__ == "__main__":
    main()
