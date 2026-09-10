"""Deterministic scorers. These run before any judge because they are free.

Order matters for cost, not just for tidiness. A judged eval on a 500 case set
is 500 model calls per commit; a `contains` check is a string operation. Running
the cheap unambiguous checks first means the judge only ever sees the cases the
cheap checks could not settle, and the harness reports how many judge calls that
saved.

Every scorer returns a `ScoreResult` in [0.0, 1.0] plus a human readable detail
string, because "exact_match: 0.0" with no detail is the reason nobody trusts
eval dashboards.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from llmkit import tokenize

from .dataset import EvalCase

# token F1 is a graded score, so it needs a pass line. 0.5 is a judgement call:
# below half the content words shared, the answer is usually about something else.
TOKEN_F1_PASS = 0.5


@dataclass
class ScoreResult:
    scorer: str
    score: float
    passed: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"scorer": self.scorer, "score": round(self.score, 4),
                "passed": self.passed, "detail": self.detail}


def _norm(text: str) -> str:
    """Casefold, collapse whitespace, drop terminal punctuation.

    Deliberately not stripping internal punctuation: "3, 7 and 14" and "429"
    are answers where punctuation and digits carry the meaning.
    """
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".!?")


def exact_match(prediction: str, case: EvalCase) -> ScoreResult:
    """Normalised string equality. Correct for classification, cruel for prose."""
    ok = _norm(prediction) == _norm(case.expected)
    return ScoreResult("exact_match", 1.0 if ok else 0.0, ok,
                       "" if ok else f"expected {case.expected!r}, got {prediction[:80]!r}")


def contains(prediction: str, case: EvalCase) -> ScoreResult:
    """Required phrase present. The workhorse for open-ended factual answers."""
    ok = _norm(case.expected) in _norm(prediction)
    return ScoreResult("contains", 1.0 if ok else 0.0, ok,
                       "" if ok else f"missing required phrase {case.expected!r}")


def regex(prediction: str, case: EvalCase) -> ScoreResult:
    """Pattern match, used where a family of phrasings is acceptable.

    A bad pattern is a silent scorer failure, so a pattern that does not compile
    fails the case loudly instead of being treated as a miss.
    """
    try:
        pattern = re.compile(case.expected)
    except re.error as exc:
        return ScoreResult("regex", 0.0, False, f"invalid pattern in case {case.id}: {exc}")
    ok = bool(pattern.search(prediction or ""))
    return ScoreResult("regex", 1.0 if ok else 0.0, ok,
                       "" if ok else f"no match for /{case.expected}/")


def token_f1(prediction: str, case: EvalCase) -> ScoreResult:
    """SQuAD-style token F1 against a reference, on content words only.

    Stopwords are dropped (llmkit.tokenize does this) so that padding an answer
    with connective prose cannot inflate the score. Multiset counting means a
    repeated word cannot be matched twice, which is the standard fix for the
    degenerate "the the the the" answer.
    """
    reference = case.metadata.get("reference") or case.expected
    pred_toks = tokenize(prediction or "")
    ref_toks = tokenize(reference or "")
    if not pred_toks or not ref_toks:
        ok = pred_toks == ref_toks
        return ScoreResult("token_f1", 1.0 if ok else 0.0, ok, "empty prediction or reference")
    common = 0
    pool = list(ref_toks)
    for tok in pred_toks:
        if tok in pool:
            pool.remove(tok)
            common += 1
    if common == 0:
        return ScoreResult("token_f1", 0.0, False, "no content word overlap with the reference")
    precision = common / len(pred_toks)
    recall = common / len(ref_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return ScoreResult("token_f1", f1, f1 >= TOKEN_F1_PASS,
                       f"p={precision:.2f} r={recall:.2f}")


# --------------------------------------------------------------------------
# JSON schema validity
# --------------------------------------------------------------------------

def validate_schema(obj: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Validate `obj` against the subset of JSON Schema this repo uses.

    A subset validator instead of the `jsonschema` package because runtime code
    here is standard library only. It covers type, required, properties, enum,
    minimum/maximum, minItems and items, which is every keyword the structured
    output paths in this repo actually emit. Anything richer belongs in a real
    validator and this function would be the wrong place to grow it.
    """
    errors: List[str] = []
    expected = schema.get("type")
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    if expected and expected in checks and not checks[expected](obj):
        return [f"{path}: expected {expected}, got {type(obj).__name__}"]

    if "enum" in schema and obj not in schema["enum"]:
        errors.append(f"{path}: {obj!r} is not one of {schema['enum']}")

    if expected in ("integer", "number") and isinstance(obj, (int, float)):
        if "minimum" in schema and obj < schema["minimum"]:
            errors.append(f"{path}: {obj} < minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            errors.append(f"{path}: {obj} > maximum {schema['maximum']}")

    if isinstance(obj, dict):
        for key in schema.get("required", []):
            if key not in obj:
                errors.append(f"{path}: missing required field {key!r}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in obj:
                errors.extend(validate_schema(obj[key], sub, f"{path}.{key}"))

    if isinstance(obj, list):
        if len(obj) < int(schema.get("minItems", 0)):
            errors.append(f"{path}: {len(obj)} items < minItems {schema['minItems']}")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(obj):
                errors.extend(validate_schema(item, item_schema, f"{path}[{i}]"))
    return errors


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)


def extract_json(text: str) -> Optional[Any]:
    """Pull a JSON value out of a model response, fenced or not.

    Repairing chatty wrappers here rather than failing is deliberate: the eval
    harness measures the system under test, and if the system ships a parser
    that strips fences then the harness must not score it as broken. It does not
    repair invalid JSON, because that would hide a real defect.
    """
    if not text:
        return None
    for candidate in ([m.group(1) for m in _FENCE_RE.finditer(text)] + [text]):
        try:
            return json.loads(candidate.strip())
        except ValueError:
            continue
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            return None
    return None


def json_schema(prediction: str, case: EvalCase) -> ScoreResult:
    """Parse the response and validate it against `case.metadata['schema']`."""
    schema = case.metadata.get("schema")
    if not schema:
        return ScoreResult("json_schema", 0.0, False, f"case {case.id} has no schema in metadata")
    obj = extract_json(prediction)
    if obj is None:
        return ScoreResult("json_schema", 0.0, False, "response did not contain parseable JSON")
    errors = validate_schema(obj, schema)
    return ScoreResult("json_schema", 0.0 if errors else 1.0, not errors,
                       "; ".join(errors[:3]) if errors else "")


ScorerFn = Callable[[str, EvalCase], ScoreResult]

DETERMINISTIC_SCORERS: Dict[str, ScorerFn] = {
    "exact_match": exact_match,
    "contains": contains,
    "regex": regex,
    "token_f1": token_f1,
    "json_schema": json_schema,
}


def score_case(prediction: str, case: EvalCase,
               registry: Optional[Dict[str, ScorerFn]] = None) -> List[ScoreResult]:
    """Run every scorer the case asks for. Unknown scorer names fail loudly."""
    reg = registry or DETERMINISTIC_SCORERS
    results: List[ScoreResult] = []
    for name in case.scorers:
        if name not in reg:
            raise KeyError(f"case {case.id} requests unknown scorer {name!r}")
        results.append(reg[name](prediction, case))
    return results
