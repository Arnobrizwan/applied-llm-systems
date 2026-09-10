"""Complexity scoring over explicit features.

The output is a feature record, not a number. A router that returns 0.71 cannot
be debugged, cannot be argued with, and cannot be corrected when it sends every
SQL question to the expensive tier. This one returns the features it measured
and each feature's contribution to the score, so a routing decision can be read
back in one line and a wrong decision points at the feature that caused it.

The weights below are hand set and are the honest weak point of this module.
They encode a claim ("code is harder than summarisation") that is defensible but
not measured. The right way to set them is a labelled sample of production
traffic where the cheap tier's answers were graded, then fit the weights to
predict where the cheap tier failed. That needs traffic this repo does not have,
so the weights are legible round numbers and this paragraph exists so nobody
mistakes them for calibrated ones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llmkit import count_tokens

TASK_TYPES = ("classification", "extraction", "summarisation", "reasoning", "code")

# Per task type difficulty, on the same 0 to 1 scale as the final score.
TASK_DIFFICULTY: Dict[str, float] = {
    "classification": 0.05,
    "extraction": 0.15,
    "summarisation": 0.25,
    "code": 0.45,
    "reasoning": 0.55,
}

WEIGHTS: Dict[str, float] = {
    "task_type": 1.0,        # applied to TASK_DIFFICULTY, the dominant signal
    "input_length": 0.15,    # long inputs need more attention, not more reasoning
    "output_length": 0.10,
    "math": 0.15,
    "code": 0.10,
    "tools": 0.10,
    "ambiguity": 0.20,
}

_CODE_FENCE = re.compile(r"```|\bdef \w+\(|\bclass \w+\b|\bSELECT\b.+\bFROM\b|=>|\bimport \w+", re.I)
_MATH = re.compile(r"\d+\s*[-+*/^%]\s*\d+|\b\d+(\.\d+)?\s*(percent|%)|\bcompound\b|\bderivative\b"
                   r"|\bcalculate\b|\bcompute\b|\bhow much\b|\bhow many\b", re.I)
_TOOLS = re.compile(r"\bsearch (the )?(web|internet)\b|\blook ?up\b|\bcall the\b|\bquery the\b"
                    r"|\bcurrent\b|\btoday'?s\b|\blatest\b|\breal[- ]time\b|\bfetch\b", re.I)
# "that", "some" and "any" were in this list and are now not: they appear in
# perfectly precise sentences ("a function that parses a header") and made every
# well specified code request look ambiguous.
_AMBIGUOUS = re.compile(r"\b(it|this|they|them|thing|things|stuff|something|somehow"
                        r"|etc|maybe|probably|whatever|unclear|not sure)\b", re.I)
# The prefix ("in 100 words", "no more than 5 bullets") is optional: a bare
# number immediately followed by a length unit is a length instruction wherever
# it appears, and requiring the prefix missed "give me 5 bullets".
_OUTPUT_HINT = re.compile(r"\b(\d{1,4})\s*(word|token|line|bullet|sentence|paragraph)s?\b", re.I)

_TASK_PATTERNS = (
    # "plan" needs an article after it: a bare \bplan\b matched "the plan tiers"
    # and classified a plan-comparison lookup as a reasoning task.
    ("reasoning", re.compile(r"\bwhy\b|\bexplain\b|\bcompare\b|\btrade[- ]?offs?\b|\bdesign\b"
                             r"|\bplan (a|an|the|our|this|for)\b|\bshould (i|we)\b"
                             r"|\bwhich .* better\b|\bprove\b|\broot cause\b"
                             r"|\bstep by step\b|\bimplications?\b|\bstrategy\b", re.I)),
    ("code", re.compile(r"```|\bwrite (a |the )?(function|script|query|regex|class)\b|\brefactor\b"
                        r"|\bunit test\b|\bstack trace\b|\bdebug\b|\bsql\b|\bpython\b|\bjavascript\b", re.I)),
    ("extraction", re.compile(r"\bextract\b|\bpull out\b|\bparse\b|\bas json\b|\bfields?\b"
                              r"|\blist all\b|\bfind (all|every)\b|\bidentify (all|every)\b", re.I)),
    ("summarisation", re.compile(r"\bsummar(y|ise|ize)\b|\btl;?dr\b|\bcondense\b|\bshorten\b"
                                 r"|\bkey points?\b|\bin brief\b|\brewrite\b", re.I)),
    ("classification", re.compile(r"\bclassify\b|\bcategor(y|ise|ize)\b|\blabel\b|\bsentiment\b"
                                  r"|\bspam\b|\bis this\b|\byes or no\b|\bwhich category\b"
                                  r"|\broute this\b|\btag\b", re.I)),
)

# Above this many input tokens the length term is saturated. 4000 rather than a
# model's real context limit: the point is "long enough that a small model starts
# losing the middle", which happens far below the advertised window.
LENGTH_SATURATION_TOKENS = 4000
OUTPUT_SATURATION_TOKENS = 800

DEFAULT_OUTPUT_TOKENS: Dict[str, int] = {
    "classification": 8,
    "extraction": 120,
    "summarisation": 200,
    "code": 400,
    "reasoning": 500,
}


@dataclass
class ComplexityFeatures:
    """Everything the router measured, plus what each measurement contributed."""

    text: str
    input_tokens: int
    task_type: str
    has_math: bool
    has_code: bool
    needs_tools: bool
    ambiguity: float
    required_output_tokens: int
    contributions: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    def explain(self) -> str:
        """One line a human can check against the prompt."""
        drivers = sorted(self.contributions.items(), key=lambda kv: -kv[1])[:3]
        detail = ", ".join(f"{k} +{v:.2f}" for k, v in drivers if v > 0)
        flags = "".join([
            "math " if self.has_math else "", "code " if self.has_code else "",
            "tools " if self.needs_tools else "",
        ]).strip()
        return (f"score {self.score:.2f} [{self.task_type}, {self.input_tokens} in, "
                f"{self.required_output_tokens} out, ambiguity {self.ambiguity:.2f}"
                + (f", {flags}" if flags else "") + f"] driven by {detail or 'nothing'}")

    def to_dict(self) -> Dict[str, object]:
        return {"score": round(self.score, 4), "task_type": self.task_type,
                "input_tokens": self.input_tokens,
                "required_output_tokens": self.required_output_tokens,
                "has_math": self.has_math, "has_code": self.has_code,
                "needs_tools": self.needs_tools, "ambiguity": round(self.ambiguity, 4),
                "contributions": {k: round(v, 4) for k, v in self.contributions.items()}}


def classify_task(text: str) -> str:
    """First matching pattern wins, hardest first.

    Ordering matters more than the patterns do. "Explain why this SQL query is
    slow" is both code and reasoning, and routing it as code would send it to a
    cheaper tier than it needs. Rejected: scoring every pattern and taking the
    best match, which sounds more principled but makes the outcome depend on how
    many synonyms happen to be in each pattern.
    """
    for name, pattern in _TASK_PATTERNS:
        if pattern.search(text or ""):
            return name
    return "summarisation"  # the middle of the difficulty range, not the cheapest


def has_explicit_task_signal(text: str) -> bool:
    """True when a task pattern matched, rather than the default being used.

    Reported by the demo. A request that falls through to the default task type
    is routed on length and ambiguity alone, and knowing how often that happens
    is the difference between "the classifier works" and "the classifier is
    silently abstaining on a sixth of production traffic".
    """
    return any(pattern.search(text or "") for _, pattern in _TASK_PATTERNS)


def measure_ambiguity(text: str) -> float:
    """Share of vague reference words, with a penalty for very short requests.

    Vague words are a proxy, not a definition. The real signal is whether the
    request can be answered without asking a clarifying question, and detecting
    that reliably needs a model call, which would defeat the purpose of a cheap
    pre-routing heuristic.
    """
    words = re.findall(r"[A-Za-z']+", text or "")
    if not words:
        return 1.0
    vague = len(_AMBIGUOUS.findall(text))
    density = min(1.0, vague / max(6, len(words)) * 4.0)
    brevity = 0.3 if len(words) < 6 else 0.0
    return round(min(1.0, density + brevity), 4)


def required_output_tokens(text: str, task_type: str) -> int:
    """Honour an explicit length instruction, otherwise use the task default."""
    match = _OUTPUT_HINT.search(text or "")
    if match:
        count, unit = int(match.group(1)), match.group(2).lower()
        per_unit = {"word": 1.3, "token": 1.0, "line": 12, "bullet": 14,
                    "sentence": 20, "paragraph": 80}[unit]
        return max(1, int(count * per_unit))
    return DEFAULT_OUTPUT_TOKENS.get(task_type, 200)


def score_complexity(text: str, task_type: Optional[str] = None) -> ComplexityFeatures:
    """Measure a request. `task_type` can be supplied when the caller knows it.

    An endpoint that only ever does classification should say so rather than let
    a regex guess, and this is the cheapest correctness win available: the caller
    usually knows the task type and the classifier usually does not.
    """
    text = text or ""
    task = task_type if task_type in TASK_TYPES else classify_task(text)
    input_tokens = count_tokens(text)
    has_code = bool(_CODE_FENCE.search(text))
    has_math = bool(_MATH.search(text))
    needs_tools = bool(_TOOLS.search(text))
    ambiguity = measure_ambiguity(text)
    output_tokens = required_output_tokens(text, task)

    contributions = {
        "task_type": WEIGHTS["task_type"] * TASK_DIFFICULTY[task],
        "input_length": WEIGHTS["input_length"] * min(1.0, input_tokens / LENGTH_SATURATION_TOKENS),
        "output_length": WEIGHTS["output_length"] * min(1.0, output_tokens / OUTPUT_SATURATION_TOKENS),
        "math": WEIGHTS["math"] if has_math else 0.0,
        "code": WEIGHTS["code"] if has_code else 0.0,
        "tools": WEIGHTS["tools"] if needs_tools else 0.0,
        "ambiguity": WEIGHTS["ambiguity"] * ambiguity,
    }
    score = min(1.0, sum(contributions.values()))

    return ComplexityFeatures(
        text=text, input_tokens=input_tokens, task_type=task, has_math=has_math,
        has_code=has_code, needs_tools=needs_tools, ambiguity=ambiguity,
        required_output_tokens=output_tokens, contributions=contributions, score=score,
    )


def distribution(features: List[ComplexityFeatures]) -> Dict[str, int]:
    """Task type counts over a workload. Used in the demo report."""
    out: Dict[str, int] = {t: 0 for t in TASK_TYPES}
    for f in features:
        out[f.task_type] = out.get(f.task_type, 0) + 1
    return out
