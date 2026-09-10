"""LLM-as-judge: a rubric judge, a pairwise judge, and the bias controls.

A judge is a measuring instrument, and an uncalibrated instrument is worse than
no instrument because it produces numbers people act on. The two biases that
matter most in practice are handled here in code rather than in a caveat:

Position bias. A pairwise judge prefers whichever answer it read first, by 5 to
15 points in most published measurements. `PairwiseJudge` therefore asks twice,
with the order swapped, and only calls a winner when both orders agree. When
they disagree the result is recorded as a tie AND counted as a position-bias
event, so the flip rate is a reported metric rather than a hidden error term.
Rejected alternative: randomising the order per case. That halves the call cost
but converts a measurable bias into unmeasurable variance.

Verbosity bias. Judges reward long answers. This is measured rather than
assumed: `verbosity_bias` correlates the judge score against the answer's token
length across the run. A strong positive correlation on a set where length is
not supposed to matter is a signal to distrust the judge, and the harness prints
it next to the score.

Self-preference bias is not handled here. The only real fix is a judge from a
different model family than the system under test, which is a deployment
decision, not something this file can do.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llmkit import EchoLLM, LLMProvider, count_tokens, system, user

from .dataset import EvalCase
from .scorers import extract_json
from .stats import mean, pearson_r

RUBRIC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["faithfulness", "relevance", "completeness", "reason"],
    "properties": {
        "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
        "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
}

PAIRWISE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["winner", "reason"],
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string"},
    },
}

RUBRIC_INSTRUCTIONS = """You are grading one answer against a reference. Score each
dimension on an integer scale of 1 to 5 and return JSON only.

faithfulness: is every claim supported by the reference? A fluent answer that adds
an unsupported specific scores 1, not 3.
relevance: does it answer the question that was asked, without padding?
completeness: does it contain the whole answer, not the first half of it?

Length is not a dimension. Do not reward a longer answer for being longer."""

PAIRWISE_INSTRUCTIONS = """Two systems answered the same question. Decide which
answer is better on correctness first and concision second, and return JSON only.
Answer "tie" if neither is clearly better. The order in which the answers appear
is arbitrary and carries no information."""


def _rescale(raw: int) -> float:
    """Map a 1 to 5 rubric integer onto [0, 1].

    Not raw/5: a score of 1 is the floor of the scale, not 20 percent quality, and
    treating it as 0.2 compresses the reportable range and flatters a bad system.
    """
    return (max(1, min(5, int(raw))) - 1) / 4.0


@dataclass
class RubricVerdict:
    case_id: str
    faithfulness: float
    relevance: float
    completeness: float
    reason: str = ""
    parse_error: str = ""
    answer_tokens: int = 0

    @property
    def overall(self) -> float:
        """Unweighted mean. Weighting the dimensions needs evidence about which
        one predicts user harm for a given product, and inventing weights here
        would be a fabricated number in a file about not fabricating numbers."""
        return (self.faithfulness + self.relevance + self.completeness) / 3.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "faithfulness": round(self.faithfulness, 4),
            "relevance": round(self.relevance, 4),
            "completeness": round(self.completeness, 4),
            "overall": round(self.overall, 4),
            "reason": self.reason,
            "parse_error": self.parse_error,
            "answer_tokens": self.answer_tokens,
        }


class RubricJudge:
    """Scores one answer on faithfulness, relevance and completeness."""

    def __init__(self, llm: Optional[LLMProvider] = None, temperature: float = 0.0):
        # Temperature 0 because a judge that disagrees with itself on a re-run
        # makes every downstream interval too narrow: the run-to-run variance
        # never enters the bootstrap, which only resamples cases.
        self.llm = llm or EchoLLM()
        self.temperature = temperature
        self.calls = 0

    def _prompt(self, case: EvalCase, prediction: str) -> List[Any]:
        reference = case.metadata.get("reference") or case.expected or "(no reference supplied)"
        return [
            system(RUBRIC_INSTRUCTIONS),
            user(
                f"Question:\n{case.input.strip() or '(empty)'}\n\n"
                f"Reference:\n{reference}\n\n"
                f"Answer under review:\n{prediction.strip() or '(empty)'}"
            ),
        ]

    def judge(self, case: EvalCase, prediction: str) -> RubricVerdict:
        self.calls += 1
        resp = self.llm.complete(self._prompt(case, prediction),
                                 json_schema=RUBRIC_SCHEMA, temperature=self.temperature)
        obj = extract_json(resp.text)
        tokens = count_tokens(prediction)
        if not isinstance(obj, dict):
            # A judge that fails to parse scores zero and says so. Defaulting to
            # a mid score would quietly lift the average of a broken run.
            return RubricVerdict(case.id, 0.0, 0.0, 0.0, parse_error="judge output was not JSON",
                                 answer_tokens=tokens)
        try:
            return RubricVerdict(
                case_id=case.id,
                faithfulness=_rescale(obj["faithfulness"]),
                relevance=_rescale(obj["relevance"]),
                completeness=_rescale(obj["completeness"]),
                reason=str(obj.get("reason", ""))[:200],
                answer_tokens=tokens,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return RubricVerdict(case.id, 0.0, 0.0, 0.0,
                                 parse_error=f"malformed judge output: {exc}", answer_tokens=tokens)


def verbosity_bias(verdicts: Sequence[RubricVerdict]) -> Dict[str, float]:
    """Correlate judge score with answer length across a run.

    Interpretation, stated so the number is not over-read: this is a diagnostic,
    not a proof. On a set where longer answers really are better, a positive r is
    correct behaviour. It is a red flag when it is high on a set of short factual
    questions, which is exactly the set this harness ships with.
    """
    if len(verdicts) < 2:
        return {"r": 0.0, "n": len(verdicts), "flagged": False}
    r = pearson_r([float(v.answer_tokens) for v in verdicts], [v.overall for v in verdicts])
    return {"r": round(r, 4), "n": len(verdicts), "flagged": abs(r) > 0.3}


@dataclass
class PairwiseVerdict:
    case_id: str
    winner: str            # "A" | "B" | "tie"
    consistent: bool       # both orderings agreed
    forward: str = ""      # verdict with A shown first
    reversed_: str = ""    # verdict with B shown first
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"case_id": self.case_id, "winner": self.winner, "consistent": self.consistent,
                "forward": self.forward, "reversed": self.reversed_, "reason": self.reason}


@dataclass
class PairwiseSummary:
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    position_flips: int = 0
    verdicts: List[PairwiseVerdict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.a_wins + self.b_wins + self.ties

    @property
    def position_bias_rate(self) -> float:
        """Share of cases where swapping the order changed the answer."""
        return self.position_flips / self.total if self.total else 0.0

    @property
    def a_win_rate(self) -> float:
        return self.a_wins / self.total if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"total": self.total, "a_wins": self.a_wins, "b_wins": self.b_wins,
                "ties": self.ties, "a_win_rate": round(self.a_win_rate, 4),
                "position_flips": self.position_flips,
                "position_bias_rate": round(self.position_bias_rate, 4)}


class PairwiseJudge:
    """Compares two systems on the same case, in both orders."""

    def __init__(self, llm: Optional[LLMProvider] = None, temperature: float = 0.0):
        self.llm = llm or EchoLLM()
        self.temperature = temperature
        self.calls = 0

    def _ask(self, case: EvalCase, first: str, second: str) -> Tuple[str, str]:
        self.calls += 1
        resp = self.llm.complete(
            [
                system(PAIRWISE_INSTRUCTIONS),
                user(
                    f"Question:\n{case.input.strip() or '(empty)'}\n\n"
                    f"Answer A:\n{first.strip() or '(empty)'}\n\n"
                    f"Answer B:\n{second.strip() or '(empty)'}"
                ),
            ],
            json_schema=PAIRWISE_SCHEMA,
            temperature=self.temperature,
        )
        obj = extract_json(resp.text)
        if not isinstance(obj, dict) or obj.get("winner") not in ("A", "B", "tie"):
            return "tie", "unparseable judge output"
        return obj["winner"], str(obj.get("reason", ""))[:200]

    def compare(self, case: EvalCase, answer_a: str, answer_b: str) -> PairwiseVerdict:
        """Two calls per case: A first, then B first.

        The second call's labels are flipped back before comparison, so "both
        orders said the same system won" is a statement about systems, not about
        slot names.
        """
        fwd, reason = self._ask(case, answer_a, answer_b)
        rev_raw, _ = self._ask(case, answer_b, answer_a)
        rev = {"A": "B", "B": "A", "tie": "tie"}[rev_raw]
        consistent = fwd == rev
        return PairwiseVerdict(
            case_id=case.id,
            winner=fwd if consistent else "tie",
            consistent=consistent,
            forward=fwd,
            reversed_=rev,
            reason=reason,
        )

    def compare_many(self, cases: Sequence[EvalCase], answers_a: Sequence[str],
                     answers_b: Sequence[str]) -> PairwiseSummary:
        if not (len(cases) == len(answers_a) == len(answers_b)):
            raise ValueError("cases and both answer lists must be the same length")
        summary = PairwiseSummary()
        for case, a, b in zip(cases, answers_a, answers_b):
            v = self.compare(case, a, b)
            summary.verdicts.append(v)
            if not v.consistent:
                summary.position_flips += 1
            if v.winner == "A":
                summary.a_wins += 1
            elif v.winner == "B":
                summary.b_wins += 1
            else:
                summary.ties += 1
        return summary


def aggregate_rubric(verdicts: Sequence[RubricVerdict]) -> Dict[str, float]:
    """Mean of each rubric dimension across a run."""
    if not verdicts:
        return {"faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0, "overall": 0.0}
    return {
        "faithfulness": round(mean([v.faithfulness for v in verdicts]), 4),
        "relevance": round(mean([v.relevance for v in verdicts]), 4),
        "completeness": round(mean([v.completeness for v in verdicts]), 4),
        "overall": round(mean([v.overall for v in verdicts]), 4),
    }
