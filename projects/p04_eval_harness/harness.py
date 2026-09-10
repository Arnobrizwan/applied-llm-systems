"""The harness: run a system over the dataset and produce a comparable report.

The report is the product of this project. It is JSON, it is versioned, and the
CI gate consumes it. Anything the gate needs to make a decision has to survive a
round trip through that file, which is why every metric carries its interval and
its n rather than a bare number.

The system under test is any callable taking an `EvalCase` and returning a
string. Passing the whole case rather than just the question is deliberate: real
systems receive metadata (a tenant, a locale, a requested schema) and a harness
that hides it can only evaluate a toy.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .dataset import EvalCase
from .judges import RubricJudge, RubricVerdict, aggregate_rubric, verbosity_bias
from .scorers import DETERMINISTIC_SCORERS, ScoreResult, ScorerFn, score_case
from .stats import summarise

SystemFn = Callable[[EvalCase], str]

REPORT_VERSION = 1


@dataclass
class CaseResult:
    case_id: str
    input: str
    prediction: str
    scores: List[ScoreResult]
    passed: bool
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    latency_ms: float = 0.0
    error: str = ""
    judge: Optional[RubricVerdict] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "case_id": self.case_id,
            "input": self.input[:200],
            "prediction": self.prediction[:400],
            "passed": self.passed,
            "tags": self.tags,
            "difficulty": self.difficulty,
            "latency_ms": round(self.latency_ms, 3),
            "scores": [s.to_dict() for s in self.scores],
        }
        if self.error:
            d["error"] = self.error
        if self.judge is not None:
            d["judge"] = self.judge.to_dict()
        return d


@dataclass
class EvalReport:
    system: str
    metrics: Dict[str, Dict[str, float]]
    results: List[CaseResult]
    slices: Dict[str, Dict[str, float]] = field(default_factory=dict)
    judge_summary: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    version: int = REPORT_VERSION

    @property
    def pass_rate(self) -> float:
        return self.metrics.get("pass_rate", {}).get("mean", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "system": self.system,
            "metrics": self.metrics,
            "slices": self.slices,
            "judge": self.judge_summary,
            "meta": self.meta,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return path

    def format_table(self) -> str:
        """Human readable summary. The interval is printed next to every mean
        because a mean without one invites the 2 point overreaction this whole
        project exists to prevent."""
        lines = [f"system: {self.system}   cases: {len(self.results)}",
                 f"{'metric':<16}{'mean':>8}{'95% CI':>20}{'n':>5}"]
        lines.append("-" * 49)
        for name in sorted(self.metrics):
            m = self.metrics[name]
            ci = f"[{m['ci_low']:.3f}, {m['ci_high']:.3f}]"
            lines.append(f"{name:<16}{m['mean']:>8.3f}{ci:>20}{int(m['n']):>5}")
        if self.slices:
            lines.append("")
            lines.append(f"{'slice':<24}{'pass rate':>10}{'n':>5}")
            lines.append("-" * 39)
            for name in sorted(self.slices):
                s = self.slices[name]
                lines.append(f"{name:<24}{s['mean']:>10.3f}{int(s['n']):>5}")
        return "\n".join(lines)


class EvalHarness:
    """Runs a system over a dataset and scores it.

    Public API, kept deliberately small so other projects can depend on it:
        EvalHarness(judge=None, judge_policy="all").run(system_fn, cases, name)
        -> EvalReport
    """

    JUDGE_POLICIES = ("never", "all", "on_failure")

    def __init__(self, judge: Optional[RubricJudge] = None, judge_policy: str = "never",
                 registry: Optional[Dict[str, ScorerFn]] = None,
                 confidence: float = 0.95, seed: int = 20260910):
        if judge_policy not in self.JUDGE_POLICIES:
            raise ValueError(f"judge_policy must be one of {self.JUDGE_POLICIES}")
        if judge_policy != "never" and judge is None:
            raise ValueError("judge_policy requires a judge")
        self.judge = judge
        self.judge_policy = judge_policy
        self.registry = registry or DETERMINISTIC_SCORERS
        self.confidence = confidence
        self.seed = seed

    def _run_case(self, system: SystemFn, case: EvalCase) -> CaseResult:
        start = time.perf_counter()
        error = ""
        try:
            prediction = system(case)
        except Exception as exc:
            # A system that raises scores zero rather than aborting the run. One
            # crashing case must not cost you the other 22 cases of signal.
            prediction, error = "", f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter() - start) * 1000.0

        scores = [] if error else score_case(prediction, case, self.registry)
        passed = bool(scores) and all(s.passed for s in scores)
        return CaseResult(
            case_id=case.id, input=case.input, prediction=prediction, scores=scores,
            passed=passed, tags=list(case.tags), difficulty=case.difficulty,
            latency_ms=elapsed, error=error,
        )

    def run(self, system: SystemFn, cases: Sequence[EvalCase], name: str = "system") -> EvalReport:
        results: List[CaseResult] = []
        judged: List[RubricVerdict] = []
        settled_cheaply = 0

        for case in cases:
            result = self._run_case(system, case)
            if result.passed:
                settled_cheaply += 1
            # Cheap checks first, judge second. `on_failure` is the setting that
            # matters on a large set: it turns a per-commit judge bill into a
            # bill proportional to the failure count, which is usually 10x less.
            wants_judge = (
                self.judge_policy == "all"
                or (self.judge_policy == "on_failure" and not result.passed)
            )
            if wants_judge and self.judge is not None:
                result.judge = self.judge.judge(case, result.prediction)
                judged.append(result.judge)
            results.append(result)

        metrics = self._metrics(results, judged)
        report = EvalReport(
            system=name,
            metrics=metrics,
            results=results,
            slices=self._slices(results),
            judge_summary=self._judge_summary(judged, len(results), settled_cheaply),
            meta={
                "cases": len(results),
                "errors": sum(1 for r in results if r.error),
                "confidence": self.confidence,
                "bootstrap_seed": self.seed,
                "judge_policy": self.judge_policy,
                "total_latency_ms": round(sum(r.latency_ms for r in results), 3),
            },
        )
        return report

    def _metrics(self, results: Sequence[CaseResult],
                 judged: Sequence[RubricVerdict]) -> Dict[str, Dict[str, float]]:
        """One entry per scorer that actually ran, plus the case-level pass rate.

        Scorers are only reported over the cases that requested them. Averaging a
        scorer over cases it never ran on (as a zero) would let a dataset change
        move a metric without any behaviour changing.
        """
        by_scorer: Dict[str, List[float]] = {}
        for r in results:
            for s in r.scores:
                by_scorer.setdefault(s.scorer, []).append(s.score)
        metrics = {name: summarise(vals, self.confidence, self.seed)
                   for name, vals in by_scorer.items()}
        metrics["pass_rate"] = summarise([1.0 if r.passed else 0.0 for r in results],
                                         self.confidence, self.seed)
        if judged:
            metrics["judge_overall"] = summarise([v.overall for v in judged],
                                                 self.confidence, self.seed)
            metrics["judge_faithfulness"] = summarise([v.faithfulness for v in judged],
                                                      self.confidence, self.seed)
        return metrics

    def _slices(self, results: Sequence[CaseResult]) -> Dict[str, Dict[str, float]]:
        buckets: Dict[str, List[float]] = {}
        for r in results:
            score = 1.0 if r.passed else 0.0
            buckets.setdefault(f"difficulty:{r.difficulty}", []).append(score)
            for tag in r.tags:
                buckets.setdefault(f"tag:{tag}", []).append(score)
        # A slice of one is a headline waiting to happen, so only slices with at
        # least three cases are reported.
        return {k: summarise(v, self.confidence, self.seed)
                for k, v in buckets.items() if len(v) >= 3}

    def _judge_summary(self, judged: Sequence[RubricVerdict], total: int,
                       settled: int) -> Dict[str, Any]:
        if not judged:
            return {"calls": 0, "coverage": 0.0,
                    "cases_settled_by_cheap_checks": settled,
                    "calls_saved_by_on_failure_policy": settled}
        summary: Dict[str, Any] = dict(aggregate_rubric(judged))
        summary.update({
            "calls": self.judge.calls if self.judge else len(judged),
            "coverage": round(len(judged) / total, 4) if total else 0.0,
            "cases_settled_by_cheap_checks": settled,
            "calls_saved_by_on_failure_policy": settled if self.judge_policy == "all" else 0,
            "parse_errors": sum(1 for v in judged if v.parse_error),
            "verbosity_bias": verbosity_bias(judged),
        })
        return summary


def load_report(path: str) -> Dict[str, Any]:
    """Read a saved report or baseline. Returned as a dict, not an EvalReport:
    the gate only needs metrics, and rebuilding case objects would make the gate
    fail on a report written by a newer harness version."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
