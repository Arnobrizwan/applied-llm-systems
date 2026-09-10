"""The CI gate: compare a run against a stored baseline and block the merge.

A gate that fires on noise is a gate that gets disabled. On a 23 case set one
flipped case moves the pass rate by 4.3 points, so a naive "fail if the number
went down by more than 2 points" rule fires constantly, everyone learns to
re-run CI until it goes green, and the gate stops protecting anything.

So a metric drop blocks a merge only when both things are true:
  1. the drop is larger than the metric's threshold, and
  2. the drop is statistically supported, meaning the baseline mean sits outside
     the candidate run's bootstrap confidence interval.
A drop that fails test 2 is reported as a warning, not a failure, and the run
still records it so a slow drift across many merges is visible.

Two escape hatches exist because both failure modes are real:
  * `min_values` is a hard floor with no significance test. Some metrics (schema
    validity, refusal on unanswerable questions) are contractual, and "the drop
    was not significant" is not an answer when the number is below the floor.
  * a metric present in the baseline and missing from the candidate is a
    failure. Deleting the failing metric is the easiest way to make any gate
    pass, and it should not be.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_MAX_DROP = 0.02


@dataclass
class Breach:
    metric: str
    baseline: float
    candidate: float
    delta: float
    threshold: float
    kind: str          # "regression" | "floor" | "missing_metric"
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateResult:
    passed: bool
    breaches: List[Breach] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    compared: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "exit_code": self.exit_code,
            "breaches": [b.to_dict() for b in self.breaches],
            "warnings": self.warnings,
            "compared": self.compared,
        }

    def write(self, path: str) -> str:
        """Machine-readable output for the CI system to attach to the PR."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return path

    def format_text(self) -> str:
        head = "GATE PASS" if self.passed else "GATE FAIL"
        lines = [f"{head} (exit {self.exit_code})",
                 f"{'metric':<20}{'base':>8}{'cand':>8}{'delta':>9}  verdict"]
        lines.append("-" * 62)
        for name in sorted(self.compared):
            c = self.compared[name]
            lines.append(f"{name:<20}{c['baseline']:>8.3f}{c['candidate']:>8.3f}"
                         f"{c['delta']:>+9.3f}  {c['verdict']}")
        for b in self.breaches:
            lines.append(f"  BLOCK {b.metric}: {b.message}")
        for w in self.warnings:
            lines.append(f"  warn  {w}")
        return "\n".join(lines)


class CIGate:
    """Compares two report dicts produced by `EvalHarness.run(...).to_dict()`."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None,
                 default_max_drop: float = DEFAULT_MAX_DROP,
                 min_values: Optional[Dict[str, float]] = None,
                 require_significance: bool = True):
        self.thresholds = dict(thresholds or {})
        self.default_max_drop = default_max_drop
        self.min_values = dict(min_values or {})
        self.require_significance = require_significance

    def threshold_for(self, metric: str) -> float:
        return self.thresholds.get(metric, self.default_max_drop)

    @staticmethod
    def _significant(baseline_mean: float, candidate: Dict[str, float]) -> bool:
        """True when the baseline mean lies outside the candidate's interval.

        This is the cheap version of a paired significance test. The correct test
        for two runs over the same cases is a paired bootstrap on the per-case
        differences, which needs both runs' per-case scores. The gate only reads
        summary metrics from the baseline file, so it uses the one-sample form
        and is stated here as the approximation it is. It is conservative in the
        direction that matters: it under-reports regressions rather than
        inventing them.
        """
        lo = candidate.get("ci_low", candidate.get("mean", 0.0))
        hi = candidate.get("ci_high", candidate.get("mean", 0.0))
        return not (lo <= baseline_mean <= hi)

    def check(self, baseline: Dict[str, Any], candidate: Dict[str, Any]) -> GateResult:
        base_metrics: Dict[str, Any] = baseline.get("metrics", baseline)
        cand_metrics: Dict[str, Any] = candidate.get("metrics", candidate)
        result = GateResult(passed=True)

        for name in sorted(base_metrics):
            base = base_metrics[name]
            base_mean = float(base["mean"] if isinstance(base, dict) else base)
            if name not in cand_metrics:
                result.breaches.append(Breach(
                    metric=name, baseline=base_mean, candidate=0.0, delta=-base_mean,
                    threshold=self.threshold_for(name), kind="missing_metric",
                    message="present in the baseline and missing from this run"))
                continue

            cand = cand_metrics[name]
            cand_dict = cand if isinstance(cand, dict) else {"mean": float(cand)}
            cand_mean = float(cand_dict["mean"])
            delta = cand_mean - base_mean
            limit = self.threshold_for(name)
            verdict = "ok"

            if delta < -limit:
                if not self.require_significance or self._significant(base_mean, cand_dict):
                    verdict = "regression"
                    result.breaches.append(Breach(
                        metric=name, baseline=base_mean, candidate=cand_mean, delta=delta,
                        threshold=limit, kind="regression",
                        message=(f"dropped {abs(delta):.3f} (limit {limit:.3f}); baseline "
                                 f"{base_mean:.3f} is outside the candidate interval "
                                 f"[{cand_dict.get('ci_low', 0):.3f}, "
                                 f"{cand_dict.get('ci_high', 0):.3f}]")))
                else:
                    verdict = "noise"
                    result.warnings.append(
                        f"{name} dropped {abs(delta):.3f} but the baseline sits inside the "
                        f"candidate interval [{cand_dict.get('ci_low', 0):.3f}, "
                        f"{cand_dict.get('ci_high', 0):.3f}]; n={int(cand_dict.get('n', 0))} "
                        f"is too small to call this a regression")

            floor = self.min_values.get(name)
            if floor is not None and cand_mean < floor:
                verdict = "below floor"
                result.breaches.append(Breach(
                    metric=name, baseline=base_mean, candidate=cand_mean, delta=delta,
                    threshold=floor, kind="floor",
                    message=f"{cand_mean:.3f} is below the hard floor {floor:.3f}"))

            result.compared[name] = {
                "baseline": round(base_mean, 4), "candidate": round(cand_mean, 4),
                "delta": round(delta, 4), "limit": limit, "verdict": verdict,
                "n": int(cand_dict.get("n", 0)),
            }

        for name in sorted(cand_metrics):
            if name not in base_metrics:
                result.warnings.append(f"{name} is new and has no baseline; not gated this run")

        result.passed = not result.breaches
        return result


def write_baseline(report: Dict[str, Any], path: str) -> str:
    """Store only what the gate reads.

    Baselines are committed, and committing every prediction makes the diff
    unreadable and tempts people to regenerate the baseline to make CI green.
    """
    payload = {"version": report.get("version", 1), "system": report.get("system"),
               "metrics": report.get("metrics", {}), "meta": report.get("meta", {})}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Block a merge on an eval regression.")
    parser.add_argument("--baseline", required=True, help="committed baseline JSON")
    parser.add_argument("--candidate", required=True, help="report JSON from this run")
    parser.add_argument("--out", default="", help="write the machine-readable gate report here")
    parser.add_argument("--max-drop", type=float, default=DEFAULT_MAX_DROP)
    parser.add_argument("--min-value", action="append", default=[],
                        metavar="METRIC=VALUE", help="hard floor, repeatable")
    args = parser.parse_args(argv)

    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.candidate, encoding="utf-8") as f:
        candidate = json.load(f)

    floors: Dict[str, float] = {}
    for item in args.min_value:
        key, _, value = item.partition("=")
        floors[key.strip()] = float(value)

    result = CIGate(default_max_drop=args.max_drop, min_values=floors).check(baseline, candidate)
    print(result.format_text())
    if args.out:
        result.write(args.out)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
