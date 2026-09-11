"""Web adapter for project 04: the evaluation harness and its CI quality gate.

The visitor picks which version of the answering system gets graded, and the
page runs the real harness over the real 23 case dataset, then runs the real
gate against the committed baseline file. Nothing here is precomputed.

Read only: the harness writes nothing, and the baseline is opened for reading.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

NUMBER = 4
SLUG = "eval-harness"
TITLE = "LLM Evaluation Harness with a CI Quality Gate"
TAGLINE = ("Pick a version of the answering system, watch it get graded on 23 questions, "
           "and see whether the merge gate lets it through.")

WHAT_IT_DOES = """Type healthy, noisy or degraded to choose which version of the question
answering system gets tested. All three are real code. The healthy one reads two sentences
of evidence per question and refuses to answer anything the documents do not cover. The
noisy one reads three sentences instead of two. The degraded one never refuses and cuts
every answer down to six words.

The page grades that version against 23 questions, shows you the result of every single
one, and then gives you the average with a confidence interval, so you can see how much
of the number is real and how much is the small sample size.

Then it runs the check that would run on a pull request. The gate compares this run
against a stored baseline and blocks only when the drop is bigger than the allowed limit
and the evidence actually supports calling it a regression. A drop that could just be noise
is reported as a warning instead of blocking. The exit code printed at the bottom is the
one the build would return."""

INPUT_LABEL = "Which version of the system should be graded"
PLACEHOLDER = "healthy, noisy or degraded"
EXAMPLES = ["healthy", "noisy", "degraded"]
SOURCE = "projects/p04_eval_harness"

_BASELINE_PATH = os.path.join(_ROOT, "projects", "p04_eval_harness", "data", "baseline.json")

_VARIANTS = {
    "healthy": ("healthy", "k=2 evidence sentences, refuses when the docs do not cover it"),
    "noisy": ("noisy", "k=3 evidence sentences, one retrieval knob moved"),
    "degraded": ("degraded", "refusal switched off, every answer clipped to 6 words"),
}

_ALIASES = {
    "healthy": "healthy", "baseline": "healthy", "good": "healthy", "ok": "healthy",
    "fine": "healthy", "reference": "healthy", "clean": "healthy",
    "noisy": "noisy", "noise": "noisy", "nois": "noisy", "tweaked": "noisy",
    "degraded": "degraded", "degrade": "degraded", "broken": "degraded",
    "bad": "degraded", "regression": "degraded", "worse": "degraded",
}

_CACHE: dict = {}


def _pick(user_input: str):
    """Return (variant, note). Any unrecognised input falls back to healthy."""
    words = "".join(c.lower() if c.isalpha() else " " for c in (user_input or "")).split()
    for word in words:
        if word in _ALIASES:
            return _ALIASES[word], ""
    if not words:
        return "healthy", "no input given, so the healthy version was graded"
    return "healthy", (f"'{' '.join(words)[:40]}' is not one of healthy, noisy or degraded, "
                       "so the healthy version was graded")


def _dataset():
    if "cases" not in _CACHE:
        from projects.p04_eval_harness.dataset import build_dataset
        _CACHE["cases"] = build_dataset()
    return _CACHE["cases"]


def _baseline():
    if "baseline" not in _CACHE:
        from projects.p04_eval_harness.harness import load_report
        _CACHE["baseline"] = load_report(_BASELINE_PATH)
    return _CACHE["baseline"]


def _build(variant: str):
    from projects.p04_eval_harness import systems
    if variant == "noisy":
        return systems.noisy_system()
    if variant == "degraded":
        return systems.degraded_system()
    return systems.baseline_system()


def _one_line(text: str, limit: int = 96) -> str:
    flat = " ".join((text or "").split())
    return flat[:limit] + ("..." if len(flat) > limit else "")


def _run(user_input: str) -> str:
    from projects.p04_eval_harness.gate import CIGate
    from projects.p04_eval_harness.harness import EvalHarness

    variant, note = _pick(user_input)
    name, description = _VARIANTS[variant]
    cases = _dataset()

    harness = EvalHarness()
    report = harness.run(_build(variant), cases, name)

    out = []
    if note:
        out.append(f"note: {note}")
        out.append("")

    gold = sum(1 for c in cases if "gold" in c.tags)
    adversarial = sum(1 for c in cases if "adversarial" in c.tags)
    out.append("SYSTEM UNDER TEST")
    out.append(f"  version    {name}")
    out.append(f"  config     {description}")
    out.append(f"  dataset    {len(cases)} cases ({gold} from the docs, "
               f"{adversarial} written to be hard)")
    out.append("  scoring    deterministic string checks only, the same ones the "
               "baseline was measured with")

    out.append("")
    out.append("EVERY CASE, ONE ROW EACH")
    out.append(f"{'case':<12}{'difficulty':<11}{'result':<8}scores")
    out.append("-" * 74)
    for r in report.results:
        scores = ", ".join(f"{s.scorer} {s.score:.2f}" for s in r.scores) or "no scorer ran"
        out.append(f"{r.case_id[:11]:<12}{r.difficulty:<11}"
                   f"{'pass' if r.passed else 'FAIL':<8}{scores[:43]}")

    passed_n = sum(1 for r in report.results if r.passed)
    out.append(f"{passed_n} of {len(report.results)} cases passed every scorer they asked for")

    out.append("")
    out.append("AGGREGATE, WITH A BOOTSTRAP CONFIDENCE INTERVAL")
    out.append("(2000 resamples, fixed seed, so this page is reproducible)")
    out.append(f"{'metric':<16}{'mean':>8}{'95% CI':>20}{'cases':>7}")
    out.append("-" * 51)
    for metric in sorted(report.metrics):
        m = report.metrics[metric]
        ci = f"[{m['ci_low']:.3f}, {m['ci_high']:.3f}]"
        out.append(f"{metric:<16}{m['mean']:>8.3f}{ci:>20}{int(m['n']):>7}")

    if report.slices:
        out.append("")
        out.append(f"{'slice (3+ cases only)':<24}{'pass rate':>10}{'cases':>7}")
        out.append("-" * 41)
        for slice_name in sorted(report.slices):
            s = report.slices[slice_name]
            out.append(f"{slice_name:<24}{s['mean']:>10.3f}{int(s['n']):>7}")

    settled = report.judge_summary.get("cases_settled_by_cheap_checks", 0)
    out.append("")
    out.append(f"{settled} of {len(cases)} cases were settled by free string checks, so an LLM "
               f"judge running only on failures would cost {len(cases) - settled} calls "
               f"instead of {len(cases)}.")

    sample_pass = next((r for r in report.results if r.passed), None)
    sample_fail = next((r for r in report.results if not r.passed), None)
    out.append("")
    out.append("A COUPLE OF THE ACTUAL ANSWERS")
    if sample_pass is not None:
        out.append(f"  passed  {sample_pass.case_id}: {_one_line(sample_pass.prediction)}")
    if sample_fail is not None:
        out.append(f"  failed  {sample_fail.case_id}: {_one_line(sample_fail.prediction)}")
    if sample_pass is None and sample_fail is None:
        out.append("  no answers were produced")

    gate = CIGate(default_max_drop=0.02, min_values={"json_schema": 1.0})
    verdict = gate.check(_baseline(), report.to_dict())

    out.append("")
    out.append("THE MERGE GATE, AGAINST THE COMMITTED BASELINE")
    out.append("baseline file: projects/p04_eval_harness/data/baseline.json")
    out.append(f"{'metric':<16}{'baseline':>10}{'this run':>10}{'change':>9}"
               f"{'limit':>8}  verdict")
    out.append("-" * 74)
    for metric in sorted(verdict.compared):
        c = verdict.compared[metric]
        out.append(f"{metric:<16}{c['baseline']:>10.3f}{c['candidate']:>10.3f}"
                   f"{c['delta']:>+9.3f}{c['limit']:>8.3f}  {c['verdict']}")

    out.append("")
    if verdict.breaches:
        out.append(f"VERDICT: BLOCKED. {len(verdict.breaches)} metric(s) breached.")
        for b in verdict.breaches:
            out.append(f"  blocked on {b.metric}: {b.message}")
    else:
        drops = [(c["delta"], m) for m, c in verdict.compared.items() if c["delta"] < 0]
        if drops:
            worst_delta, worst_metric = min(drops)
            out.append(f"VERDICT: ALLOWED. The largest drop was {worst_metric} at "
                       f"{worst_delta:+.3f} against a limit of "
                       f"{verdict.compared[worst_metric]['limit']:.3f}.")
        else:
            out.append("VERDICT: ALLOWED. No metric dropped at all, so the "
                       "0.020 limit was never approached.")

    for w in verdict.warnings:
        out.append(f"  warning: {w}")

    out.append("")
    out.append(f"the build would exit with code {verdict.exit_code} "
               f"({'merge allowed' if verdict.exit_code == 0 else 'merge blocked'})")
    out.append("a drop only blocks when it is bigger than the limit AND the baseline sits "
               "outside this run's confidence interval; anything else is a warning, "
               "because a gate that fires on noise gets switched off")
    return "\n".join(out)


def run(user_input: str) -> str:
    try:
        return _run(user_input)
    except Exception as exc:  # an adapter must never take the page down
        return (f"This demo could not complete: {type(exc).__name__}: {exc}\n"
                "Try typing healthy, noisy or degraded.")
