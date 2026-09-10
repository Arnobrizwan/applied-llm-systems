"""End to end demo: score three systems, judge one, and run the gate twice.

Run it with:  python3 projects/p04_eval_harness/demo.py
Every number printed here is measured in this process. Nothing is hard coded.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from projects.p04_eval_harness import systems  # noqa: E402
from projects.p04_eval_harness.dataset import build_dataset, slice_by  # noqa: E402
from projects.p04_eval_harness.gate import CIGate, write_baseline  # noqa: E402
from projects.p04_eval_harness.harness import EvalHarness, load_report  # noqa: E402
from projects.p04_eval_harness.judges import PairwiseJudge, RubricJudge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINE_PATH = os.path.join(HERE, "data", "baseline.json")
OUT_DIR = os.path.join(ROOT, "artifacts", "p04_eval_harness")


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = build_dataset()

    rule("1. DATASET")
    print(f"cases: {len(cases)}  "
          f"(gold {len(slice_by(cases, tag='gold'))}, "
          f"adversarial {len(slice_by(cases, tag='adversarial'))})")
    print(f"difficulty: easy {len(slice_by(cases, difficulty='easy'))}, "
          f"medium {len(slice_by(cases, difficulty='medium'))}, "
          f"hard {len(slice_by(cases, difficulty='hard'))}")
    scorer_use = {}
    for c in cases:
        for s in c.scorers:
            scorer_use[s] = scorer_use.get(s, 0) + 1
    print("scorers requested: " + ", ".join(f"{k}={v}" for k, v in sorted(scorer_use.items())))

    rule("2. DETERMINISTIC SCORING, THREE SYSTEMS")
    harness = EvalHarness()
    reports = {}
    for name, factory in (("baseline", systems.baseline_system),
                          ("noisy", systems.noisy_system),
                          ("degraded", systems.degraded_system)):
        reports[name] = harness.run(factory(), cases, name)
        reports[name].save(os.path.join(OUT_DIR, f"report_{name}.json"))
    print(reports["baseline"].format_table())
    print("\nsame three systems, pass rate side by side:")
    print(f"{'system':<12}{'pass rate':>11}{'95% CI':>18}{'contains':>10}{'token_f1':>10}")
    print("-" * 61)
    for name, rep in reports.items():
        m = rep.metrics
        ci = f"[{m['pass_rate']['ci_low']:.3f}, {m['pass_rate']['ci_high']:.3f}]"
        print(f"{name:<12}{m['pass_rate']['mean']:>11.3f}{ci:>18}"
              f"{m['contains']['mean']:>10.3f}{m['token_f1']['mean']:>10.3f}")

    rule("3. RETRIEVAL DEPTH SWEEP (why the baseline uses k=2)")
    print(f"{'k':>3}{'pass rate':>12}{'contains':>11}")
    print("-" * 26)
    for k in (1, 2, 3, 4):
        rep = harness.run(systems.baseline_system(k=k), cases, f"k={k}")
        print(f"{k:>3}{rep.metrics['pass_rate']['mean']:>12.3f}"
              f"{rep.metrics['contains']['mean']:>11.3f}")

    rule("4. LLM JUDGE (rubric) AND ITS BIAS CONTROLS")
    judged = EvalHarness(judge=RubricJudge(), judge_policy="all").run(
        systems.baseline_system(), cases, "baseline-judged")
    js = judged.judge_summary
    print(f"faithfulness {js['faithfulness']:.3f}   relevance {js['relevance']:.3f}   "
          f"completeness {js['completeness']:.3f}   overall {js['overall']:.3f}")
    jm = judged.metrics["judge_overall"]
    print(f"judge_overall {jm['mean']:.3f}  95% CI [{jm['ci_low']:.3f}, {jm['ci_high']:.3f}]  "
          f"n={int(jm['n'])}")
    print(f"judge calls this run: {js['calls']}  (coverage {js['coverage']:.0%}, "
          f"parse errors {js['parse_errors']})")
    print(f"cases settled by the cheap deterministic checks: "
          f"{js['cases_settled_by_cheap_checks']} of {len(cases)}; switching the policy to "
          f"on_failure would drop the judge bill to "
          f"{js['calls'] - js['calls_saved_by_on_failure_policy']} calls")
    vb = js["verbosity_bias"]
    print(f"verbosity bias: r={vb['r']:+.3f} between judge score and answer length "
          f"(n={vb['n']}, flagged={vb['flagged']})")

    rule("5. PAIRWISE JUDGE WITH ORDER SWAPPING")
    answers_a = systems.answers_for(systems.baseline_system(), cases)
    answers_b = systems.answers_for(systems.degraded_system(), cases)
    pairwise = PairwiseJudge()
    summary = pairwise.compare_many(cases, answers_a, answers_b)
    d = summary.to_dict()
    print(f"baseline vs degraded over {d['total']} cases, {pairwise.calls} judge calls "
          f"(two per case, order swapped)")
    print(f"baseline wins {d['a_wins']}, degraded wins {d['b_wins']}, ties {d['ties']}")
    print(f"position bias: the verdict flipped when the order was swapped on "
          f"{d['position_flips']} of {d['total']} cases ({d['position_bias_rate']:.1%})")
    print("every flipped case is scored as a tie, so a verdict that was an artefact of")
    print("ordering never reaches the report as a win")

    rule("6. CI GATE, PASSING RUN")
    if not os.path.exists(BASELINE_PATH):
        write_baseline(reports["baseline"].to_dict(), BASELINE_PATH)
    baseline = load_report(BASELINE_PATH)
    gate = CIGate(default_max_drop=0.02, min_values={"json_schema": 1.0})
    passing = gate.check(baseline, reports["noisy"].to_dict())
    print(passing.format_text())
    print(f"\nprocess exit code: {passing.exit_code}")

    rule("7. CI GATE, FAILING RUN")
    failing = gate.check(baseline, reports["degraded"].to_dict())
    print(failing.format_text())
    path = failing.write(os.path.join(OUT_DIR, "gate_report.json"))
    print(f"\nprocess exit code: {failing.exit_code}")
    print(f"machine-readable gate report: {os.path.relpath(path, ROOT)}")

    rule("SUMMARY")
    print(f"the same 2 point rule blocked the {abs(failing.compared['pass_rate']['delta']):.3f} "
          f"drop and let the {abs(passing.compared['pass_rate']['delta']):.3f} drop through, "
          f"because only one of them is outside the interval")
    print(f"artifacts written to {os.path.relpath(OUT_DIR, ROOT)}/")


if __name__ == "__main__":
    main()
