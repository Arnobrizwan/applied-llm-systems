"""Harness, judge bias controls and the CI gate."""
import pytest

from llmkit import EchoLLM, ScriptedLLM

from projects.p04_eval_harness import systems
from projects.p04_eval_harness.dataset import EvalCase, build_dataset
from projects.p04_eval_harness.gate import CIGate
from projects.p04_eval_harness.harness import EvalHarness
from projects.p04_eval_harness.judges import PairwiseJudge, RubricJudge, verbosity_bias


def tiny_cases():
    return [
        EvalCase(id="c1", input="q1", expected="alpha", scorers=["contains"]),
        EvalCase(id="c2", input="q2", expected="beta", scorers=["contains"]),
        EvalCase(id="c3", input="q3", expected="gamma", scorers=["contains"]),
        EvalCase(id="c4", input="q4", expected="delta", scorers=["contains"]),
    ]


def test_a_crashing_system_scores_zero_without_aborting_the_run():
    def flaky(case):
        if case.id == "c2":
            raise RuntimeError("upstream exploded")
        return case.expected
    report = EvalHarness().run(flaky, tiny_cases(), "flaky")
    assert len(report.results) == 4
    assert report.metrics["pass_rate"]["mean"] == pytest.approx(0.75)
    assert report.meta["errors"] == 1
    assert "RuntimeError" in [r.error for r in report.results if r.case_id == "c2"][0]


def test_on_failure_policy_only_pays_the_judge_for_unsettled_cases():
    judge = RubricJudge(llm=EchoLLM())
    harness = EvalHarness(judge=judge, judge_policy="on_failure")
    report = harness.run(lambda c: c.expected if c.id in ("c1", "c2") else "wrong",
                         tiny_cases(), "half-right")
    assert judge.calls == 2, "the two passing cases must not reach the judge"
    assert report.judge_summary["coverage"] == pytest.approx(0.5)


def test_unknown_scorer_name_is_a_loud_failure():
    with pytest.raises(KeyError):
        EvalHarness().run(lambda c: "x", [EvalCase(id="c", input="q", scorers=["vibes"])], "s")


def test_rubric_judge_scores_zero_and_says_why_on_unparseable_output():
    judge = RubricJudge(llm=ScriptedLLM(["I think it was pretty good, honestly."]))
    verdict = judge.judge(EvalCase(id="c", input="q", expected="a"), "some answer")
    assert verdict.overall == 0.0
    assert "not JSON" in verdict.parse_error


def test_rubric_scale_maps_the_floor_to_zero_not_to_one_fifth():
    judge = RubricJudge(llm=ScriptedLLM([
        '{"faithfulness": 1, "relevance": 1, "completeness": 1, "reason": "bad"}',
        '{"faithfulness": 5, "relevance": 5, "completeness": 5, "reason": "good"}',
    ]))
    c = EvalCase(id="c", input="q", expected="a")
    assert judge.judge(c, "bad answer").overall == 0.0
    assert judge.judge(c, "good answer").overall == 1.0


def test_pairwise_order_swap_turns_a_position_biased_verdict_into_a_tie():
    """A judge that always says "the first one" must never produce a winner."""
    always_first = ScriptedLLM(['{"winner": "A", "reason": "position"}'])
    judge = PairwiseJudge(llm=always_first)
    summary = judge.compare_many(tiny_cases(), ["a"] * 4, ["b"] * 4)
    assert judge.calls == 8, "two calls per case, order swapped"
    assert summary.ties == 4 and summary.a_wins == 0 and summary.b_wins == 0
    assert summary.position_bias_rate == 1.0


def test_pairwise_agreement_across_both_orders_is_reported_as_a_win():
    consistent = ScriptedLLM([
        '{"winner": "A", "reason": "better"}',   # A shown first, A wins
        '{"winner": "B", "reason": "better"}',   # order swapped, same system wins
    ])
    verdict = PairwiseJudge(llm=consistent).compare(
        EvalCase(id="c", input="q"), "good answer", "bad answer")
    assert verdict.consistent and verdict.winner == "A"


def test_verbosity_bias_flags_a_length_rewarding_judge():
    class V:
        def __init__(self, overall, answer_tokens):
            self.overall, self.answer_tokens = overall, answer_tokens
    rising = [V(i / 10.0, i * 20) for i in range(10)]
    flat = [V(0.5, i * 20) for i in range(10)]
    assert verbosity_bias(rising)["flagged"] is True
    assert verbosity_bias(flat)["flagged"] is False


def test_gate_blocks_a_real_regression_and_lets_small_noise_through():
    cases = build_dataset()
    harness = EvalHarness()
    baseline = harness.run(systems.baseline_system(), cases, "baseline").to_dict()
    noisy = harness.run(systems.noisy_system(), cases, "noisy").to_dict()
    degraded = harness.run(systems.degraded_system(), cases, "degraded").to_dict()
    gate = CIGate(default_max_drop=0.02)

    noise = gate.check(baseline, noisy)
    assert noise.passed, "a change inside the interval must not block a merge"
    assert noise.warnings, "but it must still be reported"
    assert noise.compared["pass_rate"]["delta"] < -0.02, "the point estimate really did move"

    real = gate.check(baseline, degraded)
    assert not real.passed and real.exit_code == 1
    assert any(b.metric == "pass_rate" and b.kind == "regression" for b in real.breaches)


def test_gate_cannot_be_passed_by_deleting_the_failing_metric():
    baseline = {"metrics": {"pass_rate": {"mean": 0.9, "n": 50, "ci_low": 0.8, "ci_high": 0.95}}}
    candidate = {"metrics": {"something_else": {"mean": 1.0, "n": 50, "ci_low": 1.0, "ci_high": 1.0}}}
    result = CIGate().check(baseline, candidate)
    assert not result.passed
    assert result.breaches[0].kind == "missing_metric"
    assert any("no baseline" in w for w in result.warnings)


def test_hard_floor_blocks_even_when_the_drop_is_not_significant():
    baseline = {"metrics": {"json_schema": {"mean": 1.0, "n": 4, "ci_low": 1.0, "ci_high": 1.0}}}
    candidate = {"metrics": {"json_schema": {"mean": 0.75, "n": 4, "ci_low": 0.25, "ci_high": 1.0}}}
    assert CIGate().check(baseline, candidate).passed, "small n, interval covers the baseline"
    floored = CIGate(min_values={"json_schema": 1.0}).check(baseline, candidate)
    assert not floored.passed and floored.breaches[0].kind == "floor"


def test_abstention_is_the_difference_on_the_unanswerable_case():
    unanswerable = [c for c in build_dataset() if c.id == "adv-01"][0]
    answer = systems.baseline_system()(unanswerable)
    assert "do not know" in answer.lower()
    assert "do not know" not in systems.degraded_system()(unanswerable).lower()


def test_empty_input_is_answered_with_a_question_not_an_answer():
    blank = [c for c in build_dataset() if c.id == "adv-08"][0]
    assert "rephrase" in systems.baseline_system()(blank).lower()
