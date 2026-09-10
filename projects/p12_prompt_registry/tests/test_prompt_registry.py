"""Tests for the prompt registry, splitter, outcomes and promotion gate."""
import json

import pytest

from llmkit import EchoLLM

from projects.p12_prompt_registry.gate import PromotionGate
from projects.p12_prompt_registry.outcomes import Outcome, OutcomeStore
from projects.p12_prompt_registry.registry import (
    NothingToRollBack,
    PromptRegistry,
    RegistryError,
    UnknownVersion,
)
from projects.p12_prompt_registry.splitter import Arm, TrafficSplitter
from projects.p12_prompt_registry.stats import (
    normal_cdf,
    normal_cdf_as26_2_17,
    two_proportion_z_test,
    wilson_interval,
)
from projects.p12_prompt_registry.templates import (
    MissingVariable,
    TemplateError,
    UnexpectedVariable,
    build_version,
)
from projects.p12_prompt_registry.workload import (
    PROMPT_NAME,
    register_variants,
    run_workload,
    unit_ids,
)

EXPERIMENT = "test-experiment"


@pytest.fixture()
def registry():
    reg = PromptRegistry()
    register_variants(reg)
    return reg


@pytest.fixture()
def splitter(registry):
    control_id, challenger_id = register_variants(registry)
    return TrafficSplitter(
        experiment=EXPERIMENT,
        arms=[Arm("control", control_id, 0.5), Arm("challenger", challenger_id, 0.5)],
    )


# -- immutable versions -------------------------------------------------


def test_editing_a_prompt_cannot_keep_the_old_version_id():
    a = build_version("p", "Answer using {evidence}.", ["evidence"])
    b = build_version("p", "Answer using {evidence}. Be concise.", ["evidence"])
    assert a.version_id != b.version_id
    assert a.version_id == build_version("p", "Answer using {evidence}.", ["evidence"]).version_id


def test_config_is_part_of_the_version_id():
    a = build_version("p", "{evidence}", ["evidence"], config={"temperature": 0.0})
    b = build_version("p", "{evidence}", ["evidence"], config={"temperature": 0.9})
    assert a.template == b.template
    assert a.version_id != b.version_id


def test_registering_identical_content_twice_is_a_no_op(registry):
    before = len(registry.versions)
    register_variants(registry)
    assert len(registry.versions) == before


def test_a_version_is_frozen():
    version = build_version("p", "{evidence}", ["evidence"])
    with pytest.raises(Exception):
        version.template = "something else"


# -- rendering ----------------------------------------------------------


def test_render_fails_on_a_missing_variable():
    version = build_version("p", "Use {evidence} for {question}.", ["evidence", "question"])
    with pytest.raises(MissingVariable):
        version.render(evidence="e")


def test_render_fails_on_an_unexpected_variable():
    version = build_version("p", "Use {evidence}.", ["evidence"])
    with pytest.raises(UnexpectedVariable) as exc:
        version.render(evidence="e", tone="friendly")
    assert "tone" in str(exc.value)


def test_declaring_a_variable_the_template_never_uses_is_an_error():
    with pytest.raises(TemplateError):
        build_version("p", "Use {evidence}.", ["evidence", "tone"])


def test_using_an_undeclared_variable_is_an_error():
    with pytest.raises(TemplateError):
        build_version("p", "Use {evidence} and {tone}.", ["evidence"])


def test_render_returns_the_filled_template():
    version = build_version("p", "Evidence:\n{evidence}", ["evidence"])
    assert version.render(evidence="[S1] a fact") == "Evidence:\n[S1] a fact"
    assert version.render({"evidence": "x"}) == "Evidence:\nx"


# -- labels, audit, rollback --------------------------------------------


def test_labels_are_pointers_and_promotion_moves_one(registry):
    control_id, challenger_id = register_variants(registry)
    registry.set_label(PROMPT_NAME, "prod", control_id, actor="arnob")
    assert registry.resolve(PROMPT_NAME, "prod").version_id == control_id
    registry.set_label(PROMPT_NAME, "prod", challenger_id, actor="ci", reason="gate passed")
    assert registry.resolve(PROMPT_NAME, "prod").version_id == challenger_id
    # Promotion copied nothing: both versions still exist independently.
    assert registry.get(control_id).template != registry.get(challenger_id).template


def test_rollback_is_one_call_and_walks_the_audit_trail(registry):
    control_id, challenger_id = register_variants(registry)
    registry.set_label(PROMPT_NAME, "prod", control_id, actor="arnob")
    registry.set_label(PROMPT_NAME, "prod", challenger_id, actor="ci")
    entry = registry.rollback(PROMPT_NAME, "prod", actor="oncall", reason="regression")
    assert entry.to_version == control_id
    assert registry.label_of(PROMPT_NAME, "prod") == control_id
    assert registry.history(PROMPT_NAME, "prod")[-1].actor == "oncall"


def test_rollback_refuses_when_there_is_nothing_behind_the_label(registry):
    control_id, _ = register_variants(registry)
    with pytest.raises(NothingToRollBack):
        registry.rollback(PROMPT_NAME, "prod", actor="oncall")
    registry.set_label(PROMPT_NAME, "prod", control_id, actor="arnob")
    with pytest.raises(NothingToRollBack):
        registry.rollback(PROMPT_NAME, "prod", actor="oncall")


def test_audit_records_who_when_from_and_to(registry):
    control_id, challenger_id = register_variants(registry)
    registry.set_label(PROMPT_NAME, "prod", control_id, actor="arnob", reason="initial")
    registry.set_label(PROMPT_NAME, "prod", challenger_id, actor="ci-bot", reason="gate passed")
    entries = registry.history(PROMPT_NAME, "prod")
    assert [e.from_version for e in entries] == [None, control_id]
    assert [e.to_version for e in entries] == [control_id, challenger_id]
    assert [e.actor for e in entries] == ["arnob", "ci-bot"]
    assert all(e.at > 0 for e in entries)


def test_unknown_versions_and_environments_are_rejected(registry):
    control_id, _ = register_variants(registry)
    with pytest.raises(UnknownVersion):
        registry.get("vdeadbeefdead")
    with pytest.raises(RegistryError):
        registry.set_label(PROMPT_NAME, "canary", control_id, actor="arnob")


def test_registry_round_trips_through_json(tmp_path, registry):
    control_id, challenger_id = register_variants(registry)
    registry.set_label(PROMPT_NAME, "prod", control_id, actor="arnob")
    registry.set_label(PROMPT_NAME, "prod", challenger_id, actor="ci")
    path = str(tmp_path / "registry.json")
    registry.save(path)
    reloaded = PromptRegistry.load(path)
    assert reloaded.label_of(PROMPT_NAME, "prod") == challenger_id
    assert len(reloaded.audit) == len(registry.audit)
    assert reloaded.resolve(PROMPT_NAME, "prod").template == registry.resolve(PROMPT_NAME, "prod").template


def test_a_hand_edited_registry_file_fails_to_load(tmp_path, registry):
    path = str(tmp_path / "registry.json")
    registry.save(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["versions"][0]["template"] += " (sneaky edit)"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    with pytest.raises(RegistryError):
        PromptRegistry.load(path)


# -- traffic splitting --------------------------------------------------


def test_assignment_is_sticky_across_calls_and_across_instances(splitter):
    units = unit_ids(500)
    first = {u: splitter.assign(u).name for u in units}
    again = {u: splitter.assign(u).name for u in units}
    assert first == again
    rebuilt = TrafficSplitter(experiment=splitter.experiment, arms=list(splitter.arms))
    assert {u: rebuilt.assign(u).name for u in units} == first


def test_the_split_is_roughly_even_but_not_exactly(splitter):
    units = unit_ids(2000)
    distribution = splitter.distribution(units)
    assert abs(distribution["control"] - 0.5) < 0.05
    assert splitter.max_deviation(units) < 0.05
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_weights_are_honoured(splitter):
    weighted = TrafficSplitter(
        experiment="weighted",
        arms=[Arm("control", "v1", 0.9), Arm("canary", "v2", 0.1)],
    )
    distribution = weighted.distribution(unit_ids(4000))
    assert distribution["canary"] == pytest.approx(0.1, abs=0.02)
    assert distribution["control"] == pytest.approx(0.9, abs=0.02)


def test_different_experiments_do_not_correlate(splitter):
    units = unit_ids(2000)
    other = TrafficSplitter(experiment="other", arms=list(splitter.arms))
    agreement = sum(1 for u in units if other.assign(u).name == splitter.assign(u).name) / len(units)
    assert 0.45 < agreement < 0.55, "assignments across experiments should look independent"


def test_salt_re_randomises_the_same_experiment(splitter):
    units = unit_ids(1000)
    salted = TrafficSplitter(experiment=splitter.experiment, arms=list(splitter.arms), salt="relaunch-2")
    agreement = sum(1 for u in units if salted.assign(u).name == splitter.assign(u).name) / len(units)
    assert 0.45 < agreement < 0.55


def test_a_splitter_rejects_bad_configuration():
    with pytest.raises(ValueError):
        TrafficSplitter(experiment="e", arms=[])
    with pytest.raises(ValueError):
        Arm("a", "v1", 0.0)
    with pytest.raises(ValueError):
        TrafficSplitter(experiment="e", arms=[Arm("a", "v1", 1.0), Arm("a", "v2", 1.0)])


# -- statistics ---------------------------------------------------------


def test_the_two_normal_cdf_implementations_agree():
    for z in (-4.0, -1.96, -0.5, 0.0, 0.5, 1.96, 4.0):
        assert abs(normal_cdf(z) - normal_cdf_as26_2_17(z)) < 7.5e-8
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-6)


def test_z_test_matches_a_hand_worked_example():
    # 40/200 against 60/200: pooled p = 0.25, se = sqrt(0.25*0.75*0.01) = 0.0433,
    # z = 0.10 / 0.0433 = 2.309, two-sided p = 0.0209.
    result = two_proportion_z_test(40, 200, 60, 200)
    assert result.z == pytest.approx(2.3094, abs=1e-3)
    assert result.p_value == pytest.approx(0.0209, abs=1e-3)
    assert result.absolute_lift == pytest.approx(0.10)
    assert result.relative_lift == pytest.approx(0.5)


def test_z_test_handles_degenerate_and_invalid_input():
    assert two_proportion_z_test(0, 50, 0, 50).p_value == 1.0
    with pytest.raises(ValueError):
        two_proportion_z_test(1, 0, 1, 10)
    with pytest.raises(ValueError):
        two_proportion_z_test(11, 10, 1, 10)


def test_wilson_interval_stays_inside_zero_and_one():
    low, high = wilson_interval(0, 30)
    assert low == 0.0 and 0.0 < high < 0.2
    low, high = wilson_interval(30, 30)
    assert high == 1.0 and 0.8 < low < 1.0


# -- outcomes and the gate ----------------------------------------------


def _store(rate_a, n_a, rate_b, n_b, cost_a=0.001, cost_b=0.001):
    store = OutcomeStore()
    for arm, rate, n, cost in (("control", rate_a, n_a, cost_a), ("challenger", rate_b, n_b, cost_b)):
        successes = int(round(rate * n))
        for i in range(n):
            store.record(
                Outcome(
                    experiment=EXPERIMENT, arm=arm, version_id=f"v-{arm}", unit_id=f"u{i}",
                    success=i < successes, latency_ms=10.0 + i % 5, prompt_tokens=100,
                    completion_tokens=20, cost_usd=cost,
                )
            )
    return store


def test_outcome_summary_reports_rate_latency_tokens_and_cost():
    summary = _store(0.5, 100, 0.6, 100).summarise(EXPERIMENT, "control")
    assert summary.samples == 100
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.latency_p50 > 0 and summary.latency_p95 >= summary.latency_p50
    assert summary.mean_total_tokens == pytest.approx(120.0)
    assert summary.cost_usd == pytest.approx(0.1, abs=1e-6)


def test_gate_refuses_below_the_minimum_sample_size():
    decision = PromotionGate(min_samples_per_arm=200).evaluate(
        _store(0.2, 30, 0.9, 30), EXPERIMENT, "control", "challenger"
    )
    assert not decision.promote
    assert "not enough data" in decision.reasons[0]
    # The effect is enormous and significant, and it is still held.
    assert decision.test.p_value < 0.05


def test_gate_refuses_a_real_but_tiny_lift():
    decision = PromotionGate(min_samples_per_arm=100, min_absolute_lift=0.05).evaluate(
        _store(0.500, 6000, 0.520, 6000), EXPERIMENT, "control", "challenger"
    )
    assert not decision.promote
    assert decision.test.significant_at(0.05)
    assert "below the" in decision.reasons[0]


def test_gate_refuses_an_insignificant_difference():
    decision = PromotionGate(min_samples_per_arm=100, min_absolute_lift=0.01).evaluate(
        _store(0.50, 120, 0.56, 120), EXPERIMENT, "control", "challenger"
    )
    assert not decision.promote
    assert "not significant" in decision.reasons[0]
    assert decision.needed_per_arm and decision.needed_per_arm > 120


def test_gate_refuses_a_challenger_that_is_behind():
    decision = PromotionGate(min_samples_per_arm=100).evaluate(
        _store(0.60, 300, 0.40, 300), EXPERIMENT, "control", "challenger"
    )
    assert not decision.promote
    assert "not ahead" in decision.reasons[0]


def test_gate_promotes_a_clear_significant_win():
    decision = PromotionGate(min_samples_per_arm=200, min_absolute_lift=0.05).evaluate(
        _store(0.40, 400, 0.55, 400), EXPERIMENT, "control", "challenger"
    )
    assert decision.promote
    assert decision.test.p_value < 0.05
    assert decision.verdict == "PROMOTE"
    assert "beats" in decision.reasons[0]


def test_cost_ceiling_blocks_an_otherwise_winning_challenger():
    store = _store(0.40, 400, 0.55, 400, cost_a=0.001, cost_b=0.004)
    permissive = PromotionGate(min_samples_per_arm=200, min_absolute_lift=0.05)
    strict = PromotionGate(min_samples_per_arm=200, min_absolute_lift=0.05, max_cost_ratio=1.5)
    assert permissive.evaluate(store, EXPERIMENT, "control", "challenger").promote
    decision = strict.evaluate(store, EXPERIMENT, "control", "challenger")
    assert not decision.promote
    assert "cost ratio" in decision.reasons[0]


# -- end to end ---------------------------------------------------------


def test_end_to_end_workload_produces_a_real_measured_difference(registry, splitter):
    store = OutcomeStore()
    run_workload(registry, splitter, store, unit_ids(240), EchoLLM(), experiment=EXPERIMENT)
    assert len(store) == 240
    summaries = {s.arm: s for s in store.summaries(EXPERIMENT)}
    assert set(summaries) == {"control", "challenger"}
    # Both arms do real work, and neither result is a coin flip from a generator.
    for summary in summaries.values():
        assert summary.samples > 80
        assert 0.0 < summary.success_rate < 1.0
        assert summary.total_tokens > 0
    assert summaries["challenger"].success_rate > summaries["control"].success_rate
    # Every request is attributed to the version that actually served it.
    for outcome in store.for_experiment(EXPERIMENT):
        assert registry.get(outcome.version_id).name == PROMPT_NAME


def test_outcomes_round_trip_through_json(tmp_path):
    store = _store(0.4, 20, 0.6, 20)
    path = str(tmp_path / "outcomes.json")
    store.save(path)
    reloaded = OutcomeStore.load(path)
    assert len(reloaded) == len(store)
    assert reloaded.summarise(EXPERIMENT, "challenger").success_rate == pytest.approx(0.6)
