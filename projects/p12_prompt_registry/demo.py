"""Prompt Versioning and A/B System demo.

Registers two prompt variants, promotes one to prod, runs a simulated workload
through the deterministic splitter against llmkit.EchoLLM, checks the split is
sticky and roughly even, evaluates the promotion gate twice (once on a small
sample, once on the full one), acts on the result, then rolls back.

    python3 projects/p12_prompt_registry/demo.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmkit import EchoLLM  # noqa: E402

from projects.p12_prompt_registry.gate import PromotionGate  # noqa: E402
from projects.p12_prompt_registry.outcomes import ArmSummary, OutcomeStore  # noqa: E402
from projects.p12_prompt_registry.registry import PromptRegistry  # noqa: E402
from projects.p12_prompt_registry.splitter import Arm, TrafficSplitter  # noqa: E402
from projects.p12_prompt_registry.stats import normal_cdf, normal_cdf_as26_2_17  # noqa: E402
from projects.p12_prompt_registry.templates import MissingVariable, UnexpectedVariable  # noqa: E402
from projects.p12_prompt_registry.workload import (  # noqa: E402
    PROMPT_NAME,
    register_variants,
    run_workload,
    unit_ids,
)

EXPERIMENT = "evidence-format-2026-09"
POPULATION = 600
EARLY_CHECKPOINT = 40
STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "p12_registry.json")


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    llm = EchoLLM()
    registry = PromptRegistry()

    rule("1. IMMUTABLE VERSIONS")
    control_id, challenger_id = register_variants(registry)
    print()
    print(f"  control     {control_id}")
    print(f"  challenger  {challenger_id}")
    print(f"  same prompt name, two content hashes: {control_id != challenger_id}")

    again_control, _ = register_variants(registry)
    print(f"  re-registering identical content is a no-op: {again_control == control_id}, "
          f"{len(registry.versions_of(PROMPT_NAME))} versions in the registry")

    edited = registry.register(
        PROMPT_NAME,
        registry.get(control_id).template + " Be concise.",
        variables=["evidence"],
        config=registry.get(control_id).config_dict,
        author="arnob",
        notes="four extra words",
    )
    print(f"  four extra words produce a different id: {edited.version_id}")
    print("  an edited prompt cannot keep the old id, because the id is the hash of the content")

    rule("2. RENDERING FAILS LOUDLY")
    print()
    version = registry.get(challenger_id)
    print(f"  declared variables: {list(version.variables)}")
    try:
        version.render()
    except MissingVariable as exc:
        print(f"  missing variable    -> {type(exc).__name__}: {exc}")
    try:
        version.render(evidence="[S1] something", tone="friendly")
    except UnexpectedVariable as exc:
        print(f"  unexpected variable -> {type(exc).__name__}: {exc}")
    print(f"  correct call renders {len(version.render(evidence='[S1] something'))} characters")

    rule("3. LABELS ARE POINTERS")
    print()
    registry.set_label(PROMPT_NAME, "dev", challenger_id, actor="arnob", reason="new variant under test")
    registry.set_label(PROMPT_NAME, "staging", challenger_id, actor="arnob", reason="promote dev to staging")
    registry.set_label(PROMPT_NAME, "prod", control_id, actor="arnob", reason="initial production prompt")
    for env in registry.environments:
        print(f"  {env:<8} -> {registry.label_of(PROMPT_NAME, env)}")
    print("  promotion and rollback move a pointer; no prompt text is copied between environments")

    rule("4. THE SPLIT IS STICKY")
    print()
    splitter = TrafficSplitter(
        experiment=EXPERIMENT,
        arms=[Arm("control", control_id, 0.5), Arm("challenger", challenger_id, 0.5)],
    )
    units = unit_ids(POPULATION)
    first_pass = splitter.assign_many(units)
    second_pass = splitter.assign_many(units)
    stable = sum(1 for u in units if first_pass[u].name == second_pass[u].name)
    print(f"  {stable} of {len(units)} units land in the same arm when asked twice")

    rebuilt = TrafficSplitter(
        experiment=EXPERIMENT,
        arms=[Arm("control", control_id, 0.5), Arm("challenger", challenger_id, 0.5)],
    )
    same_after_restart = sum(1 for u in units if rebuilt.assign(u).name == first_pass[u].name)
    print(f"  {same_after_restart} of {len(units)} still match after rebuilding the splitter from scratch")
    print("  there is no assignment table: the arm is a pure function of the ids")

    other = TrafficSplitter(
        experiment="unrelated-experiment",
        arms=[Arm("control", control_id, 0.5), Arm("challenger", challenger_id, 0.5)],
    )
    overlap = sum(1 for u in units if other.assign(u).name == first_pass[u].name)
    print(f"  a different experiment reshuffles the same units: {overlap} of {len(units)} coincide "
          f"({overlap / len(units) * 100:.1f}%, chance is 50%)")

    observed = splitter.distribution(units)
    print()
    for name, share in observed.items():
        print(f"  {name:<12} {share * 100:.1f}% of traffic (intended {splitter.expected()[name] * 100:.0f}%)")
    print(f"  largest deviation from the intended split: {splitter.max_deviation(units) * 100:.2f} points")

    rule("5. WORKLOAD AGAINST EchoLLM")
    print()
    store = OutcomeStore()
    run_workload(registry, splitter, store, units, llm, experiment=EXPERIMENT)
    print(f"  {len(store)} requests routed and judged")
    print("  success = the model's output contains the gold phrase for that question")
    print()
    print(ArmSummary.header())
    print("  " + "-" * 89)
    for summary in store.summaries(EXPERIMENT):
        print(summary.row())

    rule("6. THE GATE, ON A SMALL SAMPLE")
    print()
    early = OutcomeStore(store.for_experiment(EXPERIMENT)[:EARLY_CHECKPOINT])
    gate = PromotionGate(min_samples_per_arm=200, alpha=0.05, min_absolute_lift=0.05)
    early_decision = gate.evaluate(early, EXPERIMENT, control="control", challenger="challenger")
    print(f"  after {len(early)} requests:")
    print(early_decision.render())
    if early_decision.needed_per_arm:
        print(f"  estimated sample needed per arm for this effect size: {early_decision.needed_per_arm}")
    if early_decision.test and early_decision.test.p_value < gate.alpha:
        print(f"  note: p is already {early_decision.test.p_value:.5f}, below the {gate.alpha} threshold.")
        print("  this is exactly the moment a live dashboard says 'ship it'. the sample floor is the")
        print("  only thing holding the decision, and the full-sample lift below is less than half")
        print("  the lift this checkpoint shows.")

    rule("7. THE GATE, ON THE FULL SAMPLE")
    print()
    decision = gate.evaluate(store, EXPERIMENT, control="control", challenger="challenger")
    print(decision.render())

    rule("8. THE SAME NUMBERS UNDER A COST CEILING")
    print()
    control_summary, challenger_summary = store.summaries(EXPERIMENT)
    cost_ratio = challenger_summary.cost_usd / control_summary.cost_usd
    token_ratio = challenger_summary.mean_total_tokens / control_summary.mean_total_tokens
    print(f"  the challenger sends {token_ratio:.2f}x the tokens and costs {cost_ratio:.2f}x as much")
    strict = PromotionGate(min_samples_per_arm=200, alpha=0.05, min_absolute_lift=0.05, max_cost_ratio=1.5)
    strict_decision = strict.evaluate(store, EXPERIMENT, control="control", challenger="challenger")
    print(strict_decision.render())
    print("  same data, same significance, different policy. a quality gate that ignores cost")
    print("  approves changes that a finance review then reverses.")

    rule("9. ACTING ON THE DECISION")
    print()
    if decision.promote:
        registry.set_label(
            PROMPT_NAME,
            "prod",
            decision.challenger.version_id,
            actor="ci-bot",
            reason=f"gate passed, p={decision.test.p_value:.5f}",
        )
        print(f"  promoted {decision.challenger.arm} to prod: {registry.label_of(PROMPT_NAME, 'prod')}")
    else:
        print(f"  refused to promote; prod stays on {registry.label_of(PROMPT_NAME, 'prod')}")
        print(f"  reason: {decision.reasons[0] if decision.reasons else 'no reason recorded'}")

    rule("10. ROLLBACK")
    print()
    before = registry.label_of(PROMPT_NAME, "prod")
    try:
        entry = registry.rollback(PROMPT_NAME, "prod", actor="oncall", reason="cost ceiling breach flagged in review")
        print(f"  prod was {before}, rolled back to {entry.to_version} in one call")
    except Exception as exc:  # the gate refused, so there was nothing to roll back
        print(f"  rollback refused: {exc}")
    print(f"  prod now resolves to: {registry.resolve(PROMPT_NAME, 'prod').version_id}")

    rule("11. AUDIT TRAIL")
    print()
    for entry in registry.history(PROMPT_NAME):
        print(entry.row())

    rule("12. PERSISTENCE")
    print()
    path = os.path.abspath(STORE_PATH)
    registry.save(path)
    reloaded = PromptRegistry.load(path)
    print(f"  wrote {path}")
    print(f"  reloaded {len(reloaded.versions)} versions, {len(reloaded.audit)} audit entries")
    print(f"  prod pointer survives the round trip: "
          f"{reloaded.label_of(PROMPT_NAME, 'prod') == registry.label_of(PROMPT_NAME, 'prod')}")

    rule("13. THE NORMAL CDF, BOTH WAYS")
    print()
    print(f"  {'z':>6}{'math.erf':>14}{'A&S 26.2.17':>16}{'difference':>14}")
    for z in (-2.5, -1.0, 0.0, 1.0, 1.96, 3.0):
        exact = normal_cdf(z)
        approx = normal_cdf_as26_2_17(z)
        print(f"  {z:>6.2f}{exact:>14.9f}{approx:>16.9f}{abs(exact - approx):>14.2e}")
    print()
    print("  the gate uses math.erf. the Abramowitz and Stegun polynomial is kept because it")
    print("  is what you write when erf is unavailable, and having both makes the test checkable.")


if __name__ == "__main__":
    main()
