"""Web adapter for project 12, the prompt registry and A/B system.

The visitor types a unit id. The adapter shows which arm that id lands in and
that the answer never changes, runs the real 600-request workload through the
splitter, and puts the result in front of the promotion gate twice.
"""
from __future__ import annotations

import os
import tempfile

NUMBER = 12
SLUG = "prompt-registry"
TITLE = "Prompt Versioning and A/B System"
TAGLINE = "Type a customer name and see which version of a prompt they get, and whether the new one earns its promotion."
WHAT_IT_DOES = """When a team changes the wording of a prompt, two questions
follow. Which customers are seeing the new version, and is it actually better? Both
are easy to get wrong: a split that reshuffles on every request means one customer
sees both versions and the comparison measures nothing.

Type a customer or company name. The page works out which of the two versions that
name gets, shows that the answer is the same every time it is asked, and shows the
same name landing somewhere else in a different experiment. There is no table of
assignments anywhere. The arm is worked out from the name itself, so it survives a
restart and can be recalculated later from a log.

Then six hundred customers go through both versions for real, each one answering a
question with a known correct answer, and the results are put in front of a gate that
decides whether the new version ships. The page shows the gate refusing on a small
sample even though the numbers already look convincing, and agreeing later on the full
sample, with the odds that the difference is chance."""
INPUT_LABEL = "A customer, tenant or user name"
PLACEHOLDER = "acme-corp"
EXAMPLES = ["acme-corp", "meridian-labs", "tenant-4471", "arnob"]
SOURCE = "projects/p12_prompt_registry"

EXPERIMENT = "evidence-format-2026-09"
POPULATION = 600
EARLY_CHECKPOINT = 40
MIN_SAMPLES = 200


def _unit(user_input: str) -> str:
    unit = " ".join((user_input or "").split())[:60]
    return unit or EXAMPLES[0]


def run(user_input: str) -> str:
    try:
        from llmkit import EchoLLM

        from projects.p12_prompt_registry.gate import PromotionGate
        from projects.p12_prompt_registry.outcomes import ArmSummary, OutcomeStore
        from projects.p12_prompt_registry.registry import PromptRegistry
        from projects.p12_prompt_registry.splitter import Arm, TrafficSplitter
        from projects.p12_prompt_registry.workload import (
            PROMPT_NAME,
            question_for,
            register_variants,
            run_workload,
            unit_ids,
        )

        unit = _unit(user_input)

        # The registry persists to JSON. In a serverless function the only
        # writable place is a temporary directory.
        workdir = tempfile.mkdtemp(prefix="p12-")
        registry = PromptRegistry(path=os.path.join(workdir, "registry.json"))
        control_id, challenger_id = register_variants(registry)
        registry.set_label(PROMPT_NAME, "dev", challenger_id, actor="arnob", reason="new variant under test")
        registry.set_label(PROMPT_NAME, "staging", challenger_id, actor="arnob", reason="promote dev to staging")
        registry.set_label(PROMPT_NAME, "prod", control_id, actor="arnob", reason="initial production prompt")

        arms = [Arm("control", control_id, 0.5), Arm("challenger", challenger_id, 0.5)]
        splitter = TrafficSplitter(experiment=EXPERIMENT, arms=arms)
        assigned = splitter.assign(unit)
        rebuilt = TrafficSplitter(experiment=EXPERIMENT, arms=arms).assign(unit)
        elsewhere = TrafficSplitter(experiment="pricing-copy-2026-10", arms=arms).assign(unit)
        repeats = {splitter.assign(unit).name for _ in range(5)}

        out = []
        out.append("THE TWO VERSIONS OF THIS PROMPT")
        out.append(f"  control     {control_id}   one supporting document, answer it")
        out.append(f"  challenger  {challenger_id}   three tagged documents, quote and cite the one you used")
        out.append("  the id is a hash of the wording and the settings, so an edit cannot keep the old id")
        out.append(f"  live right now: prod -> {registry.label_of(PROMPT_NAME, 'prod')}"
                   f",  staging -> {registry.label_of(PROMPT_NAME, 'staging')}")

        out.append("")
        out.append("=" * 78)
        out.append(f"WHICH VERSION DOES \"{unit}\" GET")
        out.append("=" * 78)
        out.append("")
        out.append(f"  the name is hashed with the experiment name into a number from 0 to 9999")
        out.append(f"  \"{unit}\" lands on {splitter.bucket(unit)}, which is "
                   f"{splitter.position(unit) * 100:.1f}% of the way along the line")
        out.append(f"  the line is split 50/50, so this name gets:  {assigned.name.upper()}"
                   f"  ({assigned.version_id})")
        out.append("")
        out.append(f"  asked five times in a row:                   {', '.join(sorted(repeats))}")
        out.append(f"  asked again from a splitter built from scratch: {rebuilt.name}"
                   f"  (a restart changes nothing)")
        shuffle_note = "a different arm" if elsewhere.name != assigned.name else "the same arm this time, which happens about half the time"
        out.append(f"  the same name in a different experiment:     {elsewhere.name}"
                   f"  ({shuffle_note})")
        out.append("  no assignment is written down anywhere. the answer is recalculated from the name,")
        out.append("  which is why an analyst can rebuild who saw what months later from a log alone.")

        population = unit_ids(POPULATION)
        if unit not in population:
            population = [unit] + population[: POPULATION - 1]
        store = OutcomeStore()
        run_workload(registry, splitter, store, population, EchoLLM(), experiment=EXPERIMENT)

        distribution = splitter.distribution(population)
        mine = next(o for o in store.for_experiment(EXPERIMENT) if o.unit_id == unit)
        gold = question_for(unit)

        out.append("")
        out.append("=" * 78)
        out.append(f"{len(store)} CUSTOMERS PUT THROUGH BOTH VERSIONS FOR REAL")
        out.append("=" * 78)
        out.append("")
        out.append("  each one asks a question with a known correct answer, and the answer either")
        out.append("  contains the fact it should or it does not. nothing here is a random number.")
        out.append("")
        out.append(f"  your row:  \"{unit}\" asked \"{gold['question']}\"")
        out.append(f"             served the {mine.arm} version, correct answer present: "
                   f"{'yes' if mine.success else 'no'}"
                   f"  ({mine.prompt_tokens + mine.completion_tokens} tokens)")
        out.append("")
        out.append(ArmSummary.header())
        out.append("  " + "-" * 89)
        summaries = {}
        for summary in store.summaries(EXPERIMENT):
            summaries[summary.arm] = summary
            out.append(summary.row())
        out.append("")
        split_line = ", ".join(f"{name} {share * 100:.1f}%" for name, share in distribution.items())
        out.append(f"  traffic split: {split_line} against an intended 50/50, largest gap "
                   f"{splitter.max_deviation(population) * 100:.2f} points")
        out.append("  a hash split is never exactly even, so the thing to watch is the size of the gap")

        gate = PromotionGate(min_samples_per_arm=MIN_SAMPLES, alpha=0.05, min_absolute_lift=0.05)
        early = OutcomeStore(store.for_experiment(EXPERIMENT)[:EARLY_CHECKPOINT])
        early_decision = gate.evaluate(early, EXPERIMENT, control="control", challenger="challenger")
        decision = gate.evaluate(store, EXPERIMENT, control="control", challenger="challenger")

        out.append("")
        out.append("=" * 78)
        out.append("SHOULD THE NEW VERSION SHIP")
        out.append("=" * 78)
        out.append("")
        out.append(f"  AFTER THE FIRST {EARLY_CHECKPOINT} REQUESTS")
        out.append(early_decision.render())
        if early_decision.test is not None:
            early_p = early_decision.test.p_value
            if early_p < gate.alpha:
                out.append(f"  the odds this gap is chance are already {early_p * 100:.1f}%, under the usual")
                out.append("  5% mark, so a live dashboard would be calling this a winner right now")
            else:
                out.append(f"  the odds this gap is chance are {early_p * 100:.1f}% here, but the gate never got")
                out.append("  as far as looking: the sample floor is checked first")
        if early_decision.needed_per_arm:
            out.append(f"  for an effect this size the sample really needed is about "
                       f"{early_decision.needed_per_arm} per version")
        if early_decision.test is not None and decision.test is not None:
            out.append(f"  this checkpoint shows a lift of {early_decision.test.absolute_lift * 100:+.1f} points. "
                       f"the full sample below settles at {decision.test.absolute_lift * 100:+.1f}.")
        out.append("  the floor is a promise made before the numbers arrived, which is the only reason")
        out.append("  the decision is still open at this point.")

        out.append("")
        out.append(f"  AFTER ALL {len(store)} REQUESTS")
        out.append(decision.render())
        if decision.test is not None:
            out.append(f"  enough data: yes, {summaries['control'].samples} and "
                       f"{summaries['challenger'].samples} against a floor of {MIN_SAMPLES} each")
            out.append(f"  odds this gap is chance: {decision.test.p_value * 100:.3f}%")

        control_summary, challenger_summary = summaries["control"], summaries["challenger"]
        cost_ratio = challenger_summary.cost_usd / max(control_summary.cost_usd, 1e-9)
        token_ratio = challenger_summary.mean_total_tokens / max(control_summary.mean_total_tokens, 1e-9)
        strict = PromotionGate(min_samples_per_arm=MIN_SAMPLES, alpha=0.05,
                               min_absolute_lift=0.05, max_cost_ratio=1.5)
        strict_decision = strict.evaluate(store, EXPERIMENT, control="control", challenger="challenger")
        out.append("")
        out.append("  THE SAME RESULT UNDER A COST CEILING")
        out.append(f"  the new version sends {token_ratio:.2f}x the words and costs {cost_ratio:.2f}x as much")
        out.append(strict_decision.render())

        out.append("")
        out.append("=" * 78)
        out.append("WHAT HAPPENS NEXT")
        out.append("=" * 78)
        out.append("")
        if decision.promote:
            registry.set_label(PROMPT_NAME, "prod", challenger_id, actor="ci-bot",
                               reason=f"gate passed, p={decision.test.p_value:.5f}")
            out.append(f"  promoted: prod now points at {registry.label_of(PROMPT_NAME, 'prod')}")
        else:
            out.append(f"  not promoted: prod stays on {registry.label_of(PROMPT_NAME, 'prod')}")
        entry = registry.rollback(PROMPT_NAME, "prod", actor="oncall", reason="cost ceiling flagged in review")
        out.append(f"  rolled back in one call: prod is {entry.to_version} again")
        out.append("  nothing was copied between environments. a label is a pointer, and rollback moves")
        out.append("  it back to whatever the written-down history says it pointed at before.")
        out.append("")
        path = registry.save()
        reloaded = PromptRegistry.load(path)
        out.append(f"  saved and reloaded: {len(reloaded.versions)} versions, "
                   f"{len(reloaded.audit)} recorded changes, prod pointer intact: "
                   f"{reloaded.label_of(PROMPT_NAME, 'prod') == registry.label_of(PROMPT_NAME, 'prod')}")
        out.append("  editing that file by hand breaks the id it claims, and loading it fails on purpose")
        out.append("")
        out.append("  who changed what:")
        for record in registry.history(PROMPT_NAME)[-4:]:
            out.append(record.row())
        return "\n".join(out)
    except Exception as exc:  # a demo page must never 500
        return f"This demo could not run: {type(exc).__name__}: {exc}"
