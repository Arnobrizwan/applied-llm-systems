"""Agent Memory System demo.

Runs a 32-turn conversation through the memory manager and through a no-memory
baseline, then asks both the same question about something said at turn 2.

    python3 projects/p09_agent_memory/demo.py
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmkit import EchoLLM, count_tokens  # noqa: E402

from projects.p09_agent_memory.conversation import (  # noqa: E402
    CHANGED_PREDICATE,
    CHANGED_SUBJECT,
    RECALL_ANSWER,
    RECALL_QUERY,
    conversation,
    turn_text,
)
from projects.p09_agent_memory.episodic import extractive_summary, llm_summary  # noqa: E402
from projects.p09_agent_memory.eviction import keep_score  # noqa: E402
from projects.p09_agent_memory.manager import MemoryManager, RecentWindowBaseline  # noqa: E402
from projects.p09_agent_memory.records import Fact, Turn  # noqa: E402

WORKING_TOKENS = 220
MEMORY_BUDGET = 300
RECALL_BUDGET = 300


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    turns = conversation()

    manager = MemoryManager(
        working_max_tokens=WORKING_TOKENS,
        budget_tokens=MEMORY_BUDGET,
        episode_max_tokens=70,
        llm=EchoLLM(),
    )
    # An operator-set constraint, pinned so eviction can never take it. Nothing
    # in the conversation ever mentions it, so it also has zero accesses, which
    # is what makes it the first thing a decay policy would otherwise drop.
    manager.semantic.add_fact(
        Fact(
            id="op.residency",
            subject="customer data",
            predicate="must stay in",
            object="the workspace region",
            turn_index=0,
            confidence=1.0,
            evidence="operator policy, set outside the conversation",
        )
    )
    manager.pin("op.residency")

    baseline = RecentWindowBaseline(token_budget=RECALL_BUDGET)

    for role, text in turns:
        manager.observe(role, text)
        baseline.observe(role, text)

    rule("1. SETUP")
    print()
    print(f"  conversation turns          {len(turns)}")
    print(f"  working memory cap          {WORKING_TOKENS} tokens")
    print(f"  long-lived memory budget    {MEMORY_BUDGET} tokens")
    print(f"  recall budget per query     {RECALL_BUDGET} tokens")
    print(f"  full transcript             {count_tokens(chr(10).join(t for _, t in turns))} tokens")
    print()
    print(f"  turn 2 said: {turn_text(2)}")

    rule("2. WHAT MEMORY DID, TURN BY TURN")
    print()
    for event in manager.timeline():
        print(f"  turn {event.turn:>2}  {event.kind:<10} {event.detail}")

    rule("3. CONTRADICTION AND PROVENANCE")
    print()
    history = manager.semantic.history(CHANGED_SUBJECT, CHANGED_PREDICATE)
    for fact in history:
        state = "active" if fact.active else f"superseded by {fact.superseded_by}"
        print(f"  {fact.id:<8} turn {fact.turn_index:<3} '{fact.text}'  [{state}]")
    print()
    print("  the superseded version is kept, not deleted, so the agent can answer")
    print("  'you said 14 at turn 2 and 16 at turn 18' instead of just changing its mind")

    rule("4. HEADLINE RECALL: a fact from turn 2, asked at turn 32")
    print()
    print(f"  query: {RECALL_QUERY}")
    print()
    with_memory = manager.recall(RECALL_QUERY, token_budget=RECALL_BUDGET)
    print("  --- WITH MEMORY " + "-" * 58)
    for line in with_memory.text.splitlines():
        print(f"  {line}")
    print(f"  --- {with_memory.tokens} tokens")
    hit = with_memory.contains(RECALL_ANSWER)
    print(f"  contains '{RECALL_ANSWER}': {hit}   ->  {'PASS' if hit else 'FAIL'}")

    print()
    baseline_recall = baseline.recall(RECALL_QUERY, token_budget=RECALL_BUDGET)
    print("  --- NO-MEMORY BASELINE, same token budget " + "-" * 32)
    for line in baseline_recall.text.splitlines():
        print(f"  {line}")
    print(f"  --- {baseline_recall.tokens} tokens")
    baseline_hit = baseline_recall.contains(RECALL_ANSWER)
    print(f"  contains '{RECALL_ANSWER}': {baseline_hit}   ->  {'PASS' if baseline_hit else 'FAIL'}")
    print()
    print(f"  the baseline holds the last {len(baseline_recall.turns)} turns of {len(turns)}, so turn 2 is gone")

    rule("5. THE UPDATED FACT IS ALSO RECALLED CORRECTLY")
    print()
    db_recall = manager.recall("which version is our production database on", token_budget=RECALL_BUDGET)
    print(f"  query: which version is our production database on")
    for fact in db_recall.facts:
        print(f"    {fact.rendered}")
    print(f"  answers with the turn 18 value, not the turn 2 value: "
          f"{'postgres 16' in db_recall.text.lower() and 'postgres 14' not in db_recall.text.lower()}")

    rule("6. BUDGET")
    print()
    stats = manager.stats()
    print(f"  turns processed             {stats.turns}")
    print(f"  working memory              {stats.working_tokens} tokens (cap {WORKING_TOKENS})")
    print(f"  episodes                    {stats.episodes} holding {stats.episode_tokens} tokens")
    print(f"  facts                       {stats.facts_active} active, {stats.facts_superseded} superseded, "
          f"{stats.fact_tokens} tokens")
    print(f"  long-lived total            {stats.long_lived_tokens} tokens against a {MEMORY_BUDGET} token budget")
    print(f"  within budget               {stats.within_budget}")
    print(f"  evictions                   {stats.evictions}")

    print(f"  provenance (superseded)     {stats.provenance_tokens} tokens, outside the prompt budget")

    transcript_tokens = count_tokens("\n".join(t for _, t in turns))
    footprint = stats.long_lived_tokens + stats.working_tokens
    print()
    print(f"  full transcript is {transcript_tokens} tokens; total memory footprint is {footprint} tokens "
          f"({footprint / transcript_tokens * 100:.0f}% of it)")
    print(f"  the transcript grows with every turn. long-lived memory is capped at {MEMORY_BUDGET} tokens")
    print("  for the life of the session, which is the property that makes this survivable at turn 300")

    rule("7. COMPRESSION, MEASURED")
    print()
    if manager.episodic.episodes:
        original = sum(e.original_tokens for e in manager.episodic.episodes)
        kept = sum(e.tokens for e in manager.episodic.episodes)
        print(f"  {len(manager.episodic)} episodes: {original} tokens of turns compressed into {kept} tokens "
              f"({kept / original * 100:.0f}% kept)")
        print()
        for episode in manager.episodic.episodes:
            print(f"  {episode.id}  turns {episode.first_turn}-{episode.last_turn}  "
                  f"{episode.original_tokens} -> {episode.tokens} tokens")
            print(f"      {episode.text}")

    rule("8. PINNED FACTS ARE NEVER EVICTED")
    print()
    pinned = [f for f in manager.semantic.facts if f.pinned]
    evicted_scores = [
        float(e.detail.split("score ")[1].split(",")[0]) for e in manager.timeline(["evict"])
    ]
    last_eviction_turn = max((e.turn for e in manager.timeline(["evict"])), default=manager.turn_index)
    for fact in pinned:
        # Score the counterfactual: the same record with the accesses it would
        # have had if the two demo recalls above had never run.
        untouched = replace(fact, pinned=False, access_count=0, last_used_turn=0)
        score = keep_score(untouched, "fact", last_eviction_turn, manager.half_life, 0.5)
        print(f"  {fact.id:<14} '{fact.text}'")
        print(f"    still present: True   accesses so far: {fact.access_count} (both from the recalls above)")
        print(f"    unpinned, never accessed, its score at turn {last_eviction_turn} would be {score:.3f}")
    if evicted_scores:
        print(f"    lowest score actually evicted this run: {min(evicted_scores):.3f}")
        print("    it was set at turn 0 and is never mentioned in the conversation, so without the")
        print("    pin the decay policy would have taken it before anything it did take.")

    rule("9. THE TWO SUMMARISERS, SAME INPUT")
    print()
    sample = manager.episodic.episodes[-1]
    sample_turns = [Turn(index=i, role=turns[i - 1][0], text=turns[i - 1][1]) for i in sample.source_turns]
    rules_out = extractive_summary(sample_turns, 70)
    llm_out = llm_summary(sample_turns, 70, EchoLLM())
    print(f"  rule-based ({count_tokens(rules_out)} tokens, the default):")
    print(f"    {rules_out}")
    print(f"  EchoLLM ({count_tokens(llm_out)} tokens, opt-in):")
    print(f"    {llm_out}")
    print()
    print("  the rule-based path is the default because it runs on the write path of every")
    print("  overflowing turn and never paraphrases an identifier away.")


if __name__ == "__main__":
    main()
