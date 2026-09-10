"""Tests for the agent memory system. Everything runs offline."""
import pytest

from llmkit import EchoLLM, count_tokens

from projects.p09_agent_memory.conversation import (
    CHANGED_PREDICATE,
    CHANGED_SUBJECT,
    NEW_VALUE,
    OLD_VALUE,
    RECALL_ANSWER,
    RECALL_QUERY,
    conversation,
)
from projects.p09_agent_memory.episodic import EpisodicMemory, extractive_summary, llm_summary
from projects.p09_agent_memory.eviction import decay_score, keep_score, plan_evictions
from projects.p09_agent_memory.manager import MemoryManager, RecentWindowBaseline
from projects.p09_agent_memory.records import Episode, Fact, Turn
from projects.p09_agent_memory.semantic import FactExtractor, SemanticMemory
from projects.p09_agent_memory.working import WorkingMemory

WORKING_TOKENS = 220
MEMORY_BUDGET = 300
RECALL_BUDGET = 300


@pytest.fixture(scope="module")
def run():
    manager = MemoryManager(
        working_max_tokens=WORKING_TOKENS, budget_tokens=MEMORY_BUDGET, episode_max_tokens=70, llm=EchoLLM()
    )
    manager.semantic.add_fact(
        Fact(id="op.residency", subject="customer data", predicate="must stay in",
             object="the workspace region", turn_index=0, confidence=1.0)
    )
    manager.pin("op.residency")
    baseline = RecentWindowBaseline(token_budget=RECALL_BUDGET)
    for role, text in conversation():
        manager.observe(role, text)
        baseline.observe(role, text)
    return manager, baseline


# -- the headline claim -------------------------------------------------


def test_a_fact_from_turn_2_is_recalled_at_turn_32(run):
    manager, _ = run
    assert manager.turn_index >= 30
    result = manager.recall(RECALL_QUERY, token_budget=RECALL_BUDGET)
    assert result.contains(RECALL_ANSWER)
    assert any(RECALL_ANSWER in f.object for f in result.facts)
    assert result.tokens <= RECALL_BUDGET


def test_the_no_memory_baseline_fails_the_same_recall(run):
    _, baseline = run
    result = baseline.recall(RECALL_QUERY, token_budget=RECALL_BUDGET)
    assert not result.contains(RECALL_ANSWER)
    # It fails because it is holding recent turns, not because it holds nothing.
    assert len(result.turns) > 5
    assert result.tokens <= RECALL_BUDGET


def test_long_lived_memory_stays_under_budget_for_the_whole_run(run):
    manager, _ = run
    stats = manager.stats()
    assert stats.within_budget
    assert stats.long_lived_tokens <= MEMORY_BUDGET
    assert stats.working_tokens <= WORKING_TOKENS
    # And it is genuinely smaller than keeping the transcript.
    transcript = count_tokens("\n".join(t for _, t in conversation()))
    assert stats.long_lived_tokens < transcript


# -- working memory and compression -------------------------------------


def test_working_memory_drains_to_the_low_water_mark_in_batches():
    wm = WorkingMemory(max_tokens=60, min_turns=1, low_water=0.6)
    evicted_batches = []
    for i in range(12):
        out = wm.add(Turn(index=i + 1, role="user", text=f"turn number {i} with some filler words here"))
        if out:
            evicted_batches.append(out)
    assert evicted_batches, "a 60 token buffer must overflow"
    assert any(len(batch) > 1 for batch in evicted_batches), "overflow should drain a run of turns"
    assert wm.tokens <= 60


def test_overflowing_turns_become_an_episode_rather_than_disappearing(run):
    manager, _ = run
    episode_events = manager.timeline(["episode"])
    assert episode_events
    covered = {t for ep in manager.episodic.episodes for t in ep.source_turns}
    assert covered, "episodes must carry provenance back to their turns"
    for episode in manager.episodic.episodes:
        assert episode.tokens < episode.original_tokens
        assert 0.0 < episode.compression_ratio < 1.0


def test_extractive_summary_keeps_the_numbers_and_drops_the_acknowledgements():
    turns = [
        Turn(1, "user", "Our sustained limit is 600 requests per minute."),
        Turn(2, "assistant", "Understood."),
        Turn(3, "user", "Thanks."),
    ]
    summary = extractive_summary(turns, 40)
    assert "600" in summary
    assert "Understood" not in summary
    assert count_tokens(summary) <= 40


def test_llm_summary_is_capped_and_falls_back_without_a_model():
    turns = [Turn(i, "user", f"Sentence {i} about the export pipeline and its limits.") for i in range(1, 6)]
    with_llm = llm_summary(turns, 30, EchoLLM())
    without = llm_summary(turns, 30, None)
    assert count_tokens(with_llm) <= 30
    assert count_tokens(without) <= 30
    assert without  # the fallback is the rule-based summariser, not an empty string


# -- semantic memory ----------------------------------------------------


def test_two_facts_are_extracted_from_one_compound_sentence():
    facts = FactExtractor().extract(
        "Our workspace is pinned to eu-west and our production database is Postgres 14.", 2
    )
    objects = {f.object for f in facts}
    assert "pinned to eu-west" in objects
    assert "postgres 14" in objects


def test_a_restated_fact_is_deduplicated_and_gains_confidence():
    store = SemanticMemory()
    store.observe("Our region is eu-west.", 1)
    before = store.active[0].confidence
    added, superseded = store.observe("Our region is eu-west.", 5)
    assert added == [] and superseded == []
    assert len(store.active) == 1
    assert store.active[0].confidence > before


def test_a_newer_fact_supersedes_the_older_one_and_the_old_version_survives(run):
    manager, _ = run
    history = manager.semantic.history(CHANGED_SUBJECT, CHANGED_PREDICATE)
    assert len(history) == 2
    old, new = history
    assert OLD_VALUE in old.object and NEW_VALUE in new.object
    assert old.superseded_by == new.id
    assert new.supersedes == old.id
    assert not old.active and new.active
    # Provenance points back at the turn that produced each version.
    assert old.turn_index < new.turn_index
    assert old.evidence


def test_search_never_returns_a_superseded_fact(run):
    manager, _ = run
    hits = manager.semantic.search("production database version", k=5, now_turn=manager.turn_index)
    assert hits
    assert all(f.active for f in hits)
    assert any(NEW_VALUE in f.object for f in hits)
    assert not any(OLD_VALUE in f.object for f in hits)


def test_provenance_chain_length_is_bounded():
    store = SemanticMemory(max_history=3)
    for turn, value in enumerate(["v1", "v2", "v3", "v4", "v5", "v6"], start=1):
        store.observe(f"Our api version is {value}.", turn)
    chain = store.history("our api version", "is")
    assert len(chain) <= 3
    assert chain[-1].object == "v6"
    assert chain[-1].active


def test_a_pinned_fact_is_not_overwritten_by_a_later_user_claim():
    store = SemanticMemory()
    store.observe("Our region is eu-west.", 1)
    store.pin(store.active[0].id)
    added, superseded = store.observe("Our region is us-east.", 9)
    assert added == [] and superseded == []
    assert store.active[0].object == "eu-west"


# -- episodic retrieval -------------------------------------------------


def test_episodic_retrieval_blends_relevance_and_recency():
    memory = EpisodicMemory(recency_half_life=5.0)
    old_relevant = Episode(id="old", text="the nightly export hits the rate limit at 900 requests per minute",
                           first_turn=1, last_turn=2, source_turns=[1, 2], original_tokens=60, last_used_turn=2)
    new_irrelevant = Episode(id="new", text="we discussed invoice currency and billing dates",
                             first_turn=20, last_turn=21, source_turns=[20, 21], original_tokens=60,
                             last_used_turn=21)
    memory.append(old_relevant)
    memory.append(new_irrelevant)

    on_topic = memory.search("rate limit during the nightly export", now_turn=22, k=2)
    assert on_topic[0][0].id == "old"  # relevance beats recency when the query is specific
    ambient = memory.search("what were we just talking about", now_turn=22, k=2)
    assert ambient[0][0].id == "new"  # recency wins when the query carries no signal


def test_episode_ids_are_never_reused_after_an_eviction(run):
    manager, _ = run
    created = [e.detail.split(" into ")[1].split(" ")[0] for e in manager.timeline(["episode"])]
    assert len(created) == len(set(created))
    assert manager.episodic.sequence >= len(manager.episodic)


# -- eviction -----------------------------------------------------------


def test_decay_prefers_recently_used_over_recently_created():
    old_but_used = Episode(id="a", text="x", first_turn=1, last_turn=2, source_turns=[1],
                           original_tokens=10, last_used_turn=29, access_count=4)
    new_but_ignored = Episode(id="b", text="x", first_turn=27, last_turn=28, source_turns=[27],
                              original_tokens=10, last_used_turn=28, access_count=0)
    now = 30
    assert decay_score(old_but_used, now) > decay_score(new_but_ignored, now)


def test_eviction_brings_the_store_under_budget_and_never_takes_a_pin():
    facts = [
        Fact(id="pinned", subject="policy", predicate="is", object="data stays in region",
             turn_index=1, pinned=True),
        Fact(id="stale", subject="we", predicate="use", object="an old thing", turn_index=1),
    ]
    episodes = [
        Episode(id=f"ep{i}", text="filler " * 30, first_turn=i, last_turn=i, source_turns=[i],
                original_tokens=90, last_used_turn=i)
        for i in range(1, 6)
    ]
    evicted, remaining = plan_evictions(facts, episodes, now_turn=30, budget_tokens=120)
    assert remaining <= 120
    assert evicted
    assert "pinned" not in {c.id for c in evicted}


def test_a_fact_is_not_evicted_to_keep_a_bulkier_episode():
    fact = Fact(id="f1", subject="our workspace", predicate="is", object="pinned to eu-west", turn_index=2)
    episode = Episode(id="ep1", text="filler " * 40, first_turn=2, last_turn=5, source_turns=[2, 3, 4, 5],
                      original_tokens=140, last_used_turn=5)
    now = 30
    assert keep_score(fact, "fact", now, 15.0, 0.75) > keep_score(episode, "episode", now, 15.0, 0.55)
    evicted, _ = plan_evictions([fact], [episode], now_turn=now, budget_tokens=30)
    assert [c.id for c in evicted] == ["ep1"]


def test_superseded_facts_do_not_consume_the_prompt_budget(run):
    manager, _ = run
    stats = manager.stats()
    assert stats.provenance_tokens > 0
    assert stats.long_lived_tokens == manager.episodic.tokens + manager.semantic.active_tokens
    assert manager.semantic.tokens > manager.semantic.active_tokens


# -- recall assembly ----------------------------------------------------


def test_recall_respects_its_token_budget_at_several_sizes(run):
    manager, _ = run
    for budget in (80, 150, 300, 600):
        result = manager.recall(RECALL_QUERY, token_budget=budget)
        assert result.tokens <= budget, f"recall overran a {budget} token budget"


def test_recall_puts_facts_before_episodes_and_turns(run):
    manager, _ = run
    result = manager.recall(RECALL_QUERY, token_budget=RECALL_BUDGET)
    text = result.text
    assert text.startswith("Known facts:")
    if "Earlier in this conversation:" in text and "Recent turns:" in text:
        assert text.index("Earlier in this conversation:") < text.index("Recent turns:")


def test_manager_serialises_its_whole_state(run):
    manager, _ = run
    payload = manager.to_dict()
    assert payload["turn_index"] == manager.turn_index
    assert payload["facts"] and payload["episodes"]
    assert all("superseded_by" in f for f in payload["facts"])
