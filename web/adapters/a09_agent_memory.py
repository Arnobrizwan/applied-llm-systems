"""Web adapter for project 09, the agent memory system.

The visitor plants a fact early in a 32-turn conversation and asks about it at
the end, or just asks the scripted conversation a question. The adapter runs the
real manager and the no-memory baseline side by side and prints both answers.
"""
from __future__ import annotations

NUMBER = 9
SLUG = "agent-memory"
TITLE = "Agent Memory System"
TAGLINE = "Tell an assistant something early in a long chat, then ask about it thirty turns later."
WHAT_IT_DOES = """Most chat assistants remember by keeping the last few messages
and forgetting everything before that. It works until the conversation gets long, and
then something you said near the start is simply gone, with nothing in any log to say
it was ever there.

Type a fact, a vertical bar, then a question. The fact is dropped into the second turn
of a thirty-two turn support conversation, buried under thirty more turns, and your
question is asked at the very end. You can also type a question on its own and ask it
against the conversation as written.

The page shows three kinds of memory filling up: the recent messages held word for
word, older stretches squeezed into short summaries, and single facts pulled out and
stored on their own. It shows what got thrown away to stay inside a fixed budget, and
it answers your question twice, once with memory and once with only the recent
messages, so you can see which one still knows."""
INPUT_LABEL = "A fact to plant, a vertical bar, then a question to ask thirty turns later"
PLACEHOLDER = "our billing region is ap-south | which region is our billing in?"
EXAMPLES = [
    "our billing region is ap-south | which region is our billing in?",
    "which region is our workspace pinned to?",
    "our event bus is kafka | what is our event bus?",
    "our on-call rotation is weekly | how often does on-call rotate?",
]
SOURCE = "projects/p09_agent_memory"

WORKING_TOKENS = 220
MEMORY_BUDGET = 300
RECALL_BUDGET = 300
PLANT_AT = 2  # the planted turn lands right after the opening exchange
MAX_EVENT_LINES = 20


def _parse(user_input: str):
    """Split '<fact> | <question>' into its two halves."""
    text = (user_input or "").strip()
    if not text:
        text = EXAMPLES[0]
    if "|" in text:
        fact, _, question = text.partition("|")
        fact, question = fact.strip(), question.strip()
    else:
        fact, question = "", text
    if not question:
        question = "which region is our workspace pinned to?"
    return fact[:220], question[:220]


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def run(user_input: str) -> str:
    try:
        from llmkit import EchoLLM, count_tokens

        from projects.p09_agent_memory.conversation import conversation
        from projects.p09_agent_memory.manager import MemoryManager, RecentWindowBaseline
        from projects.p09_agent_memory.records import Fact

        planted_text, question = _parse(user_input)

        turns = conversation()
        if planted_text:
            turns.insert(PLANT_AT, ("user", planted_text))

        manager = MemoryManager(
            working_max_tokens=WORKING_TOKENS,
            budget_tokens=MEMORY_BUDGET,
            episode_max_tokens=70,
            llm=EchoLLM(),
        )
        # An operator constraint, set before the conversation and pinned so the
        # eviction policy can never take it.
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

        planted_ids = []
        for index, (role, text) in enumerate(turns, start=1):
            known = {f.id for f in manager.semantic.facts}
            manager.observe(role, text)
            baseline.observe(role, text)
            if planted_text and index == PLANT_AT + 1:
                planted_ids = [f.id for f in manager.semantic.facts if f.id not in known]
        planted_facts = [manager.semantic.get(i) for i in planted_ids]
        planted_facts = [f for f in planted_facts if f is not None]
        evicted_plants = len(planted_ids) - len(planted_facts)
        # A planted sentence the conversation already said is merged into the
        # existing fact rather than stored twice, so it produces no new id.
        merged_facts = []
        if planted_text and not planted_ids:
            keys = {c.key for c in manager.semantic.extractor.extract(planted_text, PLANT_AT + 1)}
            merged_facts = [f for f in manager.semantic.active if f.key in keys]

        stats = manager.stats()
        transcript_tokens = count_tokens("\n".join(t for _, t in turns))

        out = []
        out.append("THE CONVERSATION")
        out.append(f"  turns played:            {len(turns)}")
        out.append(f"  full transcript:         {transcript_tokens} tokens, and it grows every turn")
        out.append(f"  recent-message buffer:   capped at {WORKING_TOKENS} tokens")
        out.append(f"  long-term memory:        capped at {MEMORY_BUDGET} tokens, for the whole session")
        if planted_text:
            out.append("")
            out.append(f"  planted as turn {PLANT_AT + 1}:      \"{_clip(planted_text, 90)}\"")
            if planted_ids:
                for fact in planted_facts:
                    state = "stored" if fact.active else f"later corrected by {fact.superseded_by}"
                    out.append(f"    the extractor pulled out: {fact.text}  [{state}]")
                if evicted_plants:
                    out.append(f"    {evicted_plants} extracted fact(s) were later evicted to stay inside the budget")
            elif merged_facts:
                out.append("    the conversation already makes this claim, so it was merged into the fact")
                out.append("    that was there rather than stored twice. that claim now reads:")
                for fact in merged_facts:
                    out.append(f"      {fact.text}  (from turn {fact.turn_index})")
            else:
                out.append("    the extractor found no fact in that sentence. it is a small set of rules,")
                out.append("    not a model: it understands 'our X is Y', 'we use Y', 'I work at Y',")
                out.append("    'I prefer Y' and 'call me X'. anything else survives only as conversation.")
        else:
            out.append("  nothing planted, so this is the conversation exactly as written")
        out.append(f"  asked after turn {len(turns)}: \"{_clip(question, 80)}\"")

        out.append("")
        out.append("=" * 78)
        out.append("WHAT MEMORY DID WHILE THE CONVERSATION RAN")
        out.append("=" * 78)
        out.append("")
        events = manager.timeline()
        for event in events[:MAX_EVENT_LINES]:
            out.append(f"  turn {event.turn:>2}  {event.kind:<10} {_clip(event.detail, 88)}")
        if len(events) > MAX_EVENT_LINES:
            out.append(f"  ... and {len(events) - MAX_EVENT_LINES} more events")

        out.append("")
        out.append("=" * 78)
        out.append("WHAT IS IN MEMORY AT THE END")
        out.append("=" * 78)
        out.append("")
        out.append(f"  RECENT MESSAGES, word for word: {len(manager.working)} turns, "
                   f"{stats.working_tokens} of {WORKING_TOKENS} tokens")
        recent = manager.working.turns[-2:]
        for turn in recent:
            out.append(f"    {_clip(turn.rendered, 88)}")

        out.append("")
        out.append(f"  OLDER STRETCHES, squeezed into summaries: {stats.episodes} kept, "
                   f"{stats.episode_tokens} tokens")
        for episode in manager.episodic.episodes:
            out.append(f"    {episode.id}  turns {episode.first_turn}-{episode.last_turn}  "
                       f"{episode.original_tokens} -> {episode.tokens} tokens "
                       f"({episode.compression_ratio * 100:.0f}% kept)")
            out.append(f"      {_clip(episode.text, 86)}")

        out.append("")
        out.append(f"  SINGLE FACTS, pulled out and stored on their own: "
                   f"{stats.facts_active} live, {stats.facts_superseded} corrected")
        for fact in manager.semantic.facts:
            if fact.active:
                tag = "pinned" if fact.pinned else "live"
            else:
                tag = f"corrected by {fact.superseded_by}"
            out.append(f"    {fact.id:<10} turn {fact.turn_index:<3} {_clip(fact.text, 52):<52} [{tag}]")

        evictions = manager.timeline(["evict"])
        out.append("")
        out.append(f"  THROWN AWAY to stay inside the budget: {len(evictions)}")
        for event in evictions:
            out.append(f"    {_clip(event.detail, 86)}")
        out.append(f"  long-term memory finished at {stats.long_lived_tokens} of {MEMORY_BUDGET} tokens"
                   f", within budget: {stats.within_budget}")
        out.append(f"  the whole memory footprint is {stats.long_lived_tokens + stats.working_tokens} tokens"
                   f" against a {transcript_tokens} token transcript, and it stops growing here")

        out.append("")
        out.append("=" * 78)
        out.append("THE ANSWER, WITH MEMORY AND WITHOUT")
        out.append("=" * 78)
        out.append("")
        out.append(f"  question: {question}")
        out.append("")

        with_memory = manager.recall(question, token_budget=RECALL_BUDGET)
        without = baseline.recall(question, token_budget=RECALL_BUDGET)

        out.append(f"  --- WITH MEMORY ({with_memory.tokens} tokens) " + "-" * 34)
        for line in with_memory.text.splitlines()[:14]:
            out.append(f"  {_clip(line, 88)}")
        out.append("")
        out.append(f"  --- LAST MESSAGES ONLY, same {RECALL_BUDGET} token budget " + "-" * 20)
        out.append(f"  holds the last {len(without.turns)} turns of {len(turns)}, which is everything that fits")
        for line in without.text.splitlines()[1:4]:
            out.append(f"  {_clip(line, 88)}")
        out.append("  ...")
        for line in without.text.splitlines()[-2:]:
            out.append(f"  {_clip(line, 88)}")

        out.append("")
        target = None
        if planted_facts or merged_facts:
            candidates = planted_facts or merged_facts
            target = next((f for f in candidates if f.active), candidates[0])
            lead = f"  the fact you planted at turn {PLANT_AT + 1}"
        elif with_memory.facts:
            target = with_memory.facts[0]
            lead = "  memory's closest stored fact for this question"
        if target is not None:
            in_memory = any(f.id == target.id for f in with_memory.facts)
            in_window = target.object.lower() in without.text.lower()
            out.append(f"{lead}: {target.text}  (turn {target.turn_index})")
            out.append(f"  surfaced by the memory lookup for your question:  {'yes' if in_memory else 'no'}")
            out.append(f"  still present in the last-messages window:        {'yes' if in_window else 'no'}")
            if in_memory and not in_window:
                out.append("  that is the whole point: the sentence scrolled out of the window turns ago,")
                out.append("  nothing broke, and the assistant with only recent messages does not know.")
            elif in_window:
                out.append("  this one was said recently enough that both versions still have it. plant a fact")
                out.append("  and ask about it and only the memory version will still answer.")
            else:
                out.append("  the fact is stored but this question did not retrieve it. lookup is on shared")
                out.append("  wording, so a question using the words of the original sentence works best.")
        else:
            out.append("  no stored fact matched this question, so memory fell back to the older summaries")
            out.append("  and the recent messages. facts are looked up on shared wording, so a question")
            out.append("  phrased with the same words as the original sentence retrieves best.")
        return "\n".join(out)
    except Exception as exc:  # a demo page must never 500
        return f"This demo could not run: {type(exc).__name__}: {exc}"
