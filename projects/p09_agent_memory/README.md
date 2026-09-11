# 09. Agent Memory System

**Run it live: [https://applied-llm-systems.vercel.app/s/agent-memory](https://applied-llm-systems.vercel.app/s/agent-memory)** - the hosted page runs this code and shows the real output.

Working, episodic and semantic memory for a conversational agent, with
compression on overflow, decay-based eviction under a hard token budget, and
contradiction handling that keeps the superseded version.

## The problem

The default memory strategy for a chat feature is a sliding window: keep the last
N turns that fit and forget the rest. It works until a session gets long, and
then it fails in two ways that users notice immediately.

The first is the one this demo measures. A user states something at turn 2, the
conversation moves on, and at turn 32 the agent has no idea. Nothing is broken in
any log; the fact simply fell out of the window because the only thing deciding
what survives is how recently it was said.

The second is the opposite failure: a system that never forgets. Memory grows
with session length, the prompt grows with it, and eventually the context
assembly step starts dropping retrieved documents to make room for ambient chat
from an hour ago. A memory store with no ceiling is a memory leak with a nicer
name.

Then there is the case both approaches get wrong. A user corrects themselves at
turn 18. A sliding window keeps whichever version is more recent and loses the
history; a naive fact store overwrites and cannot answer "why did you change your
mind".

## What this builds

- `working.py` - the recent turn buffer under a token cap, draining to a
  low-water mark so overflow produces a run of turns rather than a dribble.
- `episodic.py` - an append-only episode log, retrieved by a blend of relevance
  (llmkit hashing embeddings, cosine) and recency (exponential decay over turns),
  plus the two summarisers used to compress overflowing turns.
- `semantic.py` - subject/predicate/object facts with deduplication,
  supersession, a bounded provenance chain and lexical search.
- `eviction.py` - the decay score over recency, access frequency and importance,
  the fact-over-episode weighting, and the pin rule.
- `manager.py` - the write path and the read path, plus `RecentWindowBaseline`,
  the no-memory comparison.
- `conversation.py` - the 32-turn conversation the demo and the tests both use.

## Architecture

```
  turn
    |
    v
  [ working memory ]  cap 220 tokens, drain to 60 percent on overflow
    |         |
    |         +--overflow--> [ summarise ] --> [ episodic log ] (append only)
    |                                               |
    +--if user turn--> [ extract facts ]            |
                            |                       |
                    dedupe / supersede              |
                            |                       |
                            v                       v
                     [ semantic facts ] <---> [ eviction: decay score ]
                            |                   hard budget, pins exempt
                            |                       |
                            +-----------+-----------+
                                        |
                     recall(query) -> facts, then episodes, then recent turns
```

## Design decisions

**Facts, episodes and turns are three stores, not one.**
A single similarity-searched blob is simpler and loses the distinctions that
matter. The recent buffer needs to be verbatim, episodes need to be append-only
so "what did the agent believe at turn 12" is answerable, and facts need a
(subject, predicate) key so a contradiction is visible as a conflict rather than
as two unrelated strings.

**A superseded fact is kept with a pointer, not deleted.**
Overwriting is one line shorter and destroys the answer to the question users
actually ask when an agent changes its mind. The chain is bounded at a fixed
depth per claim rather than by tokens, because provenance that nobody can read is
not provenance.

**Provenance sits outside the prompt-token budget.**
Superseded facts can never be returned by search, so they can never reach a
prompt, so charging them against a budget that exists to protect the prompt makes
no sense. They are bounded separately by chain depth. Getting this wrong is what
the first version of this project did: the budget evicted the turn-2 value of a
corrected fact and the provenance trail vanished at exactly the moment it became
interesting.

**Facts outrank episodes at equal decay.**
A fact is roughly fifteen tokens carrying one deduplicated claim. An episode is
roughly seventy tokens of compressed conversation that may contain no claim at
all. Evicting the fact to keep the episode spends more budget to retain less
information, so eviction applies an explicit `KIND_WEIGHT` multiplier instead of
leaving the trade to whichever record was touched most recently. Without it, the
demo's headline recall fails: the eu-west fact from turn 2 gets evicted at turn
31 to keep an episode about pagination.

**Recency decays from last use, not from creation.**
A fact from turn 2 that was looked up at turn 29 is recent. A fact from turn 28
that nobody has touched is not. Frequency is log-scaled so a record accessed
twenty times does not become permanently unevictable.

**Overflow drains a batch, not a turn.**
Evicting one turn per overflowing turn produces single-turn "episodes" that
compress nothing, because the summary of one sentence is that sentence. Draining
to a low-water mark gives the summariser a run of related turns and means
compression runs every few turns rather than on every turn past the cap.

**The default summariser is rules, not a model call.**
Compression happens on the write path of a conversation, so a model round trip
adds latency to every overflowing turn, and a paraphrasing model can quietly drop
the one identifier the user asks about later. The rule-based extractor is lossy
in a predictable direction: it keeps sentences with numbers, names and
identifiers and drops acknowledgements. The model path exists, is hard-capped,
and is opt-in.

**Pinned facts are exempt from scoring entirely.**
Not given a high score, exempt. A pin is a caller saying "this is a correctness
requirement", and a policy that can outvote a pin is a policy nobody will pin
against.

## Running it

```bash
python3 projects/p09_agent_memory/demo.py
python3 -m pytest projects/p09_agent_memory -q
```

The demo prints: the setup, a turn-by-turn timeline of every memory event, the
contradiction and its provenance chain, the headline recall with and without
memory, the budget, the compression numbers, the pinned-fact check and a
side-by-side of the two summarisers.

## Results

All figures are printed by `demo.py` running the 32-turn conversation in
`conversation.py`. Working memory cap 220 tokens, long-lived budget 300 tokens,
recall budget 300 tokens. Token counts use `llmkit.count_tokens`, an estimator
rather than a real BPE tokenizer.

**The headline recall.** Turn 2 states that the workspace is pinned to eu-west.
The fact is never repeated. At turn 32 the user asks which region the workspace
is pinned to.

| | Recall contains "eu-west" | Recall size |
|---|---|---|
| With memory | yes, PASS | 299 tokens |
| No-memory baseline | no, FAIL | 290 tokens |

Both are given the same 300-token recall budget. The baseline is not empty and is
not misconfigured: it holds the last 12 turns of 32, which is everything that
fits. Turn 2 is simply not among them.

**Compression.** Six episodes were created over the run, compressing 111, 131,
106, 101, 111 and 104 tokens of turns into 70, 70, 69, 68, 70 and 58 tokens
respectively, between 53 and 67 percent kept. The three episodes still resident
at the end hold 196 tokens compressed from 316 tokens of turns, 62 percent kept.

**Budget.** Long-lived memory finished at 265 tokens against the 300-token
budget, after 3 evictions, all of them episodes. Working memory finished at 149
tokens against its 220-token cap. The full transcript is 593 tokens and grows
every turn; the memory footprint is 414 tokens, 70 percent of the transcript at
turn 32 and capped at 300 tokens of long-lived state for the rest of the session
however long it runs. 15 tokens of superseded-fact provenance sit outside that
budget by design.

**Contradiction.** Turn 2 said Postgres 14, turn 18 said Postgres 16. Both
versions are in the store at the end of the run: the turn-2 version marked
superseded by the turn-18 version, the turn-18 version active. Search returns
only the active one, and a query about the database version answers with 16 and
not 14.

**Pinning.** The pinned operator fact ("customer data must stay in the workspace
region") is set at turn 0 and never mentioned in the conversation. Scored as an
ordinary unpinned record with no accesses, it would have scored 0.290 at turn 31,
below the lowest score actually evicted that run, 0.301. It is still present.

**Tests: 22**, including the turn-2-to-turn-32 recall, the baseline's failure of
the same recall, the budget invariant, batch overflow, both summarisers,
compound-sentence extraction, deduplication, supersession and its provenance,
chain bounding, pin protection, the relevance-versus-recency blend in both
directions, episode id reuse, the fact-over-episode eviction trade, and recall
budget compliance at four sizes.

## Limits

- **Fact extraction is a set of regular expressions.** It handles "our X is Y",
  "we use Y", "I work at Y" and a few more. It will miss most of what a real
  conversation contains, and it has no notion of negation or hypotheticals: "we
  are not moving to Enterprise" is not understood as a negation. In production
  this is an LLM extraction call, which changes one method in `semantic.py` and
  nothing else. The triple shape, the keying, the supersession and the provenance
  are the parts worth keeping.
- **Relevance is lexical.** `llmkit`'s default embedder is a hashing embedder
  over character n-grams and word n-grams, so it matches shared wording rather
  than meaning. A paraphrase with no shared vocabulary will not retrieve. Setting
  `EMBED_PROVIDER=ollama` swaps in a real encoder without touching this project.
- **`EchoLLM` is a rule engine, not a summariser.** The opt-in model path is
  exercised and hard-capped, and the demo prints its output honestly, but its
  content is deterministic reference output and should not be read as a quality
  comparison against the rule-based summariser.
- **Eviction weights are chosen, not learned.** The 0.45/0.25/0.30 split and the
  1.4 fact multiplier come from reasoning about what costs what, not from an
  experiment. They are a starting point that a real deployment should tune
  against recall failures it can actually observe.
- **No persistence.** Everything is in process. The stores are plain lists behind
  small interfaces, so a database is a swap of those two classes, but that swap
  brings concurrency questions this project does not address.
- **One conversation.** There is no notion of a user id, a tenant or a shared
  memory across sessions, and no access control on facts. Multi-tenant memory
  needs isolation guarantees that are not modelled here.
