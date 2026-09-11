# 05. Semantic Cache Layer

**Run it live: [https://applied-llm-systems.vercel.app/s/semantic-cache](https://applied-llm-systems.vercel.app/s/semantic-cache)** - the hosted page runs this code and shows the real output.

An embedding cache that serves a stored answer for a similar-enough query, with a
salience check that refuses the hit when a single token flips the meaning.

## The problem

An exact-string cache in front of an LLM has a hit rate near zero, because two
users never type the same sentence. A semantic cache fixes that by matching on
embedding similarity, and in doing so it creates a failure mode that the exact
cache could not have.

"How long is the free trial" and "how long is the paid trial" are one word apart.
Every embedding model puts them close together, because they *are* close
together: same topic, same shape, same intent, one flipped attribute. Serve the
second from the first and the user gets a fluent, confident, wrong answer, with
nothing in the response indicating anything went wrong, and they keep getting it
for the whole TTL. Nobody logs a false hit, because from the system's point of
view it was a hit.

The instinct is to raise the threshold until the problem goes away. On the
labelled fixture in this project that does not work: the highest-scoring
near-miss scores 0.894 and the lowest-scoring genuine paraphrase scores 0.703, so
the two distributions overlap across a 0.191 cosine band. Any cut point inside
that band serves wrong answers and rejects right ones simultaneously. Raising the
threshold to 0.90 removes every false hit and also drops recall from 0.92 to
0.42, which is most of the reason the cache existed.

So the cache needs a second, different kind of check.

## What this builds

1. **`SemanticCache`** (`cache.py`): cosine lookup over cached query embeddings
   with a tunable threshold, TTL expiry enforced on read as well as by sweep,
   LRU eviction under a per-namespace entry cap, and per-namespace isolation.
2. **`SalienceGuard`** (`salience.py`): the pre-serve safety check. Compares
   numbers, mid-sentence capitalised entities, negation markers and contrastive
   word groups between the incoming query and the cached one, and refuses the hit
   on any mismatch.
3. **A labelled fixture** (`fixtures.py`): 12 paraphrase pairs that should hit and
   14 near-miss pairs that must not, each near-miss tagged with the class of
   difference it turns on.
4. **Measurement** (`metrics.py`): hit rate, false-hit rate, precision, recall,
   F1, the threshold sweep with the guard on and off, the itemised list of
   near-misses a cosine-only cache would serve, the itemised list of genuine
   paraphrases the guard costs, and a token-saving measurement over a workload.

## Architecture

```
  query + namespace ("tenant-a:prompt-v3")
        |
        v
  [ namespace lookup ] --- unknown namespace ---> miss
        |                                          (no cross-tenant read is
        v                                           even attempted)
  [ TTL expiry on read ] --- all expired ---------> miss
        |
        v
  [ cosine scan over that namespace's entries ]
        |
        +--- best score < threshold --------------> miss
        |
        v  best candidate
  [ salience guard ]
        numbers        90 vs 30
        entities       Growth vs Starter
        negation       allowed vs not allowed
        contrastive    free/paid, enable/disable, min/max, before/after,
                       add/remove, sandbox/production, region, plan, role
        |
        +--- mismatch ----------------------------> refuse, report which check
        |                                            fired and with what values
        v
  [ serve ] mark used, move to LRU tail, add tokens and latency to saved
```

## Design decisions

**A second check, not a better threshold.** The sweep is the argument. Without the
guard the best achievable F1 across eight thresholds is 0.79, at threshold 0.80,
still serving 5 false hits. With the guard, precision is 1.00 at every threshold
tested from 0.60 to 0.85 and the best F1 is 0.96. The guard does not make the
similarity function better; it makes the threshold stop being the only line of
defence, which is what lets the threshold be set for recall.

**Four classes of salient token, chosen by what changes the answer.** Numbers,
capitalised entities and negation are the obvious three. The fourth, contrastive
word groups, exists specifically because the canonical example of this whole
problem is invisible to the other three: "free trial" against "paid trial" has no
number, no capital and no negation. There is a test asserting exactly that.

**A curated list, not a model.** An NLI or cross-encoder check would generalise
past the list, and would cost a model call per lookup, which is the thing the
cache exists to avoid. The list is cheap, auditable, and extendable per
deployment through `extra_groups`. It will miss contrasts nobody wrote down, and
that limit is stated below rather than papered over.

**Refuse rather than fall through to the runner-up.** When the guard blocks the
closest entry, the cache returns a miss instead of trying the second-closest. The
runner-up is by definition less similar, so if the best match is semantically
incompatible the others are not better candidates, they are worse ones that
happen to lack a token the guard knows how to check. Falling through would trade
a caught error for a quieter one.

**The entry cap is per namespace, not global.** A single global LRU cap is
simpler and introduces a noisy-neighbour bug: a high-traffic tenant fills the
cache and evicts a quiet tenant's entries, so the quiet tenant's model spend goes
up because of someone else's traffic and nothing in their own metrics explains
it. Per-namespace caps make each tenant's cache behaviour a function of their own
load.

**Namespace is `{tenant}:{prompt_version}`, not just tenant.** Shipping a new
system prompt otherwise keeps serving answers generated under the old one, which
is silent and very hard to diagnose from the outside. Making the prompt version
part of the namespace turns a subtle correctness bug into an expected cold cache.

**TTL is enforced on read, not only by a background sweep.** A cache that expires
only on sweep serves stale answers for up to one sweep interval, which is the one
thing a TTL exists to prevent. `sweep()` exists as well, for reclaiming memory
from namespaces nobody is querying.

## Running it

```bash
python3 projects/p05_semantic_cache/demo.py
python3 -m pytest projects/p05_semantic_cache -q
```

The demo prints eight sections: the similarity distributions of the two labelled
sets, every near-miss a cosine-only cache would have served with the guard's
reason for refusing each, guard on versus off at the shipped threshold, the full
threshold sweep, tenant and prompt-version isolation, TTL and LRU behaviour, a
measured token saving over a 38-request workload, and the guard's own cost in
refused paraphrases.

## Results

Measured by `demo.py` with the default `HashingEmbedder` (dim 384) over the
12 paraphrase pairs and 14 near-miss pairs in `fixtures.py`. A "true hit" requires
the cache to return the *correct* base entry, not merely to return something.

**The two labelled sets are not separable by any threshold.**

| set | min | mean | max |
|---|---|---|---|
| paraphrases (should hit) | 0.703 | 0.867 | 0.922 |
| near misses (must not hit) | 0.591 | 0.776 | 0.894 |

The best near-miss outscores the worst paraphrase by 0.191 cosine.

**At the shipped threshold of 0.80:**

| | hit rate (recall) | false hits | false-hit rate | precision | F1 |
|---|---|---|---|---|---|
| guard off | 0.92 | 5 | 0.36 | 0.69 | 0.79 |
| guard on | 0.92 | 0 | 0.00 | 1.00 | 0.96 |

The guard prevented 5 of 5 false hits and cost 0 genuine paraphrases at this
threshold.

**Threshold sweep** (12 paraphrases, 14 near misses):

| threshold | guard | served | true | false | blocked | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0.60 | off | 25 | 12 | 13 | 0 | 0.48 | 1.00 | 0.65 |
| 0.60 | on | 11 | 11 | 0 | 13 | 1.00 | 0.92 | 0.96 |
| 0.65 | off | 24 | 12 | 12 | 0 | 0.50 | 1.00 | 0.67 |
| 0.65 | on | 11 | 11 | 0 | 12 | 1.00 | 0.92 | 0.96 |
| 0.70 | off | 24 | 12 | 12 | 0 | 0.50 | 1.00 | 0.67 |
| 0.70 | on | 11 | 11 | 0 | 12 | 1.00 | 0.92 | 0.96 |
| 0.75 | off | 22 | 11 | 11 | 0 | 0.50 | 0.92 | 0.65 |
| 0.75 | on | 11 | 11 | 0 | 11 | 1.00 | 0.92 | 0.96 |
| 0.80 | off | 16 | 11 | 5 | 0 | 0.69 | 0.92 | 0.79 |
| 0.80 | on | 11 | 11 | 0 | 5 | 1.00 | 0.92 | 0.96 |
| 0.85 | off | 12 | 9 | 3 | 0 | 0.75 | 0.75 | 0.75 |
| 0.85 | on | 9 | 9 | 0 | 3 | 1.00 | 0.75 | 0.86 |
| 0.90 | off | 5 | 5 | 0 | 0 | 1.00 | 0.42 | 0.59 |
| 0.90 | on | 5 | 5 | 0 | 0 | 1.00 | 0.42 | 0.59 |
| 0.95 | off | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| 0.95 | on | 0 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |

Best F1 without the guard is 0.79 at threshold 0.80, still with 5 false hits.
Best F1 with the guard is 0.96 at threshold 0.60, with 0 false hits. The
threshold that is safe without a guard (0.90) throws away 58 percent of the
genuine paraphrase traffic.

**The five false hits the guard prevented at 0.80,** each with the check that
fired:

| cached | incoming | cosine | refused by |
|---|---|---|---|
| rotate a token after 90 days | rotate a token after 30 days | 0.844 | numbers, 30 vs 90 |
| enable single sign-on | disable single sign-on | 0.855 | contrastive enablement |
| maximum page size | minimum page size | 0.881 | contrastive bound |
| uptime target on the Growth plan | uptime target on the Starter plan | 0.820 | entities, starter vs growth |
| which role is allowed to delete | which role is **not** allowed to delete | 0.894 | negation |

**What the guard costs.** At threshold 0.80 it refuses 0 genuine paraphrases. At
0.60 it refuses 1 of 12: "how do I rotate a token once it is 90 days old" against
the cached "how do I rotate a token after 90 days", because the cached form
contains the sequence word "after" and the paraphrase does not. That is the
`contrastive:sequence` group over-triggering on a function word, it is printed by
the demo rather than hidden, and it costs one model call.

**Measured savings** over a 38-request workload (12 base queries, their 12
paraphrases, and 14 near misses) at threshold 0.80 with the guard on:

- served from cache: 11 of 38 requests, 28.95 percent hit rate
- model calls made: 27
- tokens with the cache: 898
- tokens without the cache: 1280
- token saving: 29.84 percent
- tokens attributed to hits by the cache's own counter: 360

The hit rate is low by design of the workload: 14 of the 38 requests are
adversarial near-misses that *should* miss, so they count against the hit rate
while representing correct behaviour. Over the paraphrase traffic alone the
recall is 0.92.

**Wall-clock latency is not a meaningful figure here and the demo says so.**
Against `EchoLLM`, a local rule engine, the cache lookups for the whole workload
cost more wall-clock time than every model call they replaced. The cache is
slower than the thing it is caching. Exact millisecond figures are printed by the
demo and are not quoted here because they vary run to run on an idle machine by
more than the quantity being measured. That result is a true statement about an
offline reference model and tells you nothing about a hosted one; the
transferable number is the token saving, which is a direct proxy for spend.

## Limits

- **The contrastive list is hand-written and finite.** It covers free/paid,
  enable/disable, add/remove, delete/restore, min/max, before/after, cadence,
  sandbox/production, read/write, plan tier, role and region. A deployment with a
  different vocabulary needs its own groups passed through `extra_groups`. The
  guard cannot catch a contrast nobody wrote down.
- **Entity detection is capitalisation, not NER.** An entity that begins a
  sentence is skipped, because sentence-initial capitals carry no signal. There is
  a test asserting that behaviour and its cost.
- **Units are not parsed.** "90 days" against "90 hours" passes the number check.
  It is usually caught by similarity instead, but not always.
- **The embedder is lexical.** `HashingEmbedder` is a hashing bag of word and
  character n-grams. The paraphrase fixture is deliberately built from realistic
  surface variation rather than distant rewrites, because a pair sharing no
  vocabulary would be measuring the embedder rather than the cache. With
  `EMBED_PROVIDER=ollama` the paraphrase similarities should rise and, importantly,
  so should the near-miss similarities, which makes the guard more load-bearing
  rather than less.
- **Lookup is a linear scan.** Fine to roughly ten thousand entries per namespace,
  after which it wants an ANN index. The `lookup` signature does not change when
  it gets one.
- **Nothing here validates that the cached answer was correct when it was
  stored.** The cache guarantees that the question it is answering is the same
  question. Whether the stored answer was right is the retrieval and evaluation
  problem, which is projects 01 and 04.
- **Token counts are estimates** from `llmkit.count_tokens`, calibrated at 1.3
  tokens per word. Set `LLMKIT_TOKENIZER=tiktoken` where a native wheel is
  allowed.
