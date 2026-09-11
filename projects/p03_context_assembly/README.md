# 03. Context Assembly Service

**Run it live: [https://applied-llm-systems.vercel.app/s/context-assembly](https://applied-llm-systems.vercel.app/s/context-assembly)** - the hosted page runs this code and shows the real output.

A dynamic context builder that budgets a model window across memory, documents,
tool output and chat history on every request, and emits a receipt explaining
every token it spent.

## The problem

A feature starts with one source of context and ends with five. Memory, retrieval,
tool results, chat history and the system prompt are each formatted by different
code, joined into one string, and sent. Nothing in that path knows the total.

Three failures follow, in this order:

1. **The window overflows.** The prompt plus the completion exceeds the model
   window. The provider either rejects the call or truncates the answer
   mid-sentence, which downstream JSON parsing then reports as a model quality
   problem. The completion reserve is the part everyone forgets: a prompt that
   fills the window leaves the model no room to answer.
2. **The quick fix makes it worse.** Cutting the joined string down to the window
   keeps whatever ran first and deletes whatever ran last. In the request this
   demo uses, the thing at the end of the string is the user's own question.
3. **Nobody can debug it.** A user reports that the assistant "forgot" something.
   The retriever logs say the right document was retrieved. Nothing recorded that
   it was cut to two sentences to make room for a chat history nobody needed.

## What this builds

- `sections.py` - the declaration: named sections with a priority, a guaranteed
  minimum, a ceiling expressed as a share of the window, a compression strategy
  and a pinned position. `ContextRequest.available_tokens` is the window minus an
  explicit completion reserve.
- `dedupe.py` - cross-source deduplication by token-shingle containment, so a
  fact arriving from both retrieval and memory is paid for once.
- `budget.py` - the allocator: floors first, then water-filling by priority times
  diminishing marginal value, capped by `max_share`, with shedding when the
  floors alone do not fit.
- `compression.py` - keep whole, truncate head, truncate tail, sentence-level
  extractive compression, summarise via the model, drop. Plus the hourglass
  ordering used to mitigate lost-in-the-middle.
- `assembler.py` - the orchestrator, including the packing policy that decides
  whether a section keeps two items whole or six items compressed.
- `receipt.py` - the structured record: per item, per section, with reasons.
- `baseline.py` - naive concatenation and prefix truncation, measured exactly.
- `workload.py` - a support request with six competing sources, using real cosine
  scores from `llmkit`'s vector store over `llmkit`'s corpus.

## Architecture

```
  memory   retrieval   tools   history   instructions   question
     |         |         |        |           |            |
     +---------+----+----+--------+-----------+------------+
                    |
                    v
            [ 1. deduplicate ]  containment over 4-token shingles
                    |           superset copy wins, then priority
                    v
            [ 2. allocate ]     floors -> water-fill by priority x
                    |           marginal value -> cap -> shed
                    v
            [ 3. pack ]         per-item budgets, then compress:
                    |           whole | head | tail | extractive | summary | drop
                    v
            [ 4. arrange ]      pinned first, hourglass in the middle,
                    |           pinned last
                    v
            [ 5. verify ]       count the real string; if the structural
                    |           overhead pushed it over, shrink and repack
                    v
          prompt  +  receipt (JSON, attachable to a trace span)
```

## Design decisions

**The completion reserve is subtracted before anything else is decided.**
The alternative is budgeting against the full window and hoping the answer is
short. That works until a user asks for a table. Reserving completion tokens up
front turns a class of intermittent production failures into an allocation that
is simply smaller.

**Floors before priority, and shedding before floors.**
A single proportional split by priority is shorter and fails twice: a section
that needs 40 tokens still gets its proportional 900, and a small high-priority
section can be handed less than the minimum it declared it needs to be useful at
all. Floors express "below this, do not bother including me". Shedding handles
the case where the floors themselves do not fit, by dropping whole low-priority
sections rather than starving everything equally.

**Marginal value, not raw priority, drives the remainder.**
The tenth retrieved document is worth much less than the first, so a section at
90 percent funded loses the next token to a section at 10 percent even when its
raw priority is higher. Weight is `priority * (1 + value_density) * (1 - 0.5 *
fill)`. Without the decay term the highest-priority section absorbs the entire
remainder and every other source arrives as a stub.

**Deduplication is lexical containment, not embedding similarity.**
An embedding threshold loose enough to catch a restatement is also loose enough
to merge "tokens expire after 90 days" with "tokens expire after 30 days".
Silently dropping the correct half of a contradiction is far worse than paying
for a duplicate, so matching only fires when one text genuinely repeats almost
all of the other's wording. There is a test for the contradiction case.

**Compressible sections share their grant across items; truncatable ones do not.**
Six documents reduced to their query-relevant sentences beat two documents kept
whole and four thrown away, because the answer-bearing sentence is often in the
fourth document. The opposite is true for chat history: truncating eight short
turns to a third each produces eight fragments and no readable conversation, so
history keeps a greedy whole-then-stop policy.

**Verification counts the real string rather than trusting the arithmetic.**
Token counting is an estimate, and joining strings can round up. The assembler
measures the assembled prompt, and if the structural overhead pushed it over it
reduces the budget and repacks. The guarantee is checked, not asserted.

## Running it

```bash
python3 projects/p03_context_assembly/demo.py
python3 -m pytest projects/p03_context_assembly -q
```

The demo prints, in order: the full receipt for the tightest window, the prompt
that receipt describes, what naive concatenation would have sent, a comparison
table across three window sizes, totals, an explicit budget check, a starved-window
edge case, and the receipt as JSON.

Receipt output looks like this:

```
  section                prio  offered  granted   used   keep  in/cmp/drop   note
  -------------------------------------------------------------------------------
  instructions           10.0      105      105    105   100%  2/0/0         fully funded
  retrieved_docs          5.0      459      176    164    36%  0/6/0         partially funded, compression required
  chat_history            2.0      147       37     35    24%  2/0/6         partially funded, compression required
```

## Results

All figures below are printed by `demo.py` on the request in `workload.py`: one
support question, six sections, 13 high-priority items, run against three window
sizes. Token counts use `llmkit.count_tokens`, which is a calibrated estimator
rather than a real BPE tokenizer.

| Window | Reserve | Naive prompt | Naive overflow | Assembled | Tokens saved |
|---|---|---|---|---|---|
| 800 | 250 | 976 | +426 over budget | 508 | 468 |
| 1600 | 400 | 976 | fits | 952 | 24 |
| 4000 | 600 | 976 | fits | 952 | 24 |

- **Window overflows avoided: 1 of 3 requests.** Naive concatenation overran the
  800-token window's prompt budget by 426 tokens. The assembler produced 508
  tokens against a 550-token budget, 42 tokens of headroom. Across all three runs
  the assembler overflowed 0 times.
- **Tokens saved: 516 across the three runs** (2928 naive against 2412
  assembled), of which 468 came from the tight window alone.
- **On the tight window, savings split as 331 tokens from compression and 24
  tokens from deduplication**, the rest from dropping low-value history turns.
  The deduplicated item was a memory restatement of the rate-limit document at
  containment 1.0; the fuller retrieved copy survived.
- **High-priority source coverage: 100 percent assembled against 77 percent for
  prefix truncation** on the tight window, and 100 against 92.3 percent averaged
  over the three runs. Coverage counts a high-priority item as represented when at
  least 12 of its tokens survive.
- **High-priority token retention: 59 percent assembled against 66 percent for
  prefix truncation** on the tight window (86.3 against 88.8 percent averaged).
  This one goes the other way and is worth stating plainly: prefix truncation
  keeps the front of the prompt whole, which scores well on raw token share, and
  pays for it by deleting the items at the end of the string entirely, including
  the user's question. The assembler spends some of that budget spreading across
  every source instead. Which trade is right depends on the application, and the
  point of the receipt is that the trade is visible rather than accidental.
- **Starved window edge case:** a 140-token budget against 340 tokens of declared
  minimums. Three sections were shed by priority, the instruction block was
  reported as dropped rather than truncated into half an instruction, and the
  result was 51 tokens, inside budget.
- **Tests: 26**, covering the budget guarantee at four window sizes, floors,
  shedding, `max_share` caps, surplus redistribution, dedupe including the
  contradiction case, all five compression strategies, hourglass ordering, and
  full receipt accounting.

## Limits

- **Ordering is an empirical heuristic.** The hourglass arrangement follows the
  lost-in-the-middle finding (Liu et al., 2023) that models attend more reliably
  to evidence at the start and end of a long context. The effect size depends on
  the model, the window and the prompt format. This is a defensible default, not
  a guarantee, and it is worth re-measuring per model rather than trusting.
- **Token counts are estimates.** `llmkit.count_tokens` is calibrated against BPE
  ratios, not a real tokenizer. The structural reserve and the verify-and-repack
  loop exist because of that. With `LLMKIT_TOKENIZER=tiktoken` available the
  reserve can shrink.
- **The summarise strategy is only as good as the model behind it.** With
  `EchoLLM` it produces deterministic extractive output, which is enough to prove
  the plumbing, the token cap and the receipt entry. With a real model the same
  path costs a round trip per item on the latency path of every request, which is
  why extraction is the default and summarisation is opt-in per section.
- **Deduplication is lexical.** Two paraphrases with no shared wording are not
  caught. Catching those needs an encoder and a threshold, and that threshold
  brings the contradiction-merging risk described above.
- **Item value comes from the caller.** The allocator trusts retrieval scores as
  a marginal-value signal. If a retriever's scores are badly calibrated the
  allocator will faithfully spend the budget on the wrong documents.
