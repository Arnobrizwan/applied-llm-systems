# 15. Self-Correcting RAG Agent

**Run it live: [https://applied-llm-systems.vercel.app/s/self-correcting-rag](https://applied-llm-systems.vercel.app/s/self-correcting-rag)** - the hosted page runs this code and shows the real output.

A bounded retrieve, critique, decide, act loop that rewrites its own query, widens
its search, escalates to a tool outside its index, and abstains when none of that
worked.

## The problem

Project 01 answers in one shot. Retrieve, check the evidence, answer or refuse.
That is the right default and it has one structural weakness: it gets exactly one
attempt at retrieval, so a phrasing mismatch is a permanent failure. The user asks
"how do I stop a retried payment request from charging twice", the corpus says
"Idempotency-Key", nothing in the query resembles that word, and the system either
answers from the wrong document or declines. Either way the answer was sitting in
the corpus the whole time.

Worse, single-shot RAG has no way to know which of those two things just happened.
Retrieval returned five chunks, the chunks looked plausible, an answer was
produced. Nothing in the pipeline asked whether that evidence was actually capable
of answering the question that was asked.

And when the answer genuinely is not in the corpus, single-shot RAG answers
anyway. On the eight adversarial questions in this project, single-shot answered
four of them.

## What this builds

1. **A bounded control loop** (`agent.py`): retrieve, critique, decide, act, with
   `max_steps` enforced by the loop and a four-rung escalation ladder.
2. **Query rewriting** (`rewrite.py`): decomposition of compound questions,
   keyword-only reduction, and a hypothetical-answer reformulation, with the
   results of all reformulations fused by reciprocal rank.
3. **Retrieval critique** (`critique.py`): idf-weighted question-term coverage,
   inter-chunk agreement, and an LLM judge verdict, combined into a confidence.
4. **An escalation tool interface** (`tools.py`): `SearchTool` with one method,
   an offline `FallbackSearch` over a second corpus, and a `NullSearch` used to
   prove abstention is not an accident of a weak tool.
5. **A second corpus** (`fallback_corpus.py`): status-page and release-note
   material the primary index does not contain, with four questions answerable
   only from it.
6. **Adversarial questions** (`adversarial.py`): eight questions across four
   failure kinds, all of which should produce an abstention.
7. **Four-outcome evaluation** (`evaluate.py`) against single-shot RAG on the
   same questions, plus the two threshold sweeps that set the shipped defaults.

## Architecture

```
  question
     |
     v
  +--> [ rung ] ------------------------------------+
  |      1 retrieve   hybrid + rerank, k=5           |
  |      2 rewrite    3 reformulations, RRF fused    |
  |      3 widen      same query, k=10               |
  |      4 fallback   SearchTool outside the index   |
  |        |                                         |
  |        v                                         |
  |    [ critique ]                                  |
  |      coverage   idf-weighted question terms      |
  |      agreement  pairwise overlap of chunks       |
  |      judge      LLM pass/fail verdict            |
  |        |                                         |
  |        v                                         |
  |    confidence = 0.55 cov + 0.15 agr + 0.30 judge |
  |        |                                         |
  |        +-- >= accept_above (0.70) --> stop       |
  |        |                                         |
  |        +-- keep if better than best so far ------+
  |                                            (max_steps)
  v
  best evidence seen across every rung
     |
     +-- confidence < abstain_below (0.50) --> "I cannot answer that"
     |
     v
  [ p01 CitedAnswerer ] -> validated citations, including fallback chunks
```

Every rung, every critique and the final answer run inside a `llmkit.tracer`
span, all sharing one trace id, so a single question is reconstructable from a
log.

## Design decisions

**`max_steps` is enforced by the loop, not by the model's judgement.** An agent
that decides its own stopping condition from its own confidence will, on the
questions where its confidence estimate is broken, keep going until a timeout, a
rate limit or a bill stops it. The ladder is finite, each rung runs at most once,
and there is a test asserting that `max_steps=n` produces exactly the first `n`
rungs.

**Escalation cannot make the answer worse.** Each rung's evidence is kept only if
it critiques better than the best seen so far. Without that, a rewrite that
retrieves confidently wrong material overwrites a mediocre but correct first pass,
and the loop actively harms the questions it was supposed to leave alone. The
final confidence is therefore the maximum across all rungs, which a test asserts
directly.

**The ladder is ordered by cost, and widening comes second-to-last.** Rewriting is
cheap and fixes the most common failure, which is phrasing. Widening `k` mostly
returns more of the same bad results for the same bad query while consuming
context budget, so it sits below rewriting. The external tool is last because it
is the only rung with an outside dependency and, in a real deployment, a
per-call price.

**"Corrected" is keyed on which rung won, not on how many rungs ran.** The first
version counted any question that escalated as corrected, which inflated the
headline claim with questions where the loop escalated, found nothing better, and
fell back on its own first result. Keying on the winning rung means the corrected
column contains only questions the loop actually repaired.

**The rewrite rung deliberately excludes the original ranking from its fusion.**
This was found by watching the rung do nothing: fusing three reformulations
together with the original top-5 returned the original top-5 on every question in
the evaluation set, because four mostly-agreeing lists fuse to their consensus.
Each reformulation now retrieves `2 * k` and the original is left out. Nothing is
lost by leaving it out, because the best-so-far tracking already protects the
first result. After the fix, the winning rung is step 1 for 9 questions, step 2
for 5, step 3 for 1 and step 4 for 4.

**A failing judge verdict contributes zero, not its own score.** A judge that
replies "fail, confidence 0.9" is 0.9 confident of failure. Reading that 0.9 as
quality is a sign error that makes the confidence highest exactly where the judge
is most certain the evidence is bad. An unparseable judge is treated as an
abstention worth 0.5, so a broken provider can neither veto well-covered evidence
nor rubber-stamp bad evidence.

**A fourth confidence signal was built, measured, and deleted.** A specificity
penalty (multiply confidence down when the highest-idf question term appears in no
retrieved chunk) reads well and was swept at 0.0, 0.2, 0.35, 0.5 and 0.6 with the
abstention threshold recalibrated for each. Every setting produced identical
outcomes on all 27 evaluation questions. It rescaled the confidence axis and moved
nothing across a decision boundary, so it was removed rather than kept as
decoration.

**Both thresholds come from a sweep, not from a round number in a constructor.**
The abstention floor is the most consequential number in the system and it depends
on the corpus, the retriever and the relative cost of a wrong answer against a
refusal. It is swept against both labelled question sets and the demo prints the
sweep that chose it. `accept_above` is swept separately because it is a different
kind of dial: it does not change what the agent can find, only how early it stops
looking, so its sweep is an accuracy-against-cost table.

**HyDE is implemented as pseudo-relevance feedback, and that is stated plainly.**
True HyDE generates a hypothetical answer from the model's own knowledge with no
retrieval involved. `EchoLLM` is a deterministic rule engine with no knowledge to
draw on, so asking it to invent documentation text returns the prompt. The
implementation instead grounds the hypothetical document in the first-pass top
hit. Dropping the evidence from `_hyde_messages` turns it back into HyDE proper
with a real model; nothing else changes. Its measured contribution here is
covered in Results, honestly.

## Running it

```bash
python3 projects/p15_self_correcting_rag/demo.py
python3 -m pytest projects/p15_self_correcting_rag -q
```

The demo prints eleven sections: a question settled by the first retrieval, a
question repaired by the ladder, a question only the fallback tool can answer,
an abstention, an abstention with the tool disabled, `max_steps` enforcement, both
threshold sweeps, the four-outcome comparison against single-shot RAG on both
question sets, where the ladder settled each question, the remaining failures by
name, and a trace summary.

```python
from projects.p01_rag_pipeline.pipeline import RagPipeline, preset
from projects.p15_self_correcting_rag.agent import SelfCorrectingRAG

agent = SelfCorrectingRAG(RagPipeline.build(config=preset("hybrid+rerank")))
result = agent.answer("What caused the eu-west outage in March 2026?")
print(agent.explain(result))
```

## Results

Measured by `demo.py` over 19 answerable questions (15 from `llmkit.corpus`, 4
answerable only from the fallback corpus) and 8 adversarial questions, using the
default offline stack. Correctness is judged on citations: an answer is correct
when the gold document appears among the documents it cited. Grading the prose
would be grading `EchoLLM`, which is a rule engine, and the number would say
nothing about a real deployment.

**Answerable questions:**

| system | n | correct | corrected | abstained | wrong |
|---|---|---|---|---|---|
| self-correcting | 19 | 9 (0.47) | 7 (0.37) | 1 (0.05) | 2 (0.11) |
| single-shot (p01) | 19 | 13 (0.68) | 0 | 3 (0.16) | 3 (0.16) |

16 of 19 right for the agent against 13 of 19 for single-shot. The three
questions the agent gets right that single-shot does not are all questions whose
answer lives outside the primary corpus, reached through the fallback rung:
"what caused the eu-west outage in March 2026", "how many CIDR ranges can an
allowlist hold", and "are the mobile SDKs covered by the SLA". No question is
lost in the other direction.

**Adversarial and unanswerable questions:**

| system | n | abstained | wrong |
|---|---|---|---|
| self-correcting | 8 | 5 (0.62) | 3 (0.38) |
| single-shot (p01) | 8 | 4 (0.50) | 4 (0.50) |

**Where the ladder settled each answerable question** (winning rung, not rungs
run): step 1 for 9 questions, step 2 for 5, step 3 for 1, step 4 for 4.

**Abstention floor sweep** (score = correct answers plus justified refusals, out
of 27 decisions):

| floor | correct | abstained | wrong | adversarial refused | adversarial answered | score |
|---|---|---|---|---|---|---|
| 0.30 | 17 | 0 | 2 | 1 | 7 | 18 |
| 0.35 | 17 | 0 | 2 | 2 | 6 | 19 |
| 0.40 | 17 | 0 | 2 | 3 | 5 | 20 |
| 0.45 | 16 | 1 | 2 | 4 | 4 | 20 |
| **0.50** | 16 | 1 | 2 | 5 | 3 | **21** |
| 0.55 | 12 | 5 | 2 | 5 | 3 | 17 |
| 0.60 | 11 | 6 | 2 | 5 | 3 | 16 |
| 0.65 | 10 | 8 | 1 | 6 | 2 | 16 |

0.50 is shipped. The cliff between 0.50 and 0.55 is the interesting part: four
answerable questions sit in that 0.05 band, so this threshold is genuinely
sensitive on a corpus this size and would want re-calibrating on a larger one.

**Early-stopping sweep** (`accept_above`, on the 19 answerable questions):

| accept_above | correct | corrected | wrong | abstained | total right | rungs run | judge calls |
|---|---|---|---|---|---|---|---|
| 0.50 | 10 | 4 | 4 | 1 | 14 | 31 | 31 |
| 0.60 | 9 | 6 | 3 | 1 | 15 | 44 | 44 |
| **0.70** | 9 | 7 | 2 | 1 | **16** | 53 | 53 |
| 0.80 | 6 | 10 | 2 | 1 | 16 | 60 | 60 |

0.70 is shipped: it reaches the maximum accuracy on this set at 53 rungs, where
0.80 spends 60 rungs for the same 16. Dropping to 0.50 saves 42 percent of the
work and costs 2 correct answers, which is the trade a cost-constrained
deployment would evaluate.

**The three adversarial questions that still get answered,** named rather than
averaged away:

- "How many compute-seconds does a single webhook delivery consume?" (adjacent
  but absent). Retrieval returns the billing and webhook documents, both genuinely
  on topic, and coverage is high because every content word is present. The fact
  itself is not in either.
- "What is the rate limit on the Starter plan in requests per second?" (adjacent
  but absent). The rate-limits document covers the default limit in requests per
  minute. Every term matches; the specific combination does not exist.
- "Given that offset pagination is the default, how do I switch to cursors?"
  (false premise). The pagination document is genuinely the right document, and it
  says offset pagination is not supported at all. Answering from it is defensible
  and is still scored as wrong here, because the premise should have been
  challenged rather than the question answered.

All three need semantic understanding of what the evidence does and does not
claim. They are precisely the cases a real judge model closes and a lexical
heuristic does not, and the `use_judge` path already exists to be pointed at one.

**Cost:** 182 judge calls across the whole demo, one per rung per question. The
critique is the dominant cost of this design, and with a hosted model it would be
the dominant bill.

**The hypothetical-answer reformulation earns nothing measurable here.** Running
the evaluation with `max_variants=2` (decomposition and keywords only) and with
`max_variants=3` produces an identical confusion matrix and identical winning-rung
histogram. With an offline rule engine that cannot invent domain text, the third
reformulation is grounded in the same top hit the first retrieval already found,
so it contributes little new signal. It is kept because the strategy interface is
where a real model's HyDE would plug in, and because it costs one call inside a
rung that is already running, but it is not carrying weight on these numbers.

## Limits

- **The judge is `EchoLLM`.** It produces well-shaped, deterministic pass/fail
  verdicts, which is enough to build and test the plumbing, and it has no semantic
  understanding. Every remaining adversarial failure above is a judgement failure.
  Point `LLM_PROVIDER` at Ollama or an OpenAI-compatible endpoint and the judge
  becomes real with no code change.
- **Thresholds are calibrated on 27 questions.** That is enough to demonstrate the
  method and far too few to trust the specific values. The 0.05-wide cliff between
  0.50 and 0.55 makes that concrete.
- **The critic's idf table comes from the primary corpus** and is applied to
  fallback evidence too, so terms common in the second corpus and rare in the
  first are over-weighted at the fallback rung. Correct behaviour would maintain a
  frequency table per source.
- **The fallback tool is offline.** `FallbackSearch` is a real hybrid retriever
  over a real second corpus, not a stub, but it is not the internet. A real
  deployment implements `SearchTool.search` against a search API; nothing above
  that method changes.
- **False premises are not detected, only sometimes survived.** Nothing here
  checks whether a question's presupposition contradicts the corpus. That is a
  separate mechanism.
- **The loop costs one critique per rung.** At `accept_above=0.70` the median
  answerable question runs close to the full ladder. This is a real cost and the
  sweep above is there so it can be traded deliberately rather than discovered on
  an invoice.
