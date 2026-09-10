# Applied LLM Systems

Fifteen production-shaped systems an AI engineer is actually asked to build:
retrieval, structured output, context budgeting, evaluation, caching, routing,
multi-tenancy, adaptation, memory, guardrails, streaming, prompt versioning,
observability, tool calling, and an agent that repairs its own retrieval.

**Every one of them runs offline, on a stock Python install, with no API key and
no paid service.** That is a hard constraint, and CI enforces it.

```bash
git clone https://github.com/Arnobrizwan/applied-llm-systems
cd applied-llm-systems
python3 -m pip install -e ".[dev]"   # pytest, and nothing else

python3 -m pytest                    # 494 tests
python3 scripts/run_all_demos.py     # all 15 demos, end to end, ~18s
```

No `pip install openai`. No `.env`. No network call. If that stops being true,
`scripts/check_no_runtime_deps.py` fails the build.

---

## The fifteen

| # | System | What it proves | Headline measurement |
|---|---|---|---|
| 01 | [Production RAG Pipeline](projects/p01_rag_pipeline) | Ingest, chunk, hybrid search, rerank, cite | Reranking lifted recall@1 from **0.67 to 0.93** and MRR from 0.756 to **0.967**; every citation emitted resolved to a real chunk |
| 02 | [Structured Output Engine](projects/p02_structured_output) | Schema enforcement, repair, retry, fallback | With **100% of responses deliberately corrupted**, success went from 50% at one attempt to **88% at three**; local repair fixed half of it without a second model call |
| 03 | [Context Assembly Service](projects/p03_context_assembly) | Token budgeting across memory, docs, tools | Naive concatenation overran an 800-token window by **426 tokens**; the assembler landed 508 against a 550 budget with **0 overflows** and kept **100% of high-priority sources vs 77%** |
| 04 | [LLM Evaluation Harness](projects/p04_eval_harness) | Golden set, judges, bootstrap CI, CI gate | Gate **passes a -0.087 drop and fails a -0.522 one**; 16 of 23 cases settled by free deterministic checks; pairwise judging showed **78.3% position bias**, so those verdicts are recorded as ties |
| 05 | [Semantic Cache Layer](projects/p05_semantic_cache) | Similarity cache that does not answer the wrong question | The two sets are **not separable by any threshold**; a salience guard removed **all 5 false hits** (precision 0.69 to **1.00**) while losing no real hits. 29.8% tokens saved |
| 06 | [Model Routing Gateway](projects/p06_model_router) | Complexity routing, budgets, fallback | **55.7% cheaper** than always-large across 44 requests, and **44 of 44 served during a total large-tier outage** |
| 07 | [Multi-Tenant LLM API](projects/p07_multi_tenant_api) | Keys, rate limits, budgets, isolation | A cross-tenant read returns a **byte-identical 404** to a fictional id, so the API is not an existence oracle. 429 and **402** are distinct outcomes; 21 of 21 log lines carry a request id |
| 08 | [Fine-Tuning Pipeline](projects/p08_finetuning_pipeline) | LoRA against a prompt-only baseline | LoRA trains **1,048 parameters vs 1,542** for a full fine-tune and matches it at 1.000 accuracy; merging the adapter moved the largest logit by **1.8e-15** |
| 09 | [Agent Memory System](projects/p09_agent_memory) | Working, episodic and semantic memory | A fact stated at **turn 2 is still recalled at turn 32** inside a 300-token budget; the same-budget no-memory baseline fails the identical question |
| 10 | [Guardrails Middleware](projects/p10_guardrails) | Injection detection, PII redaction, policy | **140 PII decisions, 0 false positives and 0 false negatives**, including four deliberate lookalikes; **0.08-0.13 ms** of overhead per request |
| 11 | [Streaming Infrastructure](projects/p11_streaming) | SSE, backpressure, resume, cancellation | TTFT p50 **2.9 ms**; a connection killed mid-stream resumes to **277 contiguous event ids with zero duplicates and no regeneration** |
| 12 | [Prompt Versioning and A/B](projects/p12_prompt_registry) | Immutable versions, sticky splits, a promotion gate | Assignment was sticky **600 of 600** times across a registry rebuild; the gate **refused to promote at n=40 despite p=0.031** and promoted at n=600 (p=0.00054) |
| 13 | [LLM Observability Stack](projects/p13_observability) | Traces, cost, anomaly alerting | Cooldown turned **7 alert events into 3 pages**; **0 of 196 spans carried raw prompt text**; 36% of calls were 98% of the bill |
| 14 | [Tool-Calling Framework](projects/p14_tool_calling) | Typed schemas, discovery, sandboxing | **12 of 12 sandbox escape attempts rejected** (dunder walks, `__import__`, huge exponents) while real arithmetic still works; a 0.2s timeout returned control in **205 ms** |
| 15 | [Self-Correcting RAG Agent](projects/p15_self_correcting_rag) | Rewrite, critique, escalate, abstain | **16 of 19 answerable questions right vs 13 single-shot**, three gained and none lost, and it abstains on 5 of 8 unanswerable ones instead of inventing an answer |

Each project has its own README with the architecture, the design decisions and
the alternatives rejected, the exact commands to run it, and the measured
results with the method stated.

## How it is put together

```
llmkit/          shared core: providers, embeddings, BM25, vector store,
                 chunking, token accounting, tracing, retry, eval corpus
projects/pNN_*/  one system each: implementation, README, demo.py, tests/
scripts/         demo runner and the no-dependency gate that runs in CI
docs/            the build spec every project is written against, and the
                 provider matrix
```

`llmkit` exists so the fifteen read as one codebase. Retrieval, cost accounting,
tracing and retry are written once and reused, which is also why a project can
depend on another one: project 15 uses project 01's retriever, and project 06
uses project 04's judge rather than reimplementing it.

## Why no paid API

Two reasons, and only one of them is money.

The first is that a portfolio a reviewer cannot run is a screenshot. Clone,
install pytest, run the demos: no signup, no key, no quota, nothing to expire.

The second is that most of what makes an LLM system production-grade is not the
model. Chunking, fusion, schema repair, token budgets, isolation, backoff,
cooldowns, sticky bucketing, abstention: none of it needs a frontier model to be
correct, and all of it is where the outages come from.

So the default provider is `llmkit.EchoLLM`, a deterministic offline reference
model. It is not a language model and the READMEs never pretend it is. It
returns output of the right *shape* (grounded answers with citations,
schema-valid JSON, judge verdicts), it is reproducible from a hash of the
prompt, and it can be told to corrupt a fixed fraction of its responses in the
ways real models fail. That last property is what lets project 02 measure a
repair path instead of asserting that one exists.

Where a result depends on real model quality rather than on the surrounding
system, the project README says so and reports the system-level metric instead.

**Free upgrades, no code change:**

```bash
export LLM_PROVIDER=ollama EMBED_PROVIDER=ollama      # local, free
# or any OpenAI-compatible endpoint, including a local llama.cpp server
export LLM_PROVIDER=openai OPENAI_COMPAT_BASE_URL=http://localhost:8080/v1
```

Full matrix, including what the hashing embedder can and cannot do: [docs/PROVIDERS.md](docs/PROVIDERS.md).

## What is in here

- 15 systems, ~20,100 lines of implementation, ~4,800 lines of tests
- **494 tests**, all passing, no network and no fixtures recorded from a paid API
- 15 demos runnable end to end in about 18 seconds
- CI on Python 3.9, 3.11 and 3.13, running the tests, the demos and the
  dependency gate on every push

## Honest limits

- `HashingEmbedder` is lexical, not semantic. It gives partial credit for shared
  sub-words but will not match a paraphrase with no shared vocabulary. Hybrid
  retrieval in project 01 exists partly because of that. Point `EMBED_PROVIDER`
  at Ollama for real semantics.
- The corpus is 20 fictional documents. Numbers on it are internally valid and
  say nothing about your corpus. The evaluation harness is the part designed to
  travel; the scores are not.
- The reranker is a feature scorer standing in for a cross-encoder, behind a
  swappable interface.
- Project 08 is LoRA arithmetic on a linear classifier, not a GPU fine-tune of a
  transformer. The mechanism is identical (frozen base, low-rank adapter,
  alpha/r scaling, merge) and its README gives the exact `peft` config to run
  the real thing.
- The tool sandbox uses a thread timeout, which cannot kill a wedged C call. A
  real sandbox is a subprocess or a container, and its README says so.

## Licence

MIT. Built by [Arnob Rizwan Ahmad](https://arnobrizwan.github.io).
