# Providers: how this repo stays free

Every demo, test and benchmark in this repository runs with **no API key, no
account and no network**. That is a design constraint, not a limitation I worked
around, and it is enforced in CI by `scripts/check_no_runtime_deps.py`.

## The default: `echo`

`llmkit.EchoLLM` is a deterministic, offline reference model. It is not a
language model. It is a rule engine that returns output of the *right shape*:

| Ask it for | It returns |
|---|---|
| an answer with `[S1] ...` evidence blocks in the prompt | an extractive answer that cites the block it drew from |
| a `json_schema` | a schema-valid JSON instance, with `answer`-ish fields filled from the evidence |
| a judge or grading prompt | a JSON verdict with a score and a reason |
| anything else | a deterministic restatement of the request |

Two properties make it useful rather than a stub:

1. **Determinism.** The same prompt always produces the same output, seeded from
   a hash of the prompt. Tests assert on behaviour instead of tolerating noise.
2. **Configurable faults.** `EchoLLM(fault_rate=0.6)` deterministically corrupts
   a fraction of responses in the ways real models actually fail: a fenced code
   block, a chatty preamble, a truncated object, single quotes and trailing
   commas. That is how project 02 proves its repair path works instead of
   asserting that it exists.

What it does **not** do is reason. Wherever a result depends on real model
quality rather than on the surrounding system, the project README says so and
reports the system-level metric instead.

## Free upgrades, no code change

```bash
# Local model, free, offline. https://ollama.com
ollama pull llama3.2:1b
ollama pull nomic-embed-text
export LLM_PROVIDER=ollama EMBED_PROVIDER=ollama

# Any OpenAI-compatible endpoint: llama.cpp, vLLM, LM Studio, or a free-tier gateway
export LLM_PROVIDER=openai
export OPENAI_COMPAT_BASE_URL="http://localhost:8080/v1"
export OPENAI_COMPAT_MODEL="your-model"
export OPENAI_COMPAT_API_KEY="not-needed-for-local-servers"
```

Then re-run any demo. The pipelines, budgets, guardrails, evaluation and routing
are unchanged; only the model behind `get_llm()` moves.

## Embeddings

`HashingEmbedder` is the default: signed hashing of word unigrams, word bigrams
and character 4-grams into an L2-normalised vector. **It is lexical, not
semantic.** It gives partial credit for shared sub-words, so it behaves enough
like an encoder to build and test a retrieval stack, but it will not match a
paraphrase that shares no vocabulary. Every project that leans on it says so in
its Limits section, and the hybrid retrieval in project 01 exists partly because
lexical signal alone is not enough.

Swap in `nomic-embed-text` through Ollama for real semantics, still free.

## Cost model

`llmkit.tokens.PRICE_PER_1K_USD` is an editable price book with `small`,
`medium` and `large` tiers, and zero for every free provider. Projects 06, 07 and
13 do real arithmetic against it, so the cost and budget logic stays meaningful
and immediately becomes accurate the moment a paid model is plugged in.
