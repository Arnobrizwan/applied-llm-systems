# 01. Production RAG Pipeline

**Run it live: [https://applied-llm-systems.vercel.app/s/rag-pipeline](https://applied-llm-systems.vercel.app/s/rag-pipeline)** - the hosted page runs this code and shows the real output.

Ingestion, chunking, hybrid retrieval, reranking and answer synthesis where every
citation in the output is verified against the evidence that was actually sent to
the model.

## The problem

The demo version of RAG is twenty lines: embed the corpus, cosine search, stuff
the top five chunks into a prompt, print whatever comes back. It works on the
questions you tried while building it and fails in three specific ways once real
users arrive.

**It cannot find exact tokens.** Dense retrieval is a similarity match on
meaning. A user searching for `429`, `Idempotency-Key` or an internal service
name gets whatever was semantically nearby, which is often nothing useful. The
answer exists in the corpus and retrieval never sees it.

**Its citations are decorative.** Asking a model to cite its sources and printing
the result is not a citation feature. Models emit `[S4]` when four sources were
supplied and the claim came from none of them, and they emit `[S7]` when only
five were supplied at all. Both render identically to a user. Nothing in the
naive pipeline notices.

**It always answers.** With no floor on evidence quality, an off-topic question
retrieves the five least-irrelevant chunks and gets a confident, fluent answer
built from them. That is the failure users actually report, because unlike a
retrieval miss it is invisible to everyone except the person who knows the true
answer.

This project fixes all three, and measures the fix.

## What this builds

1. **Ingestion** (`ingest.py`) reads the built-in Meridian corpus and any
   `.txt`/`.md` file or directory passed in, chunks it sentence-aware with token
   overlap, assigns position-derived citable ids, and drops exact-duplicate
   boilerplate.
2. **Two indexes** (`retrieval.py`): an `InMemoryVectorStore` over hashing
   embeddings and a `BM25` lexical index, built from one shared pass of
   embedding so the reranker can reuse the vectors.
3. **Reciprocal rank fusion** over the two candidate lists, with per-chunk
   provenance recorded: the rank each retriever gave it, its raw score in each,
   and whether the two agreed.
4. **A reranker** (`rerank.py`): idf-weighted query term coverage, longest
   contiguous phrase hit, a chunk position prior and a length penalty, followed
   by MMR selection for diversity. It stands in for a cross-encoder and does not
   pretend otherwise.
5. **Term folding** (`terms.py`), shared by the reranker and the answerer, so
   "expires" and "expire" compare equal and interrogatives earn no coverage
   credit.
6. **Cited answer synthesis** (`answer.py`): numbered `[S1]` evidence blocks, an
   explicit citation contract in the prompt, validation of every marker that
   comes back, and a refusal gate evaluated before the model is called.
7. **Evaluation** (`evaluate.py`) over the 15 gold questions across four
   configurations, printed as one comparison table by `demo.py`.

## Architecture

```
  documents (llmkit.corpus + any .txt/.md path)
        |
        v
  [ ingest ]  sentence-aware chunking, overlap, dedupe, stable ids
        |
        +---------------------------+
        v                           v
  [ dense index ]              [ BM25 index ]
  InMemoryVectorStore          lexical, exact tokens
        |                           |
        | top-20                    | top-20
        +------------+--------------+
                     v
            [ reciprocal rank fusion ]     score(d) = sum 1/(60 + rank)
                     |
                     v  20 candidates
            [ feature reranker ]           coverage / phrase / position / length
                     |
                     v
            [ MMR selection ]              relevance minus redundancy
                     |
                     v  top-5 evidence
            [ grounding gate ] --- below floor ---> "insufficient evidence"
                     |
                     v
            [ prompt: [S1]..[S5] + citation contract ] -> LLM
                     |
                     v
            [ citation validator ]
              resolves marker -> chunk   -> unresolvable markers stripped
              no valid markers at all    -> deterministic extractive fallback
                     |
                     v
              cited answer + provenance
```

## Design decisions

**Reciprocal rank fusion, not a weighted score blend.** The obvious approach is
`alpha * cosine + (1 - alpha) * bm25`. It requires normalising two scores that
are not comparable: cosine is bounded in [-1, 1], BM25 is unbounded and grows
with corpus idf, so the same query scores differently after an ingest. Every
normaliser available is unsound in this setting. Min-max normalisation is
computed over the candidate list, which means a document's fused score changes
depending on which other documents happened to be retrieved alongside it.
Z-scores assume a distribution a three-item candidate list does not have. RRF
discards the scores and fuses the rankings, which is the only thing the two
retrievers agree on the meaning of. It has one constant, it cannot be
destabilised by a scale change, and it rewards agreement between two independent
retrievers. On this corpus fusion puts `auth-rotation` first for the token
expiry question when the dense index puts `pagination` first and BM25 puts
`billing` first. Neither retriever alone gets it; agreement does.

**The refusal gate does not threshold on the retriever score.** `if top_score <
0.4: refuse` means three different things across the three modes here, and in
hybrid mode it is meaningless: an RRF score depends only on rank, so the top hit
scores about 1/61 whether it is perfect or noise. The gate instead uses
idf-weighted coverage of the question's content vocabulary by the retrieved
text, which is computed from the question and the chunk and is therefore
identical across retrieval modes. That comparability is what makes the four-way
table in Results an apples-to-apples comparison. It also lets the refusal happen
*before* the model call, so a question that was always going to produce a hedge
costs nothing.

**Term folding was added because measurement said so, not because it looked
tidy.** The first working version refused 47 percent of the gold questions,
including ones whose answer sat in the top retrieved chunk: the question said
"expires" and "tokens", the document said "expire" and "token", and set
intersection scored that as zero overlap. Separately, a wholly off-corpus
question ("what is the capital city of Iceland") scored 0.19 grounding because
"what" is rare in a technical corpus and therefore carried a high idf. Folding
morphology and removing interrogatives took the refusal rate on the gold set from
0.47 to 0.00 and the off-corpus grounding from 0.19 to 0.00. A later one-character
widening of the trailing-`e` rule, so that "move" and "moved" fold together, moved
recall@1 from 0.87 to 0.93 and MRR from 0.922 to 0.967 on its own. The stemmer
over-stems on purpose; a real deployment uses Snowball, which is the same idea
with better rules and a dependency this repo cannot take.

**An unresolvable citation is stripped, and an uncited answer falls back rather
than shipping.** Leaving `[S9]` in the text when nine sources were never supplied
manufactures confidence that nothing supports, so invalid markers are removed and
counted. When the model returns no resolvable marker at all, the system quotes the
most on-topic sentence from the top evidence and attaches its marker by
construction. It is worse prose than the model would have written and it is
always attributable. The alternative considered and rejected was retrying the
model with a sterner prompt: it costs a second call, it is not guaranteed to
converge, and there is already a correct answer sitting in the evidence.

**MMR instead of straight top-k.** Top-k by relevance on a chunked corpus returns
five near-identical chunks, because a document that mentions the answer once
mentions it across three overlapping chunks. That burns the context budget and
makes one source cited five times look like five sources. The test
`test_mmr_returns_distinct_documents_not_five_views_of_one` builds exactly that
situation: greedy top-k returns the same sentence twice, MMR returns two distinct
ones.

**The reranker is a feature scorer, stated plainly.** In a funded system this slot
holds a cross-encoder. That is strictly better and it is a model download or a
paid API call. `Reranker` is an ABC with one method; `FeatureReranker` and
`IdentityReranker` implement it and a real cross-encoder would be a third
subclass and a one-line change in `pipeline.py`. The feature scorer does have one
genuine advantage: every component is written back onto the result, so a bad
ranking is explainable from a log line instead of by re-running the query.

## Running it

```bash
python3 projects/p01_rag_pipeline/demo.py
python3 -m pytest projects/p01_rag_pipeline -q
```

The demo prints seven sections: ingestion counts, the same query resolved by each
retrieval mode with the winning chunk's feature breakdown, one cited answer with
each marker resolved to a chunk id, a refusal on an out-of-corpus question, the
four-configuration comparison table, the questions the best configuration still
gets wrong, and a trace summary by span.

To index your own material:

```python
from projects.p01_rag_pipeline.pipeline import RagPipeline, preset

pipeline = RagPipeline.build(paths=["./docs"], config=preset("hybrid+rerank"))
answer = pipeline.ask("How long is a token valid?")
print(answer.text, answer.cited_doc_ids, answer.refused)
```

## Results

Measured by `demo.py` on the 20-document `llmkit.corpus` and its 15 hand-written
gold questions, using the default offline stack (`HashingEmbedder` at dim 384,
`EchoLLM`). Recall is scored at document level: the gold set names the document
that answers each question, so chunk-level scoring would reward an arbitrary
chunking decision rather than the retriever. Every configuration runs against the
same chunks, the same vectors and the same BM25 counts; only the routing changes.

| configuration | R@1 | R@3 | R@5 | MRR | citation validity | contract break | refusal | grounded answer |
|---|---|---|---|---|---|---|---|---|
| vector only | 0.67 | 0.87 | 0.87 | 0.756 | 1.00 | 0.29 | 0.07 | 0.33 |
| BM25 only | 0.73 | 1.00 | 1.00 | 0.844 | 1.00 | 0.20 | 0.00 | 0.40 |
| hybrid (RRF) | 0.67 | 0.93 | 0.93 | 0.800 | 1.00 | 0.36 | 0.07 | 0.40 |
| **hybrid + rerank** | **0.93** | **1.00** | **1.00** | **0.967** | 1.00 | 0.20 | 0.00 | **0.47** |

- Best configuration is hybrid + rerank at MRR 0.967, which is 27.9 percent above
  vector-only and 14.5 percent above BM25-only.
- Citation validity is 1.00 in all four configurations: across 27 markers emitted
  by the best configuration, 0 were unresolvable. Validity is measured per marker
  emitted, not per question, so a system that emits ten markers and gets one
  wrong is not scored the same as one that emits a single marker.
- The gold document appears among the cited documents for 0.87 of answered
  questions in the best configuration.
- Contract break rate 0.20 means 3 of the 15 answered questions produced no
  resolvable citation and were served by the deterministic extractive fallback.
  The cause here is worth stating because it is a real production failure mode in
  miniature: `EchoLLM` routes on prompt keywords, and any prompt containing the
  word "rate", "score", "judge" or "grade" is treated as an evaluation prompt and
  answered with a JSON verdict instead of prose. The corpus contains a document
  about rate limits, so retrieving it hijacks the synthesis call. The validator
  catches every one of those and the user never sees a JSON blob. This is exactly
  the class of provider-side behaviour that citation validation exists to absorb.
- Refusal rate is 0.00 on the gold set for the best configuration and the
  off-corpus control question ("What is the capital city of Iceland?") scores
  0.000 grounding and is refused. Refusal rate must always be read against
  accuracy: a system that refuses everything has perfect citation validity.
- Grounded answer rate 0.47 is a substring check for the gold phrase in the
  answer text. It under-counts every correct answer that paraphrases, so treat it
  as a floor, not an accuracy score.
- Average prompt size is 507 estimated tokens per question at 5 pieces of
  evidence.
- Two gold questions remain wrong in the best configuration and the demo prints
  them: the uptime question cites `sla` rather than `plans`, which is defensible
  but not the gold answer, and the audit-log question cites `logs` and `audit`
  rather than `roles`.

## Limits

- **The embedder is lexical.** `HashingEmbedder` is a hashing-trick bag of word
  and character n-grams. It gives partial credit for morphology and typos and it
  cannot match a true paraphrase that shares no sub-words. Setting
  `EMBED_PROVIDER=ollama` swaps in `nomic-embed-text` with no code change and the
  vector-only column should improve most.
- **The reranker is not a cross-encoder.** It has no notion of entailment,
  negation or answer type. A chunk that contains every query term while stating
  the opposite of the answer will rank highly. This is the single biggest quality
  gap in the project and the interface exists so it can be closed.
- **`EchoLLM` is not a language model.** It is a deterministic rule engine that
  produces correctly shaped grounded output, which is what makes retrieval and
  validation testable offline. Answer fluency and the grounded answer rate would
  both change with a real model; retrieval recall and MRR would not, since no
  model is involved in producing them.
- **Token counts are estimates.** `llmkit.count_tokens` approximates BPE at 1.3
  tokens per word. Set `LLMKIT_TOKENIZER=tiktoken` where the environment allows a
  native wheel.
- **Exact brute-force search.** 20 chunks here, thousands in a demo, and no ANN
  index. Past roughly 100k chunks this needs FAISS or pgvector; the
  `InMemoryVectorStore` interface mirrors theirs so it is a class swap.
- **Citation validity proves attribution, not truth.** A verified citation means
  the marker resolves to a chunk that was in the prompt. It does not mean the
  sentence is supported by that chunk. Claim-level faithfulness needs an entailment
  check over each sentence and its cited chunk, which is a separate system.
