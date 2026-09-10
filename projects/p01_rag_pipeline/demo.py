"""End-to-end demo for the production RAG pipeline.

Run:  python3 projects/p01_rag_pipeline/demo.py
Every number printed here is measured at run time. Nothing is hard coded.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmkit import Tracer  # noqa: E402
from llmkit.corpus import gold_questions  # noqa: E402

from projects.p01_rag_pipeline.evaluate import evaluate_all, format_comparison  # noqa: E402
from projects.p01_rag_pipeline.pipeline import RagPipeline, preset  # noqa: E402


def rule(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


def main() -> None:
    tracer = Tracer("p01_rag_pipeline")
    pipeline = RagPipeline.build(config=preset("hybrid+rerank"), tracer=tracer)

    rule("1. Ingestion")
    for line in pipeline.ingest_stats.as_lines():
        print("  " + line)
    print(f"  chunks indexed       : {len(pipeline.retriever)}")
    print(f"  embedder             : {pipeline.retriever.embedder.name} "
          f"(dim {pipeline.retriever.embedder.dim})")

    rule("2. Retrieval on one query, mode by mode")
    query = "How long is a Meridian token valid before it expires?"
    print(f"  query: {query}\n")
    for mode in ("vector", "bm25", "hybrid"):
        hits = pipeline.retrieve(query, k=3, mode=mode, rerank=False)
        rendered = ", ".join(f"{h.chunk.doc_id}({h.score:.3f})" for h in hits)
        print(f"  {mode:<8} -> {rendered}")
    reranked = pipeline.retrieve(query, k=3, mode="hybrid", rerank=True)
    print("  reranked -> " + ", ".join(f"{h.chunk.doc_id}({h.score:.3f})" for h in reranked))
    top = reranked[0]
    print("\n  top chunk feature breakdown:")
    for key in ("coverage", "phrase", "position", "length", "relevance", "mmr", "agreement"):
        if key in top.components:
            print(f"    {key:<12} {top.components[key]:.3f}")

    rule("3. Cited answer")
    answer = pipeline.ask(query)
    print(f"  answer   : {answer.text}")
    print(f"  grounding: {answer.grounding:.3f}   reason: {answer.reason}")
    for citation in answer.citations:
        state = "valid" if citation.valid else "INVALID"
        print(f"  {citation.marker} -> {citation.chunk_id} ({citation.title}) [{state}]")

    rule("4. Refusal on an out-of-corpus question")
    off_topic = "What is the capital city of Iceland?"
    refused = pipeline.ask(off_topic)
    print(f"  query    : {off_topic}")
    print(f"  answer   : {refused.text}")
    print(f"  refused  : {refused.refused}   reason: {refused.reason}   "
          f"grounding: {refused.grounding:.3f}")

    rule("5. Configuration comparison over the gold set")
    questions = gold_questions()
    reports = evaluate_all(pipeline, questions)
    print(f"  {len(questions)} gold questions, document-level recall\n")
    print(format_comparison(reports))

    best = max(reports, key=lambda r: (r.retrieval.mrr, r.retrieval.recall_at_1))
    baseline = {r.label: r for r in reports}
    print("\n  citation accounting for " + best.label + ":")
    print(f"    markers emitted          : {best.answer.citations_emitted}")
    print(f"    markers unresolvable     : {best.answer.citations_invalid}")
    print(f"    gold doc among cited     : {best.answer.cited_doc_precision:.2f}")
    print(f"    avg prompt tokens/question: {best.answer.avg_prompt_tokens:.0f}")
    print(f"\n  best MRR: {best.label} at {best.retrieval.mrr:.3f}")
    if "vector" in baseline and baseline["vector"].retrieval.mrr:
        lift = (best.retrieval.mrr / baseline["vector"].retrieval.mrr - 1.0) * 100.0
        print(f"  lift over vector-only MRR: {lift:+.1f} percent")
    if "bm25" in baseline and baseline["bm25"].retrieval.mrr:
        lift = (best.retrieval.mrr / baseline["bm25"].retrieval.mrr - 1.0) * 100.0
        print(f"  lift over bm25-only MRR  : {lift:+.1f} percent")

    rule("6. Questions the best configuration still gets wrong")
    misses = [
        row for row in max(reports, key=lambda r: r.retrieval.mrr).per_question
        if row["gold_doc"] not in row["cited"]
    ]
    if not misses:
        print("  none")
    for row in misses:
        print(f"  - {row['question']}")
        print(f"      gold={row['gold_doc']} cited={row['cited'] or 'none'} "
              f"reason={row['reason']} grounding={row['grounding']}")

    rule("7. Trace summary")
    for name, agg in sorted(tracer.summary().items()):
        print(f"  {name:<16} calls={agg['count']:<5} avg_ms={agg['avg_ms']:<8} "
              f"tokens={agg['tokens']}")


if __name__ == "__main__":
    main()
