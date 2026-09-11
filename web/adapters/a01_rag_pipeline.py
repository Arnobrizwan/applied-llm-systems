"""Web adapter for project 01, the production RAG pipeline.

The page shows the retrieval work, not only the answer. Three retrievers race on
the same question, the reranker reorders what they agreed on, and every citation
in the final answer is resolved back to the chunk it came from.
"""
from __future__ import annotations

import os
import sys
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

NUMBER = 1
SLUG = "rag-pipeline"
TITLE = "Production RAG Pipeline"
TAGLINE = "Ask a question about a small product manual and watch the search find, reorder and cite its evidence."

WHAT_IT_DOES = """Type a question about Meridian, a made up product with a twenty
page manual loaded on this server. The page then shows you the whole search,
step by step, instead of just handing you an answer.

Three different searches run on your question at once. One matches on meaning,
one matches on exact words, and the third merges the two rankings. A reranking
stage then reorders the merged list and the page shows you which documents moved
and why, using the plain scores it worked from: how many of your words the
document covers, whether it contains a matching phrase, and whether both searches
agreed on it.

The answer at the end has a numbered marker on each sentence. Every marker is
checked against the evidence that was actually sent to the model, so you can see
which document each sentence came from. If nothing found is good enough to answer
from, the system says so before it writes anything. Try the Iceland question to
see that happen."""

INPUT_LABEL = "Ask a question about the Meridian docs"
PLACEHOLDER = "How long is a Meridian token valid before it expires?"
EXAMPLES = [
    "How long is a Meridian token valid before it expires?",
    "How do I stop a retried payment request from charging twice?",
    "What happens if I go over the request rate limit?",
    "What is the capital city of Iceland?",
]
SOURCE = "projects/p01_rag_pipeline"

_LOCK = threading.Lock()
_PIPELINE = None


def _pipeline():
    """Build the indexes once per process and reuse them on warm invocations."""
    global _PIPELINE
    with _LOCK:
        if _PIPELINE is None:
            from projects.p01_rag_pipeline.pipeline import RagPipeline, preset

            _PIPELINE = RagPipeline.build(config=preset("hybrid+rerank"))
        return _PIPELINE


def _clip(text: str, width: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "\u2026"


def _cell(hits, index: int, width: int = 22) -> str:
    if index >= len(hits):
        return " " * width
    hit = hits[index]
    return f"{_clip(hit.chunk.doc_id, 13):<14}{hit.score:>6.3f}  "


def run(user_input: str) -> str:
    try:
        return _run(user_input)
    except Exception as exc:  # never take the page down
        return f"This demo could not finish: {type(exc).__name__}: {exc}"


def _run(user_input: str) -> str:
    question = " ".join((user_input or "").split()) or EXAMPLES[0]
    if len(question) > 400:
        question = question[:400]

    pipe = _pipeline()
    out = []
    add = out.append

    add("QUESTION")
    add(f"  {question}")

    stats = pipe.ingest_stats
    docs = getattr(stats, "documents", None)
    add("")
    add("CORPUS ON THIS SERVER")
    add(f"  {docs if docs is not None else 20} documents, {len(pipe.retriever)} chunks indexed, "
        f"embedder {pipe.retriever.embedder.name} at {pipe.retriever.embedder.dim} dimensions")

    # -- step 1: three retrievers, same question -------------------------
    modes = ("vector", "bm25", "hybrid")
    lists = {m: pipe.retrieve(question, k=5, mode=m, rerank=False) for m in modes}

    add("")
    add("STEP 1  THREE SEARCHES RUN ON THE SAME QUESTION")
    add("  meaning search matches ideas, word search matches exact tokens,")
    add("  merged fuses the two rankings so agreement between them wins.")
    add("")
    add(f"  {'rank':<6}{'meaning search':<22}{'word search':<22}{'merged':<22}")
    add("  " + "-" * 70)
    for i in range(5):
        add(f"  {i + 1:<6}{_cell(lists['vector'], i)}{_cell(lists['bm25'], i)}"
            f"{_cell(lists['hybrid'], i)}")

    top_by_mode = {m: (lists[m][0].chunk.doc_id if lists[m] else "none") for m in modes}
    if len({top_by_mode["vector"], top_by_mode["bm25"]}) == 2:
        for line in _wrap(
            f"the two searches disagree on first place, {top_by_mode['vector']} against "
            f"{top_by_mode['bm25']}. Merging throws the two scores away and keeps only "
            f"the rank each one gave, so a document both searches rate well beats one "
            f"only a single search liked. Merged first place: {top_by_mode['hybrid']}.", 84
        ):
            add(f"  {line}")

    # -- step 2: reranking -----------------------------------------------
    candidates = pipe.retriever.candidates(question, mode="hybrid",
                                           candidate_k=pipe.config.candidate_k)
    old_rank = {h.chunk.id: n for n, h in enumerate(candidates, start=1)}
    reranked = pipe.retrieve(question, k=5, mode="hybrid", rerank=True)

    add("")
    add(f"STEP 2  RERANKER REORDERS THE TOP {len(candidates)} MERGED CANDIDATES")
    add(f"  {'new':<5}{'was':<5}{'document':<16}{'relevance':>10}{'picked':>9}"
        f"{'covers':>8}{'phrase':>8}{'agreed':>8}")
    add("  " + "-" * 70)
    moved = 0
    for new, hit in enumerate(reranked, start=1):
        comp = hit.components
        was = old_rank.get(hit.chunk.id, 0)
        if was and was != new:
            moved += 1
        add(f"  {new:<5}{(was or '-'):<5}{_clip(hit.chunk.doc_id, 15):<16}{hit.score:>10.3f}"
            f"{comp.get('mmr', hit.score):>9.3f}"
            f"{comp.get('coverage', 0.0):>8.3f}{comp.get('phrase', 0.0):>8.3f}"
            f"{('yes' if comp.get('agreement', 0.0) >= 1.0 else 'one'):>8}")
    add(f"  {moved} of {len(reranked)} results changed position after reranking")
    add("  covers = share of your question's rarer words present. phrase = longest run of")
    add("  your words found intact. agreed = both searches returned it. picked = relevance")
    add("  minus overlap with what was already chosen, which is the order used, so five")
    add("  views of one document cannot fill the whole answer.")

    # -- step 3: the evidence actually sent ------------------------------
    add("")
    add("STEP 3  EVIDENCE HANDED TO THE MODEL")
    for n, hit in enumerate(reranked[: pipe.config.max_evidence], start=1):
        title = hit.chunk.metadata.get("title") or hit.chunk.doc_id
        add(f"  [S{n}] {hit.chunk.id:<18} {_clip(title, 34)}")
        add(f"        {_clip(hit.chunk.text, 80)}")

    # -- step 4: the answer and its citations ----------------------------
    answer = pipe.ask(question)
    add("")
    add("STEP 4  ANSWER, WITH EVERY CITATION CHECKED")
    add(f"  evidence quality {answer.grounding:.3f} against a floor of "
        f"{pipe.config.grounding_floor:.2f}  ->  {answer.reason}")
    if answer.refused:
        add("")
        add(f"  REFUSED: {answer.text}")
        add("  Nothing retrieved covered enough of the question, so the refusal happened")
        add("  before the model was called and cost nothing.")
        return "\n".join(out)[:8000]

    add("")
    for line in _wrap(answer.text, 82):
        add(f"  {line}")
    add("")
    add(f"  {'marker':<8}{'resolves to':<20}{'document':<34}{'status':<8}")
    add("  " + "-" * 68)
    for cite in answer.citations:
        add(f"  {cite.marker:<8}{_clip(cite.chunk_id or 'unresolved', 19):<20}"
            f"{_clip(cite.title or '-', 33):<34}"
            f"{('valid' if cite.valid else 'STRIPPED'):<8}")
    if not answer.citations:
        add("  the model returned no marker at all, so the answer above is the most")
        add("  on-topic sentence quoted straight out of the top evidence")
    invalid = len(answer.invalid_citations)
    add(f"  {len(answer.valid_citations)} markers resolved to real evidence, "
        f"{invalid} unresolvable and removed from the text")
    if answer.fallback_used:
        add("  the model broke the citation contract, so the deterministic fallback answered")
    add(f"  prompt size {answer.prompt_tokens} tokens for "
        f"{len(answer.evidence)} pieces of evidence")

    return "\n".join(out)[:8000]


def _wrap(text: str, width: int):
    words = str(text).split()
    lines, current = [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]
