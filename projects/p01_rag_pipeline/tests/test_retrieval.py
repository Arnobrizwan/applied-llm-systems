"""Retrieval, fusion and reranking behaviour."""
import pytest

from llmkit import Chunk
from llmkit.corpus import gold_questions

from projects.p01_rag_pipeline.evaluate import evaluate_retrieval
from projects.p01_rag_pipeline.ingest import IngestConfig, Ingestor
from projects.p01_rag_pipeline.pipeline import RagPipeline, preset
from projects.p01_rag_pipeline.rerank import FeatureReranker, IdentityReranker
from projects.p01_rag_pipeline.retrieval import HybridRetriever
from projects.p01_rag_pipeline.terms import content_terms, fold


@pytest.fixture(scope="module")
def pipeline():
    """One built index shared by the module; construction is the slow part."""
    return RagPipeline.build(config=preset("hybrid+rerank"))


def test_ingests_external_text_files_with_citable_ids(tmp_path):
    (tmp_path / "runbook.md").write_text(
        "# Failover runbook\n\n"
        "Promote the standby replica before draining the primary. "
        "Draining first strands in-flight writes with nowhere to land.\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("Retention for cold storage is 400 days.\n", encoding="utf-8")
    (tmp_path / "ignored.pdf").write_bytes(b"%PDF-1.4 not text")

    ingestor = Ingestor()
    chunks = ingestor.ingest(paths=[str(tmp_path)], include_builtin=False)

    doc_ids = {c.doc_id for c in chunks}
    assert doc_ids == {"runbook", "notes"}, "the .pdf must be skipped, not parsed"
    assert all(c.id == f"{c.doc_id}#{c.ordinal}" for c in chunks)
    # Markdown H1 becomes the citation title, so a citation reads like a person
    # wrote it rather than like a filename.
    runbook = [c for c in chunks if c.doc_id == "runbook"][0]
    assert runbook.metadata["doc_title"] == "Failover runbook"


def test_duplicate_boilerplate_is_dropped_once():
    boilerplate = "Copyright Meridian. All rights reserved. Redistribution is prohibited."
    ingestor = Ingestor(IngestConfig(target_tokens=40, overlap_tokens=0, min_chunk_chars=10))
    from llmkit import Document

    docs = [
        Document(id="a", text=boilerplate, metadata={"title": "A"}),
        Document(id="b", text=boilerplate, metadata={"title": "B"}),
    ]
    chunks = ingestor.chunk(docs)
    assert len(chunks) == 1
    assert ingestor.stats.duplicates_dropped == 1


def test_fusion_is_rank_based_and_records_agreement(pipeline):
    """Fused order must be reproducible from the two rank lists alone.

    This is the property that makes RRF safe: the output depends on the ordering
    each retriever produced and on nothing else, so neither retriever's score
    scale can drift and silently re-weight the blend.
    """
    query = "how are webhook deliveries authenticated"
    depth = 10
    dense = [h.chunk.id for h in pipeline.retriever.vector_search(query, depth)]
    lexical = [h.chunk.id for h in pipeline.retriever.bm25_search(query, depth)]

    expected = {}
    for ranking in (dense, lexical):
        for rank, chunk_id in enumerate(ranking, start=1):
            expected[chunk_id] = expected.get(chunk_id, 0.0) + 1.0 / (60 + rank)
    expected_order = [cid for cid, _ in sorted(expected.items(), key=lambda kv: (-kv[1], kv[0]))]

    fused = pipeline.retriever.hybrid_search(query, k=depth, candidate_k=depth)
    assert [h.chunk.id for h in fused] == expected_order[:depth]

    agreed = [h for h in fused if h.components["agreement"] == 1.0]
    only_one = [h for h in fused if h.components["agreement"] == 0.0]
    assert agreed, "both retrievers should agree on something for this query"
    assert all(h.components["vector_rank"] and h.components["bm25_rank"] for h in agreed)
    assert all(not (h.components["vector_rank"] and h.components["bm25_rank"]) for h in only_one)
    assert fused[0].chunk.doc_id == "webhooks"


def test_fusion_beats_both_retrievers_where_each_one_alone_is_wrong(pipeline):
    """The concrete reason both indexes exist, measured on the shipped corpus.

    On this query the dense index puts an unrelated document first and BM25 puts
    a different unrelated document first, because each is distracted by a
    different part of the query. Neither is confidently wrong in the same
    direction, so the document they both rank moderately well wins the fusion.
    That is agreement doing the work, and it is not reproducible by taking the
    better of the two retrievers.
    """
    query = "How long is a Meridian token valid before it expires?"
    dense = [h.chunk.doc_id for h in pipeline.retriever.vector_search(query, 5)]
    lexical = [h.chunk.doc_id for h in pipeline.retriever.bm25_search(query, 5)]
    fused = [h.chunk.doc_id for h in pipeline.retriever.retrieve(query, k=5, mode="hybrid")]

    assert dense[0] != "auth-rotation"
    assert lexical[0] != "auth-rotation"
    assert fused[0] == "auth-rotation"


def test_lexical_and_dense_fail_in_different_places(pipeline):
    """The complementarity claim in the README, asserted rather than asserted at."""
    # BM25 nails an exact status code, which is the case a dense index loses.
    lexical = pipeline.retriever.bm25_search("429 Retry-After", k=3)
    assert lexical[0].chunk.doc_id == "rate-limits"

    # BM25 returns literally nothing when no query term is in the vocabulary.
    # The dense side still returns candidates, so hybrid degrades instead of
    # handing the user an empty page.
    nonsense = "qqqzzz vlorpen thaxil"
    assert pipeline.retriever.bm25_search(nonsense, k=3) == []
    assert len(pipeline.retriever.vector_search(nonsense, k=3)) == 3
    assert len(pipeline.retriever.retrieve(nonsense, k=3, mode="hybrid")) == 3


def test_reranker_beats_raw_fusion_on_the_gold_set(pipeline):
    """The reranker has to earn its place in the pipeline on measured numbers."""
    fused = evaluate_retrieval(pipeline.with_config(preset("hybrid")), gold_questions())
    reranked = evaluate_retrieval(pipeline.with_config(preset("hybrid+rerank")), gold_questions())
    assert reranked.recall_at_1 > fused.recall_at_1
    assert reranked.mrr > fused.mrr


def test_mmr_returns_distinct_documents_not_five_views_of_one():
    near_duplicates = [
        Chunk(id=f"dup#{i}", doc_id=f"dup{i}",
              text="The default rate limit is 600 requests per minute per workspace.",
              metadata={"title": f"Dup {i}"})
        for i in range(4)
    ]
    distinct = Chunk(id="other#0", doc_id="other",
                     text="Exceeding the limit returns HTTP 429 with a Retry-After header.",
                     metadata={"title": "Other"})
    retriever = HybridRetriever(near_duplicates + [distinct])
    candidates = retriever.candidates("rate limit per workspace", mode="hybrid")

    greedy = IdentityReranker().rerank("rate limit per workspace", candidates, k=2)
    diverse = FeatureReranker(
        *retriever.document_frequencies(), mmr_lambda=0.3, similarity_fn=retriever.similarity
    ).rerank("rate limit per workspace", candidates, k=2)

    assert len({c.chunk.text for c in greedy}) == 1, "top-k by score returns the same sentence twice"
    assert len({c.chunk.text for c in diverse}) == 2, "MMR must cover distinct material"


def test_length_penalty_stops_a_padded_chunk_winning_on_word_count():
    reranker = FeatureReranker(target_chars=200)
    short = Chunk(id="s#0", doc_id="s", text="Audit log export is available only to workspace owners.")
    padded = Chunk(
        id="l#0", doc_id="l",
        text="Audit log export is available only to workspace owners. " + ("filler text here. " * 60),
    )
    assert reranker.length_score(short.text) == 1.0
    assert reranker.length_score(padded.text) < 0.3
    assert reranker.relevance("who can export the audit log", short)["relevance"] > \
        reranker.relevance("who can export the audit log", padded)["relevance"]


def test_term_folding_survives_plurals_and_tense():
    assert fold("tokens") == fold("token")
    assert fold("expires") == fold("expire")
    assert fold("authenticated") == fold("authenticates")
    # Interrogatives are removed entirely: they are rare in a technical corpus,
    # so leaving them in would give an off-topic question a high idf free ride.
    assert content_terms("How long is a token valid?") == content_terms("token valid")


def test_unknown_retrieval_mode_is_rejected_loudly(pipeline):
    with pytest.raises(ValueError):
        pipeline.retriever.retrieve("anything", mode="magic")


def test_empty_query_returns_nothing_rather_than_arbitrary_chunks(pipeline):
    assert pipeline.retriever.retrieve("   ", k=5) == []
