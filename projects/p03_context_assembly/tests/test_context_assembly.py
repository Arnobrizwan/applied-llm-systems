"""Tests for the context assembly service. Everything runs offline."""
import pytest

from llmkit import EchoLLM, count_tokens

from projects.p03_context_assembly.assembler import ContextAssembler
from projects.p03_context_assembly.baseline import (
    assembled_metrics,
    compare,
    naive_prompt,
    prefix_truncation_metrics,
)
from projects.p03_context_assembly.budget import BudgetAllocator
from projects.p03_context_assembly.compression import arrange_hourglass, compress, truncate_head
from projects.p03_context_assembly.dedupe import containment, deduplicate
from projects.p03_context_assembly.receipt import (
    STATUS_COMPRESSED,
    STATUS_DEDUPED,
    STATUS_DROPPED,
    STATUS_INCLUDED,
)
from projects.p03_context_assembly.sections import (
    Compression,
    ContextItem,
    ContextRequest,
    Section,
    SectionSpec,
)
from projects.p03_context_assembly.workload import build_request

QUERY = "Why do we keep getting 429 responses at night and will more tokens help?"


def _section(name, priority, items, **kwargs):
    spec = SectionSpec(name=name, priority=priority, **kwargs)
    return Section(spec=spec, items=items)


def _items(prefix, count, tokens_each=40, value=0.5):
    body = " ".join(f"{prefix}word{i}" for i in range(tokens_each))
    return [ContextItem(id=f"{prefix}.{i}", text=body, source=prefix, value=value) for i in range(count)]


# -- the guarantee ------------------------------------------------------


@pytest.mark.parametrize("window,reserve", [(400, 150), (800, 250), (1600, 400), (6000, 1000)])
def test_never_exceeds_window_minus_reserve(window, reserve):
    request = build_request(QUERY, model_window=window, reserve_tokens=reserve)
    assembled = ContextAssembler(llm=EchoLLM()).assemble(request)
    assert assembled.tokens <= window - reserve
    # And the receipt agrees with a fresh count of the string it describes.
    assert assembled.receipt.prompt_tokens == count_tokens(assembled.text)
    assert assembled.receipt.headroom >= 0


def test_naive_concatenation_would_have_overflowed_the_same_window():
    request = build_request(QUERY, model_window=800, reserve_tokens=250)
    assembled = ContextAssembler().assemble(request)
    result = compare(request, assembled.receipt)
    assert result.naive_overflows
    assert result.naive_tokens > request.available_tokens
    assert assembled.tokens <= request.available_tokens
    assert result.tokens_saved > 0


def test_reserve_is_actually_subtracted_not_decorative():
    small = build_request(QUERY, model_window=1200, reserve_tokens=100)
    large = build_request(QUERY, model_window=1200, reserve_tokens=700)
    assembler = ContextAssembler()
    assert assembler.assemble(large).tokens < assembler.assemble(small).tokens


def test_request_rejects_a_reserve_larger_than_the_window():
    with pytest.raises(ValueError):
        ContextRequest(query="q", sections=[], model_window=500, reserve_tokens=500)


# -- allocation ---------------------------------------------------------


def test_minimums_are_honoured_before_priority():
    tiny_but_guaranteed = _section("guarded", 1.0, _items("g", 1, 30), min_tokens=40)
    greedy = _section("greedy", 9.0, _items("h", 10, 60))
    plan = BudgetAllocator().allocate([tiny_but_guaranteed, greedy], available=300)
    assert plan.allocations["guarded"].granted >= min(40, tiny_but_guaranteed.demand)
    assert plan.granted_total <= 300


def test_lowest_priority_section_is_shed_when_minimums_do_not_fit():
    high = _section("high", 9.0, _items("a", 2, 50), min_tokens=100)
    low = _section("low", 1.0, _items("b", 2, 50), min_tokens=100)
    plan = BudgetAllocator().allocate([high, low], available=110)
    assert plan.allocations["low"].granted == 0
    assert "shed" in plan.allocations["low"].note
    assert plan.allocations["high"].granted > 0
    assert plan.granted_total <= 110


def test_max_share_caps_a_greedy_section():
    hog = _section("hog", 9.0, _items("a", 20, 50), max_share=0.25)
    other = _section("other", 1.0, _items("b", 20, 50))
    plan = BudgetAllocator().allocate([hog, other], available=1000)
    assert plan.allocations["hog"].granted <= 250
    assert plan.allocations["other"].granted > 0


def test_surplus_from_a_small_section_is_redistributed_not_stranded():
    small = _section("small", 9.0, _items("a", 1, 20))
    big = _section("big", 1.0, _items("b", 20, 60))
    plan = BudgetAllocator().allocate([small, big], available=900)
    assert plan.allocations["small"].granted == small.demand  # never more than it asked for
    # The tokens the small section did not need went somewhere useful.
    assert plan.allocations["big"].granted > 600


# -- deduplication ------------------------------------------------------


def test_cross_source_duplicate_is_paid_for_once():
    request = build_request(QUERY, model_window=1600, reserve_tokens=400)
    assembled = ContextAssembler().assemble(request)
    dups = assembled.receipt.duplicates
    assert dups, "the memory restatement of the rate-limit doc should be caught"
    assert assembled.receipt.dedup_tokens_saved() > 0
    # The surviving copy is the fuller one, and the removal is on the receipt.
    assert any(d["kept_section"] == "retrieved_docs" for d in dups)
    assert any(i.status == STATUS_DEDUPED for i in assembled.receipt.items())


def test_dedupe_keeps_contradicting_numbers_apart():
    a = ContextItem(id="a", text="Meridian tokens expire 90 days after creation.", source="doc")
    b = ContextItem(id="b", text="Meridian tokens expire 30 days after creation.", source="mem")
    section = _section("s", 1.0, [a, b])
    removed = deduplicate([section], threshold=0.8)
    assert removed == []
    assert len(section.items) == 2
    assert containment(a.text, b.text) < 0.8


# -- compression --------------------------------------------------------


def test_keep_whole_drops_rather_than_truncates():
    long_item = ContextItem(id="policy", text="word " * 200, source="system")
    result = compress(long_item.text, 50, Compression.KEEP_WHOLE)
    assert result is None


def test_truncate_head_keeps_the_end_and_tail_keeps_the_start():
    text = " ".join(f"word{i}" for i in range(60))
    head_kept = compress(text, 20, Compression.TRUNCATE_TAIL).text
    tail_kept = compress(text, 20, Compression.TRUNCATE_HEAD).text
    assert head_kept.startswith("word0")
    assert tail_kept.endswith("word59")
    assert count_tokens(head_kept) <= 20
    assert count_tokens(tail_kept) <= 20
    assert truncate_head(text, 0) == ""


def test_extractive_keeps_query_relevant_sentences_in_document_order():
    text = (
        "Meridian ships SDKs for three languages. "
        "The default rate limit is 600 requests per minute per workspace. "
        "Invoices are issued monthly in arrears. "
        "Exceeding the limit returns HTTP 429 with a Retry-After header."
    )
    out = compress(text, 30, Compression.EXTRACTIVE, query="which status code does a 429 response return").text
    assert "429" in out
    assert "Invoices" not in out  # the irrelevant sentence is the one that goes
    assert count_tokens(out) <= 30
    assert out.index("Meridian") < out.index("429")  # original document order preserved


def test_compression_refuses_to_emit_a_useless_fragment():
    assert compress("word " * 100, 3, Compression.EXTRACTIVE, query="x") is None


def test_summarize_output_is_capped_even_if_the_model_overshoots():
    class Chatty:
        def complete(self, messages, **kwargs):
            class R:
                text = "word " * 500

            return R()

    result = compress("word " * 300, 40, Compression.SUMMARIZE, query="q", llm=Chatty())
    assert result is not None
    assert result.after_tokens <= 40


# -- ordering -----------------------------------------------------------


def test_hourglass_puts_the_best_at_the_ends_and_filler_in_the_middle():
    order = arrange_hourglass([0.9, 0.1, 0.8, 0.2, 0.7])
    assert order[0] == 0  # best first
    assert order[-1] == 2  # second best last
    assert 1 in order[1:-1] and 3 in order[1:-1]  # weakest buried


def test_pinned_sections_keep_their_declared_position():
    request = build_request(QUERY, model_window=1600, reserve_tokens=400)
    receipt = ContextAssembler().assemble(request).receipt
    assert receipt.order[0] == "instructions"
    assert receipt.order[-1] == "question"


# -- receipt ------------------------------------------------------------


def test_receipt_accounts_for_every_item_exactly_once():
    request = build_request(QUERY, model_window=800, reserve_tokens=250)
    receipt = ContextAssembler().assemble(request).receipt
    offered = {i.id for s in request.sections for i in s.items}
    recorded = [i.item_id for i in receipt.items()]
    assert set(recorded) == offered
    assert len(recorded) == len(offered)
    assert all(
        i.status in (STATUS_INCLUDED, STATUS_COMPRESSED, STATUS_DROPPED, STATUS_DEDUPED)
        for i in receipt.items()
    )


def test_receipt_explains_every_drop_and_serialises():
    request = build_request(QUERY, model_window=800, reserve_tokens=250)
    receipt = ContextAssembler().assemble(request).receipt
    dropped = receipt.items(STATUS_DROPPED)
    assert dropped, "a 550 token budget cannot hold this request"
    assert all(d.reason for d in dropped)
    payload = receipt.to_dict()
    assert payload["prompt_tokens"] == receipt.prompt_tokens
    assert "sections" in payload and payload["sections"]
    assert receipt.render().startswith("CONTEXT RECEIPT")


def test_compressed_items_report_a_real_ratio():
    request = build_request(QUERY, model_window=800, reserve_tokens=250)
    receipt = ContextAssembler().assemble(request).receipt
    compressed = receipt.items(STATUS_COMPRESSED)
    assert compressed
    for item in compressed:
        assert 0 < item.after_tokens < item.before_tokens
    assert receipt.compression_tokens_saved() == sum(i.saved_tokens for i in compressed)


# -- measured comparison ------------------------------------------------


def test_assembler_covers_more_high_priority_sources_than_prefix_truncation():
    request = build_request(QUERY, model_window=800, reserve_tokens=250)
    receipt = ContextAssembler().assemble(request).receipt
    _, naive_coverage = prefix_truncation_metrics(request)
    _, ours_coverage = assembled_metrics(request, receipt)
    assert ours_coverage > naive_coverage
    assert ours_coverage == 1.0


def test_the_live_question_survives_a_budget_that_naive_truncation_would_cut():
    request = build_request(QUERY, model_window=800, reserve_tokens=250)
    assembled = ContextAssembler().assemble(request)
    assert QUERY in assembled.text
    # The naive prompt puts the question last, so cutting to fit removes it.
    naive_cut_length = len(naive_prompt(request))
    assert naive_cut_length > len(assembled.text)


def test_empty_sections_are_handled_without_emitting_a_blank_block():
    empty = _section("empty", 5.0, [])
    filled = _section("filled", 5.0, _items("a", 2, 20))
    request = ContextRequest(
        query="q", sections=[empty, filled], model_window=600, reserve_tokens=100, request_id="r"
    )
    assembled = ContextAssembler().assemble(request)
    assert "empty" not in assembled.receipt.order
    assert assembled.text.strip()
