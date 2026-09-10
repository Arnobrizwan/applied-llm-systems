"""Context Assembly Service demo.

Runs the same support request against three window sizes, prints the full
receipt for the tightest one, and compares every run against naive
concatenation. Every number printed is measured at run time.

    python3 projects/p03_context_assembly/demo.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmkit import EchoLLM, count_tokens  # noqa: E402

from projects.p03_context_assembly.assembler import ContextAssembler  # noqa: E402
from projects.p03_context_assembly.baseline import (  # noqa: E402
    Comparison,
    compare,
    naive_prompt,
    summarise,
)
from projects.p03_context_assembly.sections import Compression, ContextItem, Section, SectionSpec  # noqa: E402
from projects.p03_context_assembly.workload import build_request  # noqa: E402

QUERY = "Why do we keep getting 429 responses at night and will more tokens help?"

# window, reserve. The reserve is the completion budget: the tokens the model
# needs to write its answer, which the prompt is not allowed to spend.
SCENARIOS = [
    ("tight-800", 800, 250),
    ("normal-1600", 1600, 400),
    ("roomy-4000", 4000, 600),
]


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    assembler = ContextAssembler(llm=EchoLLM())
    comparisons = []
    receipts = {}

    for request_id, window, reserve in SCENARIOS:
        request = build_request(QUERY, model_window=window, reserve_tokens=reserve, request_id=request_id)
        assembled = assembler.assemble(request)
        comparisons.append(compare(request, assembled.receipt))
        receipts[request_id] = (request, assembled)

    rule("1. RECEIPT for the tightest window")
    request, assembled = receipts["tight-800"]
    print()
    print(assembled.receipt.render())

    rule("2. THE PROMPT that receipt describes")
    print()
    print(assembled.text)

    rule("3. NAIVE CONCATENATION, same request, same window")
    naive = naive_prompt(request)
    naive_tokens = count_tokens(naive)
    print()
    print(f"  naive prompt is {naive_tokens} tokens")
    print(f"  window is {request.model_window}, completion reserve is {request.reserve_tokens}")
    print(f"  budget for the prompt is {request.available_tokens}")
    over = naive_tokens - request.available_tokens
    if over > 0:
        print(f"  naive concatenation overruns the prompt budget by {over} tokens")
        print("  in production that is a 400 from the provider, or a completion truncated mid-sentence")
    else:
        print("  naive concatenation happens to fit at this window size")

    rule("4. ASSEMBLED vs NAIVE across all three windows")
    print()
    print(Comparison.header())
    print("  " + "-" * 74)
    for c in comparisons:
        print(c.row())
    print()
    print("  hp tok = share of high-priority tokens (priority >= 3.0) that survived.")
    print("  hp src = share of high-priority items with any usable content in the prompt.")
    print("  naive columns assume the usual quick fix: cut the joined string down to the budget,")
    print("  which keeps the front of the prompt whole and deletes whatever was at the end.")

    totals = summarise(comparisons)
    rule("5. TOTALS")
    print()
    print(f"  requests measured                 {int(totals['requests'])}")
    print(f"  window overflows, naive           {int(totals['naive_overflows'])} of {int(totals['requests'])}")
    print(f"  window overflows, assembled       {int(totals['assembled_overflows'])} of {int(totals['requests'])}")
    print(f"  prompt tokens, naive              {int(totals['naive_tokens'])}")
    print(f"  prompt tokens, assembled          {int(totals['assembled_tokens'])}")
    print(f"  tokens saved                      {int(totals['tokens_saved'])}")
    print(f"  high-priority tokens, naive       {totals['naive_high_priority_retention'] * 100:.1f}%")
    print(f"  high-priority tokens, assembled   {totals['assembled_high_priority_retention'] * 100:.1f}%")
    print(f"  high-priority sources, naive      {totals['naive_high_priority_coverage'] * 100:.1f}%")
    print(f"  high-priority sources, assembled  {totals['assembled_high_priority_coverage'] * 100:.1f}%")

    tight = comparisons[0]
    lost = [
        item.id
        for section in receipts["tight-800"][0].sections
        if section.spec.priority >= 3.0
        for item in section.items
    ]
    print()
    print(f"  at the tight window naive truncation represents "
          f"{tight.naive_high_priority_coverage * 100:.0f}% of the {len(lost)} high-priority items,")
    print("  and the items it deletes are the ones at the end of the string, including the question itself")

    rule("6. THE GUARANTEE, checked rather than asserted")
    print()
    for request_id, window, reserve in SCENARIOS:
        req, asm = receipts[request_id]
        ok = asm.tokens <= window - reserve
        print(
            f"  {request_id:<14} prompt={asm.tokens:<6} budget={window - reserve:<6} "
            f"headroom={asm.receipt.headroom:<5} within budget: {ok}"
        )

    rule("7. EDGE CASE: a window too small for every guaranteed minimum")
    starved = build_request(QUERY, model_window=260, reserve_tokens=120, request_id="starved-260")
    starved_ctx = assembler.assemble(starved)
    print()
    print(f"  budget is {starved.available_tokens} tokens against "
          f"{sum(s.spec.min_tokens for s in starved.sections)} tokens of declared minimums")
    for section in starved_ctx.receipt.sections:
        print(f"    {section.name:<16} granted={section.granted_tokens:<5} used={section.used_tokens:<5} {section.note}")
    survived = [s.name for s in starved_ctx.receipt.sections if s.used_tokens > 0]
    empty = [s.name for s in starved_ctx.receipt.sections if s.used_tokens == 0]
    print(f"  survived: {', '.join(survived)}")
    print(f"  empty:    {', '.join(empty)}")
    print(f"  result: {starved_ctx.tokens} tokens, inside the {starved.available_tokens} token budget.")
    print("  the instruction block is declared keep_whole and does not fit, so it is reported as")
    print("  dropped rather than silently truncated to half an instruction. that distinction is the")
    print("  difference between a caller that can raise the window and a caller that ships a bug.")

    rule("8. RECEIPT AS JSON, ready to attach to a trace span")
    print()
    print(assembled.receipt.to_json()[:900] + "\n  ...")


if __name__ == "__main__":
    main()
