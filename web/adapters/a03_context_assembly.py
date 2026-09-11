"""Web adapter for project 03, the context assembly service.

The visitor sets a token budget and optionally a question. The adapter builds
the real six-source request from `workload.py`, assembles it, and prints the
receipt the assembler produced plus the same request under naive concatenation.
"""
from __future__ import annotations

import re

NUMBER = 3
SLUG = "context-assembly"
TITLE = "Context Assembly Service"
TAGLINE = "Give the model a token budget and watch six sources of context compete for it."
WHAT_IT_DOES = """A chat answer is built from several sources at once: the system
instructions, what the assistant remembers about you, documents it looked up, live
tool readings, the conversation so far, and your actual question. Together they are
usually bigger than the space the model has to read them in.

Type a number and that becomes the size of the window. The service works out how much
each source gets, shortens what can be shortened, drops what will not fit, and prints a
receipt explaining every decision. Type a question after the number and the documents
are re-ranked against it, so the sentences that survive are the ones about your question.

Underneath the receipt is the same request done the usual way, by gluing every source
together and cutting the end off when it is too long. That version regularly throws away
the question itself, because the question happens to be at the end of the string."""
INPUT_LABEL = "A token budget, and optionally a question after it"
PLACEHOLDER = "800 why do we keep getting 429 responses at night?"
EXAMPLES = [
    "800",
    "800 why do we keep getting 429 responses at night and will more tokens help?",
    "1600 how do I export my data and how long does the job take?",
    "320",
]
SOURCE = "projects/p03_context_assembly"

DEFAULT_QUERY = "Why do we keep getting 429 responses at night and will more tokens help?"
MIN_WINDOW = 200
MAX_WINDOW = 6000
RESERVE_SHARE = 0.30
PROMPT_PREVIEW_CHARS = 900
# Only a *leading* number is read as the budget. A digit in the middle of a
# sentence is part of the question, not a window size.
_LEADING_NUMBER = re.compile(r"^\s*(\d+)\b")


def _parse(user_input: str):
    """Pull a window size and a question out of whatever the visitor typed."""
    text = (user_input or "").strip()
    if not text:
        text = EXAMPLES[0]
    match = _LEADING_NUMBER.match(text)
    if match:
        window = int(match.group(1))
        question = text[match.end():].strip()
    else:
        window = 800
        question = text
    window = max(MIN_WINDOW, min(MAX_WINDOW, window))
    question = question.strip(" -:,") or DEFAULT_QUERY
    if len(question) > 300:
        question = question[:300]
    return window, question


def _prefix_walk(request):
    """Which sections survive when the joined string is cut down to the budget.

    Mirrors `baseline.prefix_truncation_metrics`, but keeps the per-section
    numbers so the page can name what the cut deleted.
    """
    budget = request.available_tokens
    rows = []
    for section in request.sections:
        if not section.items:
            continue
        budget -= section.spec.header_tokens
        offered = kept = 0
        for item in section.items:
            offered += item.tokens
            take = min(item.tokens, budget) if budget > 0 else 0
            budget -= take
            kept += take
        rows.append((section.name, offered, kept))
    return rows


def run(user_input: str) -> str:
    try:
        from llmkit import EchoLLM, count_tokens

        from projects.p03_context_assembly.assembler import ContextAssembler
        from projects.p03_context_assembly.baseline import compare, naive_prompt
        from projects.p03_context_assembly.workload import build_request

        window, question = _parse(user_input)
        reserve = max(40, min(int(window * RESERVE_SHARE), window - 100))
        request = build_request(question, model_window=window, reserve_tokens=reserve,
                                request_id=f"web-{window}")
        assembled = ContextAssembler(llm=EchoLLM()).assemble(request)
        receipt = assembled.receipt
        naive = naive_prompt(request)
        naive_tokens = count_tokens(naive)
        comparison = compare(request, receipt)

        out = []
        out.append("THE REQUEST")
        out.append(f"  question:            {question}")
        out.append(f"  model window:        {window} tokens")
        out.append(f"  completion reserve:  {reserve} tokens, held back so the model can answer")
        out.append(f"  budget for context:  {request.available_tokens} tokens")
        out.append(f"  sources competing:   {len([s for s in request.sections if s.items])}"
                   f", offering {receipt.offered_tokens} tokens between them")
        out.append("")
        out.append("=" * 78)
        out.append("WHAT THE ASSEMBLER DECIDED")
        out.append("=" * 78)
        out.append("")
        out.append(receipt.render())

        question_record = next((s for s in receipt.sections if s.name == "question"), None)
        if question_record is not None and question_record.used_tokens == 0:
            out.append("")
            out.append("  note: your question is short enough that its slice of the budget fell below the")
            out.append("  smallest useful block, so it is reported as dropped rather than half included.")
            out.append("  a longer question gets a bigger slice, because the grant follows the demand.")

        out.append("")
        out.append("=" * 78)
        out.append("THE PROMPT THAT RECEIPT DESCRIBES")
        out.append("=" * 78)
        out.append("")
        preview = assembled.text
        if len(preview) > PROMPT_PREVIEW_CHARS:
            preview = preview[:PROMPT_PREVIEW_CHARS].rstrip() + "\n  ... (preview cut here, the real prompt continues)"
        for line in preview.splitlines():
            out.append("  " + line)

        out.append("")
        out.append("=" * 78)
        out.append("THE SAME REQUEST, GLUED TOGETHER THE USUAL WAY")
        out.append("=" * 78)
        out.append("")
        out.append(f"  everything joined, no budget awareness:  {naive_tokens} tokens")
        out.append(f"  budget:                                  {request.available_tokens} tokens")
        if comparison.naive_overflow_tokens > 0:
            out.append(f"  over the budget by:                      {comparison.naive_overflow_tokens} tokens")
            out.append("  in production that is a rejected call, or an answer that stops mid-sentence")
        else:
            out.append("  it happens to fit at this window size, so the usual quick fix is not triggered")
        out.append(f"  assembled instead:                       {assembled.tokens} tokens"
                   f", {receipt.headroom} to spare")
        out.append(f"  tokens saved:                            {comparison.tokens_saved}")
        out.append("")
        out.append("  cut the joined string down to the budget and this is what is left of each source:")
        out.append("")
        out.append(f"    {'source':<18}{'offered':>9}{'kept':>8}{'':>4}what the cut did")
        out.append("    " + "-" * 68)
        for name, offered, kept in _prefix_walk(request):
            if kept == 0:
                verdict = "deleted entirely"
            elif kept >= offered:
                verdict = "kept whole"
            else:
                verdict = f"cut mid-source, {kept * 100 // max(1, offered)}% left"
            out.append(f"    {name:<18}{offered:>9}{kept:>8}    {verdict}")
        out.append("")
        out.append(f"  high-value sources still represented:  {comparison.naive_high_priority_coverage * 100:.0f}%"
                   f" cutting the string, {comparison.assembled_high_priority_coverage * 100:.0f}% assembling it")
        out.append("  the cut keeps the front of the prompt whole and deletes whatever ran last, which")
        out.append("  is why the question is the thing at risk. the assembler spends a little of every")
        out.append("  source's budget instead, and writes down what it spent it on.")
        return "\n".join(out)
    except Exception as exc:  # a demo page must never 500
        return f"This demo could not run: {type(exc).__name__}: {exc}"
