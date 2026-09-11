"""Web adapter for project 15, the self-correcting RAG agent.

The page shows the whole loop: each attempt, the query it used, the critique it
scored, the decision to try harder, and the final answer or refusal. The same
question is also put to the one-shot pipeline from project 01, so the visitor can
see what the loop bought.
"""
from __future__ import annotations

import os
import sys
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

NUMBER = 15
SLUG = "self-correcting-rag"
TITLE = "Self-Correcting RAG Agent"
TAGLINE = "Ask anything, including something the documents cannot answer, and watch the system try again, then admit defeat."

WHAT_IT_DOES = """A normal question answering system gets one attempt at finding
evidence. If your wording does not match the wording in the documents, it either
answers from the wrong page or gives up, and it cannot tell those two apart.

This one grades its own evidence and tries again. There are four attempts, in
order of cost. Search as asked. Rewrite the question and search again. Widen the
net. Finally, go to a second document set outside its own index. After each
attempt it scores what it found on how much of your question the text covers,
whether the passages agree with each other, and a pass or fail verdict from the
model. It stops early once it is confident, and it keeps whichever attempt scored
best, so trying again can never make the answer worse.

Below a set confidence floor it refuses to answer at all. That is the part worth
watching, so ask it something the documents genuinely do not contain, like which
regions launch in 2028. The page also runs the same question through the one shot
system from project 01, so you can compare a refusal against a confident guess."""

INPUT_LABEL = "Ask a question, or try one that cannot be answered"
PLACEHOLDER = "What caused the eu-west outage in March 2026?"
EXAMPLES = [
    "What caused the eu-west outage in March 2026?",
    "Which new regions will Meridian launch in 2028?",
    "How are webhook deliveries authenticated?",
    "How do I stop a retried payment request from charging twice?",
]
SOURCE = "projects/p15_self_correcting_rag"

_ACTION_LABEL = {
    "retrieve": "search as asked",
    "rewrite": "reword and retry",
    "widen": "widen the net",
    "fallback": "search outside the index",
}

_LOCK = threading.Lock()
_PARTS = None


def _parts():
    """Build the indexes and the agent once per process."""
    global _PARTS
    with _LOCK:
        if _PARTS is None:
            from projects.p01_rag_pipeline.pipeline import RagPipeline, preset
            from projects.p15_self_correcting_rag.agent import SelfCorrectingRAG
            from projects.p15_self_correcting_rag.tools import FallbackSearch

            pipeline = RagPipeline.build(config=preset("hybrid+rerank"))
            agent = SelfCorrectingRAG(pipeline, search_tool=FallbackSearch())
            _PARTS = (pipeline, agent)
        return _PARTS


def _clip(text, width: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "\u2026"


def _wrap(text: str, width: int):
    lines, current = [], ""
    for word in str(text).split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def run(user_input: str) -> str:
    try:
        return _run(user_input)
    except Exception as exc:  # never take the page down
        return f"This demo could not finish: {type(exc).__name__}: {exc}"


def _run(user_input: str) -> str:
    question = " ".join((user_input or "").split()) or EXAMPLES[0]
    if len(question) > 400:
        question = question[:400]

    pipeline, agent = _parts()
    result = agent.answer(question)

    out = []
    add = out.append
    add("QUESTION")
    add(f"  {question}")
    add("")
    add(f"THE LOOP   stop early at confidence {agent.accept_above:.2f}, "
        f"refuse below {agent.abstain_below:.2f}")
    add(f"           confidence = 0.55 word coverage + 0.15 passage agreement "
        "+ 0.30 model verdict")

    ladder = ["retrieve", "rewrite", "widen", "fallback"]
    for step in result.steps:
        crit = step.critique
        add("")
        add(f"  ATTEMPT {step.index}  {_ACTION_LABEL.get(step.action, step.action)}")
        if step.action == "rewrite":
            add(f"    reworded as : {_clip(step.query, 74)}")
        elif step.action == "widen":
            add(f"    same question, {step.query} instead of k={agent.k}")
        elif step.action == "fallback":
            add(f"    second document set: {_clip(step.query, 58)}")
        add(f"    found       : {_clip(', '.join(step.evidence_ids) or 'nothing', 74)}")
        add(f"    critique    : coverage {crit.coverage:.3f}   agreement "
            f"{crit.agreement:.3f}   model says {crit.judge_verdict} "
            f"({crit.judge_score:.2f})")
        if crit.missing_terms:
            add(f"    not found   : {_clip(', '.join(crit.missing_terms[:5]), 68)}")
        add(f"    CONFIDENCE  : {crit.confidence:.3f}")
        if step.accepted:
            add(f"    decision    : good enough, stop here")
        elif step.index < len(result.steps):
            nxt = ladder[step.index] if step.index < len(ladder) else "next"
            add(f"    decision    : not confident enough, escalate to "
                f"{_ACTION_LABEL.get(nxt, nxt)}")
        else:
            add("    decision    : still not confident, and the ladder is exhausted")

    winning = result.steps[result.winning_step - 1]
    add("")
    add("DECISION")
    add(f"  attempts run     : {result.step_count} of {agent.max_steps}")
    add(f"  best evidence    : attempt {result.winning_step}, "
        f"{_ACTION_LABEL.get(winning.action, winning.action)}")
    add(f"  final confidence : {result.confidence:.3f}")
    if result.corrected:
        add("  the first search was not the one that answered, so the loop repaired itself")

    if result.abstained:
        add(f"  {result.confidence:.3f} is below the {agent.abstain_below:.2f} floor, "
            "so it refuses rather than guess")
        add("")
        add("ANSWER")
        add(f"  {result.answer.text}")
    else:
        band = ("confident" if result.confidence >= agent.accept_above
                else "answering, but flagged low confidence")
        add(f"  {result.confidence:.3f} clears the {agent.abstain_below:.2f} floor -> {band}")
        add("")
        add("ANSWER")
        for line in _wrap(result.answer.text, 82):
            add(f"  {line}")
        cites = result.answer.valid_citations
        if cites:
            add("")
            add("  sources, each checked against the evidence that was sent:")
            for cite in cites:
                add(f"    {cite.marker:<5}{_clip(cite.chunk_id, 30):<31}"
                    f"{_clip(cite.title or '-', 40)}")
        else:
            add("  no source marker survived checking, so nothing above is attributed")

    # -- the honest comparison -------------------------------------------
    single = pipeline.ask(question)
    add("")
    add("THE SAME QUESTION WITHOUT THE LOOP   (project 01, one attempt only)")
    if single.refused:
        add(f"  one shot: refused, {single.reason}, evidence quality "
            f"{single.grounding:.3f}")
    else:
        add(f"  one shot: answered from {', '.join(single.cited_doc_ids) or 'no source'}")
        for line in _wrap(_clip(single.text, 300), 78):
            add(f"    {line}")

    loop_docs = result.cited_doc_ids
    if result.abstained and not single.refused:
        add("  The loop refused where one shot answered. On a question with no answer in")
        add("  the documents, that refusal is the correct outcome.")
    elif not result.abstained and single.refused:
        add("  The loop found usable evidence where one shot gave up.")
    elif loop_docs and loop_docs != single.cited_doc_ids:
        add(f"  Different sources: the loop cited {', '.join(loop_docs)}, one shot cited "
            f"{', '.join(single.cited_doc_ids) or 'none'}.")
    else:
        add("  Both landed in the same place, which is what should happen when the first")
        add("  search was already right.")

    return "\n".join(out)[:8000]
