"""Web adapter for project 05, the semantic cache with a salience guard.

The page runs the visitor's questions through two caches at once: one that
decides on similarity alone, and the shipped one that also checks the tokens
where a single word flips the meaning. The difference between them is the point.
"""
from __future__ import annotations

import os
import sys
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

NUMBER = 5
SLUG = "semantic-cache"
TITLE = "Semantic Cache Layer"
TAGLINE = "Type two questions and see whether the cache reuses the first answer for the second, and whether it should."

WHAT_IT_DOES = """A cache in front of a language model saves money by reusing an
old answer when a new question means the same thing. The risk is that two
questions can look almost identical and mean opposite things. Free trial against
paid trial. Enable against disable. After 90 days against after 30 days. Serve
the wrong one and the visitor gets a confident, fluent, wrong answer with nothing
to warn them.

Type one question per line. The first is answered and stored. The second is
checked against everything stored, and the page shows you the similarity score,
the closest stored question, and the decision. Twelve answers are already stored
here, so a single line works too.

Every question runs through two caches side by side. One serves on similarity
alone. The shipped one adds a second check over the words that change an answer:
numbers, names, negations, and opposite pairs. Where the two disagree, the page
says which check fired and on which words. At the bottom you get the running hit
rate and tokens saved, plus the same comparison measured live over a labelled set
of twenty six question pairs."""

INPUT_LABEL = "Type one question per line, up to three"
PLACEHOLDER = "How do I rotate a token after 90 days?\nHow do I rotate a token after 30 days?"
EXAMPLES = [
    "How do I rotate a token after 90 days?\nHow do I rotate a token after 30 days?",
    "On list endpoints, what is the maximum page size?\n"
    "What is the minimum page size on list endpoints?",
    "Which role is allowed to delete a workspace account?\n"
    "Which role is not allowed to delete a workspace?",
    "What is the default per workspace rate limit?",
]
SOURCE = "projects/p05_semantic_cache"

MAX_QUESTIONS = 3

_LOCK = threading.Lock()
_WARM = None


def _warm_data():
    """Embed the twelve stored answers once per process."""
    global _WARM
    with _LOCK:
        if _WARM is None:
            from llmkit import get_embedder
            from projects.p05_semantic_cache.fixtures import BASE_ANSWERS, NAMESPACE

            embedder = get_embedder()
            vectors = {q: embedder.embed_one(q) for q in BASE_ANSWERS}
            _WARM = (embedder, dict(BASE_ANSWERS), NAMESPACE, vectors)
        return _WARM


def _build(embedder, answers, namespace, vectors, use_guard: bool):
    from projects.p05_semantic_cache.cache import SemanticCache
    from projects.p05_semantic_cache.salience import SalienceGuard

    cache = SemanticCache(
        threshold=0.80,
        use_guard=use_guard,
        embedder=embedder,
        guard=SalienceGuard(ignore_entities=("meridian",)),
    )
    for query, answer in answers.items():
        cache.put(namespace, query, answer, vector=list(vectors[query]))
    return cache


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
    from llmkit import count_tokens, get_llm, user as user_message

    raw = user_input if (user_input or "").strip() else EXAMPLES[0]
    questions = [" ".join(line.split()) for line in raw.splitlines()]
    questions = [q[:200] for q in questions if q][:MAX_QUESTIONS]
    if not questions:
        questions = [q for q in EXAMPLES[0].splitlines() if q.strip()]

    embedder, answers, namespace, vectors = _warm_data()
    guarded = _build(embedder, answers, namespace, vectors, use_guard=True)
    naive = _build(embedder, answers, namespace, vectors, use_guard=False)
    llm = get_llm()

    out = []
    add = out.append
    add(f"CACHE STATE   namespace {namespace!r}, {guarded.size(namespace)} answers stored,")
    add(f"              similarity threshold {guarded.threshold:.2f}, "
        f"guard checks numbers, names, negation, opposites")

    tokens_spent = 0
    tokens_saved = 0
    model_calls = 0
    false_hits_avoided = 0

    for n, question in enumerate(questions, start=1):
        add("")
        add(f"REQUEST {n}")
        add(f"  asked   : {question}")

        result = guarded.lookup(namespace, question)
        loose = naive.lookup(namespace, question)
        nearest = result.entry or result.candidate or loose.entry or loose.candidate
        similarity = max(result.similarity, loose.similarity)

        add(f"  nearest : {_clip(nearest.query, 74) if nearest else 'nothing stored yet'}")
        add(f"  similar : {similarity:.3f}   "
            f"({'above' if similarity >= guarded.threshold else 'below'} the "
            f"{guarded.threshold:.2f} threshold)")

        if result.hit and result.entry is not None:
            saved = result.entry.total_tokens
            tokens_saved += saved
            add("  guard   : salient words agree, safe to reuse")
            add(f"  DECISION: HIT, served from cache, no model call, {saved} tokens saved")
            for line in _wrap(result.entry.answer, 74):
                add(f"            {line}")
        else:
            if result.reason == "blocked_by_salience" and result.salience is not None:
                report = result.salience
                add(f"  guard   : REFUSED, {report.check} check fired")
                add(f"            incoming has {sorted(report.incoming) or 'nothing'}, "
                    f"stored has {sorted(report.cached) or 'nothing'}")
                add("  DECISION: MISS on purpose. Similar enough to serve, different enough")
                add("            to be wrong, so it goes to the model instead.")
            elif result.reason == "below_threshold":
                add("  guard   : not reached, similarity alone already said no")
                add("  DECISION: MISS, nothing stored is close enough, so the model runs")
            else:
                add(f"  guard   : not reached ({result.reason})")
                add("  DECISION: MISS, the model runs")

            response = llm.complete([user_message(question)])
            model_calls += 1
            tokens_spent += response.total_tokens
            guarded.put(namespace, question, response.text,
                        response.prompt_tokens, response.completion_tokens)
            naive.put(namespace, question, response.text,
                      response.prompt_tokens, response.completion_tokens)
            add(f"            model answered in {response.total_tokens} tokens, "
                "now stored for next time")

        if loose.hit and not result.hit and loose.entry is not None:
            false_hits_avoided += 1
            add("  WITHOUT THE GUARD, a similarity-only cache would have answered your")
            add("  question with this stored answer instead:")
            for line in _wrap(loose.entry.answer, 72):
                add(f"      {line}")
            add(f"  stored against the question: {_clip(loose.entry.query, 60)}")

    stats = guarded.stats
    add("")
    add("RUNNING TOTALS FOR THIS PAGE VIEW")
    add(f"  lookups                  : {stats.lookups}")
    add(f"  served from cache        : {stats.hits} ({stats.hit_rate:.0%} hit rate)")
    add(f"  refused by the guard     : {stats.blocked_by_salience}")
    add(f"  too different to reuse   : {stats.misses_below_threshold}")
    add(f"  model calls made         : {model_calls}")
    add(f"  tokens spent on the model: {tokens_spent}")
    add(f"  tokens saved by cache    : {tokens_saved}")
    total = tokens_spent + tokens_saved
    if total:
        add(f"  spend avoided            : {tokens_saved / total:.0%} of the tokens this "
            "page would otherwise have cost")
    if false_hits_avoided:
        add(f"  wrong answers prevented  : {false_hits_avoided}")
    add(f"  average lookup cost      : {stats.avg_lookup_ms:.3f} ms")

    add("")
    add("THE SAME COMPARISON OVER A LABELLED SET, MEASURED RIGHT NOW")
    try:
        from projects.p05_semantic_cache.fixtures import NEAR_MISSES, PARAPHRASES
        from projects.p05_semantic_cache.metrics import run_probes

        off = run_probes(0.80, False, embedder)
        on = run_probes(0.80, True, embedder)
        add(f"  {len(PARAPHRASES)} reworded questions that should reuse an answer, "
            f"{len(NEAR_MISSES)} lookalikes that must not")
        add("")
        add(f"  {'':<12}{'reuse rate':>12}{'wrong answers':>15}{'correctness':>13}")
        add("  " + "-" * 52)
        add(f"  {'similarity':<12}{off.recall:>12.2f}{off.false_hits:>15}"
            f"{off.precision:>13.2f}")
        add(f"  {'plus guard':<12}{on.recall:>12.2f}{on.false_hits:>15}"
            f"{on.precision:>13.2f}")
        add("")
        add(f"  The guard stops {off.false_hits - on.false_hits} of {off.false_hits} wrong "
            f"answers and still reuses {on.recall:.0%} of the genuine repeats.")
        add("  Raising the similarity bar instead would throw away most of the reuse,")
        add("  because the closest lookalike scores higher than the loosest real repeat.")
    except Exception as exc:
        add(f"  comparison unavailable: {type(exc).__name__}: {exc}")

    return "\n".join(out)[:8000]
