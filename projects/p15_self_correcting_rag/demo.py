"""End-to-end demo for the self-correcting RAG agent.

Run:  python3 projects/p15_self_correcting_rag/demo.py
Every number printed here is measured at run time. Nothing is hard coded.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmkit import Tracer  # noqa: E402
from llmkit.corpus import gold_questions  # noqa: E402

from projects.p01_rag_pipeline.pipeline import RagPipeline, preset  # noqa: E402
from projects.p15_self_correcting_rag.adversarial import adversarial_questions  # noqa: E402
from projects.p15_self_correcting_rag.agent import SelfCorrectingRAG  # noqa: E402
from projects.p15_self_correcting_rag.critique import RetrievalCritic  # noqa: E402
from projects.p15_self_correcting_rag.evaluate import (  # noqa: E402
    calibrate_abstention, escalation_histogram, evaluate_agent, evaluate_single_shot,
    format_acceptance, format_calibration, format_confusion, rows_with_outcome,
    sweep_acceptance, CORRECT, CORRECTED, WRONG,
)
from projects.p15_self_correcting_rag.fallback_corpus import FALLBACK_GOLD  # noqa: E402
from projects.p15_self_correcting_rag.tools import FallbackSearch, NullSearch  # noqa: E402


def rule(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


def main() -> None:
    tracer = Tracer("p15_self_correcting_rag")
    pipeline = RagPipeline.build(config=preset("hybrid+rerank"), tracer=tracer)
    search = FallbackSearch()
    agent = SelfCorrectingRAG(pipeline, search_tool=search, tracer=tracer)

    rule("1. A question the first retrieval already answers")
    print(agent.explain(agent.answer("How are webhook deliveries authenticated?")))

    rule("2. A question the first retrieval gets wrong and the ladder repairs")
    escalated = None
    for gold in gold_questions():
        result = agent.answer(gold["question"])
        if result.corrected and gold["doc_id"] in result.cited_doc_ids:
            escalated = result
            break
    print(agent.explain(escalated) if escalated else
          "  every gold question was settled on the first retrieval")

    rule("3. A question only the fallback tool can answer")
    fallback_case = FALLBACK_GOLD[0]
    reached = agent.answer(fallback_case["question"])
    print(agent.explain(reached))
    print(f"  fallback tool called {search.calls} times so far")
    print(f"  cited from the second corpus: "
          f"{[c.chunk_id for c in reached.answer.valid_citations]}")

    rule("4. Abstention on an unanswerable question")
    print(agent.explain(agent.answer(
        "What is the exact discount on a three year prepaid Enterprise contract?")))

    rule("5. Abstention still happens when the fallback tool finds nothing")
    null = NullSearch()
    blind = SelfCorrectingRAG(pipeline, search_tool=null, tracer=tracer)
    result = blind.answer("Which new regions will Meridian launch in 2028?")
    print(agent.explain(result))
    print(f"  null tool called {null.calls} times, abstained={result.abstained}")

    rule("6. max_steps is enforced")
    for limit in (1, 2, 4):
        capped = SelfCorrectingRAG(pipeline, search_tool=FallbackSearch(),
                                   max_steps=limit, tracer=tracer)
        capped_result = capped.answer("What is the exact discount on a three year contract?")
        print(f"  max_steps={limit} -> steps taken={capped_result.step_count} "
              f"actions={[s.action for s in capped_result.steps]}")

    rule("7. Calibrating the abstention floor")
    answerable = list(gold_questions()) + list(FALLBACK_GOLD)
    unanswerable = adversarial_questions()
    calibration = calibrate_abstention(agent, answerable, unanswerable)
    print(f"  {len(answerable)} answerable, {len(unanswerable)} unanswerable, "
          "score = correct answers plus justified refusals\n")
    print(format_calibration(calibration))
    print(f"\n  shipped default abstain_below = {agent.abstain_below:.2f}")

    df, corpus_size = pipeline.retriever.document_frequencies()

    def fresh_agent(accept_above: float) -> SelfCorrectingRAG:
        return SelfCorrectingRAG(
            pipeline,
            critic=RetrievalCritic(document_frequencies=df, corpus_size=corpus_size),
            search_tool=FallbackSearch(), accept_above=accept_above, tracer=tracer,
        )

    print("\n  early stopping is an accuracy against cost dial, swept separately:\n")
    print(format_acceptance(sweep_acceptance(fresh_agent, answerable)))
    print(f"\n  shipped default accept_above = {agent.accept_above:.2f}")

    rule("8. Evaluation: agent versus single-shot RAG")

    agent_answerable, agent_rows = evaluate_agent(agent, answerable, "self-correcting")
    single_answerable, single_rows = evaluate_single_shot(pipeline, answerable, "single-shot (p01)")
    agent_unanswerable, agent_adv_rows = evaluate_agent(agent, unanswerable, "self-correcting")
    single_unanswerable, single_adv_rows = evaluate_single_shot(
        pipeline, unanswerable, "single-shot (p01)"
    )

    print(f"  answerable questions: {len(answerable)} "
          f"({len(gold_questions())} in the primary corpus, "
          f"{len(FALLBACK_GOLD)} only in the fallback corpus)\n")
    print(format_confusion([agent_answerable, single_answerable]))

    print(f"\n  unanswerable and adversarial questions: {len(unanswerable)}\n")
    print(format_confusion([agent_unanswerable, single_unanswerable]))

    rule("9. Where the ladder settled each answerable question")
    print(f"  winning step -> question count: {escalation_histogram(agent_rows)}")
    corrected = rows_with_outcome(agent_rows, CORRECTED)
    print(f"  answered from a later rung than the first retrieval: {len(corrected)}")
    for row in corrected:
        print(f"    - step {row.winning_step} conf={row.confidence:.3f} "
              f"cited={row.cited} | {row.question}")

    single_right = {
        row.question for row in single_rows if row.outcome in (CORRECT, CORRECTED)
    }
    agent_right = {
        row.question for row in agent_rows if row.outcome in (CORRECT, CORRECTED)
    }
    gained = sorted(agent_right - single_right)
    lost = sorted(single_right - agent_right)
    print(f"\n  questions the agent gets right that single-shot does not: {len(gained)}")
    for question in gained:
        print(f"    + {question}")
    print(f"  questions single-shot gets right that the agent does not: {len(lost)}")
    for question in lost:
        print(f"    - {question}")

    rule("10. Remaining failures")
    for label, rows in (("agent", agent_rows), ("single-shot", single_rows)):
        wrong = rows_with_outcome(rows, WRONG)
        print(f"  {label}: {len(wrong)} wrong on answerable questions")
        for row in wrong:
            print(f"    - {row.question[:70]} cited={row.cited or 'none'}")
    for label, rows in (("agent", agent_adv_rows), ("single-shot", single_adv_rows)):
        wrong = rows_with_outcome(rows, WRONG)
        print(f"  {label}: {len(wrong)} answered an unanswerable question")
        for row in wrong:
            print(f"    - [{row.kind}] {row.question[:64]}")

    rule("11. Trace summary")
    print(f"  judge calls made: {agent.critic.judge_calls}")
    for name, agg in sorted(tracer.summary().items()):
        print(f"  {name:<18} calls={agg['count']:<5} avg_ms={agg['avg_ms']:<8} "
              f"tokens={agg['tokens']}")


if __name__ == "__main__":
    main()
