"""The bounded control loop: retrieve, critique, decide, act.

Why bounded, and bounded hard
-----------------------------
An agent that decides its own stopping condition from its own confidence will,
on the questions where its confidence estimate is broken, loop until something
else stops it. The something else is usually a timeout, a rate limit or a bill.
`max_steps` is enforced by the loop itself and is not advisory: the ladder is
finite, each rung is attempted at most once, and the loop exits with the best
evidence it has seen even if nothing ever cleared the acceptance bar.

The escalation ladder, in order
-------------------------------
1. **Retrieve.** Baseline hybrid retrieval with reranking.
2. **Rewrite.** Up to three reformulations (decomposition, keywords-only,
   hypothetical answer), fused by rank. Cheap and fixes the most common failure,
   which is phrasing.
3. **Widen k.** Double the candidate depth and the returned k. This is second,
   not first, because a wider net over the same bad query mostly returns more of
   the same bad results, and it costs context budget on the way.
4. **Fallback search.** A `SearchTool` outside the primary index. Last because it
   is the only rung with an external dependency and, in a real deployment, a
   per-call cost.

Each rung keeps the best evidence seen so far by confidence, so escalation can
never make the final answer worse than an earlier step. That is a real risk
otherwise: a rewrite that retrieves confidently wrong material would otherwise
overwrite a merely mediocre first pass.

Abstention
----------
Below `abstain_below`, the agent says it cannot answer. This is the behaviour
that makes the rest of it trustworthy, and it is the behaviour most likely to be
quietly removed when someone complains that the bot says "I don't know" too
often. Between `abstain_below` and `accept_above` it answers but marks the result
low confidence, because "probably this, and I am not sure" is a legitimate and
useful answer that a binary gate throws away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from llmkit import LLMProvider, ScoredChunk, Tracer, get_llm, tracer as default_tracer

from projects.p01_rag_pipeline.answer import Answer, CitedAnswerer
from projects.p01_rag_pipeline.pipeline import RagPipeline

from .critique import Critique, RetrievalCritic
from .rewrite import QueryRewriter, fuse
from .tools import FallbackSearch, SearchTool

ABSTAIN_TEXT = "I cannot answer that from the sources available."


@dataclass
class Step:
    index: int
    action: str
    query: str
    evidence_ids: List[str]
    critique: Critique
    accepted: bool = False

    def as_line(self) -> str:
        mark = "accept" if self.accepted else "continue"
        return (f"  step {self.index} {self.action:<12} {self.critique.as_line()} -> {mark}\n"
                f"           query: {self.query[:88]}")


@dataclass
class AgentResult:
    question: str
    answer: Answer
    confidence: float
    abstained: bool
    steps: List[Step] = field(default_factory=list)
    escalations: List[str] = field(default_factory=list)
    evidence: List[ScoredChunk] = field(default_factory=list)
    winning_step: int = 1
    trace_id: Optional[str] = None

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def corrected(self) -> bool:
        """True when the evidence that was answered from came from a later rung.

        Keyed on which step *won*, not on how many steps ran. Counting steps
        would mark a question as "corrected" whenever the loop escalated and then
        fell back on its first result, which inflates the headline claim of the
        whole project with questions the loop did not actually repair.
        """
        return self.winning_step > 1 and not self.abstained

    @property
    def cited_doc_ids(self) -> List[str]:
        return self.answer.cited_doc_ids


class SelfCorrectingRAG:
    def __init__(
        self,
        pipeline: RagPipeline,
        critic: Optional[RetrievalCritic] = None,
        rewriter: Optional[QueryRewriter] = None,
        search_tool: Optional[SearchTool] = None,
        llm: Optional[LLMProvider] = None,
        max_steps: int = 4,
        # Both defaults come from the sweep in `evaluate.calibrate_abstention`,
        # printed by the demo. `abstain_below=0.50` maximised correct decisions
        # across the 19 answerable and 8 adversarial questions; `accept_above`
        # only controls early stopping, so it is set high enough that a strong
        # first retrieval short-circuits the ladder and everything else pays for
        # at least one repair attempt.
        accept_above: float = 0.70,
        abstain_below: float = 0.50,
        k: int = 5,
        tracer: Optional[Tracer] = None,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if not 0.0 <= abstain_below <= accept_above <= 1.0:
            raise ValueError("need 0 <= abstain_below <= accept_above <= 1")
        self.pipeline = pipeline
        self.llm = llm or get_llm()
        df, corpus_size = pipeline.retriever.document_frequencies()
        self.critic = critic or RetrievalCritic(
            llm=self.llm, document_frequencies=df, corpus_size=corpus_size
        )
        self.rewriter = rewriter or QueryRewriter(llm=self.llm)
        self.search_tool = search_tool if search_tool is not None else FallbackSearch()
        self.max_steps = max_steps
        self.accept_above = accept_above
        self.abstain_below = abstain_below
        self.k = k
        self.tracer = tracer or default_tracer
        # A dedicated answerer so the escalation path can cite fallback chunks the
        # primary pipeline's index has never seen. Its grounding floor is 0.0
        # because the agent's own confidence gate has already made that decision,
        # and having two independent refusal gates disagree is how a system ends
        # up refusing questions it has good evidence for.
        self.answerer = CitedAnswerer(
            llm=self.llm, grounding_floor=0.0, max_evidence=k,
            document_frequencies=df, corpus_size=corpus_size,
        )

    # -- rungs -----------------------------------------------------------
    def _retrieve(self, question: str) -> List[ScoredChunk]:
        return self.pipeline.retrieve(question, k=self.k)

    def _rewrite(self, question: str, evidence: Sequence[ScoredChunk]) -> tuple:
        """Retrieve for each reformulation and fuse, deeper than the final k.

        Two details matter here and both were found by watching this rung do
        nothing. Each reformulation retrieves `2 * k` rather than `k`, because
        fusing four top-5 lists that mostly agree just returns the same top 5.
        And the original ranking is deliberately *not* included in the fusion:
        including it made the fused result identical to step 1 on every question
        in the evaluation set, which is a rung that costs model calls and changes
        nothing. The first result is not lost by leaving it out, because the loop
        keeps the best evidence seen across all rungs regardless.
        """
        variants = self.rewriter.rewrite(question, evidence)
        if not variants:
            return [], "no reformulation available"
        result_sets = [self.pipeline.retrieve(v.query, k=self.k * 2) for v in variants]
        fused = fuse(result_sets, k=self.k)
        return fused, ", ".join(f"{v.strategy}:{v.query[:40]}" for v in variants)

    def _widen(self, question: str) -> List[ScoredChunk]:
        return self.pipeline.retrieve(question, k=self.k * 2)

    def _fallback(self, question: str) -> List[ScoredChunk]:
        return list(self.search_tool.search(question, k=self.k))

    # -- loop ------------------------------------------------------------
    def answer(self, question: str) -> AgentResult:
        with self.tracer.span("agent.run", question=question, max_steps=self.max_steps) as root:
            steps: List[Step] = []
            escalations: List[str] = []
            best_evidence: List[ScoredChunk] = []
            best_critique: Optional[Critique] = None
            best_step = 1
            last_query = question

            ladder = ["retrieve", "rewrite", "widen", "fallback"][: self.max_steps]
            for index, action in enumerate(ladder, start=1):
                with self.tracer.span(f"agent.{action}", step=index) as span:
                    if action == "retrieve":
                        evidence, note = self._retrieve(question), question
                    elif action == "rewrite":
                        evidence, note = self._rewrite(question, best_evidence)
                    elif action == "widen":
                        evidence, note = self._widen(question), f"k={self.k * 2}"
                    else:
                        evidence, note = self._fallback(question), self.search_tool.name
                    last_query = note if isinstance(note, str) else question

                    with self.tracer.span("agent.critique") as critique_span:
                        critique = self.critic.critique(question, evidence)
                        critique_span.attributes.update(
                            {
                                "confidence": round(critique.confidence, 4),
                                "coverage": round(critique.coverage, 4),
                                "agreement": round(critique.agreement, 4),
                                "judge": critique.judge_verdict,
                            }
                        )

                    accepted = critique.confidence >= self.accept_above
                    step = Step(
                        index=index, action=action, query=last_query,
                        evidence_ids=[e.chunk.id for e in evidence],
                        critique=critique, accepted=accepted,
                    )
                    steps.append(step)
                    span.attributes.update(
                        {"confidence": round(critique.confidence, 4), "accepted": accepted,
                         "evidence": len(evidence)}
                    )

                    # Monotonic best: escalation can never degrade the final answer.
                    if best_critique is None or critique.confidence > best_critique.confidence:
                        best_evidence, best_critique, best_step = list(evidence), critique, index

                    if accepted:
                        break
                    if index < len(ladder):
                        escalations.append(ladder[index])

            confidence = best_critique.confidence if best_critique else 0.0
            root.attributes.update(
                {"steps": len(steps), "confidence": round(confidence, 4),
                 "escalations": escalations}
            )

            if confidence < self.abstain_below or not best_evidence:
                root.attributes["outcome"] = "abstained"
                return AgentResult(
                    question=question,
                    answer=Answer(question=question, text=ABSTAIN_TEXT, refused=True,
                                  reason="below_confidence_floor", grounding=confidence,
                                  evidence=list(best_evidence)),
                    confidence=confidence, abstained=True, steps=steps,
                    escalations=escalations, evidence=list(best_evidence),
                    winning_step=best_step, trace_id=root.trace_id,
                )

            with self.tracer.span("agent.answer") as answer_span:
                answer = self.answerer.answer(question, best_evidence)
                answer_span.attributes.update(
                    {"refused": answer.refused, "reason": answer.reason,
                     "llm.total_tokens": answer.total_tokens}
                )
            root.attributes["outcome"] = "answered"
            return AgentResult(
                question=question, answer=answer, confidence=confidence,
                abstained=answer.refused, steps=steps, escalations=escalations,
                evidence=list(best_evidence), winning_step=best_step,
                trace_id=root.trace_id,
            )

    def explain(self, result: AgentResult) -> str:
        lines = [f"question: {result.question}"]
        lines += [step.as_line() for step in result.steps]
        lines.append(
            f"  final: confidence={result.confidence:.3f} "
            f"abstained={result.abstained} steps={result.step_count} "
            f"winning_step={result.winning_step} ({result.steps[result.winning_step - 1].action}) "
            f"cited={result.cited_doc_ids or 'none'}"
        )
        lines.append(f"  answer: {result.answer.text[:160]}")
        return "\n".join(lines)
