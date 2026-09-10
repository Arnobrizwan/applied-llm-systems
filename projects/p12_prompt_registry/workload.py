"""A simulated but real workload: two prompt formats, judged on actual output.

The challenger bundles a prompt change and a config change, which is what a real
prompt release usually is. The control shows the single best-matching document
and asks for an answer; the challenger shows three tagged blocks and asks the
model to quote and cite the one it used. `llmkit.EchoLLM` genuinely behaves
differently on the two, because its grounding path selects among the evidence
blocks it is given, so more blocks means a real chance of picking a better one
and a real chance of picking a worse one.

Success is not simulated. Each question comes from `llmkit.corpus` with a gold
phrase the correct answer has to contain, and an outcome is a success when the
model's actual output contains that phrase. Nothing here draws a success from a
random number generator, which is the difference between demonstrating an A/B
system and drawing a picture of one.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

from llmkit import BM25, estimate_cost
from llmkit.corpus import by_id, gold_questions

from .outcomes import Outcome, OutcomeStore
from .registry import PromptRegistry
from .splitter import TrafficSplitter

PROMPT_NAME = "support_answer"

CONTROL_TEMPLATE = "You are a support assistant. Answer using the note below.\n{evidence}"

CHALLENGER_TEMPLATE = (
    "You are a support assistant. Answer the user's question using the evidence blocks below.\n"
    "Each block carries a tag. Quote the block that supports your answer and cite its tag.\n"
    "{evidence}"
)

BASE_CONFIG = {"temperature": 0.0, "max_output_tokens": 300}
# How many evidence blocks to retrieve lives in the config rather than in a flag
# next to the call site, so it is inside the version hash: the same template
# served one document and served three is two different systems and has to be two
# different version ids.
CONTROL_CONFIG = dict(BASE_CONFIG, evidence_blocks=1)
CHALLENGER_CONFIG = dict(BASE_CONFIG, evidence_blocks=3)
COST_TIER = "small"  # the demo runs on a free provider; this prices it as if it did not


def _index() -> BM25:
    index = BM25()
    for doc_id, doc in by_id().items():
        index.add(doc_id, doc.text)
    return index


def _evidence(index: BM25, question: str, k: int) -> str:
    docs = by_id()
    hits = index.search(question, k=k)
    if not hits:  # pragma: no cover - the corpus always matches something
        return ""
    blocks = []
    for ordinal, (doc_id, _) in enumerate(hits, start=1):
        text = docs[doc_id].text
        blocks.append(f"[S{ordinal}] {text}")
    return "\n".join(blocks)


def register_variants(registry: PromptRegistry) -> Tuple[str, str]:
    """Register both variants and return their content-hash version ids."""
    control = registry.register(
        PROMPT_NAME,
        CONTROL_TEMPLATE,
        variables=["evidence"],
        config=CONTROL_CONFIG,
        author="arnob",
        notes="baseline: single best-matching document, no citation contract",
    )
    challenger = registry.register(
        PROMPT_NAME,
        CHALLENGER_TEMPLATE,
        variables=["evidence"],
        config=CHALLENGER_CONFIG,
        author="arnob",
        notes="challenger: three evidence blocks plus an explicit citation instruction",
    )
    return control.version_id, challenger.version_id


def question_for(unit_id: str) -> Dict[str, str]:
    """Deterministically pick a gold question for a unit.

    Hashed rather than round-robined so question choice is independent of arm
    assignment. Round-robin over a list while assigning arms by hash can
    correlate the two and hand one arm the easier questions.
    """
    gold = gold_questions()
    digest = hashlib.sha256(f"question:{unit_id}".encode("utf-8")).digest()
    return gold[int.from_bytes(digest[:4], "big") % len(gold)]


def run_workload(
    registry: PromptRegistry,
    splitter: TrafficSplitter,
    store: OutcomeStore,
    unit_ids: Sequence[str],
    llm,
    experiment: Optional[str] = None,
) -> OutcomeStore:
    """Route each unit through the splitter and record what actually happened."""
    index = _index()
    experiment_name = experiment or splitter.experiment
    for unit_id in unit_ids:
        arm = splitter.assign(unit_id)
        version = registry.get(arm.version_id)
        gold = question_for(unit_id)
        blocks = int(version.config_dict.get("evidence_blocks", 3))
        evidence = _evidence(index, gold["question"], k=blocks)
        system = version.render(evidence=evidence)

        response = llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": gold["question"]}]
        )
        success = gold["must_contain"].lower() in response.text.lower()
        store.record(
            Outcome(
                experiment=experiment_name,
                arm=arm.name,
                version_id=version.version_id,
                unit_id=unit_id,
                success=success,
                latency_ms=round(response.latency_ms, 4),
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost_usd=estimate_cost(COST_TIER, response.prompt_tokens, response.completion_tokens),
                detail=gold["doc_id"],
            )
        )
    return store


def unit_ids(count: int, prefix: str = "user") -> List[str]:
    return [f"{prefix}-{i:05d}" for i in range(count)]
