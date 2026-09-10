"""Questions that have no answer in any corpus this system can reach.

An evaluation set made only of answerable questions measures how often a system
is right and says nothing about how often it makes something up. That second
number is the one that gets a system pulled from production, and the only way to
measure it is to ask questions whose correct answer is "I cannot answer that".

Four kinds are represented, because they fail in different ways:

**Out of domain.** Nothing in the corpus is even about this topic. The retriever
returns its least-irrelevant chunks and a naive system answers from them.

**Adjacent but absent.** The topic is plausibly in scope and the specific fact is
not. These are the dangerous ones: retrieval returns genuinely on-topic chunks
with high lexical overlap, so a coverage-only confidence signal is happy, and the
answer is invented from material that reads as relevant.

**False premise.** The question presupposes something the corpus contradicts.
Answering the question as asked confirms the premise.

**Unknowable.** Future-dated or opinion questions that no document set can
answer, and which a model will nevertheless answer fluently.
"""
from __future__ import annotations

from typing import Dict, List

ADVERSARIAL: List[Dict[str, str]] = [
    {"question": "What is the capital city of Iceland?",
     "kind": "out of domain"},
    {"question": "Who is the chief executive of Meridian and where did they study?",
     "kind": "out of domain"},
    {"question": "What is the exact discount on a three year prepaid Enterprise contract?",
     "kind": "adjacent but absent"},
    {"question": "How many compute-seconds does a single webhook delivery consume?",
     "kind": "adjacent but absent"},
    {"question": "What is the rate limit on the Starter plan in requests per second?",
     "kind": "adjacent but absent"},
    {"question": "Since workspaces can be migrated between regions, how long does a migration take?",
     "kind": "false premise"},
    {"question": "Given that offset pagination is the default, how do I switch to cursors?",
     "kind": "false premise"},
    {"question": "Which new regions will Meridian launch in 2028?",
     "kind": "unknowable"},
]


def adversarial_questions() -> List[Dict[str, str]]:
    """Every entry expects abstention. `doc_id` is empty because none is correct."""
    return [{"question": q["question"], "doc_id": "", "must_contain": "",
             "kind": q["kind"], "expect": "abstain"} for q in ADVERSARIAL]
