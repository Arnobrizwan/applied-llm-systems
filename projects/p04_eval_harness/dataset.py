"""The golden dataset: a typed case, JSONL persistence and the adversarial set.

Why JSONL and not a Python list: an eval set is a data artefact with a review
history, not code. JSONL diffs one case per line, so a pull request that adds
three cases shows three added lines and a reviewer can approve it without
reading a diff of a Python literal. It is also append-friendly, which matters
when cases are harvested from production failures.

Why `tags` and `difficulty` are first-class: a single aggregate number hides the
regression you care about. A model change that improves easy paraphrase cases
by 4 points while losing every adversarial case is a bad change, and you can
only see that if the slice is carried on the case itself.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from llmkit import sentence_split
from llmkit.corpus import by_id, gold_questions

_HERE = os.path.dirname(os.path.abspath(__file__))
ADVERSARIAL_PATH = os.path.join(_HERE, "data", "adversarial.jsonl")


@dataclass
class EvalCase:
    """One evaluation case.

    `expected` carries whatever the case's scorers need: a literal answer for
    exact match, a required phrase for `contains`, a pattern for `regex`. That
    is a deliberate simplification over a per-scorer expectation map, which is
    more general and, in three previous versions of this file, unreadable.
    """

    id: str
    input: str
    expected: str = ""
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    scorers: List[str] = field(default_factory=lambda: ["contains"])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalCase":
        known = {f for f in cls.__dataclass_fields__}  # tolerate extra columns
        return cls(**{k: v for k, v in d.items() if k in known})


def save_jsonl(cases: Iterable[EvalCase], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict(), sort_keys=True) + "\n")
            n += 1
    return n


def load_jsonl(path: str) -> List[EvalCase]:
    """Load cases, skipping blank lines and `#` comments.

    A malformed line raises rather than being skipped: silently dropping a case
    shrinks the eval set without shrinking the reported confidence, which is the
    worst possible failure mode for a quality gate.
    """
    cases: List[EvalCase] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(EvalCase.from_dict(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: malformed case: {exc}") from exc
    return cases


def _reference_sentence(text: str, must_contain: str) -> str:
    """The sentence of the source document that actually carries the answer.

    Token F1 against the whole document would punish a correct one-sentence
    answer for everything it left out, and F1 against the required phrase alone
    would reward a bare "90 days" with no subject. The answer-bearing sentence is
    the only reference that scores a good short answer as a good short answer.
    """
    for sentence in sentence_split(text):
        if must_contain.lower() in sentence.lower():
            return sentence
    return text


def gold_cases() -> List[EvalCase]:
    """The 15 hand-written questions from `llmkit.corpus`, as eval cases.

    The corpus is fictional so a public model cannot have memorised it. Scoring
    is `contains` on the required phrase plus token F1 against the source
    sentence, because there are many correct phrasings of "90 days" and exact
    match would punish all but one of them.
    """
    docs = by_id()
    cases: List[EvalCase] = []
    for i, g in enumerate(gold_questions(), 1):
        cases.append(
            EvalCase(
                id=f"gold-{i:02d}",
                input=g["question"],
                expected=g["must_contain"],
                tags=["gold", "grounded", g["doc_id"]],
                difficulty="easy" if i % 3 else "medium",
                scorers=["contains", "token_f1"],
                metadata={
                    "doc_id": g["doc_id"],
                    "reference": _reference_sentence(docs[g["doc_id"]].text, g["must_contain"]),
                },
            )
        )
    return cases


def adversarial_cases(path: Optional[str] = None) -> List[EvalCase]:
    """Cases written to break the system rather than to confirm it works."""
    return load_jsonl(path or ADVERSARIAL_PATH)


def build_dataset(include_adversarial: bool = True) -> List[EvalCase]:
    """The full evaluation set. This is what CI runs against."""
    cases = gold_cases()
    if include_adversarial:
        cases += adversarial_cases()
    seen = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"duplicate case id {c.id!r}")
        seen.add(c.id)
    return cases


def slice_by(cases: Iterable[EvalCase], tag: Optional[str] = None,
             difficulty: Optional[str] = None) -> List[EvalCase]:
    """Filter to one slice. Used to report per-slice scores next to the total."""
    out = list(cases)
    if tag:
        out = [c for c in out if tag in c.tags]
    if difficulty:
        out = [c for c in out if c.difficulty == difficulty]
    return out
