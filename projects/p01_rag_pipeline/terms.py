"""Query and document term folding, shared by the reranker and the answerer.

Two problems this fixes, both found by measurement rather than by inspection
-----------------------------------------------------------------------------
The first version of the grounding check compared raw tokens. It refused 47
percent of the gold questions, including questions whose answer was sitting in
the top retrieved chunk. The cause was morphology: the question said "expires"
and "token", the document said "expire" and "tokens", and exact set intersection
scored those as no overlap at all. Every one of those refusals was a false
negative produced by string equality, not by missing evidence.

The second problem was the opposite. An entirely off-corpus question ("what is
the capital city of Iceland") scored 0.19 grounding, because "what" is a rare
term in a corpus of technical documents, so idf weighted it heavily and any
chunk containing the word "what" made the question look partly answered.
Interrogatives and auxiliaries carry no retrieval signal at all and must not be
allowed to earn coverage credit.

The stemmer here is crude and over-stems on purpose. "authenticate",
"authenticates" and "authenticated" all fold to "authenticat", which is not a
word. That is fine: nothing is ever displayed to a user, both sides of every
comparison are folded identically, and over-stemming costs a little precision to
buy recall on exactly the plural and tense mismatches that were causing the false
refusals. A real deployment would use a Snowball stemmer, which is the same idea
with better rules and a dependency this repo is not allowed to take.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from llmkit import tokenize

_WORD_RE = re.compile(r"[a-z0-9]+")

# Interrogatives, auxiliaries, pronouns and comparative function words. None of
# these appear in a technical document for a reason connected to the question
# being asked, so counting them as covered vocabulary is noise with a high idf.
QUESTION_WORDS: Set[str] = {
    "what", "how", "which", "who", "whom", "whose", "when", "where", "why",
    "does", "do", "did", "done", "can", "could", "should", "would", "will",
    "shall", "may", "might", "must", "have", "has", "had", "am",
    "i", "me", "my", "we", "us", "our", "you", "your", "they", "them", "their",
    "he", "she", "his", "her", "its", "it",
    "long", "much", "many", "often", "far", "about", "into", "over", "under",
    "than", "then", "also", "any", "some", "all", "more", "most", "other",
    "such", "there", "here", "if", "get", "before", "after", "later", "still",
}

_PLURAL_RULES: Sequence[Tuple[str, str]] = (
    ("ies", "y"), ("sses", "ss"), ("shes", "sh"), ("ches", "ch"),
    ("xes", "x"), ("zes", "z"),
)


def fold(term: str) -> str:
    """Fold one surface form to a comparison key. Not a linguistic stem."""
    t = term.lower()
    if len(t) <= 3:
        return t
    for suffix, replacement in _PLURAL_RULES:
        if t.endswith(suffix) and len(t) - len(suffix) + len(replacement) >= 3:
            t = t[: -len(suffix)] + replacement
            break
    else:
        if t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
    if t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    # Trailing "e" is dropped so a verb and its participle land on the same key
    # ("authenticate" and "authenticated" both become "authenticat").
    if t.endswith("e") and len(t) >= 4:
        t = t[:-1]
    return t


def content_terms(text: str) -> List[str]:
    """Folded, de-duplicated content vocabulary of a query or a chunk."""
    out: List[str] = []
    seen: Set[str] = set()
    for token in tokenize(text):
        if token in QUESTION_WORDS:
            continue
        key = fold(token)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def folded_sequence(text: str) -> List[str]:
    """Folded content terms in order, duplicates kept, for phrase matching.

    Phrase matching needs ordering, so it cannot use the de-duplicated set. Both
    sides of the comparison go through this same function, so the stopword and
    interrogative removal that makes "the rotation window" and "rotation window"
    contiguous is applied identically to the query and to the chunk.
    """
    return [fold(t) for t in tokenize(text) if t not in QUESTION_WORDS]


def term_set(text: str) -> Set[str]:
    """Folded vocabulary of a document, stopwords included.

    Stopwords are kept on the document side: dropping them there would mean a
    query term that happens to be on the stoplist can never be found, while
    keeping them costs nothing because the query side has already dropped them.
    """
    return {fold(t) for t in _WORD_RE.findall(text.lower())}


def document_frequencies(texts: Iterable[str]) -> Tuple[Dict[str, int], int]:
    """Document frequency over folded terms, plus the corpus size.

    Computed in folded space rather than reused from BM25's raw counts. Mixing
    the two would look up the idf of "expir" in a table keyed on "expires", find
    nothing, and treat the corpus's most common word as its rarest.
    """
    df: Counter = Counter()
    total = 0
    for text in texts:
        total += 1
        for term in term_set(text):
            df[term] += 1
    return dict(df), total
