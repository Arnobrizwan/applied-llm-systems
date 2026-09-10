"""A realistic request shape, built from the shared corpus.

Kept out of demo.py so the tests exercise exactly the request the demo reports
on. The retrieval scores are real cosine scores from llmkit's vector store over
llmkit's corpus, not hand-written numbers, because the allocator uses item value
as its marginal-value signal and feeding it invented scores would make the demo
prove nothing.
"""
from __future__ import annotations

from typing import List

from llmkit import InMemoryVectorStore
from llmkit.corpus import chunks

from .sections import Compression, ContextItem, ContextRequest, Section, SectionSpec

INSTRUCTIONS = [
    (
        "sys.role",
        "You are Meridian Support Copilot. Answer only from the evidence supplied in this "
        "prompt. If the evidence does not contain the answer, say so and name the document "
        "the user should read instead. Never guess a number, a limit or a date.",
    ),
    (
        "sys.tools",
        "You may cite the tool output block. Tool readings are live and override any figure "
        "in a document when the two disagree, but you must say which one you used.",
    ),
]

MEMORIES = [
    (
        "mem.plan",
        "The user's workspace is on the Growth plan and is pinned to the eu-west region.",
        0.82,
    ),
    (
        "mem.incident",
        "Two weeks ago this user hit sustained 429s during a nightly batch job and asked "
        "whether adding a second token would raise their ceiling.",
        0.71,
    ),
    (
        # Deliberately a restatement of the rate-limits document. The retriever
        # will surface the fuller original, so this copy should be deduplicated.
        "mem.limit",
        "The default rate limit is 600 requests per minute per workspace, burstable to 1000 "
        "for 10 seconds.",
        0.66,
    ),
]

TOOL_OUTPUT = (
    "workspace_metrics(workspace='wsp_4471', window='24h') -> "
    "{'requests_total': 812004, 'requests_rejected_429': 4126, 'peak_rpm': 1180, "
    "'sustained_rpm_p95': 640, 'burst_events': 37, 'burst_seconds_over_limit': 214, "
    "'retry_after_p50_seconds': 12, 'retry_after_p95_seconds': 41, "
    "'tokens_active': 6, 'region': 'eu-west', 'plan': 'growth', "
    "'largest_client': 'nightly-batch-runner', 'nightly_batch_window': '02:00-03:10 UTC', "
    "'nightly_batch_share_of_requests': 0.63, 'note': 'rejections cluster inside the batch window'}"
)

HISTORY = [
    ("user", "hi, we keep getting 429s at night and it is breaking our nightly export"),
    ("assistant", "Understood. Are the rejections spread through the day or clustered in a window?"),
    ("user", "clustered, roughly two in the morning UTC"),
    ("assistant", "That lines up with a batch job. How many tokens does the workspace have?"),
    ("user", "six, we added more last month hoping it would help"),
    ("assistant", "Noted. Adding tokens does not change a per-workspace limit."),
    ("user", "we also tried retrying immediately on the 429"),
    ("assistant", "Retrying without honouring Retry-After will extend the rejection window."),
]


def _retrieved_items(query: str, k: int = 6) -> List[ContextItem]:
    store = InMemoryVectorStore()
    store.add(chunks())
    return [
        ContextItem(
            id=f"doc.{hit.chunk.doc_id}",
            text=hit.chunk.text,
            source="retrieval",
            value=round(max(0.0, hit.score), 4),
            metadata={"citation": hit.chunk.citation},
        )
        for hit in store.search(query, k=k)
    ]


def build_request(
    query: str,
    model_window: int,
    reserve_tokens: int,
    request_id: str = "req-1",
    k: int = 6,
) -> ContextRequest:
    """One support request with six competing sources of context."""
    instructions = Section(
        spec=SectionSpec(
            name="instructions",
            priority=10.0,
            min_tokens=90,
            max_share=0.35,
            strategy=Compression.KEEP_WHOLE,
            position="first",
            header="# Instructions",
        ),
        items=[ContextItem(id=i, text=t, source="system", value=1.0) for i, t in INSTRUCTIONS],
    )
    memory = Section(
        spec=SectionSpec(
            name="user_memory",
            priority=6.0,
            min_tokens=40,
            max_share=0.25,
            strategy=Compression.EXTRACTIVE,
            header="# What we already know about this workspace",
        ),
        items=[ContextItem(id=i, text=t, source="memory", value=v) for i, t, v in MEMORIES],
    )
    retrieved = Section(
        spec=SectionSpec(
            name="retrieved_docs",
            priority=5.0,
            min_tokens=120,
            max_share=0.55,
            strategy=Compression.EXTRACTIVE,
            header="# Documentation",
        ),
        items=_retrieved_items(query, k=k),
    )
    tools = Section(
        spec=SectionSpec(
            name="tool_output",
            priority=4.0,
            min_tokens=60,
            max_share=0.3,
            strategy=Compression.TRUNCATE_TAIL,
            header="# Live tool output",
        ),
        items=[ContextItem(id="tool.metrics", text=TOOL_OUTPUT, source="tool", value=0.9)],
    )
    history = Section(
        spec=SectionSpec(
            name="chat_history",
            priority=2.0,
            min_tokens=0,
            max_share=0.25,
            strategy=Compression.TRUNCATE_HEAD,
            header="# Conversation so far",
        ),
        items=[
            ContextItem(
                id=f"turn.{idx}",
                text=f"{role}: {text}",
                source="history",
                # Recency is the value signal for history: the last turn is worth
                # more than the first, and the allocator should know that.
                value=round(0.3 + 0.7 * (idx + 1) / len(HISTORY), 3),
            )
            for idx, (role, text) in enumerate(HISTORY)
        ],
    )
    question = Section(
        spec=SectionSpec(
            name="question",
            priority=10.0,
            min_tokens=30,
            max_share=0.2,
            strategy=Compression.KEEP_WHOLE,
            position="last",
            header="# Question",
        ),
        items=[ContextItem(id="q.live", text=query, source="user", value=1.0)],
    )
    return ContextRequest(
        query=query,
        sections=[instructions, memory, retrieved, tools, history, question],
        model_window=model_window,
        reserve_tokens=reserve_tokens,
        request_id=request_id,
    )
