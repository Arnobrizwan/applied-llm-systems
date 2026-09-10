"""llmkit - the shared, dependency-free core behind all 15 systems.

Everything defaults to a free provider. Nothing here reads a paid API key
unless you explicitly select a provider that needs one.
"""
from .bm25 import BM25, reciprocal_rank_fusion, tokenize
from .embeddings import Embedder, HashingEmbedder, cosine, get_embedder, l2_normalize
from .providers import (
    EchoLLM,
    FailingLLM,
    LLMError,
    LLMProvider,
    OllamaLLM,
    OpenAICompatLLM,
    ScriptedLLM,
    get_llm,
)
from .retry import CircuitBreaker, RetryExhausted, RetryPolicy, retry
from .text import chunk_text, keyword_overlap, normalize_ws, sentence_split
from .tokens import count_message_tokens, count_tokens, estimate_cost, truncate_to_tokens
from .tracing import Span, Tracer, percentile, tracer
from .types import Chunk, Document, LLMResponse, Message, ScoredChunk, Usage, assistant, system, user
from .vectorstore import InMemoryVectorStore

__version__ = "1.0.0"

__all__ = [
    "BM25", "reciprocal_rank_fusion", "tokenize",
    "Embedder", "HashingEmbedder", "cosine", "get_embedder", "l2_normalize",
    "EchoLLM", "FailingLLM", "LLMError", "LLMProvider", "OllamaLLM",
    "OpenAICompatLLM", "ScriptedLLM", "get_llm",
    "CircuitBreaker", "RetryExhausted", "RetryPolicy", "retry",
    "chunk_text", "keyword_overlap", "normalize_ws", "sentence_split",
    "count_message_tokens", "count_tokens", "estimate_cost", "truncate_to_tokens",
    "Span", "Tracer", "percentile", "tracer",
    "Chunk", "Document", "LLMResponse", "Message", "ScoredChunk", "Usage",
    "assistant", "system", "user",
    "InMemoryVectorStore",
]
