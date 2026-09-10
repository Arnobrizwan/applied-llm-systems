"""Embeddings without a model download.

`HashingEmbedder` is the default: a deterministic character-n-gram + word
hashing embedder (the "hashing trick") with L2 normalisation. It is lexical, not
semantic - it will match paraphrases only when they share sub-words - and every
project that uses it says so in its README. It exists so retrieval, caching and
clustering code can be built, tested and demonstrated with zero dependencies,
zero downloads and zero spend, and swapped for a real encoder with one env var.

Free upgrades, both supported here:
  EMBED_PROVIDER=ollama   -> nomic-embed-text served locally by Ollama
  EMBED_PROVIDER=openai   -> any OpenAI-compatible /embeddings endpoint
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Returns 0.0 if either vector is all zeros."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def l2_normalize(v: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return v
    return [x / norm for x in v]


class Embedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Deterministic bag-of-features hashing embedder.

    Features: lowercased word unigrams, word bigrams, and character 4-grams.
    Character n-grams give partial credit for morphology and typos, which is what
    keeps this usable as a stand-in for a real encoder in a retrieval demo.
    """

    name = "hashing"

    def __init__(self, dim: int = 384, char_ngram: int = 4, use_bigrams: bool = True):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.char_ngram = char_ngram
        self.use_bigrams = use_bigrams

    def _bucket(self, feature: str) -> tuple:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "big") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0  # signed hashing cancels collision bias
        return idx, sign

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            low = (text or "").lower()
            words = _TOKEN_RE.findall(low)
            feats: List[str] = list(words)
            if self.use_bigrams:
                feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
            compact = " ".join(words)
            n = self.char_ngram
            feats += [compact[i : i + n] for i in range(max(0, len(compact) - n + 1))]
            for f in feats:
                idx, sign = self._bucket(f)
                vec[idx] += sign
            out.append(l2_normalize(vec))
        return out


class OllamaEmbedder(Embedder):  # pragma: no cover - network path
    """Local embeddings via Ollama, e.g. `ollama pull nomic-embed-text`."""

    name = "ollama"

    def __init__(self, model: str = "nomic-embed-text", host: Optional[str] = None, timeout: float = 60.0):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.dim = 768

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for t in texts:
            req = urllib.request.Request(
                f"{self.host}/api/embeddings",
                data=json.dumps({"model": self.model, "prompt": t}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            v = data.get("embedding") or []
            self.dim = len(v) or self.dim
            vectors.append(l2_normalize([float(x) for x in v]))
        return vectors


class OpenAICompatEmbedder(Embedder):  # pragma: no cover - network path
    """Any OpenAI-compatible /embeddings endpoint (including local servers)."""

    name = "openai"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 60.0):
        self.model = model or os.environ.get("EMBED_MODEL", "text-embedding-3-small")
        self.base_url = (base_url or os.environ.get("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY", "not-needed")
        self.timeout = timeout
        self.dim = 1536

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        vecs = [l2_normalize([float(x) for x in row["embedding"]]) for row in data.get("data", [])]
        if vecs:
            self.dim = len(vecs[0])
        return vecs


_REGISTRY = {"hashing": HashingEmbedder, "ollama": OllamaEmbedder, "openai": OpenAICompatEmbedder}


def get_embedder(provider: Optional[str] = None, **kwargs) -> Embedder:
    """Resolve an embedder by name or from EMBED_PROVIDER. Defaults to `hashing`."""
    key = (provider or os.environ.get("EMBED_PROVIDER") or "hashing").lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown embedder {key!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
