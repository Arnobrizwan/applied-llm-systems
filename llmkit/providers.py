"""LLM providers, all of them free.

Three real backends and two test doubles behind one interface:

  echo    - EchoLLM, a deterministic offline reference model (default). No
            network, no key, no install. It is not a language model; it is a
            rule engine that produces *shaped* output (grounded extractive
            answers with citations, schema-valid JSON, judge verdicts) so every
            demo and every test in this repo runs identically on any machine.
  ollama  - a local model served by Ollama (free, runs on the laptop).
  openai  - any OpenAI-compatible endpoint via base URL. Point it at a free
            tier, a self-hosted llama.cpp server, or a free-quota gateway.
  scripted / failing - test doubles.

Nothing in this file requires a paid account. `OPENAI_COMPAT_BASE_URL` is only
read if you explicitly select the `openai` provider.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .tokens import count_message_tokens, count_tokens
from .types import LLMResponse, Message


class LLMError(RuntimeError):
    """Raised when a provider fails in a way the caller may want to retry."""

    def __init__(self, message: str, *, retryable: bool = True, status: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


def _as_messages(messages: Sequence) -> List[Message]:
    out: List[Message] = []
    for m in messages:
        if isinstance(m, Message):
            out.append(m)
        elif isinstance(m, dict):
            out.append(Message(m.get("role", "user"), m.get("content", ""), m.get("name")))
        else:
            out.append(Message("user", str(m)))
    return out


class LLMProvider(ABC):
    """Every backend implements complete() and stream()."""

    name: str = "base"
    model: str = "unknown"
    tier: str = "small"

    @abstractmethod
    def complete(self, messages: Sequence, **kwargs: Any) -> LLMResponse: ...

    def stream(self, messages: Sequence, **kwargs: Any) -> Iterator[str]:
        """Default streaming: emit the completed text in word-sized deltas."""
        resp = self.complete(messages, **kwargs)
        for i, word in enumerate(resp.text.split(" ")):
            yield (" " if i else "") + word

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{type(self).__name__} provider={self.name} model={self.model}>"


# --------------------------------------------------------------------------
# EchoLLM: the offline reference model
# --------------------------------------------------------------------------

_SCHEMA_RE = re.compile(r"\{[\s\S]*\}")


class EchoLLM(LLMProvider):
    """Deterministic, offline, dependency-free stand-in for a chat model.

    Design rules:
      * Same input -> same output, always (seeded from a hash of the prompt).
      * Output *shape* matches what the caller asked for, so downstream parsing,
        validation, retry and evaluation code is genuinely exercised.
      * `fault_rate` deterministically corrupts a fraction of responses, which is
        how the structured-output and guardrail projects prove their repair paths.
    """

    name = "echo"
    tier = "echo"

    def __init__(self, model: str = "echo-1", fault_rate: float = 0.0, latency_ms: float = 0.0):
        self.model = model
        self.fault_rate = fault_rate
        self.latency_ms = latency_ms
        self.call_count = 0

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _seed(text: str) -> int:
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)

    @staticmethod
    def _last_user(msgs: List[Message]) -> str:
        for m in reversed(msgs):
            if m.role == "user":
                return m.content
        return msgs[-1].content if msgs else ""

    @staticmethod
    def _context_blocks(text: str) -> List[tuple]:
        """Pull `[S1] ...` style evidence blocks out of a prompt."""
        blocks = re.findall(r"\[(S\d+)\]\s*(.+?)(?=\n\[S\d+\]|\Z)", text, re.DOTALL)
        return [(tag, body.strip()) for tag, body in blocks]

    def _synth_from_schema(self, schema: Dict[str, Any], rng: random.Random, path: str = "") -> Any:
        t = schema.get("type", "object")
        if "enum" in schema:
            return schema["enum"][rng.randrange(len(schema["enum"]))]
        if t == "object":
            out: Dict[str, Any] = {}
            for key, sub in (schema.get("properties") or {}).items():
                out[key] = self._synth_from_schema(sub, rng, f"{path}.{key}")
            return out
        if t == "array":
            item = schema.get("items", {"type": "string"})
            n = max(1, int(schema.get("minItems", 2)))
            return [self._synth_from_schema(item, rng, path) for _ in range(n)]
        if t == "integer":
            lo = int(schema.get("minimum", 1))
            hi = int(schema.get("maximum", max(lo + 4, 5)))
            return rng.randint(lo, hi)
        if t == "number":
            lo = float(schema.get("minimum", 0.0))
            hi = float(schema.get("maximum", 1.0))
            return round(rng.uniform(lo, hi), 3)
        if t == "boolean":
            return bool(rng.getrandbits(1))
        if t == "null":
            return None
        name = path.rsplit(".", 1)[-1] or "value"
        return f"{name}-{rng.randrange(1000):03d}"

    @staticmethod
    def _corrupt(payload: str, mode: int) -> str:
        """Produce realistically-broken model output, not random noise."""
        if mode == 0:  # chatty prose wrapper, the classic
            return "Sure! Here is the JSON you asked for:\n```json\n" + payload + "\n```"
        if mode == 1:  # truncated mid-object
            return payload[: max(1, int(len(payload) * 0.6))]
        if mode == 2:  # single quotes / trailing comma
            return payload.replace('"', "'").replace("}", ",}")
        return payload + "\n\nLet me know if you need anything else."

    # -- main ------------------------------------------------------------
    def complete(self, messages: Sequence, **kwargs: Any) -> LLMResponse:
        start = time.perf_counter()
        msgs = _as_messages(messages)
        joined = "\n".join(m.content for m in msgs)
        prompt = self._last_user(msgs)
        rng = random.Random(self._seed(joined))
        self.call_count += 1
        schema = kwargs.get("json_schema")

        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)

        # 1. schema-constrained output
        if schema:
            obj = self._synth_from_schema(schema, rng)
            if isinstance(obj, dict):
                for key in obj:
                    lk = key.lower()
                    if lk in ("answer", "summary", "text", "content"):
                        obj[key] = self._extractive(joined, prompt, rng)
                    elif lk in ("question", "query"):
                        obj[key] = prompt[:180]
            text = json.dumps(obj, indent=2)
            if self.fault_rate > 0 and rng.random() < self.fault_rate:
                text = self._corrupt(text, rng.randrange(4))
        # 2. judge / scoring prompts
        elif re.search(r"\b(judge|grade|rate|score)\b", joined, re.I):
            faithful = 1 if "context" not in joined.lower() or rng.random() > 0.2 else 0
            text = json.dumps(
                {
                    "score": round(rng.uniform(0.55, 0.98), 3),
                    "verdict": "pass" if faithful else "fail",
                    "reason": "Deterministic reference judge: "
                    + ("claims are supported by the supplied evidence." if faithful else "an unsupported claim was found."),
                },
                indent=2,
            )
        # 3. grounded answering when evidence blocks are present
        else:
            text = self._extractive(joined, prompt, rng)

        latency = (time.perf_counter() - start) * 1000.0
        pt = count_message_tokens(msgs)
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            prompt_tokens=pt,
            completion_tokens=count_tokens(text),
            latency_ms=latency,
        )

    def _extractive(self, joined: str, prompt: str, rng: random.Random) -> str:
        blocks = self._context_blocks(joined)
        if not blocks:
            return f"{prompt.strip()[:400]}"
        q_words = {w for w in re.findall(r"\w+", prompt.lower()) if len(w) > 3}
        scored = []
        for tag, body in blocks:
            words = {w for w in re.findall(r"\w+", body.lower()) if len(w) > 3}
            overlap = len(q_words & words)
            scored.append((overlap, tag, body))
        scored.sort(key=lambda x: -x[0])
        best = [s for s in scored if s[0] > 0][:2] or scored[:1]
        sentences = []
        for _, tag, body in best:
            first = re.split(r"(?<=[.!?])\s+", body.strip())[0]
            sentences.append(f"{first.rstrip('.')} [{tag}].")
        return " ".join(sentences)


class ScriptedLLM(LLMProvider):
    """Returns canned responses in order. For tests that assert on exact text."""

    name = "scripted"

    def __init__(self, responses: Sequence[str], model: str = "scripted-1"):
        self.responses = list(responses)
        self.model = model
        self.calls: List[List[Message]] = []

    def complete(self, messages: Sequence, **kwargs: Any) -> LLMResponse:
        msgs = _as_messages(messages)
        self.calls.append(msgs)
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        text = self.responses[idx] if self.responses else ""
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            prompt_tokens=count_message_tokens(msgs),
            completion_tokens=count_tokens(text),
        )


class FailingLLM(LLMProvider):
    """Fails `fail_times` times, then delegates. Exercises retry/fallback paths."""

    name = "failing"

    def __init__(self, fail_times: int = 1, then: Optional[LLMProvider] = None, status: int = 429):
        self.fail_times = fail_times
        self.then = then or EchoLLM()
        self.status = status
        self.attempts = 0
        self.model = f"failing-{status}"

    def complete(self, messages: Sequence, **kwargs: Any) -> LLMResponse:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise LLMError(f"synthetic upstream failure (HTTP {self.status})", retryable=True, status=self.status)
        return self.then.complete(messages, **kwargs)


class OllamaLLM(LLMProvider):
    """Local model over Ollama's HTTP API. Free, offline, no account."""

    name = "ollama"
    tier = "ollama"

    def __init__(self, model: str = "llama3.2:1b", host: Optional[str] = None, timeout: float = 120.0):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - network path
            raise LLMError(f"ollama HTTP {e.code}", retryable=e.code >= 500 or e.code == 429, status=e.code) from e
        except urllib.error.URLError as e:  # pragma: no cover - network path
            raise LLMError(f"ollama unreachable at {self.host}: {e.reason}", retryable=True) from e

    def complete(self, messages: Sequence, **kwargs: Any) -> LLMResponse:  # pragma: no cover - network path
        msgs = _as_messages(messages)
        start = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in msgs],
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", 0.0)},
        }
        if kwargs.get("json_schema"):
            payload["format"] = kwargs["json_schema"]
        data = self._post("/api/chat", payload)
        text = (data.get("message") or {}).get("content", "")
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            prompt_tokens=data.get("prompt_eval_count") or count_message_tokens(msgs),
            completion_tokens=data.get("eval_count") or count_tokens(text),
            latency_ms=(time.perf_counter() - start) * 1000.0,
            raw=data,
        )


class OpenAICompatLLM(LLMProvider):
    """Any OpenAI-compatible /chat/completions endpoint.

    Set OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_API_KEY. Works with free-tier
    gateways and with a local llama.cpp / vLLM server (where the key is ignored).
    """

    name = "openai"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 90.0, tier: str = "small"):
        self.model = model or os.environ.get("OPENAI_COMPAT_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY", "not-needed")
        self.timeout = timeout
        self.tier = tier

    def complete(self, messages: Sequence, **kwargs: Any) -> LLMResponse:  # pragma: no cover - network path
        msgs = _as_messages(messages)
        start = time.perf_counter()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in msgs],
            "temperature": kwargs.get("temperature", 0.0),
        }
        if kwargs.get("max_tokens"):
            payload["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("json_schema"):
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"upstream HTTP {e.code}", retryable=e.code >= 500 or e.code == 429, status=e.code) from e
        except urllib.error.URLError as e:
            raise LLMError(f"upstream unreachable: {e.reason}", retryable=True) from e
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            prompt_tokens=usage.get("prompt_tokens") or count_message_tokens(msgs),
            completion_tokens=usage.get("completion_tokens") or count_tokens(text),
            latency_ms=(time.perf_counter() - start) * 1000.0,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )


_REGISTRY = {
    "echo": EchoLLM,
    "ollama": OllamaLLM,
    "openai": OpenAICompatLLM,
    "scripted": ScriptedLLM,
    "failing": FailingLLM,
}


def get_llm(provider: Optional[str] = None, **kwargs: Any) -> LLMProvider:
    """Resolve a provider by name or from LLM_PROVIDER. Defaults to `echo`."""
    key = (provider or os.environ.get("LLM_PROVIDER") or "echo").lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown provider {key!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
