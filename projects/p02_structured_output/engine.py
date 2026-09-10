"""The retry loop: prompt, extract, validate, feed the errors back, fall back.

This is the part that makes structured output safe to depend on. The contract
is that `generate()` never raises for a model problem. A caller in a request
path gets either a valid object or a typed fallback, plus a flag saying which,
and can decide whether a fallback is worth a 500 or worth degrading quietly.

Three design decisions are worth stating, because each had an alternative:

1. Errors go back verbatim. The alternative, "that was not valid JSON, try
   again", measurably wastes attempts: the model has no idea which field it got
   wrong, so it re-rolls the whole response and reproduces the same mistake.
   Sending "$.score: must be <= 1, got 1.4" turns a re-roll into a correction.
2. The bounded loop is `llmkit.retry`, not a hand-rolled `for` loop. A schema
   violation is raised as a retryable exception so the same backoff, jitter and
   exhaustion semantics used for transport errors apply to semantic ones. One
   retry policy in the codebase, not two that drift apart.
3. Repair is attempted before re-prompting, always. A repaired response costs
   microseconds; a re-prompt costs a model call. The measurement in the demo
   exists to check that this ordering is actually earning its place.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from llmkit import LLMProvider, RetryExhausted, RetryPolicy, retry

from .extraction import Extraction, extract_json
from .validator import SchemaError, describe_errors, validate

__all__ = ["Attempt", "StructuredResult", "StructuredOutputEngine", "SchemaViolation"]

_SYSTEM = (
    "You return a single JSON value and nothing else. No prose, no code fence, "
    "no commentary. The value must satisfy the supplied JSON Schema exactly."
)


class SchemaViolation(Exception):
    """Raised inside the retry loop so llmkit.retry drives the attempt budget."""

    retryable = True

    def __init__(self, errors: List[SchemaError], extraction: Extraction):
        super().__init__(describe_errors(errors) or (extraction.error or "unparseable"))
        self.errors = errors
        self.extraction = extraction


@dataclass
class Attempt:
    """What happened on one model call. The demo's numbers come from these."""

    index: int
    ok: bool
    parsed: bool
    repairs: List[str] = field(default_factory=list)
    errors: List[SchemaError] = field(default_factory=list)
    raw: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def outcome(self) -> str:
        if self.ok:
            return "repaired" if self.repairs else "clean"
        return "unparseable" if not self.parsed else "invalid"


@dataclass
class StructuredResult:
    """A valid object, or a fallback, and the full story of how it got there."""

    ok: bool
    value: Any
    source: str  # "clean" | "repaired" | "reprompted" | "fallback"
    attempts: List[Attempt] = field(default_factory=list)
    errors: List[SchemaError] = field(default_factory=list)

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def repairs_used(self) -> List[str]:
        return [r for a in self.attempts for r in a.repairs]

    @property
    def total_tokens(self) -> int:
        return sum(a.prompt_tokens + a.completion_tokens for a in self.attempts)


class StructuredOutputEngine:
    """Turns a schema plus a prompt into a validated object or a fallback."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        max_attempts: int = 3,
        policy: Optional[RetryPolicy] = None,
        sleep: Callable[[float], None] = lambda _seconds: None,
    ):
        self.llm = llm
        self.max_attempts = max_attempts
        # base_delay 0 by default: the wait between attempts here is not
        # protecting an overloaded upstream, it is a correction round trip, and
        # a sleep only adds latency to a user-facing request. Pass a real policy
        # when the provider is rate-limiting rather than mis-formatting.
        self.policy = policy or RetryPolicy(attempts=max_attempts, base_delay=0.0, jitter="none")
        self.sleep = sleep

    # -- prompt construction --------------------------------------------
    def _messages(self, prompt: str, schema: Dict[str, Any], feedback: Optional[str]) -> List[Dict[str, str]]:
        content = [
            prompt.strip(),
            "",
            "JSON Schema:",
            json.dumps(schema, indent=2, sort_keys=True),
        ]
        if feedback:
            content += [
                "",
                "Your previous response was rejected by the validator. Fix exactly "
                "these problems and return the whole value again:",
                feedback,
            ]
        return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "\n".join(content)}]

    # -- one attempt ------------------------------------------------------
    def _attempt(self, prompt: str, schema: Dict[str, Any], state: Dict[str, Any]) -> Any:
        index = len(state["attempts"])
        messages = self._messages(prompt, schema, state.get("feedback"))
        response = self.llm.complete(messages, json_schema=schema)

        extraction = extract_json(response.text)
        errors: List[SchemaError] = []
        if extraction.ok:
            errors = validate(extraction.value, schema)

        record = Attempt(
            index=index,
            ok=extraction.ok and not errors,
            parsed=extraction.ok,
            repairs=list(extraction.repairs),
            errors=errors,
            raw=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        state["attempts"].append(record)

        if record.ok:
            return extraction.value

        if not extraction.ok:
            # A parse failure gets a synthetic error at the root so the feedback
            # block has the same shape whether the problem was syntax or schema.
            errors = [
                SchemaError(
                    path="$",
                    rule="parse",
                    got=(response.text or "")[:120],
                    expected="a single JSON value",
                    message=f"response could not be parsed as JSON ({extraction.error})",
                )
            ]
            record.errors = errors
        state["feedback"] = describe_errors(errors)
        raise SchemaViolation(errors, extraction)

    # -- public API -------------------------------------------------------
    def generate(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        fallback: Any = None,
    ) -> StructuredResult:
        """Produce a schema-valid object, or `fallback` if every attempt fails.

        `fallback` is deep-copied on the way out so a shared default object
        cannot be mutated by one caller and observed by the next. That bug is
        tedious to find and free to prevent.
        """
        state: Dict[str, Any] = {"attempts": [], "feedback": None}
        try:
            value = retry(
                lambda: self._attempt(prompt, schema, state),
                policy=self.policy,
                retry_on=(SchemaViolation,),
                sleep=self.sleep,
            )
        except RetryExhausted as exhausted:
            last = state["attempts"][-1] if state["attempts"] else None
            return StructuredResult(
                ok=False,
                value=copy.deepcopy(fallback),
                source="fallback",
                attempts=state["attempts"],
                errors=list(last.errors) if last else [],
            )

        first = state["attempts"][0]
        winner = state["attempts"][-1]
        if len(state["attempts"]) > 1:
            source = "reprompted"
        else:
            source = "repaired" if winner.repairs else "clean"
        _ = first  # kept explicit: attempt 0 is what success@1 is measured on
        return StructuredResult(ok=True, value=value, source=source, attempts=state["attempts"])
