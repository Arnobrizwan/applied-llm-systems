"""Guarded execution: allowlist, timeout, output cap, retries, circuit breaker, audit.

Be clear about what this is and is not. The timeout runs the tool on a worker
thread and abandons the thread if it overruns. That bounds the *agent loop*,
which is the thing that was actually stalling, but it does not kill the work: a
Python thread cannot be interrupted from outside, so a tool wedged inside a C
call or a blocking socket keeps running, keeps its file handles, and keeps its
memory until the process exits. A real sandbox is a subprocess you can SIGKILL,
or a container with a cgroup and no network. That is the correct fix and it is
deliberately not what this module does, because a subprocess-per-call design
brings pickling, cold start and platform differences that would dominate a
project about tool calling.

What this module does buy, and what a surprising number of agent frameworks skip:

* an allowlist, so a compromised planner cannot reach a tool that was registered
  for a different agent,
* an output cap, so one tool returning a 40 MB blob cannot blow the context
  window and the bill along with it,
* a circuit breaker per tool, so one broken dependency degrades to a fast, clear
  error instead of burning the whole step budget on timeouts,
* an audit record per call, which is what you actually read at 3am.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from llmkit import CircuitBreaker, RetryExhausted, RetryPolicy, retry

from .registry import ToolRegistry
from .schema import ToolSpec, coerce_arguments

__all__ = ["ToolSandbox", "ToolResult", "AuditRecord", "ToolTimeout", "TransientToolError"]


class ToolTimeout(TimeoutError):
    """The tool exceeded its wall-clock budget. The worker thread is abandoned."""


class TransientToolError(RuntimeError):
    """A failure the tool believes is worth retrying, such as a flaky upstream.

    Tools raise this to opt in to retries. Everything else is treated as
    permanent, because retrying a ValueError just produces the same ValueError
    three times and triples the latency.
    """

    retryable = True


@dataclass
class AuditRecord:
    """One line of the audit log. Written for every call, including refusals."""

    tool: str
    version: str
    args: Dict[str, Any]
    outcome: str  # ok | unknown_tool | denied | invalid_args | timeout | error | output_too_large | circuit_open
    duration_ms: float
    attempts: int = 1
    result_bytes: int = 0
    error: Optional[str] = None
    ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    """The value or the structured error, plus the audit record for the call."""

    ok: bool
    value: Any = None
    error: Optional[Dict[str, Any]] = None
    audit: Optional[AuditRecord] = None

    def observation(self) -> str:
        """The single line written back into the model's context."""
        if self.ok:
            return json.dumps(self.value, default=str)
        assert self.error is not None
        detail = self.error.get("message", "")
        return f"ERROR {self.error.get('code')}: {detail}"


def run_with_timeout(func: Callable[[], Any], timeout_s: float) -> Any:
    """Run `func` on a daemon thread, raising ToolTimeout if it overruns.

    Daemon so an abandoned thread cannot keep the interpreter alive at exit. The
    exception from the worker is re-raised on the caller's thread so the caller
    sees the original traceback rather than a wrapper.
    """
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = func()
        except BaseException as exc:  # deliberately broad: it is re-raised below
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True, name="tool-call")
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise ToolTimeout(f"tool exceeded its {timeout_s:g}s budget and was abandoned")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class ToolSandbox:
    """Executes registry tools under an allowlist, a clock and a failure budget."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowlist: Optional[Sequence[str]] = None,
        max_output_bytes: int = 4096,
        retry_policy: Optional[RetryPolicy] = None,
        breaker_threshold: int = 3,
        breaker_cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.registry = registry
        # None means "everything registered". An explicit empty list means
        # "nothing", which is a useful configuration for a read-only agent and
        # would be indistinguishable from None if this used a falsy check.
        self.allowlist = None if allowlist is None else set(allowlist)
        self.max_output_bytes = max_output_bytes
        self.retry_policy = retry_policy or RetryPolicy(attempts=3, base_delay=0.0, jitter="none")
        self.clock = clock
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._breaker_config = (breaker_threshold, breaker_cooldown_s)
        self.audit_log: List[AuditRecord] = []

    # -- helpers ---------------------------------------------------------
    def breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            threshold, cooldown = self._breaker_config
            self._breakers[name] = CircuitBreaker(
                failure_threshold=threshold, cooldown_s=cooldown, clock=self.clock
            )
        return self._breakers[name]

    def _record(self, record: AuditRecord) -> AuditRecord:
        record.ts = self.clock()
        self.audit_log.append(record)
        return record

    def _fail(self, spec_name: str, version: str, args: Dict[str, Any], code: str,
              message: str, started: float, *, details: Any = None, attempts: int = 1) -> ToolResult:
        record = self._record(
            AuditRecord(
                tool=spec_name, version=version, args=args, outcome=code,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                attempts=attempts, error=message,
            )
        )
        error: Dict[str, Any] = {"code": code, "message": message}
        if details is not None:
            error["details"] = details
        return ToolResult(ok=False, error=error, audit=record)

    # -- main entry point -------------------------------------------------
    def call(self, name: str, args: Optional[Dict[str, Any]] = None, version: Optional[str] = None) -> ToolResult:
        """Execute a tool. Never raises; every failure comes back structured.

        The order of the guards matters and is cheapest-first: a denied call
        should not pay for argument coercion, and an open circuit should not pay
        for a timeout. It is also safest-first, since an unknown or denied name
        must never reach the coercion code with attacker-shaped input.
        """
        started = time.perf_counter()
        args = dict(args or {})

        spec: Optional[ToolSpec] = self.registry.find(name, version)
        if spec is None:
            available = self.registry.names()
            return self._fail(name, version or "-", args, "unknown_tool",
                              f"no tool named {name!r}", started, details={"available": available})

        if self.allowlist is not None and spec.name not in self.allowlist:
            return self._fail(spec.name, spec.version, args, "denied",
                              f"tool {spec.name!r} is not in this agent's allowlist", started)

        breaker = self.breaker(spec.name)
        if not breaker.allow():
            return self._fail(spec.name, spec.version, args, "circuit_open",
                              f"tool {spec.name!r} is failing repeatedly and is temporarily disabled", started)

        coerced, arg_errors = coerce_arguments(spec.parameters, args)
        if arg_errors:
            # Not counted as a tool failure: the tool never ran, and tripping the
            # breaker on a model's bad arguments would disable a healthy tool.
            return self._fail(spec.name, spec.version, args, "invalid_args",
                              "; ".join(e.message for e in arg_errors), started,
                              details=[e.to_dict() for e in arg_errors])

        attempts = {"n": 0}

        def invoke() -> Any:
            attempts["n"] += 1
            return run_with_timeout(lambda: spec.func(**coerced), spec.timeout_s)

        try:
            value = retry(invoke, policy=self.retry_policy,
                          retry_on=(TransientToolError,), sleep=lambda _s: None)
        except RetryExhausted as exhausted:
            breaker.record_failure()
            return self._fail(spec.name, spec.version, coerced, "error",
                              f"tool failed after {attempts['n']} attempt(s): {exhausted.last_error}",
                              started, attempts=attempts["n"])
        except ToolTimeout as timeout:
            breaker.record_failure()
            return self._fail(spec.name, spec.version, coerced, "timeout", str(timeout),
                              started, attempts=attempts["n"])
        except Exception as exc:
            breaker.record_failure()
            return self._fail(spec.name, spec.version, coerced, "error",
                              f"{type(exc).__name__}: {exc}", started, attempts=attempts["n"])

        try:
            serialised = json.dumps(value, default=str)
        except (TypeError, ValueError) as exc:
            breaker.record_failure()
            return self._fail(spec.name, spec.version, coerced, "error",
                              f"tool returned a value that cannot be serialised: {exc}",
                              started, attempts=attempts["n"])

        size = len(serialised.encode("utf-8"))
        if size > self.max_output_bytes:
            # Rejected rather than truncated. A truncated JSON payload handed
            # back to a model is worse than an explicit error: it parses as
            # nothing, or worse, parses as something subtly wrong.
            return self._fail(spec.name, spec.version, coerced, "output_too_large",
                              f"result is {size} bytes, over the {self.max_output_bytes} byte cap",
                              started, attempts=attempts["n"])

        breaker.record_success()
        record = self._record(
            AuditRecord(
                tool=spec.name, version=spec.version, args=coerced, outcome="ok",
                duration_ms=(time.perf_counter() - started) * 1000.0,
                attempts=attempts["n"], result_bytes=size,
            )
        )
        return ToolResult(ok=True, value=value, audit=record)

    # -- reporting --------------------------------------------------------
    def stats(self) -> Dict[str, Dict[str, Any]]:
        """Per-tool call counts by outcome, for the demo report and dashboards."""
        out: Dict[str, Dict[str, Any]] = {}
        for record in self.audit_log:
            agg = out.setdefault(record.tool, {"calls": 0, "ok": 0, "failed": 0, "total_ms": 0.0, "outcomes": {}})
            agg["calls"] += 1
            agg["ok"] += 1 if record.outcome == "ok" else 0
            agg["failed"] += 0 if record.outcome == "ok" else 1
            agg["total_ms"] += record.duration_ms
            agg["outcomes"][record.outcome] = agg["outcomes"].get(record.outcome, 0) + 1
        for agg in out.values():
            agg["avg_ms"] = round(agg["total_ms"] / agg["calls"], 3) if agg["calls"] else 0.0
            agg["total_ms"] = round(agg["total_ms"], 3)
        return out
