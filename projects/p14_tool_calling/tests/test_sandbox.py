"""Sandbox guarantees: allowlist, timeout, output cap, retry, breaker, audit."""
import time

from projects.p14_tool_calling.registry import ToolRegistry
from projects.p14_tool_calling.sandbox import ToolSandbox, TransientToolError
from projects.p14_tool_calling.schema import tool


@tool(timeout_s=0.05)
def sleeper(seconds: float = 0.5) -> str:
    """Sleep.

    Args:
        seconds: How long.
    """
    time.sleep(seconds)
    return "done"


@tool
def firehose(size: int = 5000) -> str:
    """Return a big payload.

    Args:
        size: Characters.
    """
    return "x" * size


@tool
def ok_tool(n: int = 1) -> int:
    """Return n doubled.

    Args:
        n: A number.
    """
    return n * 2


@tool
def always_broken() -> str:
    """Always fail."""
    raise RuntimeError("dependency is down")


def build(**kwargs):
    registry = ToolRegistry()
    registry.register_all([sleeper, firehose, ok_tool, always_broken])
    return ToolSandbox(registry, **kwargs)


def test_a_successful_call_returns_the_value_and_writes_an_audit_record():
    sandbox = build()
    result = sandbox.call("ok_tool", {"n": 21})
    assert result.ok and result.value == 42
    assert result.audit.outcome == "ok" and result.audit.result_bytes > 0
    assert len(sandbox.audit_log) == 1


def test_an_unknown_tool_is_data_not_an_exception():
    result = build().call("teleport", {"to": "mars"})
    assert not result.ok and result.error["code"] == "unknown_tool"
    assert "ok_tool" in result.error["details"]["available"]


def test_the_allowlist_blocks_before_the_tool_runs():
    sandbox = build(allowlist=["ok_tool"])
    assert sandbox.call("sleeper", {"seconds": 5.0}).error["code"] == "denied"
    assert sandbox.call("ok_tool", {"n": 1}).ok


def test_an_empty_allowlist_means_nothing_rather_than_everything():
    """The distinction only exists because the check is `is not None`."""
    assert build(allowlist=[]).call("ok_tool", {"n": 1}).error["code"] == "denied"


def test_the_timeout_returns_control_long_before_the_tool_finishes():
    sandbox = build()
    started = time.perf_counter()
    result = sandbox.call("sleeper", {"seconds": 0.5})
    elapsed = time.perf_counter() - started
    assert result.error["code"] == "timeout"
    assert elapsed < 0.3, f"waited {elapsed:.3f}s for a 0.05s budget"


def test_oversize_output_is_refused_rather_than_truncated():
    sandbox = build(max_output_bytes=256)
    result = sandbox.call("firehose", {"size": 5000})
    assert result.error["code"] == "output_too_large"
    assert result.value is None


def test_invalid_arguments_do_not_count_against_the_tool_health_budget():
    sandbox = build(breaker_threshold=2)
    for _ in range(5):
        assert sandbox.call("ok_tool", {"n": "not a number"}).error["code"] == "invalid_args"
    assert sandbox.breaker("ok_tool").state == "closed"
    assert sandbox.call("ok_tool", {"n": 2}).ok


def test_transient_failures_are_retried_and_permanent_ones_are_not():
    state = {"calls": 0}

    @tool(name="flaky")
    def flaky() -> str:
        """Fail twice then succeed."""
        state["calls"] += 1
        if state["calls"] <= 2:
            raise TransientToolError("connection reset")
        return "recovered"

    registry = ToolRegistry()
    registry.register_all([flaky, always_broken])
    sandbox = ToolSandbox(registry)

    result = sandbox.call("flaky")
    assert result.ok and result.value == "recovered" and result.audit.attempts == 3

    result = sandbox.call("always_broken")
    assert not result.ok and result.audit.attempts == 1  # not retried


def test_the_breaker_opens_after_the_threshold_and_half_opens_after_cooldown():
    now = [0.0]
    sandbox = build(breaker_threshold=3, breaker_cooldown_s=30.0, clock=lambda: now[0])
    outcomes = [sandbox.call("always_broken").audit.outcome for _ in range(5)]
    assert outcomes == ["error", "error", "error", "circuit_open", "circuit_open"]
    assert sandbox.breaker("always_broken").state == "open"
    now[0] = 31.0
    assert sandbox.breaker("always_broken").state == "half_open"
    assert sandbox.call("always_broken").audit.outcome == "error"


def test_one_broken_tool_does_not_disable_a_healthy_one():
    now = [0.0]
    sandbox = build(breaker_threshold=2, clock=lambda: now[0])
    for _ in range(4):
        sandbox.call("always_broken")
    assert sandbox.breaker("always_broken").state == "open"
    assert sandbox.call("ok_tool", {"n": 5}).ok


def test_every_call_including_refusals_produces_exactly_one_audit_record():
    sandbox = build(allowlist=["ok_tool", "always_broken"], max_output_bytes=64)
    sandbox.call("ok_tool", {"n": 1})
    sandbox.call("firehose")
    sandbox.call("nope")
    sandbox.call("ok_tool", {"n": "x"})
    sandbox.call("always_broken")
    outcomes = [r.outcome for r in sandbox.audit_log]
    assert outcomes == ["ok", "denied", "unknown_tool", "invalid_args", "error"]
    assert sandbox.stats()["ok_tool"]["calls"] == 2


def test_a_non_json_native_result_is_stringified_rather_than_failing_the_call():
    """`default=str` is used for both the size check and the observation, so the
    bytes counted are the bytes the model sees. A tool returning a datetime or a
    Decimal should not fail; it should arrive as text."""
    @tool(name="exotic")
    def exotic() -> object:
        """Return something json cannot encode natively."""
        return {"when": complex(1, 2)}

    registry = ToolRegistry()
    registry.register(exotic)
    result = ToolSandbox(registry).call("exotic")
    assert result.ok
    assert result.observation() == '{"when": "(1+2j)"}'
    assert result.audit.result_bytes == len(result.observation())
