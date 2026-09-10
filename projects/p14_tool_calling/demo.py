"""End-to-end demo for the Tool-Calling Framework.

Six sections: the registry and its generated schemas, real tool calls through
the sandbox, the adversarial calculator suite, the sandbox guarantees
(allowlist, timeout, output cap, retry, circuit breaker), the agent loop, and
the audit log rollup.

Offline and deterministic. The numbers printed here are the numbers in the README.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import time

from llmkit import EchoLLM, RetryPolicy

from projects.p14_tool_calling.agent import ToolAgent
from projects.p14_tool_calling.registry import ToolRegistry
from projects.p14_tool_calling.safe_math import UnsafeExpression, safe_eval
from projects.p14_tool_calling.sandbox import ToolSandbox, TransientToolError
from projects.p14_tool_calling.schema import tool
from projects.p14_tool_calling.tools import ALL_TOOLS, search_docs

QUESTIONS = [
    "How long is a Meridian token valid before it expires?",
    "What is the default rate limit per workspace?",
    "How many kilometres is 5 miles?",
    "What time is it in Dhaka right now?",
    "How are failed webhook deliveries retried?",
    "What does a 429 response mean and how should a client react?",
]

ATTACKS = [
    "__import__('os').system('echo pwned')",
    "().__class__.__base__.__subclasses__()",
    "(9).__class__.__mro__",
    "(9).real",
    "open('/etc/passwd').read()",
    "eval('1+1')",
    "2 ** 10000000",
    "[x for x in range(10**9)]",
    "(lambda: 1)()",
    "1 if 1 else 2",
    "globals()",
    "1/0",
]


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# -- fault-injection tools, defined here rather than shipped in tools.py ----
# They exist only to prove the sandbox guarantees. Shipping them in the real
# tool module would mean an agent could call them.

@tool(tags=("fault",), timeout_s=0.2)
def sleeper(seconds: float = 1.0) -> str:
    """Sleep for a while, to prove the wall-clock timeout fires.

    Args:
        seconds: How long to sleep.
    """
    time.sleep(seconds)
    return "finished"


@tool(tags=("fault",))
def firehose(size: int = 20000) -> str:
    """Return a payload far larger than the sandbox output cap.

    Args:
        size: Number of characters to return.
    """
    return "x" * size


_flaky_state = {"calls": 0}


@tool(tags=("fault",))
def flaky() -> str:
    """Fail twice with a transient error, then succeed."""
    _flaky_state["calls"] += 1
    if _flaky_state["calls"] <= 2:
        raise TransientToolError("upstream connection reset")
    return "recovered"


@tool(tags=("fault",))
def always_broken() -> str:
    """Always fail, to drive the circuit breaker open."""
    raise RuntimeError("dependency is down")


def section_registry(registry):
    rule("1. Registry, discovery and generated schemas")
    print(f"{len(registry)} tool version(s) registered, tags: {registry.tags()}\n")
    for spec in registry.list():
        params = ", ".join(spec.parameters["properties"])
        print(f"  {spec.qualified_name:<22} tags={list(spec.tags)!s:<22} params=({params})")
        print(f"    {spec.description}")
    print("\nFiltering by tag is how an agent gets a task-relevant subset:")
    print(f"  tags=('math',)   -> {registry.names(tags=('math',))}")
    print(f"  tags=('search',) -> {registry.names(tags=('search',))}")

    print("\nSchema generated from the search_docs signature and docstring:")
    print(json.dumps(search_docs.tool_spec.to_prompt_schema(), indent=2))


def section_versioning():
    """Uses its own registry: registering v2 into the main one would silently
    upgrade every later call in this demo, which is exactly the blast radius
    versioning exists to control."""
    rule("2. Versioning: two versions of one tool side by side")
    registry = ToolRegistry()
    registry.register_all(ALL_TOOLS)

    @tool(name="search_docs", version="2.0.0", tags=("search", "read"),
          constraints={"k": {"minimum": 1, "maximum": 10}})
    def search_docs_v2(query: str, k: int = 5) -> str:
        """Search the docs, returning up to ten passages instead of five.

        Args:
            query: What to look for.
            k: How many passages to return.
        """
        return "v2"

    registry.register(search_docs_v2)
    print(f"  registry.get('search_docs')          -> {registry.get('search_docs').qualified_name}")
    print(f"  registry.get('search_docs', '1.0.0') -> {registry.get('search_docs', '1.0.0').qualified_name}")
    print(f"  default k maximum, v1 -> {registry.get('search_docs', '1.0.0').parameters['properties']['k']['maximum']}")
    print(f"  default k maximum, v2 -> {registry.get('search_docs', '2.0.0').parameters['properties']['k']['maximum']}")
    print("  list() shows one entry per name; list(all_versions=True) shows both:")
    print(f"    {[s.qualified_name for s in registry.list()]}")
    print(f"    {[s.qualified_name for s in registry.list(all_versions=True)]}")


def section_real_calls(sandbox):
    rule("3. Real tool calls through the sandbox")
    calls = [
        ("calculate", {"expression": "600 * 60 / 1000"}),
        ("calculate", {"expression": "round(sqrt(2) * 100, 2)"}),
        ("search_docs", {"query": "how long before a token expires", "k": 2}),
        ("convert_units", {"value": 5, "from_unit": "mi", "to_unit": "km"}),
        ("convert_units", {"value": 37, "from_unit": "c", "to_unit": "f"}),
        ("current_time", {"offset_hours": 6}),
        ("current_time", {"offset_hours": "6"}),
        ("convert_units", {"value": 5, "from_unit": "km", "to_unit": "kg"}),
        ("calculate", {"expression": "1 + 1", "bogus": True}),
        ("calculate", {"precision": 2}),
        ("teleport", {"to": "mars"}),
    ]
    print(f"{'tool':<14} {'outcome':<16} observation")
    print("-" * 78)
    for name, args in calls:
        result = sandbox.call(name, args)
        outcome = result.audit.outcome if result.audit else "?"
        print(f"{name:<14} {outcome:<16} {result.observation()[:46]}")
    print("-" * 78)
    print("Note the last four: a semantic error the schema cannot express, an invented")
    print("argument, a missing required argument, and an unknown tool. All four come")
    print("back as structured errors the model can act on, none of them raise.")
    print("\nString-to-integer coercion is deliberate: current_time({'offset_hours': '6'})")
    print("succeeded, because a quoted number is a formatting slip, not a reasoning error.")


def section_adversarial():
    rule("4. Adversarial input to the calculator")
    print(f"{'expression':<42} {'rejected':<9} reason")
    print("-" * 78)
    rejected = 0
    for expression in ATTACKS:
        try:
            value = safe_eval(expression)
            print(f"{expression:<42} {'NO':<9} evaluated to {value}")
        except UnsafeExpression as exc:
            rejected += 1
            print(f"{expression:<42} {'yes':<9} {exc}")
    print("-" * 78)
    print(f"{rejected}/{len(ATTACKS)} rejected. The allowlist is over AST node types, so")
    print("anything not in the arithmetic grammar fails closed without being named.")
    return rejected


def section_guarantees():
    rule("5. Sandbox guarantees")
    registry = ToolRegistry()
    registry.register_all([sleeper, firehose, flaky, always_broken])

    print("allowlist: a tool the agent was not granted is refused before it runs")
    restricted = ToolSandbox(registry, allowlist=["flaky"])
    result = restricted.call("sleeper", {"seconds": 0.01})
    print(f"  sleeper -> {result.observation()}")

    print("\ntimeout: sleeper has a 0.2s budget and is asked to sleep for 1.0s")
    now = [0.0]
    sandbox = ToolSandbox(registry, max_output_bytes=1024, clock=lambda: now[0])
    started = time.perf_counter()
    result = sandbox.call("sleeper", {"seconds": 1.0})
    elapsed = (time.perf_counter() - started) * 1000.0
    print(f"  sleeper -> {result.observation()}")
    print(f"  returned control after {elapsed:.0f}ms, not 1000ms")
    print("  the worker thread is abandoned, not killed; see the README on why")

    print("\noutput cap: firehose returns 20000 characters against a 1024 byte cap")
    result = sandbox.call("firehose", {"size": 20000})
    print(f"  firehose -> {result.observation()}")

    print("\nretry: flaky raises TransientToolError twice, then succeeds")
    result = sandbox.call("flaky")
    print(f"  flaky -> {result.observation()} after {result.audit.attempts} attempt(s)")

    print("\ncircuit breaker: always_broken trips after 3 consecutive failures")
    for i in range(5):
        result = sandbox.call("always_broken")
        print(f"  call {i + 1}: {result.audit.outcome:<14} {result.observation()[:48]}")
    print(f"  breaker state: {sandbox.breaker('always_broken').state}")
    now[0] = 31.0
    result = sandbox.call("always_broken")
    print(f"  after the 30s cooldown the breaker half-opens and lets one call through:"
          f" {result.audit.outcome}")
    return sandbox


def section_agent(sandbox):
    rule("6. The agent loop")
    print("EchoLLM is a deterministic rule engine, not a reasoner: given an enum of")
    print("actions it samples one, so the tool it picks is often not the tool a real")
    print("model would pick. What this section demonstrates is the control flow, the")
    print("error feedback and the step bounding, which is the part this project owns.\n")

    agent = ToolAgent(EchoLLM(), sandbox, max_steps=5)
    runs = []
    for question in QUESTIONS:
        run = agent.run(question)
        runs.append(run)
        print(f"Q: {question}")
        for step in run.steps:
            print(f"   step {step.index}: {step.action:<14} {step.outcome:<14} {step.observation[:52]}")
        print(f"   stopped: {run.stopped_because}, {run.llm_calls} model calls, "
              f"{run.tool_calls} tool calls ({run.failed_tool_calls} failed)")
        print(f"   answer: {run.answer[:150]}\n")

    total_tools = sum(r.tool_calls for r in runs)
    failed = sum(r.failed_tool_calls for r in runs)
    print(f"{len(runs)} runs: {sum(r.llm_calls for r in runs)} model calls, "
          f"{total_tools} tool calls, {total_tools - failed} succeeded, {failed} failed")
    print(f"budget exhausted on {sum(1 for r in runs if r.stopped_because == 'step_budget')} of {len(runs)} runs")
    print(f"every run terminated: max steps observed = {max(len(r.steps) for r in runs)} (budget 5)")
    print("\nThe calculator's circuit opens partway through: EchoLLM cannot synthesise a")
    print("valid arithmetic string, so after three rejections the breaker stops the loop")
    print("paying for a dependency that cannot succeed for this planner. That is the")
    print("breaker doing its job, and it is visible in the rollup below.")
    return runs


def section_audit(sandbox):
    rule("7. Audit log rollup, every call the agent made")
    print(f"{'tool':<16} {'calls':>6} {'ok':>5} {'failed':>7} {'avg ms':>8}  outcomes")
    print("-" * 78)
    for name, agg in sorted(sandbox.stats().items()):
        print(f"{name:<16} {agg['calls']:>6} {agg['ok']:>5} {agg['failed']:>7} "
              f"{agg['avg_ms']:>8.3f}  {agg['outcomes']}")
    print("-" * 78)
    print(f"{len(sandbox.audit_log)} audit records; every call, including refusals, has one.")
    print("\nOne record in full:")
    print(json.dumps(sandbox.audit_log[0].to_dict(), indent=2, default=str))


def main():
    print("Tool-Calling Framework")
    print("provider: EchoLLM (offline, deterministic); clock pinned for reproducibility")

    registry = ToolRegistry()
    registry.register_all(ALL_TOOLS)
    sandbox = ToolSandbox(registry, max_output_bytes=4096,
                          retry_policy=RetryPolicy(attempts=3, base_delay=0.0, jitter="none"))

    section_registry(registry)
    section_versioning()
    section_real_calls(sandbox)
    section_adversarial()
    section_guarantees()

    agent_registry = ToolRegistry()
    agent_registry.register_all(ALL_TOOLS)
    agent_sandbox = ToolSandbox(agent_registry)
    section_agent(agent_sandbox)
    section_audit(agent_sandbox)
    print()


if __name__ == "__main__":
    main()
