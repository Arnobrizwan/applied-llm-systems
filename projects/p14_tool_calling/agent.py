"""The tool-calling loop: decide, fill arguments, execute, observe, answer.

Two model calls per step instead of one. The first picks an action from an enum
of tool names plus `final_answer`; the second fills in the arguments using that
tool's own parameter schema as the constrained-output schema. The single-call
alternative needs a union schema ("arguments is an object whose shape depends on
the value of action"), which JSON Schema can only express with `oneOf` and which
constrained decoders handle badly. Splitting the decision means every call is
constrained by a flat, concrete schema, which is the shape that actually works.

Observations are written back as `[S1] ... [S2] ...` evidence blocks. That is
not decoration: it gives the final answer something citable, so a reader can see
which tool result an answer came from. An agent that produces an unattributable
final sentence is an agent you cannot debug.

Termination is enforced by the loop, not requested of the model. The model
chooses `final_answer` when it wants to stop, and the loop forces a final answer
when the step budget runs out. A loop that trusts a model to stop is a loop that
eventually does not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llmkit import LLMProvider, Tracer

from .registry import ToolRegistry
from .sandbox import ToolSandbox

__all__ = ["ToolAgent", "AgentRun", "Step"]

_SYSTEM = (
    "You are an agent that answers questions by calling tools.\n"
    "Pick one action per step. Call a tool when you need a fact you do not have, "
    "and pick final_answer once the evidence below is enough to answer.\n"
)


@dataclass
class Step:
    """One decision plus its consequence."""

    index: int
    action: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    outcome: str = "ok"
    observation: str = ""
    thought: str = ""


@dataclass
class AgentRun:
    """The whole run: what it did, why it stopped, and what it cost."""

    question: str
    answer: str
    steps: List[Step] = field(default_factory=list)
    stopped_because: str = "final_answer"
    llm_calls: int = 0
    tokens: int = 0

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action != "final_answer")

    @property
    def failed_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action != "final_answer" and not s.ok)


class ToolAgent:
    """Drives an LLM through a registry of tools under a step budget."""

    def __init__(
        self,
        llm: LLMProvider,
        sandbox: ToolSandbox,
        *,
        tags: Sequence[str] = (),
        max_steps: int = 6,
        tracer: Optional[Tracer] = None,
    ):
        self.llm = llm
        self.sandbox = sandbox
        self.registry: ToolRegistry = sandbox.registry
        self.tags = tuple(tags)
        self.max_steps = max_steps
        self.tracer = tracer or Tracer("tool-agent")

    # -- prompt pieces ---------------------------------------------------
    def _action_names(self) -> List[str]:
        return self.registry.names(tags=self.tags) + ["final_answer"]

    def _decision_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "One sentence on why this action."},
                "action": {"type": "string", "enum": self._action_names()},
            },
            "required": ["thought", "action"],
            "additionalProperties": False,
        }

    def _context(self, steps: List[Step]) -> str:
        catalog = self.registry.render_prompt(tags=self.tags)
        if not steps:
            return _SYSTEM + catalog + "\n\nNo tool results yet."
        lines = ["", "Evidence gathered so far:"]
        for i, step in enumerate(steps, start=1):
            lines.append(f"[S{i}] {step.action} returned {step.observation}")
        return _SYSTEM + catalog + "\n".join(lines)

    # -- model calls -----------------------------------------------------
    def _ask(self, run: AgentRun, system: str, question: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        response = self.llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": question}],
            json_schema=schema,
        )
        run.llm_calls += 1
        run.tokens += response.total_tokens
        try:
            value = json.loads(response.text)
        except ValueError:
            # A malformed decision is treated as "stop": project 02 is where
            # repair and re-prompting belong, and duplicating that logic here
            # would give the repo two versions of it to keep in step.
            return {}
        return value if isinstance(value, dict) else {}

    # -- main loop -------------------------------------------------------
    def run(self, question: str) -> AgentRun:
        run = AgentRun(question=question, answer="")
        with self.tracer.span("agent.run", question=question[:120]):
            for index in range(self.max_steps):
                system = self._context(run.steps)

                with self.tracer.span("agent.decide", step=index):
                    decision = self._ask(run, system, question, self._decision_schema())
                action = decision.get("action") or "final_answer"
                thought = decision.get("thought", "")

                if action == "final_answer":
                    run.stopped_because = "final_answer"
                    break

                spec = self.registry.find(action)
                if spec is None:
                    # Only reachable if the enum and the registry disagree, which
                    # is a bug, not a model failure. Recorded rather than raised.
                    run.steps.append(Step(index=index, action=action, ok=False,
                                          outcome="unknown_tool",
                                          observation=f"ERROR unknown_tool: {action}", thought=thought))
                    continue

                with self.tracer.span("agent.arguments", step=index, tool=spec.name):
                    arguments = self._ask(run, system, question, spec.parameters)

                with self.tracer.span("tool.call", step=index, tool=spec.name) as span:
                    result = self.sandbox.call(spec.name, arguments)
                    span.attributes["outcome"] = result.audit.outcome if result.audit else "unknown"

                run.steps.append(
                    Step(
                        index=index,
                        action=spec.name,
                        arguments=arguments,
                        ok=result.ok,
                        outcome=result.audit.outcome if result.audit else "unknown",
                        observation=result.observation(),
                        thought=thought,
                    )
                )
            else:
                run.stopped_because = "step_budget"

            with self.tracer.span("agent.final"):
                final = self._ask(
                    run,
                    self._context(run.steps),
                    question,
                    {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                )
            run.answer = final.get("answer") or "No answer could be produced."
        return run
