"""Project 14: Tool-Calling Framework.

Typed function schemas derived from type hints and docstrings, a versioned
registry with discovery, argument coercion with structured errors, guarded
execution (allowlist, timeout, output cap, retries, circuit breaker, audit log)
and a bounded tool-calling loop.
"""
from .agent import AgentRun, Step, ToolAgent
from .registry import ToolNotFound, ToolRegistry
from .safe_math import UnsafeExpression, safe_eval
from .sandbox import AuditRecord, ToolResult, ToolSandbox, ToolTimeout, TransientToolError
from .schema import ArgError, ToolSpec, coerce_arguments, tool
from .tools import ALL_TOOLS, calculate, convert_units, current_time, search_docs

__all__ = [
    "AgentRun", "Step", "ToolAgent",
    "ToolNotFound", "ToolRegistry",
    "UnsafeExpression", "safe_eval",
    "AuditRecord", "ToolResult", "ToolSandbox", "ToolTimeout", "TransientToolError",
    "ArgError", "ToolSpec", "coerce_arguments", "tool",
    "ALL_TOOLS", "calculate", "convert_units", "current_time", "search_docs",
]
