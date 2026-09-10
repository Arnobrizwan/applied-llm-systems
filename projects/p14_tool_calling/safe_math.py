"""An arithmetic evaluator built on `ast`, because `eval` is not an option.

`eval("__import__('os').system('rm -rf /')")` is the obvious attack, but the
interesting ones are quieter. `().__class__.__base__.__subclasses__()` walks
from a literal to every class loaded in the process. `(9).__class__` gets you
`int`, and from there `__mro__` and the same walk. `9**9**9**9` never returns
and pins a core. None of those need an import statement, so a blocklist of
keywords does not stop them.

The approach here is an allowlist over the parsed AST: parse the expression,
walk every node, and reject anything that is not one of a small set of
arithmetic node types. A blocklist was rejected outright because it fails open,
which is the wrong direction for a component whose input is attacker-influenced
by definition (the model's input is the user's text).

Two limits that are not about parsing: the expression length is capped, and the
exponent is capped, because `2**10000000` is perfectly valid arithmetic and
still a denial of service.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Dict, Union

__all__ = ["safe_eval", "UnsafeExpression", "ALLOWED_FUNCTIONS", "ALLOWED_CONSTANTS"]

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 64


class UnsafeExpression(ValueError):
    """The expression contains something outside the arithmetic allowlist."""


_BINARY: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

ALLOWED_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}

ALLOWED_CONSTANTS: Dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _evaluate(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        # Only numbers. A bare string constant is harmless on its own but it is
        # the first half of every attribute-access trick, so it is refused.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpression(f"only numeric literals are allowed, got {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BINARY.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"operator {type(node.op).__name__} is not allowed")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise UnsafeExpression(f"exponent {right} exceeds the limit of {MAX_EXPONENT}")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"unary operator {type(node.op).__name__} is not allowed")
        return op(_evaluate(node.operand))

    if isinstance(node, ast.Name):
        if node.id in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[node.id]
        raise UnsafeExpression(f"unknown name {node.id!r}")

    if isinstance(node, ast.Call):
        # The callee must be a bare allowlisted name. Anything computed, such as
        # `getattr(x, 'y')()` or `f[0]()`, is refused before it is evaluated.
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpression("only calls to a plain allowlisted function name are permitted")
        if node.func.id not in ALLOWED_FUNCTIONS:
            raise UnsafeExpression(f"function {node.func.id!r} is not allowed")
        if node.keywords:
            raise UnsafeExpression("keyword arguments are not allowed")
        args = [_evaluate(a) for a in node.args]
        return ALLOWED_FUNCTIONS[node.func.id](*args)

    raise UnsafeExpression(f"{type(node).__name__} is not allowed in an expression")


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression, or raise UnsafeExpression.

    Every rejection path raises the same exception type so the tool layer can
    turn it into one structured error shape rather than leaking Python
    internals (a raw SyntaxError message quoting the offending source is a small
    but real information leak back to whoever wrote the prompt).
    """
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpression("expression must be a non-empty string")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise UnsafeExpression(f"expression exceeds {MAX_EXPRESSION_CHARS} characters")
    if "__" in expression:
        # Redundant with the AST walk (Attribute nodes are refused anyway) and
        # kept as a cheap outer guard, since every known escape from a literal
        # goes through a dunder.
        raise UnsafeExpression("double underscore is not allowed")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"could not parse expression: {exc.msg}") from exc
    try:
        result = _evaluate(tree)
    except ZeroDivisionError as exc:
        raise UnsafeExpression("division by zero") from exc
    except (OverflowError, ValueError) as exc:
        if isinstance(exc, UnsafeExpression):
            raise
        raise UnsafeExpression(f"arithmetic error: {exc}") from exc
    return float(result)
