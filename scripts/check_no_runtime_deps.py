#!/usr/bin/env python3
"""Fail if any runtime module imports a third-party package.

The promise this repo makes is that it runs on a stock Python install with no
paid API and no wheels to build. A promise with nothing enforcing it is a README
claim, so this check runs in CI. `pytest` is allowed, but only inside tests/.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import sysconfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED_EXTRA = {"pytest", "llmkit", "projects", "scripts", "web", "api", "web", "api", "conftest"}

# Optional accelerators that may only ever be imported lazily, inside a guard,
# and must have a working standard-library fallback. Anything here that is
# imported unconditionally at module scope is still a violation.
OPTIONAL_LAZY = {"tiktoken"}


def _is_stdlib(mod: str) -> bool:
    """True if `mod` ships with Python.

    `sys.stdlib_module_names` exists from 3.10. On 3.9 we fall back to resolving
    the module and checking where it lives: anything under the interpreter's own
    stdlib directory is standard library, anything under site-packages is not.
    Resolving beats hard-coding a list, which is what made the first version of
    this check fail CI on 3.9 by not knowing about `__future__`.
    """
    if mod in ("__future__", "__main__"):
        return True
    if mod in sys.builtin_module_names:
        return True
    names = getattr(sys, "stdlib_module_names", None)
    if names is not None:
        return mod in names
    try:  # Python 3.9
        spec = importlib.util.find_spec(mod)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    origin = spec.origin or ""
    if origin in ("built-in", "frozen"):
        return True
    stdlib_dir = sysconfig.get_paths().get("stdlib", "")
    return bool(stdlib_dir) and origin.startswith(stdlib_dir) and "site-packages" not in origin


def collect_imports(path: str):
    """Return (hard, lazy) module name sets.

    `hard` = imported unconditionally when the module is loaded.
    `lazy` = imported inside a try/if/function, i.e. behind a guard with a fallback.
    """
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    def names(node) -> set:
        if isinstance(node, ast.Import):
            return {a.name.split(".")[0] for a in node.names}
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            return {node.module.split(".")[0]}
        return set()

    hard = set()
    for stmt in tree.body:  # module scope only
        hard |= names(stmt)

    all_imports = set()
    for node in ast.walk(tree):
        all_imports |= names(node)
    return hard, all_imports - hard


def main() -> int:
    violations = []

    def allowed(mod: str) -> bool:
        return mod in ALLOWED_EXTRA or _is_stdlib(mod)

    for base in ("llmkit", "projects", "scripts", "web", "api"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, base)):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            in_tests = os.sep + "tests" in dirpath + os.sep
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, ROOT)
                hard, lazy = collect_imports(path)
                for mod in sorted(hard):
                    if allowed(mod):
                        if mod == "pytest" and not in_tests:
                            violations.append((rel, "pytest outside tests/"))
                        continue
                    violations.append((rel, mod))
                for mod in sorted(lazy):
                    if mod in OPTIONAL_LAZY or allowed(mod):
                        continue
                    violations.append((rel, f"{mod} (lazy import, still third-party)"))

    for path, mod in violations:
        print(f"{path}: disallowed runtime import {mod}")
    if violations:
        print(f"\n{len(violations)} violation(s): runtime code must use the standard library only.")
        return 1
    print("ok: no third-party runtime imports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
