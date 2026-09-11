#!/usr/bin/env python3
"""Smoke test every hosted demo: it must load, run, stay fast and return text.

The serverless function has a hard time limit and a read-only filesystem, so an
adapter that is slow or writes to disk fails here rather than in production.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from web import registry  # noqa: E402

BUDGET_S = 8.0
EXPECTED = 15


def main() -> int:
    mods = registry.all_systems()
    failures = []
    print(f"{len(mods)} adapter(s) loaded\n")

    for mod in mods:
        meta = registry.meta(mod)
        example = (meta["examples"] or [""])[0]
        for label, value in (("example", example), ("empty", "")):
            start = time.perf_counter()
            try:
                out = mod.run(value)
                err = None
            except Exception as exc:
                out, err = "", f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - start

            if err:
                failures.append((meta["slug"], label, err))
                status = "RAISED"
            elif not isinstance(out, str) or not out.strip():
                failures.append((meta["slug"], label, "empty output"))
                status = "EMPTY"
            elif elapsed > BUDGET_S:
                failures.append((meta["slug"], label, f"{elapsed:.1f}s over the {BUDGET_S}s budget"))
                status = "SLOW"
            else:
                status = "ok"
            print(f"  {meta['number']:02d} {meta['slug']:<24} {label:<8} {elapsed:6.2f}s  {len(out):>6} chars  {status}")

    print()
    if len(mods) != EXPECTED:
        failures.append(("registry", "count", f"expected {EXPECTED} adapters, found {len(mods)}"))
    for slug, label, why in failures:
        print(f"FAIL {slug} [{label}]: {why}")
    print(f"{'all adapters healthy' if not failures else str(len(failures)) + ' failure(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
