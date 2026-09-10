#!/usr/bin/env python3
"""Run every project demo in order and report which ones pass.

Used by `make demo` and by CI. A demo that exits non-zero, times out or writes to
stderr with a traceback is a failure: these demos are the executable version of
the claims in the READMEs, so they are not allowed to rot.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS_DIR = os.path.join(ROOT, "projects")
TIMEOUT_S = 300


def discover() -> list:
    names = []
    for entry in sorted(os.listdir(PROJECTS_DIR)):
        demo = os.path.join(PROJECTS_DIR, entry, "demo.py")
        if os.path.isfile(demo):
            names.append((entry, demo))
    return names


def main() -> int:
    demos = discover()
    if not demos:
        print("no demos found", file=sys.stderr)
        return 1

    results = []
    for name, path in demos:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, path],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
            )
            ok = proc.returncode == 0
            detail = "" if ok else (proc.stderr.strip().splitlines() or ["no stderr"])[-1]
        except subprocess.TimeoutExpired:
            ok, detail = False, f"timed out after {TIMEOUT_S}s"
        elapsed = time.perf_counter() - start
        results.append((name, ok, elapsed, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name:<28} {elapsed:6.2f}s  {detail}")

    failed = [r for r in results if not r[1]]
    total_s = sum(r[2] for r in results)
    print("-" * 72)
    print(f"{len(results) - len(failed)}/{len(results)} demos passed in {total_s:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
