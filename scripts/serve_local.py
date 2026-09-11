#!/usr/bin/env python3
"""Serve the hosted demos locally with the exact handler Vercel runs.

    python3 scripts/serve_local.py 8000
"""
from __future__ import annotations

import os
import sys
from http.server import HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "api"))

from api.index import handler  # noqa: E402


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"http://127.0.0.1:{port}/")
    HTTPServer(("127.0.0.1", port), handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
