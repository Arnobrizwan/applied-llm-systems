"""Vercel Python entry point.

One serverless function serves the whole site: an index of the fifteen systems
and a page per system that executes the real project code on request. Routing is
done here rather than with one function per page so a cold start warms the
shared library once for every route.
"""
from __future__ import annotations

import os
import sys
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web import registry, render  # noqa: E402

MAX_INPUT = 2000
MAX_OUTPUT = 12000


def _run_adapter(mod, user_input: str):
    started = time.perf_counter()
    try:
        out = mod.run(user_input)
    except Exception as exc:  # an adapter must never take the page down
        out = f"This demo failed: {type(exc).__name__}: {exc}"
    if not isinstance(out, str):
        out = str(out)
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + "\n\n... output truncated."
    return out, (time.perf_counter() - started) * 1000.0


class handler(BaseHTTPRequestHandler):
    server_version = "applied-llm-systems"

    def _send(self, body: str, status: int = 200) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "public, max-age=60")
        self.end_headers()
        self.wfile.write(payload)

    def _route(self, method: str) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        # Vercel rewrites every route to this function, and the function only
        # ever sees the rewritten path (/api/index). vercel.json therefore
        # carries the original path through in __path; locally there is no
        # rewrite, so fall back to the real path.
        path = (query.get("__path") or [parsed.path])[0]
        path = path.split("?")[0].rstrip("/") or "/"

        if path in ("/", "/index.html"):
            systems = [registry.meta(m) for m in registry.all_systems()]
            self._send(render.index_page(systems))
            return

        if path == "/health":
            n = len(registry.all_systems())
            self._send(f"<pre>ok: {n} adapters loaded</pre>", 200 if n == 15 else 500)
            return

        if path.startswith("/s/"):
            slug = path[3:]
            mod = registry.by_slug(slug)
            if mod is None:
                self._send(render.not_found(), 404)
                return
            meta = registry.meta(mod)

            user_input = ""
            if method == "POST":
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length).decode("utf-8", "replace") if length else ""
                user_input = (parse_qs(raw).get("q") or [""])[0]
            else:
                user_input = (query.get("q") or [""])[0]

            user_input = user_input[:MAX_INPUT]
            if method == "POST" or user_input:
                out, ms = _run_adapter(mod, user_input)
                self._send(render.system_page(meta, user_input, out, ms))
            else:
                self._send(render.system_page(meta))
            return

        self._send(render.not_found(), 404)

    def do_GET(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")

    def log_message(self, fmt, *args):  # keep the function logs readable
        print(fmt % args)
