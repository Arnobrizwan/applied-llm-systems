"""A thin http.server adapter over Gateway.

ThreadingHTTPServer rather than the single-threaded HTTPServer: the rate limit
and budget code is only interesting if concurrent requests can actually
contend for the same tenant's bucket, and a single-threaded server would
serialise that contention away and make the locking look unnecessary.

Binding to port 0 asks the kernel for a free ephemeral port. Hardcoding 8080
in a demo means the demo fails on any machine where something else already
holds 8080, which for a portfolio repo is the reviewer's machine.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Tuple

from .gateway import Gateway, Response


def make_handler(gateway: Gateway):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "multi-tenant-llm-api/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            """Silence the default stderr access log; the gateway logs structurally."""

        def _dispatch(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError:
                resp = Response(400, {"error": "bad_request", "message": "body is not valid JSON"})
            else:
                path = self.path.split("?", 1)[0]
                resp = gateway.handle(method, path, dict(self.headers), body)
            payload = resp.json_bytes()
            self.send_response(resp.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for k, v in resp.headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

    return Handler


def serve(gateway: Gateway, host: str = "127.0.0.1", port: int = 0) -> Tuple[ThreadingHTTPServer, str, threading.Thread]:
    """Start the server on an ephemeral local port. Returns (server, base_url, thread)."""
    httpd = ThreadingHTTPServer((host, port), make_handler(gateway))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, name="mt-api", daemon=True)
    thread.start()
    return httpd, f"http://{host}:{httpd.server_address[1]}", thread
