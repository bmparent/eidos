"""Local-only Observatory HTTP UI. Run: python server.py --port 8765."""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import signal
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from telemetry import Observatory, canonical

WEB_ROOT = Path(__file__).resolve().parent / "web"


def make_handler(observatory):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SentinelObservatory/0.1"

        def log_message(self, *args):
            pass  # Do not log request paths, payloads, IPs, or headers.

        def _trusted_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def _headers(self, status, mime, length):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
            self.end_headers()

        def _json(self, value, status=200):
            data = canonical(value).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", len(data))
            self.wfile.write(data)

        def do_GET(self):
            if not self._trusted_host():
                return self._json({"error": "Untrusted Host"}, 403)
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.query or parsed.fragment:
                return self._json({"error": "Query strings are not supported"}, 400)
            if parsed.path == "/api/snapshot":
                return self._json(observatory.snapshot())
            if parsed.path.startswith("/api/events/"):
                eid = parsed.path.removeprefix("/api/events/")
                if not re.fullmatch(r"evt_[a-f0-9]{32}", eid):
                    return self._json({"error": "Invalid evidence ID"}, 400)
                event = observatory.store.get(eid)
                return self._json(event if event else {"error": "Evidence not found or expired from bounded retention"}, 200 if event else 404)
            if parsed.path.startswith("/api/"):
                return self._json({"error": "Unknown endpoint"}, 404)
            requested = "index.html" if parsed.path == "/" else urllib.parse.unquote(parsed.path).lstrip("/")
            path = (WEB_ROOT / requested).resolve()
            if not path.is_relative_to(WEB_ROOT) or not path.is_file() or path.suffix.lower() not in (".html", ".js", ".css", ".svg", ".png", ".woff2", ".ico"):
                return self._json({"error": "Not found"}, 404)
            data = path.read_bytes()
            self._headers(200, mimetypes.guess_type(path)[0] or "application/octet-stream", len(data))
            self.wfile.write(data)

        def do_POST(self):
            # Exact Host/Origin equality prevents cross-site and DNS-rebinding probes.
            if not self._trusted_host() or self.headers.get("Origin") != f"http://{self.headers.get('Host')}":
                return self._json({"error": "Same-origin requests required"}, 403)
            if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin"):
                return self._json({"error": "Same-origin requests required"}, 403)
            if self.path != "/api/probe":
                return self._json({"error": "Unknown endpoint"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json({"error": "Invalid content length"}, 400)
            if length > 2 or length < 0 or self.headers.get("Transfer-Encoding"):
                return self._json({"error": "Probe accepts no request parameters"}, 400)
            body = self.rfile.read(length) if length else b""
            if body not in (b"", b"{}"):
                return self._json({"error": "Probe accepts no request parameters"}, 400)
            result = observatory.request_probe()
            return self._json(result, 202 if result["accepted"] else 429)

        def do_OPTIONS(self):
            return self._json({"error": "Cross-origin access is disabled"}, 405)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between same-route HEAD requests; minimum 5")
    parser.add_argument("--db", default=str(Path(__file__).resolve().parent / "runtime" / "observatory.sqlite3"))
    parser.add_argument("--no-collect", action="store_true", help="Disable recurring checks; explicit probe remains available")
    parser.add_argument("--export", metavar="JSON_PATH", help="Verify journal, export a snapshot, and exit without collecting")
    args = parser.parse_args()
    observatory = Observatory(args.db, args.interval, collect=not args.no_collect)
    if args.export:
        Path(args.export).write_text(json.dumps(observatory.snapshot(), indent=2), encoding="utf-8")
        print(f"Verified snapshot exported to {args.export}")
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(observatory))
    server.daemon_threads = True
    observatory.start()
    def shutdown(*_):
        observatory.stop_event.set()
        observatory.wake.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Sentinel Observatory http://127.0.0.1:{args.port} · public HEAD only · interval {args.interval:g}s", flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        observatory.stop()
        server.server_close()


if __name__ == "__main__":
    main()
