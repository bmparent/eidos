"""Loopback-only public URL/IP observer. Standard library; no daemon or model.

DNS answers are validated and one numeric address is pinned to the connection.
No redirects, cookies, authorization, response bodies or private addresses.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import json
import mimetypes
import queue
import re
import socket
import ssl
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent
TIMEOUT_SECONDS = 8.0
INTERVAL_SECONDS = 15.0
CAPABILITIES = {"mode": "local", "direct_ip": True, "interval_ms": 15000,
                "model": {"connected": False}, "recorded_url": None}
SECURITY_KEYS = ("hsts_present", "csp_present", "x_content_type_options", "frame_options", "referrer_policy", "cache_private", "cache_no_store")
_DNS_SLOTS = threading.BoundedSemaphore(2)


class BoundaryError(ValueError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Observation deadline exceeded")
    return value


def public_address(value):
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise BoundaryError("Invalid numeric address") from exc
    if (not address.is_global or address.is_private or address.is_reserved
            or address.is_loopback or address.is_link_local or address.is_multicast
            or address.is_unspecified or getattr(address, "is_site_local", False)
            or getattr(address, "ipv4_mapped", None)):
        raise BoundaryError("Only globally routable public addresses are permitted")
    # Conservative special-use exclusions also match the hosted companion.
    special = ("192.0.0.0/24", "192.88.99.0/24") if address.version == 4 else ("2001::/23", "2001:db8::/32", "2002::/16", "3fff::/20")
    if any(address in ipaddress.ip_network(network) for network in special):
        raise BoundaryError("Special-use address ranges are not permitted")
    return address


@dataclass(frozen=True)
class Target:
    url: str
    scheme: str
    host: str
    port: int
    path: str
    direct_ip: bool

    @property
    def authority(self):
        return f"[{self.host}]" if ":" in self.host else self.host

    @property
    def label(self):
        return self.authority + ("" if self.path == "/" else self.path)

    @property
    def id(self):
        value = 0xcbf29ce484222325
        for byte in self.url.encode("utf-8"):
            value = ((value ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
        return f"target_{value:016x}"


def normalize_target(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 2048:
        raise BoundaryError("Enter a public URL or IP address of at most 2048 characters")
    if value != value.strip() or any(ord(c) <= 32 or ord(c) == 127 for c in value) or "\\" in value:
        raise BoundaryError("Whitespace, control characters and backslashes are not allowed")
    if "%" in value.split("/", 1)[0]:
        raise BoundaryError("Encoded hostnames and address zones are not allowed")
    # Bare numeric IPv6 needs brackets before it can be parsed as an authority.
    bare_ip = None
    if "://" not in value:
        try:
            bare_ip = ipaddress.ip_address(value)
        except ValueError:
            pass
        value = "https://" + (f"[{bare_ip}]" if bare_ip and bare_ip.version == 6 else value)
    try:
        parts = urlsplit(value)
        port = parts.port
        host = parts.hostname
    except ValueError as exc:
        raise BoundaryError("Invalid URL or port") from exc
    if parts.scheme not in {"http", "https"} or not host:
        raise BoundaryError("Only public HTTP and HTTPS targets are supported")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment or "?" in value or "#" in value:
        raise BoundaryError("Credentials, query strings and fragments are not collected")
    expected = 443 if parts.scheme == "https" else 80
    if port not in (None, expected):
        raise BoundaryError("Only the scheme's standard port is permitted: HTTP 80 or HTTPS 443")
    if "%" in host:
        raise BoundaryError("Encoded hostnames and IPv6 zone identifiers are not allowed")
    host = host.rstrip(".").lower()
    direct_ip = False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise BoundaryError("Invalid hostname") from exc
        if len(host) > 253 or "." not in host or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in host.split(".")):
            raise BoundaryError("Enter a valid public DNS hostname")
        if host.split(".")[-1].isdigit() or host.endswith((".localhost", ".local", ".internal", ".intranet", ".lan", ".home", ".localdomain", ".invalid", ".test", ".example", ".onion", ".alt", ".arpa")):
            raise BoundaryError("Local and reserved hostnames are not permitted")
    else:
        public_address(str(address))
        host, direct_ip = str(address), True
    authority = f"[{host}]" if ":" in host else host
    path = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~")
    return Target(urlunsplit((parts.scheme, authority, path, "", "")), parts.scheme, host, expected, path, direct_ip)


def resolve_public(target, deadline, resolver=socket.getaddrinfo):
    if target.direct_ip:
        address = public_address(target.host)
        return socket.AF_INET6 if address.version == 6 else socket.AF_INET, str(address)
    if not _DNS_SLOTS.acquire(blocking=False):
        raise TimeoutError("DNS capacity is busy")
    result = queue.Queue(maxsize=1)
    def run():
        try:
            result.put((True, resolver(target.host, target.port, type=socket.SOCK_STREAM)))
        except Exception as exc:
            result.put((False, exc))
        finally:
            _DNS_SLOTS.release()
    threading.Thread(target=run, daemon=True, name="observatory-dns").start()
    try:
        ok, value = result.get(timeout=remaining(deadline))
    except queue.Empty as exc:
        raise TimeoutError("DNS deadline exceeded") from exc
    if not ok:
        raise socket.gaierror("DNS lookup failed")
    addresses = []
    for family, socktype, protocol, canonname, sockaddr in value:
        if family not in (socket.AF_INET, socket.AF_INET6) or socktype != socket.SOCK_STREAM:
            continue
        address = public_address(sockaddr[0])  # Reject the entire mixed public/private answer.
        pair = (family, str(address))
        if pair not in addresses:
            addresses.append(pair)
    if not addresses:
        raise socket.gaierror("No public addresses returned")
    remaining(deadline)
    return addresses[0]


class PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, target, address, deadline, *, socket_factory=socket.socket, tls_context=None):
        super().__init__(target.host, target.port, timeout=remaining(deadline))
        self.target, self.address, self.deadline = target, address, deadline
        self.socket_factory = socket_factory
        self.tls_context = tls_context
        self._watchdog = None

    def connect(self):
        family, numeric_ip = self.address
        sock = self.socket_factory(family, socket.SOCK_STREAM)
        self.sock = sock
        self._watchdog = threading.Timer(remaining(self.deadline), self._expire)
        self._watchdog.daemon = True
        self._watchdog.start()
        try:
            sock.settimeout(remaining(self.deadline))
            destination = (numeric_ip, self.target.port, 0, 0) if family == socket.AF_INET6 else (numeric_ip, self.target.port)
            sock.connect(destination)  # Numeric address; no second DNS lookup.
            if self.target.scheme == "https":
                context = self.tls_context or ssl.create_default_context()
                sock.settimeout(remaining(self.deadline))
                self.sock = context.wrap_socket(sock, server_hostname=self.target.host)
            self.sock.settimeout(remaining(self.deadline))
        except BaseException:
            self.close()
            raise

    def _expire(self):
        sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def close(self):
        if self._watchdog:
            self._watchdog.cancel()
        super().close()


def safe_headers(headers):
    def enum(key, allowed):
        value = headers.get(key)
        if value is None:
            return None
        normalized = value.lower().split(";", 1)[0].strip()
        return normalized if normalized in allowed else "other"
    return {"server": enum("server", {"cloudflare", "nginx", "apache", "vercel", "cloudfront"}),
            "content-type": enum("content-type", {"text/html", "text/plain", "application/json", "text/css", "application/javascript"}),
            "cf-cache-status": enum("cf-cache-status", {"hit", "miss", "dynamic", "bypass", "expired", "stale", "revalidated", "updating"})}


def security_headers(headers):
    def enum(key, allowed):
        value = headers.get(key)
        return "absent" if value is None else value.strip().lower() if value.strip().lower() in allowed else "other"
    directives = {v.split("=", 1)[0].strip().lower() for v in headers.get("cache-control", "").split(",")}
    return {"hsts_present": bool(headers.get("strict-transport-security")),
            "csp_present": bool(headers.get("content-security-policy")),
            "x_content_type_options": enum("x-content-type-options", {"nosniff"}),
            "frame_options": enum("x-frame-options", {"deny", "sameorigin"}),
            "referrer_policy": enum("referrer-policy", {"no-referrer", "same-origin", "origin", "strict-origin", "origin-when-cross-origin", "strict-origin-when-cross-origin", "unsafe-url", "no-referrer-when-downgrade"}),
            "cache_private": "private" in directives, "cache_no_store": "no-store" in directives}


def collect(target, *, resolver=socket.getaddrinfo, connection_factory=PinnedHTTPConnection, timeout=TIMEOUT_SECONDS):
    start = time.monotonic()
    deadline = start + timeout
    event = {"id": "evt_" + uuid.uuid4().hex, "source_id": target.id, "target": target.label,
             "url": target.url, "observed_at": utc_now(), "method": "HEAD", "kind": "measurement",
             "status": None, "error": None, "headers": {}, "security_headers": {key: None for key in SECURITY_KEYS},
             "redirect_count": 0, "response_chain_complete": False,
             "redirect": None, "latency_ms": None, "completed_at": None,
             "coverage": {"vantage": "Local public-only pinned-address HTTP client", "sampling": "One requested HEAD; no crawling or traffic capture",
                 "head_attempted": False, "redirect_followed": False, "body_collected": False,
                 "dns": {"state": "not_checked", "address_pinned": False},
                 "missing": ["Visitor traffic", "Origin logs", "Internal network", "Packet capture", "Certificate details", "Authenticated application state"],
                 "dns_rebinding": "Every returned DNS address is checked; one validated numeric address is pinned for this connection.",
                 "protections": "Observational only; no network protections or defensive actions changed"}}
    connection = None
    head_start = None
    try:
        address = resolve_public(target, deadline, resolver)
        event["coverage"]["dns"] = {"state": "literal_public_ip_pinned" if target.direct_ip else "public_address_pinned", "address_pinned": True,
                                        "resolver": "none: literal IP" if target.direct_ip else "system resolver; all returned addresses validated"}
        connection = connection_factory(target, address, deadline)
        event["observed_at"] = utc_now()
        head_start = time.monotonic()
        event["coverage"]["head_attempted"] = True
        connection.request("HEAD", target.path, headers={"Host": target.authority,
            "User-Agent": "Eidos-Sentinel-Observatory/0.2 (authorized read-only HEAD)", "Accept": "*/*", "Connection": "close", "Cache-Control": "no-cache", "Pragma": "no-cache"})
        response = connection.getresponse()
        remaining(deadline)
        event["status"] = response.status
        event["headers"] = safe_headers(response.headers)
        event["security_headers"] = security_headers(response.headers)
        event["redirect_present"] = 300 <= response.status < 400 and bool(response.headers.get("location"))
        event["response_chain_complete"] = not event["redirect_present"]
        if response.status in (301, 302, 303, 307, 308):
            event["redirect"] = {"followed": False, "target": None, "validation": "not_available"}
            location = response.headers.get("location")
            if location and len(location) <= 768:
                try:
                    next_target = normalize_target(urljoin(target.url, location))
                    event["redirect"].update({"target": next_target.url, "validation": "syntax_public_only_not_resolved_or_requested"})
                except BoundaryError:
                    event["redirect"]["validation"] = "blocked_or_unrepresentable"
        response.close()  # No response.read(), including on HTTP errors.
    except BoundaryError:
        event["error"] = {"code": "dns_non_public", "message": "DNS returned a non-public or reserved address; no target request was sent.", "scope": "dns_preflight"}
    except (TimeoutError, socket.timeout):
        event["error"] = {"code": "timeout", "message": "The observation deadline elapsed.", "scope": "observation_vantage"}
    except ssl.SSLError:
        event["error"] = {"code": "tls_error", "message": "TLS validation or negotiation failed; no insecure retry was made.", "scope": "observation_vantage"}
    except socket.gaierror:
        event["error"] = {"code": "dns_unavailable", "message": "No usable DNS answer was obtained; no target request was sent.", "scope": "dns_preflight"}
    except (OSError, http.client.HTTPException, UnicodeError):
        code = "timeout" if time.monotonic() >= deadline else "transport_error"
        event["error"] = {"code": code, "message": "The observer did not obtain a usable response; target availability is unknown.", "scope": "observation_vantage"}
    finally:
        if connection is not None:
            connection.close()
    event["completed_at"] = utc_now()
    event["latency_ms"] = round((time.monotonic() - head_start) * 1000, 3) if head_start is not None else None
    event["elapsed_ms"] = round((time.monotonic() - start) * 1000, 3)
    event["evidence_hash"] = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return event


class Observer:
    def __init__(self, collector=collect, clock=time.monotonic):
        self.collector, self.clock = collector, clock
        self.lock = threading.Lock()
        self.active = False
        self.attempts = OrderedDict()

    def observe(self, value):
        target = normalize_target(value)
        with self.lock:
            now = self.clock()
            if self.active:
                return 409, {"error": {"code": "busy", "message": "One observation is already running."}}
            last = self.attempts.get(target.url)
            if last is not None and now - last < INTERVAL_SECONDS:
                return 429, {"error": {"code": "cooldown", "message": "Wait at least 15 seconds between checks of this target."}, "retry_after_ms": int((INTERVAL_SECONDS - (now-last))*1000)+1}
            self.active = True
            self.attempts[target.url] = now
            self.attempts.move_to_end(target.url)
            while len(self.attempts) > 256:
                self.attempts.popitem(last=False)
        try:
            return 200, {"event": self.collector(target), "target": {"id": target.id, "url": target.url, "label": target.label},
                         "retry_after_ms": 15000, "rate_limit_scope": "this local process; one observation at a time and 15 seconds per target"}
        finally:
            with self.lock:
                self.active = False


def make_handler(observer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SentinelLocal/0.2"
        def log_message(self, format, *args):
            pass  # Never echo user targets or traffic into an access log.

        def _valid_host(self):
            hosts = self.headers.get_all("Host", [])
            return len(hosts) == 1 and hosts[0] in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def _send(self, status, body, content_type="application/json; charset=utf-8"):
            if isinstance(body, dict) and isinstance(body.get("error"), str):
                body = {**body, "error": {"code": body["error"], "message": body.get("message", body["error"].replace("_", " "))}}
                body.pop("message", None)
            payload = json.dumps(body, separators=(",", ":")).encode() if not isinstance(body, bytes) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self._valid_host():
                return self._send(403, {"error": "invalid_host"})
            if self.path == "/api/capabilities":
                return self._send(200, CAPABILITIES)
            known = {"/": ROOT / "web" / "index.html", "/index.html": ROOT / "web" / "index.html",
                     "/app.js": ROOT / "web" / "app.js", "/styles.css": ROOT / "web" / "styles.css",
                     "/brain.mjs": ROOT / "brain.mjs", "/brain-weights.mjs": ROOT / "brain-weights.mjs"}
            path = known.get(self.path)
            if path is None or not path.is_file():
                return self._send(404, {"error": "not_found"})
            content_type = "text/javascript; charset=utf-8" if path.suffix in {".js", ".mjs"} else mimetypes.guess_type(path)[0] or "application/octet-stream"
            self._send(200, path.read_bytes(), content_type)

        def do_POST(self):
            if not self._valid_host():
                return self._send(403, {"error": "invalid_host"})
            origins = self.headers.get_all("Origin", [])
            if len(origins) != 1 or origins[0] != "http://" + self.headers["Host"]:
                return self._send(403, {"error": "origin_mismatch"})
            if self.path != "/api/observe":
                return self._send(404, {"error": "not_found"})
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isdigit():
                return self._send(411, {"error": "content_length_required"})
            length = int(lengths[0])
            if not 0 < length <= 4096:
                return self._send(413, {"error": "body_too_large"})
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                return self._send(415, {"error": "json_required"})
            try:
                self.connection.settimeout(2)
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or set(data) != {"target"}:
                    raise BoundaryError("Expected one target field")
                status, result = observer.observe(data["target"])
                self._send(status, result)
            except BoundaryError as exc:
                self._send(400, {"error": "invalid_target", "message": str(exc)})
            except (ValueError, UnicodeError, socket.timeout):
                self._send(400, {"error": "invalid_request", "message": "A bounded JSON target request is required."})
            except Exception:
                self._send(500, {"error": "collector_error", "message": "The local observation could not complete."})
    return Handler


def make_server(port=8766, observer=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(observer or Observer()))
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open-browser", action="store_true", help="Open the local interface after starting; does not connect any target")
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"Sentinel local observer: http://127.0.0.1:{server.server_port} — public URL/IP checks; Ctrl-C stops.", flush=True)
    if args.open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{server.server_port}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
