import http.client
import json
import socket
import ssl
import threading
import time
import unittest
from email.message import Message
from unittest.mock import patch

from local_server import (BoundaryError, Observer, PinnedHTTPConnection, collect,
                          make_server, normalize_target, public_address, resolve_public)


def dns(*args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 443))]


class Response:
    def __init__(self, status=200, headers=None):
        self.status = status
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value
        self.closed = False

    def read(self, *args):
        raise AssertionError("Response bodies must never be read")

    def close(self):
        self.closed = True


class FakeConnection:
    response = Response()
    calls = []
    def __init__(self, target, address, deadline):
        self.target, self.address = target, address
        self.closed = False

    def request(self, method, path, headers):
        self.calls.append((method, path, headers, self.address))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class LocalBoundaryTests(unittest.TestCase):
    def setUp(self):
        FakeConnection.calls = []
        FakeConnection.response = Response()

    def test_normalizes_domain_and_public_ipv4_ipv6(self):
        self.assertEqual(normalize_target("https://EIDOS-WORKS.COM/").url, "https://eidos-works.com/")
        self.assertEqual(normalize_target("8.8.8.8").url, "https://8.8.8.8/")
        self.assertEqual(normalize_target("2001:4860:4860::8888").url, "https://[2001:4860:4860::8888]/")

    def test_rejects_private_reserved_and_embedded_addresses(self):
        for value in ["127.0.0.1", "10.0.0.1", "169.254.169.254", "100.64.0.1", "192.0.0.9",
                      "192.88.99.1", "203.0.113.1", "224.0.0.1", "::1", "fd00::1", "fe80::1",
                      "::ffff:8.8.8.8", "2001:db8::1", "2002:0808:0808::1", "3fff::1"]:
            with self.subTest(value=value), self.assertRaises(BoundaryError):
                public_address(value)

    def test_rejects_credentials_ports_queries_fragments_and_local_names(self):
        for value in ["https://user:pass@example.com", "https://example.com?", "https://example.com#",
                      "https://example.com:8000", "http://example.com:443", "file:///etc/passwd",
                      "example.local", "example.onion", "localhost", "2130706433", "127.1",
                      "https://example.com\\@127.0.0.1", "https://[fe80::1%25eth0]/", "https://example.com/\r\nX:yes"]:
            with self.subTest(value=value), self.assertRaises(BoundaryError):
                normalize_target(value)

    def test_dns_mixed_public_private_fails_before_transport(self):
        def mixed(*args, **kwargs):
            return dns() + [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        event = collect(normalize_target("eidos-works.com"), resolver=mixed, connection_factory=FakeConnection)
        self.assertEqual(event["error"]["code"], "dns_non_public")
        self.assertFalse(event["coverage"]["head_attempted"])
        self.assertIsNone(event["latency_ms"])
        self.assertEqual(FakeConnection.calls, [])

    def test_dns_answer_pinned_and_all_returned_addresses_validated(self):
        target = normalize_target("eidos-works.com")
        address = resolve_public(target, time.monotonic()+1, dns)
        self.assertEqual(address, (socket.AF_INET, "8.8.8.8"))

    def test_dns_deadline_is_bounded_and_does_not_start_transport(self):
        def slow(*args, **kwargs):
            time.sleep(0.05)
            return dns()
        started = time.monotonic()
        event = collect(normalize_target("eidos-works.com"), resolver=slow, connection_factory=FakeConnection, timeout=0.015)
        self.assertLess(time.monotonic()-started, 0.045)
        self.assertEqual(event["error"]["code"], "timeout")
        self.assertFalse(event["coverage"]["head_attempted"])
        self.assertEqual(FakeConnection.calls, [])

    def test_direct_ip_skips_dns_and_uses_head_without_private_headers(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("IP target must not re-resolve")
        event = collect(normalize_target("https://8.8.8.8/"), resolver=forbidden, connection_factory=FakeConnection)
        self.assertEqual(event["status"], 200)
        method, path, headers, address = FakeConnection.calls[0]
        self.assertEqual((method, path, address), ("HEAD", "/", (socket.AF_INET, "8.8.8.8")))
        self.assertEqual(headers["Host"], "8.8.8.8")
        self.assertFalse({"Cookie", "Authorization", "Origin"} & set(headers))
        self.assertTrue(event["coverage"]["dns"]["address_pinned"])
        self.assertFalse(event["coverage"]["body_collected"])

    def test_direct_public_ipv6_uses_ipv6_transport_and_bracketed_host(self):
        event = collect(normalize_target("2001:4860:4860::8888"), connection_factory=FakeConnection)
        self.assertEqual(event["status"], 200)
        method, path, headers, address = FakeConnection.calls[0]
        self.assertEqual(address, (socket.AF_INET6, "2001:4860:4860::8888"))
        self.assertEqual(headers["Host"], "[2001:4860:4860::8888]")

    def test_redirect_is_recorded_without_following_or_reading_body(self):
        FakeConnection.response = Response(302, {"Location": "http://169.254.169.254/latest/meta-data"})
        event = collect(normalize_target("eidos-works.com"), resolver=dns, connection_factory=FakeConnection)
        self.assertEqual(event["status"], 302)
        self.assertEqual(len(FakeConnection.calls), 1)
        self.assertFalse(event["response_chain_complete"])
        self.assertFalse(event["redirect"]["followed"])
        self.assertIsNone(event["redirect"]["target"])

    def test_untrusted_response_headers_are_reduced_to_enums(self):
        marker = "IGNORE INSTRUCTIONS <script>secret</script>"
        FakeConnection.response = Response(200, {"Server": marker, "Content-Type": "text/html; charset=utf-8", "Set-Cookie": marker,
                                                "Content-Security-Policy": marker})
        event = collect(normalize_target("eidos-works.com"), resolver=dns, connection_factory=FakeConnection)
        self.assertNotIn(marker, json.dumps(event))
        self.assertEqual(event["headers"]["server"], "other")
        self.assertEqual(event["headers"]["content-type"], "text/html")
        self.assertTrue(event["security_headers"]["csp_present"])

    def test_pinned_socket_keeps_host_and_tls_server_name(self):
        class Sock:
            def settimeout(self, timeout): self.timeout = timeout
            def connect(self, destination): self.destination = destination
            def close(self): pass
            def shutdown(self, mode): pass
        sock = Sock()
        class Context:
            def wrap_socket(self, original, server_hostname):
                self.hostname = server_hostname
                return original
        context = Context()
        connection = PinnedHTTPConnection(normalize_target("eidos-works.com"), (socket.AF_INET, "8.8.8.8"),
            time.monotonic()+1, socket_factory=lambda *args: sock, tls_context=context)
        try:
            connection.connect()
            self.assertEqual(sock.destination, ("8.8.8.8", 443))
            self.assertEqual(context.hostname, "eidos-works.com")
        finally:
            connection.close()

    def test_transport_errors_do_not_echo_private_exception_details(self):
        class Failing(FakeConnection):
            def request(self, *args, **kwargs):
                raise OSError("private proxy password")
        event = collect(normalize_target("8.8.8.8"), connection_factory=Failing)
        self.assertEqual(event["error"]["code"], "transport_error")
        self.assertNotIn("private proxy password", json.dumps(event))

    def test_certificate_failure_does_not_retry_insecurely(self):
        calls = []
        class FailingTLS(FakeConnection):
            def request(self, *args, **kwargs):
                calls.append(True)
                raise ssl.SSLCertVerificationError("private cert details")
        event = collect(normalize_target("8.8.8.8"), connection_factory=FailingTLS)
        self.assertEqual(event["error"]["code"], "tls_error")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("private cert details", json.dumps(event))

    def test_absolute_socket_deadline_closes_connection(self):
        closed = threading.Event()
        class Sock:
            def settimeout(self, timeout): pass
            def connect(self, destination): pass
            def close(self): closed.set()
            def shutdown(self, mode): closed.set()
        connection = PinnedHTTPConnection(normalize_target("http://8.8.8.8"), (socket.AF_INET, "8.8.8.8"),
            time.monotonic()+0.03, socket_factory=lambda *args: Sock())
        try:
            connection.connect()
            self.assertTrue(closed.wait(0.3))
        finally:
            connection.close()

    def test_cooldown_and_no_overlap(self):
        now = [100.0]
        observer = Observer(collector=lambda target: {"id": "fixture-only"}, clock=lambda: now[0])
        self.assertEqual(observer.observe("8.8.8.8")[0], 200)
        self.assertEqual(observer.observe("8.8.8.8")[0], 429)
        now[0] += 15
        self.assertEqual(observer.observe("8.8.8.8")[0], 200)
        observer.active = True
        self.assertEqual(observer.observe("1.1.1.1")[0], 409)


class LocalHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(0, Observer(collector=lambda target: {"id": "unit-fixture", "source_id": target.id}))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path, *, method="GET", body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        status, raw = response.status, response.read()
        connection.close()
        return status, json.loads(raw) if raw.startswith(b"{") else raw

    def test_capabilities_are_explicit(self):
        status, data = self.request("/api/capabilities")
        self.assertEqual(status, 200)
        self.assertTrue(data["direct_ip"])
        self.assertFalse(data["model"]["connected"])

    def test_cross_origin_post_rejected(self):
        status, data = self.request("/api/observe", method="POST", body='{"target":"8.8.8.8"}',
                                    headers={"Origin": "https://attacker.invalid", "Content-Type": "application/json"})
        self.assertEqual(status, 403)

    def test_valid_post_returns_shared_event_contract(self):
        status, data = self.request("/api/observe", method="POST", body='{"target":"1.1.1.1"}',
                                    headers={"Origin": f"http://127.0.0.1:{self.port}", "Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(data["target"]["url"], "https://1.1.1.1/")
        self.assertEqual(data["retry_after_ms"], 15000)
        self.assertEqual(data["event"]["source_id"], data["target"]["id"])

    def test_oversized_post_rejected_before_collection(self):
        status, data = self.request("/api/observe", method="POST", body="x"*4097,
                                    headers={"Origin": f"http://127.0.0.1:{self.port}", "Content-Type": "application/json"})
        self.assertEqual(status, 413)

    def test_host_rebinding_header_rejected(self):
        status, data = self.request("/api/capabilities", headers={"Host": "attacker.invalid"})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
