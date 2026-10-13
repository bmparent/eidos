"""Safety, evidence integrity, and asynchronous HTTP contract tests (no WAN)."""
import json
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from email.message import Message
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from telemetry import EventStore, Observatory, collect_head, deterministic_analysis, safe_redirect
from server import make_handler


def event(n=1, source="home", status=200):
    return {"id": "evt_" + f"{n:032x}", "source_id": source, "url": "https://eidos-works.com/", "observed_at": datetime.now(timezone.utc).isoformat(), "latency_ms": 12.5, "status": status, "kind": "measurement", "headers": {}, "error": None, "brain": {"state": "observing"}}


class Response:
    def __init__(self, status=200, headers=None):
        self.code, self.headers, self.closed = status, Message(), False
        for key, value in (headers or {}).items():
            self.headers[key] = value
    def close(self):
        self.closed = True
    def read(self, *_):
        raise AssertionError("HEAD body must never be read")


class Opener:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []
    def open(self, request, timeout):
        self.calls.append((request, timeout))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class Monitor:
    def observe(self, *_):
        return {"state": "observing"}
    def summary(self):
        return {"sources": []}


class TransportTests(unittest.TestCase):
    def test_only_head_bounded_headers_and_body_never_read(self):
        response = Response(200, {"Set-Cookie": "secret-cookie", "Authorization": "secret-token", "CF-Cache-Status": "HIT", "Content-Type": "text/html", "Server": "x" * 500})
        opener = Opener(response)
        result = collect_head("home", opener=opener)
        self.assertEqual(opener.calls[0][0].get_method(), "HEAD")
        self.assertEqual(result["status"], 200)
        self.assertTrue(result["response_chain_complete"])
        self.assertEqual(set(result["headers"]), {"cf-cache-status", "content-type", "server"})
        self.assertEqual(len(result["headers"]["server"]), 256)
        self.assertNotIn("secret", json.dumps(result))
        self.assertTrue(response.closed)

    def test_redirect_boundary_and_actual_response_preserved(self):
        for target in ("http://eidos-works.com/", "https://other.test/", "https://127.0.0.1/", "https://eidos-works.com:444/", "https://a:b@eidos-works.com/", "/path?token=secret", "/path#secret"):
            with self.subTest(target=target):
                opener = Opener(Response(302, {"Location": target}))
                result = collect_head("home", opener=opener)
                self.assertEqual(len(opener.calls), 1)
                self.assertEqual(result["status"], 302)
                self.assertEqual(result["error"]["code"], "redirect_outside_boundary")
                self.assertFalse(result["response_chain_complete"])
                self.assertNotIn("token=", json.dumps(result))
                self.assertNotIn("No HTTP response", json.dumps(deterministic_analysis(result)))

    def test_redirect_limit_and_method_remains_head(self):
        opener = Opener(*[Response(302, {"Location": "/next"}) for _ in range(4)])
        result = collect_head("home", opener=opener)
        self.assertEqual(len(opener.calls), 4)
        self.assertEqual(result["redirect_count"], 3)
        self.assertEqual(result["error"]["code"], "redirect_limit")
        self.assertTrue(all(call[0].get_method() == "HEAD" for call in opener.calls))

    def test_transport_failure_never_asserts_site_failure_or_leaks_error(self):
        result = collect_head("home", opener=Opener(urllib.error.URLError("proxy secret@private:123")))
        self.assertIsNone(result["status"])
        self.assertEqual(result["error"]["scope"], "observation_vantage")
        self.assertTrue(all(value is None for value in result["security_headers"].values()))
        self.assertNotIn("secret@private", json.dumps(result))
        self.assertIn("Site availability is unknown", deterministic_analysis(result)["summary"])
        with self.assertRaises(ValueError):
            collect_head("https://arbitrary.test")

    def test_security_controls_are_bounded_observations_not_raw_policies(self):
        response = Response(200, {"Strict-Transport-Security": "max-age=12345", "Content-Security-Policy": "secret-report-location", "X-Content-Type-Options": "nosniff", "X-Frame-Options": "SAMEORIGIN", "Referrer-Policy": "strict-origin", "Cache-Control": "private, no-store"})
        result = collect_head("home", opener=Opener(response))
        self.assertEqual(result["security_headers"], {"hsts_present": True, "csp_present": True, "x_content_type_options": "nosniff", "frame_options": "sameorigin", "referrer_policy": "strict-origin", "cache_private": True, "cache_no_store": True})
        self.assertNotIn("secret-report", json.dumps(result))


class JournalTests(unittest.TestCase):
    def test_retention_restart_hash_chain_and_analysis_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.sqlite3"
            store = EventStore(path, retention=3)
            for n in range(5):
                store.append(event(n))
            self.assertEqual(store.stats()["retained"], 3)
            self.assertEqual(store.stats()["total_recorded"], 5)
            first = store.recent()[0]
            store.update_analysis(first["id"], {"kind": "deterministic", "summary": "new analysis"})
            self.assertEqual(first["evidence_hash"], store.get(first["id"])["evidence_hash"])
            self.assertEqual(store.verify()["records_checked"], 3)
            store.close()
            reopened = EventStore(path, retention=3)
            self.assertEqual(reopened.verify()["records_checked"], 3)
            reopened.close()
            with sqlite3.connect(path) as db:
                db.execute("UPDATE events SET payload=replace(payload,'12.5','999.9') WHERE seq=5")
            with self.assertRaisesRegex(ValueError, "integrity"):
                EventStore(path, retention=3)

    def test_duplicate_evidence_is_rejected(self):
        store = EventStore(":memory:")
        item = event()
        store.append(item)
        with self.assertRaises(sqlite3.IntegrityError):
            store.append(item)
        self.assertEqual(store.stats()["retained"], 1)
        store.close()


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.observatory = Observatory(Path(self.temp.name) / "events.db", 30, collect=False, monitor=Monitor())
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.observatory))
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.observatory.store.close()
        self.temp.cleanup()

    def request(self, path, data=None, headers=None):
        request = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        try:
            response = self.client.open(request, timeout=2)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            return response.code, json.load(response)

    def test_probe_same_origin_payload_and_rate_limit(self):
        self.assertEqual(self.request("/api/probe", b"")[0], 403)
        self.assertEqual(self.request("/api/probe", b"", {"Origin": "https://evil.test"})[0], 403)
        headers = {"Origin": self.base}
        self.assertEqual(self.request("/api/probe", b'{"url":"https://evil.test"}', headers)[0], 400)
        self.assertEqual(self.request("/api/probe", b"", headers)[0], 202)
        self.assertEqual(self.request("/api/probe", b"", headers)[0], 429)
        self.assertEqual(self.observatory.probe_queue.get_nowait(), "probe")

    def test_get_read_only_and_host_protected(self):
        self.assertEqual(self.request("/api/snapshot", headers={"Host": "evil.test"})[0], 403)
        status, snapshot = self.request("/api/snapshot")
        self.assertEqual(status, 200)
        self.assertEqual(snapshot["coverage"]["retained_events"], 0)
        self.assertFalse(snapshot["collection_enabled"])
        self.assertEqual(self.request("/../telemetry.py")[0], 404)
        self.assertEqual(self.request("/api/events/not-an-id")[0], 400)

    def test_stale_is_derived_from_observation_time(self):
        item = event()
        item["observed_at"] = (datetime.now(timezone.utc) - timedelta(seconds=100)).isoformat()
        self.observatory.store.append(item)
        snapshot = self.observatory.snapshot()
        self.assertEqual(snapshot["sources"][0]["state"], "stale")
        self.assertEqual(snapshot["sources"][1]["state"], "waiting")

    def test_brain_uses_completed_measurement_time_and_excludes_incomplete_latency(self):
        observed = event()
        observed["completed_at"] = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
        class RecordingMonitor(Monitor):
            def observe(self, *args):
                self.arguments = args
                return super().observe(*args)
        monitor = RecordingMonitor()
        self.observatory.brain = monitor
        with patch("telemetry.collect_head", return_value=observed):
            saved = self.observatory.observe_once("home")
        self.assertEqual(monitor.arguments[2], observed["completed_at"])
        self.assertEqual(saved["brain"]["observation_timestamp_scope"], "response_completed_at")
        observed["error"] = {"code": "redirect_limit"}
        self.observatory._brain_observe(observed)
        self.assertIsNone(monitor.arguments[3])

    def test_snapshot_responsive_while_inference_is_blocked(self):
        started, unblock, finished = threading.Event(), threading.Event(), threading.Event()
        class BlockingInterpreter:
            def analyze(self, item, recent):
                started.set()
                unblock.wait(10)
                finished.set()
                return deterministic_analysis(item)
        self.observatory.interpreter = BlockingInterpreter()
        self.observatory.start()
        item = self.observatory.store.append(event())
        self.observatory.analysis_queue.put((0, 1, item["id"]))
        self.assertTrue(started.wait(1))
        try:
            self.assertEqual(self.request("/api/snapshot")[0], 200)
            self.assertFalse(finished.is_set(), "HTTP responded while inference remained blocked")
        finally:
            unblock.set()
            self.observatory.stop()


if __name__ == "__main__":
    unittest.main()
