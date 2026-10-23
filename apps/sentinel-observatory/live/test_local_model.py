"""Controller tests use a stub; actual model evidence is separately retained."""
import json
import http.client
import threading
import unittest
from local_server import Observer, make_server


class StubInterpreter:
    model_path = "test-stub-only"
    def __init__(self):
        self.calls = []
    def status(self):
        return {"state": "ready", "name": "Unit fixture; no inference receipt"}
    def analyze(self, event, context):
        self.calls.append((event, context))
        return {"kind": "deterministic", "summary": "Unit fixture", "provenance": {"model_ran": False}}


class LocalModelBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.model = StubInterpreter()
        self.observer = Observer(collector=lambda target: {"id": "unit_ev", "source_id": target.id,
            "observed_at": "2026-10-13T20:00:00Z", "status": 200}, interpreter=self.model)

    def test_only_retained_collector_evidence_can_be_analyzed(self):
        status, _ = self.observer.analyze("invented")
        self.assertEqual(status, 404)
        self.assertEqual(self.model.calls, [])
        self.observer.observe("1.1.1.1")
        status, result = self.observer.analyze("unit_ev")
        self.assertEqual(status, 200)
        self.assertEqual(result["event_id"], "unit_ev")
        self.assertEqual(self.model.calls[0][0]["id"], "unit_ev")
        self.assertFalse(result["analysis"]["provenance"]["model_ran"])

    def test_repeated_analysis_is_cached_and_concurrency_is_bounded(self):
        self.observer.observe("1.1.1.1")
        self.observer.model_slot.acquire()
        self.assertEqual(self.observer.analyze("unit_ev")[0], 409)
        self.observer.model_slot.release()
        self.assertEqual(self.observer.analyze("unit_ev")[0], 200)
        self.assertTrue(self.observer.analyze("unit_ev")[1]["cached"])
        self.assertEqual(len(self.model.calls), 1)

    def test_cross_origin_or_inline_events_cannot_trigger_inference(self):
        server = make_server(0, self.observer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for body, origin, expected in [({"event_id": "invented"}, "https://attacker.invalid", 403),
                  ({"event": {"id": "fake"}}, f"http://127.0.0.1:{server.server_port}", 400)]:
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                connection.request("POST", "/api/analyze", json.dumps(body), {"Content-Type": "application/json", "Origin": origin})
                response = connection.getresponse()
                self.assertEqual(response.status, expected)
                response.read()
                connection.close()
            self.assertEqual(self.model.calls, [])
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
