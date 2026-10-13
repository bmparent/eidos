"""Small contract tests; inputs here are synthetic and never shown as telemetry."""
import json
import unittest
from datetime import datetime, timedelta, timezone

from brain import BrainMonitor


class BrainContractTests(unittest.TestCase):
    def stamp(self, index):
        return (datetime(2026, 10, 13, tzinfo=timezone.utc) + timedelta(seconds=index)).isoformat()

    def observe(self, monitor, index, latency=100.0, source="site"):
        return monitor.observe(source, f"evidence-{index}", self.stamp(index), latency, 200)

    def test_forecasts_are_frozen_before_future_observation(self):
        monitor = BrainMonitor()
        for index in range(12):
            last = self.observe(monitor, index, 100 + index)
        frozen = last["forecast"]["issued"].copy()
        result = self.observe(monitor, 12, 1000)
        score = result["forecast"]["scored"]
        self.assertEqual(score["predicted_latency_ms"], frozen["predicted_latency_ms"])
        self.assertEqual(score["basis_evidence_id"], "evidence-11")
        self.assertEqual(score["actual_evidence_id"], "evidence-12")
        self.assertLess(score["issued_at"], score["observed_at"])
        self.assertAlmostEqual(score["absolute_error_ms"], abs(frozen["predicted_latency_ms"] - 1000))

    def test_missing_nonfinite_duplicate_and_out_of_order_are_excluded(self):
        monitor = BrainMonitor()
        self.observe(monitor, 0)
        for index, value in enumerate([None, float("nan"), float("inf"), -1, True, "100"], start=1):
            result = self.observe(monitor, index, value)
            self.assertEqual(result["state"], "insufficient_data")
        self.assertEqual(self.observe(monitor, 0)["state"], "insufficient_data")
        result = monitor.observe("site", "other", self.stamp(-1), 100, 200)
        self.assertEqual(result["state"], "insufficient_data")
        self.assertEqual(monitor.summary()["sources"][0]["sample_count"], 1)

    def test_sources_are_independent_and_summary_never_claims_skill(self):
        monitor = BrainMonitor()
        for index in range(40):
            result = self.observe(monitor, index, 100 + index % 4)
        self.observe(monitor, 40, 999, "other")
        sources = {s["source_id"]: s for s in monitor.summary()["sources"]}
        self.assertEqual(sources["site"]["sample_count"], 40)
        self.assertEqual(sources["other"]["sample_count"], 1)
        self.assertEqual(sources["site"]["forecast"]["scored_count"], 28)
        self.assertEqual(sources["site"]["forecast"]["status"], "provisional_observed_window")
        self.assertFalse(sources["site"]["forecast"]["skill_established"])
        json.dumps(result, allow_nan=False)
        json.dumps(monitor.summary(), allow_nan=False)

    def test_gap_is_unscored_and_missing_samples_clear_pending(self):
        monitor = BrainMonitor(max_forecast_gap_seconds=5)
        for index in range(12):
            self.observe(monitor, index)
        gap = self.observe(monitor, 30)
        self.assertEqual(gap["forecast"]["scored"]["status"], "unscored")
        self.observe(monitor, 31, None)
        after = self.observe(monitor, 32)
        self.assertIsNone(after["forecast"]["scored"])
        self.assertEqual(after["forecast"]["comparison"]["scored_count"], 0)

    def test_warmup_is_not_green_security_clearance(self):
        monitor = BrainMonitor()
        result = self.observe(monitor, 0)
        self.assertEqual(result["state"], "calibrating")
        self.assertIsNone(result["event_score"])
        self.assertEqual(result["signals"], [])
        self.assertEqual(result["diagnostics"]["security_verdict"], "not assessed")
        self.assertIsNone(monitor.sources["site"].detector.input_detector._period_power())


if __name__ == "__main__":
    unittest.main()
