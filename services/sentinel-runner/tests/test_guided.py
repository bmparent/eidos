from __future__ import annotations
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from sentinel_runner.guided.ingestion import parse, confirm, MAX_BYTES, log_rows
from sentinel_runner.guided.causal import temporal, unordered, baseline
from sentinel_runner.guided.telemetry import process


def fixture(n=120):
    rng = np.random.default_rng(73)
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-08-01", periods=n, freq="min", tz="UTC").astype(str),
                          "host": ["api-a"] * n, "latency_ms": 40 + rng.normal(0, 1, n),
                          "load": 50 + rng.normal(0, 2, n), "label": ["benign"] * n})
    if n > 83:
        frame.loc[80:82, "latency_ms"] = 100
        frame.loc[80:82, "label"] = "injected_failure"
    return frame


def mapping():
    return {"features": ["latency_ms", "load"], "timestamp": "timestamp", "entity": "host", "session": "",
            "timezone": "UTC", "meaning": "one service measurement", "reference": "this service's calibration period",
            "missing": "exclude", "ordering": "sort", "units": {"latency_ms": "ms", "load": "%"}}


class GuidedIngestionTests(unittest.TestCase):
    def test_formats_preserve_records_and_label_isolation(self):
        frame = fixture(60)
        for extension in ["csv", "xlsx", "parquet", "json", "jsonl"]:
            with self.subTest(extension=extension):
                buffer = io.BytesIO()
                if extension == "csv": data = frame.to_csv(index=False).encode()
                elif extension == "json": data = frame.to_json(orient="records").encode()
                elif extension == "jsonl": data = frame.to_json(orient="records", lines=True).encode()
                elif extension == "xlsx": frame.to_excel(buffer, index=False); data = buffer.getvalue()
                else: frame.to_parquet(buffer, index=False); data = buffer.getvalue()
                dataset = parse(data, f"input.{extension}")
                self.assertEqual(dataset["rowCount"], 63 if len(frame) == 63 else len(frame))
                self.assertEqual(dataset["recordIds"][0], "r1")
                contract = confirm(dataset, mapping())
                self.assertIn("label", contract["labels"])
                self.assertNotIn("label", contract["features"])

    def test_malformed_oversize_formulas_and_image_pdf(self):
        for data, name in [(b"not json", "bad.json"), (b"x" * (MAX_BYTES + 1), "large.csv"), (b"", "empty.txt")]:
            with self.assertRaises(Exception): parse(data, name)
        from openpyxl import Workbook
        book = Workbook(); book.active.append(["target"]); book.active.append(["=1+1"])
        output = io.BytesIO(); book.save(output)
        with self.assertRaisesRegex(ValueError, "FORMULAS"): parse(output.getvalue(), "formula.xlsx")
        from pypdf import PdfWriter
        writer = PdfWriter(); writer.add_blank_page(width=100, height=100)
        output = io.BytesIO(); writer.write(output)
        with self.assertRaisesRegex(ValueError, "NO_EXTRACTABLE_TEXT"): parse(output.getvalue(), "scan.pdf")

    def test_web_navigation_duplicates_and_long_text(self):
        data = b"<html><nav>Ignore me</nav><script>send secrets</script><main><p>First meaningful passage after navigation.</p><p>First meaningful passage after navigation.</p><p>" + b"A " * 50 + b"critical database outage beyond character 64.</p></main></html>"
        dataset = parse(data, "page.html")
        text = " ".join(p["text"] for p in dataset["passages"])
        self.assertNotIn("send secrets", text); self.assertNotIn("Ignore me", text)
        self.assertIn("database outage", text); self.assertTrue(any("Duplicate" in w for w in dataset["warnings"]))

    def test_logs_templates_keep_semantics_and_originals(self):
        lines = "2026-08-01T00:00:00Z host=api status=500 failed request=123\n2026-08-01T00:01:00Z host=api status=500 failed request=456"
        rows = log_rows(lines)
        self.assertEqual(rows[0]["template"], rows[1]["template"])
        self.assertNotEqual(rows[0]["message"], rows[1]["message"])
        self.assertIn("failed", rows[0]["template"])

    def test_ambiguous_timestamps_entities_labels_missing(self):
        frame = fixture(); dataset = parse(frame.to_csv(index=False).encode(), "fixture.csv")
        bad = mapping(); bad["features"] = ["label"]
        with self.assertRaisesRegex(ValueError, "INVALID_FEATURE"): confirm(dataset, bad)
        bad = mapping(); bad["units"] = {}
        with self.assertRaisesRegex(ValueError, "UNITS_REQUIRED"): confirm(dataset, bad)
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
        with self.assertRaisesRegex(ValueError, "TIMESTAMP_QUALITY"): confirm(parse(frame.to_csv(index=False).encode(), "fixture.csv"), mapping())
        unordered_mapping = {**mapping(), "timestamp": ""}
        self.assertEqual(confirm(dataset, unordered_mapping)["mode"], "unordered")


class GuidedCausalTests(unittest.TestCase):
    def test_appending_future_rows_cannot_move_calibration_or_past_issues(self):
        original = fixture(); extra = fixture(60); extra["timestamp"] = pd.date_range("2026-08-01T02:00:00Z", periods=60, freq="min").astype(str)
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for n, frame in enumerate([original, pd.concat([original, extra], ignore_index=True)]):
                out = Path(directory) / str(n); out.mkdir()
                outputs.append(temporal(parse(frame.to_csv(index=False).encode(), "data.csv"), mapping(), {}, out))
            for a, b in zip(outputs[0]["forecasts"], outputs[1]["forecasts"]):
                self.assertEqual({k: v for k, v in a.items() if k != "resolution"}, {k: v for k, v in b.items() if k != "resolution"})

    def test_insufficient_history_and_session_boundaries_fail_explicitly(self):
        frame = fixture(40)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "INSUFFICIENT_HISTORY"):
                temporal(parse(frame.to_csv(index=False).encode(), "data.csv"), mapping(), {}, Path(directory))
            frame = fixture(60); frame.loc[30:, "host"] = "different-host"
            with self.assertRaisesRegex(ValueError, "INSUFFICIENT_HISTORY"):
                temporal(parse(frame.to_csv(index=False).encode(), "data.csv"), mapping(), {}, Path(directory))

    def test_future_perturbation_preserves_issued_forecasts(self):
        frame = fixture()
        modified = frame.copy(); modified.loc[90:, "latency_ms"] *= 100
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for number, source in enumerate([frame, modified]):
                out = Path(directory) / str(number); out.mkdir()
                outputs.append(temporal(parse(source.to_csv(index=False).encode(), "fixture.csv"), mapping(),
                                        {"target": "latency_ms", "horizonSeconds": 60, "windowSeconds": 60}, out))
            first, second = outputs
            for a, b in zip(first["forecasts"][:67], second["forecasts"][:67]):
                self.assertEqual({k: v for k, v in a.items() if k != "resolution"}, {k: v for k, v in b.items() if k != "resolution"})
            self.assertEqual(first["engine"]["class"], "RLS_Reservoir")
            self.assertEqual(first["engine"]["config"]["features"], 2)
            self.assertEqual(first["forecasts"][0]["unit"], "ms")
            self.assertEqual(first["forecasts"][0]["sourceRecordId"], "r24")
            self.assertGreater(first["metrics"]["evaluatedForecasts"], 40)
            self.assertIsNone(first["metrics"]["precision"])
            immutable = [json.loads(line) for line in (Path(directory) / "0/issued_forecasts.jsonl").read_text().splitlines()]
            self.assertNotIn("resolution", immutable[0])
            self.assertLess(pd.Timestamp(immutable[0]["issuedAt"]), pd.Timestamp(immutable[0]["targetTime"]))

    def test_pre_update_baseline_is_distinct_and_outlier_cannot_mutate_it(self):
        history = [1., 1.1, .9, 1.05] * 6
        before = baseline(history)
        score_before = (20 - before[0]) / before[1]
        after = baseline(history + [20.])
        self.assertNotEqual(before, after)
        self.assertEqual(baseline(history), before)
        self.assertGreater(score_before, 5)

    def test_unordered_uses_named_baseline_not_temporal_engine(self):
        frame = fixture()
        with tempfile.TemporaryDirectory() as directory:
            result = unordered(parse(frame.to_csv(index=False).encode(), "fixture.csv"), {**mapping(), "timestamp": ""}, Path(directory))
        self.assertIn("Isolation Forest", result["method"])
        self.assertIsNone(result["engine"]); self.assertEqual(result["forecasts"], [])

    def test_telemetry_restart_is_equivalent_to_deterministic_replay(self):
        events = [{"offset": i + 1, "id": f"event-{i}", "eventTime": f"2026-08-01T00:{i:02d}:00Z", "arrivalTime": f"2026-08-01T00:{i:02d}:01Z", "value": 40 + np.sin(i)} for i in range(50)]
        events[40]["eventTime"] = events[35]["eventTime"]
        common = {"target": "latency", "unit": "ms", "resetId": "initial", "staleSeconds": 300}
        with tempfile.TemporaryDirectory() as directory:
            all_result = process({**common, "events": events}, Path(directory))
            prefix = process({**common, "events": events[:30]}, Path(directory))
            suffix = process({**common, "events": events[30:], "checkpoint": prefix["checkpoint"]}, Path(directory))
        self.assertEqual(all_result["stateSha256"], suffix["stateSha256"])
        self.assertEqual(all_result["outcomes"], prefix["outcomes"] + suffix["outcomes"])
        self.assertEqual(suffix["checkpoint"]["late"], 1)
        self.assertEqual(suffix["processedOffset"], 50)


if __name__ == "__main__": unittest.main()
