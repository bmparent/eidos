import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sensor_log import MAX_RECORDS, sanitize_sensor_record, sensor_capabilities, sensor_snapshot


class SensorLogTests(unittest.TestCase):
    def test_disconnected_without_explicit_path(self):
        with patch.dict(os.environ, {"OBSERVATORY_SENSOR_LOG": ""}):
            self.assertFalse(sensor_capabilities()["connected"])
            self.assertEqual(sensor_snapshot()["coverage"]["state"], "disconnected")

    def test_tail_reloads_source_metadata_without_payload_or_notice_messages(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "eve.json"
            alert = {"timestamp": "2026-10-13T19:00:00Z", "event_type": "alert", "flow_id": 18446744073709551610,
                     "src_ip": "10.0.0.1", "alert": {"signature_id": 1, "signature": "TEST"}, "payload": "secret"}
            path.write_text(json.dumps(alert) + "\n", encoding="utf8")
            with patch.dict(os.environ, {"OBSERVATORY_SENSOR_LOG": str(path)}):
                first = sensor_snapshot()
                self.assertTrue(first["connected"])
                self.assertEqual(first["counts"]["records"], 1)
                self.assertNotIn("secret", first["text"])
                with path.open("a") as handle:
                    handle.write(json.dumps({"ts": 1791918000, "uid": "TEST", "note": "Scan::Port_Scan", "msg": "secret notice"}) + "\n")
                second = sensor_snapshot()
                self.assertEqual(second["counts"]["records"], 2)
                self.assertEqual(second["source_id"], first["source_id"])
                self.assertNotIn("secret notice", second["text"])
                self.assertIn("18446744073709551610", second["text"])

    def test_file_limit_partial_lines_and_invalid_records_are_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "zeek.json"
            row = json.dumps({"ts": 1791918000, "uid": "TEST", "orig_bytes": 100})
            path.write_text((row + "\n") * (MAX_RECORDS + 5) + '{partial}\n{}\n', encoding="utf8")
            with patch.dict(os.environ, {"OBSERVATORY_SENSOR_LOG": str(path)}):
                result = sensor_snapshot()
                self.assertTrue(result["coverage"]["truncated"])
                self.assertEqual(result["counts"]["records"], MAX_RECORDS - 2)
                self.assertEqual(result["counts"]["malformed"], 1)
                self.assertEqual(result["counts"]["unsupported"], 1)

    def test_configured_relative_path_or_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for value in ["relative.json", folder]:
                with patch.dict(os.environ, {"OBSERVATORY_SENSOR_LOG": value}):
                    self.assertFalse(sensor_capabilities()["connected"])
                    self.assertFalse(sensor_snapshot()["connected"])

    def test_configured_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.json"
            path.write_text('{}', encoding="utf8")
            link = Path(folder) / "link.json"
            try:
                link.symlink_to(path)
            except OSError as error:
                if getattr(error, "winerror", None) == 1314:
                    self.skipTest("Windows denied symlink creation (WinError 1314); metadata rejection is tested separately")
                raise
            with patch.dict(os.environ, {"OBSERVATORY_SENSOR_LOG": str(link)}):
                self.assertFalse(sensor_capabilities()["connected"])
                self.assertFalse(sensor_snapshot()["connected"])

    def test_symlink_metadata_is_rejected_before_open(self):
        # This guard remains testable without OS permission to create symlinks.
        path = Path.cwd() / "fixture-only-sensor-link.json"
        with patch.dict(os.environ, {"OBSERVATORY_SENSOR_LOG": str(path)}), \
                patch("sensor_log.Path.lstat", return_value=SimpleNamespace(st_mode=stat.S_IFLNK)), \
                patch("sensor_log.os.open") as open_file:
            self.assertFalse(sensor_capabilities()["connected"])
            self.assertFalse(sensor_snapshot()["connected"])
            open_file.assert_not_called()

    def test_sensor_metadata_whitelist_drops_arbitrary_fields(self):
        row = {"timestamp": "2026-10-13T19:00:00Z", "event_type": "dns", "dns": {"queries": [{"rrname": "example.com", "rrtype": "A", "secret": "omit"}]},
               "http": {"headers": "secret"}, "pcap": "secret"}
        clean = sanitize_sensor_record(row)
        self.assertEqual(clean["dns"]["queries"][0]["rrname"], "example.com")
        self.assertNotIn("secret", json.dumps(clean))


if __name__ == "__main__":
    unittest.main()
