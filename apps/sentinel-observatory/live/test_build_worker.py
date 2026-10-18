import hashlib
import tempfile
import unittest
from pathlib import Path

from build_worker import build


class WorkerBuildTests(unittest.TestCase):
    def test_worker_has_stable_lf_bytes_and_a_matching_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'worker.js'
            receipt = build(output)
            raw = output.read_bytes()
            self.assertNotIn(b'\r\n', raw)
            self.assertEqual(receipt['sha256'], hashlib.sha256(raw).hexdigest())

    def test_test_transport_and_fixture_credentials_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'worker.js'
            build(output)
            raw = output.read_text(encoding='utf-8')
            for forbidden in ['createFaultTransport', 'test-only-key.pem', 'BEGIN PRIVATE KEY', 'fault-harness/transport']:
                self.assertNotIn(forbidden, raw)


if __name__ == '__main__':
    unittest.main()
