import json
import tempfile
import unittest
from pathlib import Path

from sentinel_runner.sandbox_launcher import status


class SandboxLauncherTests(unittest.TestCase):
    def test_bootstrap_status_preserves_proof_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory) / "rd-aaaaaaaaaaaa-bbbbbbbb"
            status(job_dir, "BOOTSTRAPPING_RUNTIME", lockDigest="a" * 64)
            receipt = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "BOOTSTRAPPING_RUNTIME")
            self.assertEqual(receipt["executionBackend"], "sandbox")
            self.assertEqual(receipt["evidenceClass"], "REAL_DATA_ENGINEERING")
            self.assertEqual(receipt["proofVerdict"], "BLOCKED_RESOURCE_BEFORE_HELDOUT")
            self.assertEqual(receipt["gatesAdvanced"], 0)


if __name__ == "__main__":
    unittest.main()
