import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from sentinel_runner.sandbox_launcher import CPU_TORCH_VERSION, command, status


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
            self.assertEqual(CPU_TORCH_VERSION, "2.14.0")

    def test_bootstrap_failure_names_step_and_exposes_authenticated_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = root / "jobs" / "rd-aaaaaaaaaaaa-bbbbbbbb"
            request_path = root / "request.json"
            request_path.write_text(json.dumps({"lockDigest": "a" * 64}), encoding="utf-8")
            args = Namespace(repo_root=str(root), job_dir=str(job_dir), request=str(request_path))
            with patch("sentinel_runner.sandbox_launcher.shutil.which", return_value=None):
                self.assertEqual(command(args), 1)
            receipt = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "FAILED")
            self.assertEqual(receipt["error"], "SANDBOX_BOOTSTRAP_FAILED")
            self.assertIn("locate_uv", receipt["detail"])
            self.assertEqual(receipt["artifacts"], ["runner.log", "bootstrap_failure_traceback.log"])
            self.assertEqual(receipt["gatesAdvanced"], 0)


if __name__ == "__main__":
    unittest.main()
