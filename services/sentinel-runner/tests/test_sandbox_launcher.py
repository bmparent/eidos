import json
import tempfile
import unittest
import subprocess
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from sentinel_runner.sandbox_launcher import CPU_TORCH_VERSION, command, status


class SandboxLauncherTests(unittest.TestCase):
    def fixture(self, root):
        job_dir = root / "jobs" / "rd-aaaaaaaaaaaa-bbbbbbbb"
        request_path = root / "request.json"
        request_path.write_text(json.dumps({"lockDigest": "a" * 64}), encoding="utf-8")
        return job_dir, Namespace(repo_root=str(root), job_dir=str(job_dir), request=str(request_path))

    def test_invalid_request_still_commits_a_failed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir, args = self.fixture(root)
            Path(args.request).write_text("{invalid", encoding="utf-8")
            self.assertEqual(command(args), 1)
            receipt = json.loads((job_dir / "status.json").read_text())
            self.assertEqual(receipt["status"], "FAILED")
            self.assertIn("read_request", receipt["detail"])
            self.assertEqual(receipt["artifacts"], ["bootstrap_failure_traceback.log"])

    def test_existing_python_does_not_skip_install_or_cpu_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir, args = self.fixture(root)
            python = root / ".eidos-runner-venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
            calls = []

            def run(argv, **kwargs):
                calls.append(argv)
                if "sentinel_runner.cli" in argv:
                    status(job_dir, "COMPLETED_ENGINEERING", lockDigest="a" * 64)
                return subprocess.CompletedProcess(argv, 0)

            with patch("sentinel_runner.sandbox_launcher.shutil.which", return_value="/usr/bin/uv"), patch("sentinel_runner.sandbox_launcher.subprocess.run", side_effect=run):
                self.assertEqual(command(args), 0)
            self.assertEqual(calls[0][1:3], ["pip", "install"])
            self.assertIn("--torch-backend", calls[0])
            self.assertIn("torch.version.cuda is None", calls[1][2])
            self.assertIn("sentinel_runner.cli", calls[2])

    def test_child_exit_without_terminal_receipt_fails_and_valid_failure_is_preserved(self):
        for exit_code, child_status in [(0, None), (2, None), (1, "FAILED")]:
            with self.subTest(exit_code=exit_code, child_status=child_status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                job_dir, args = self.fixture(root)

                def run(argv, **kwargs):
                    if "sentinel_runner.cli" in argv:
                        if child_status:
                            status(job_dir, child_status, error="DATASET_HASH_MISMATCH", detail="original diagnostic")
                        return subprocess.CompletedProcess(argv, exit_code)
                    return subprocess.CompletedProcess(argv, 0)

                with patch("sentinel_runner.sandbox_launcher.shutil.which", return_value="/usr/bin/uv"), patch("sentinel_runner.sandbox_launcher.subprocess.run", side_effect=run):
                    self.assertEqual(command(args), 1)
                receipt = json.loads((job_dir / "status.json").read_text())
                self.assertEqual(receipt["status"], "FAILED")
                self.assertEqual(receipt["error"], "DATASET_HASH_MISMATCH" if child_status else "SANDBOX_BOOTSTRAP_FAILED")

    def test_dependency_install_failure_never_starts_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir, args = self.fixture(root)
            calls = []

            def run(argv, **kwargs):
                calls.append(argv)
                if "install" in argv:
                    raise subprocess.CalledProcessError(1, argv)
                return subprocess.CompletedProcess(argv, 0)

            with patch("sentinel_runner.sandbox_launcher.shutil.which", return_value="/usr/bin/uv"), patch("sentinel_runner.sandbox_launcher.subprocess.run", side_effect=run):
                self.assertEqual(command(args), 1)
            receipt = json.loads((job_dir / "status.json").read_text())
            self.assertIn("install_cpu_runtime", receipt["detail"])
            self.assertFalse(any("sentinel_runner.cli" in argv for argv in calls))

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
