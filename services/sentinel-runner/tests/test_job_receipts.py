import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sentinel_runner.dataset import prepare_dataframe, sha256_file
from sentinel_runner.diagnostics import engine_telemetry
from sentinel_runner.job import atomic_json, run_job
from sentinel_runner.profiles import EXECUTION_PROFILES
from sentinel_runner.spec import ExperimentSpec, lock_digest
from test_spec import fixture_spec


class ReceiptTests(unittest.TestCase):
    def test_telemetry_never_exports_labels_or_input_snippets(self):
        row = {"step": 1, "z": 2.0, "label": "ATTACK", "snippet": "private", "vector": [1, 2], "thermo_temp": float("nan")}
        self.assertEqual(list(engine_telemetry([row])), [{"step": 1, "z": 2.0, "thermo_temp": None}])

    def test_job_commits_predictions_before_evaluation_and_exposes_receipts(self):
        spec = ExperimentSpec.from_dict(fixture_spec())
        prepared = prepare_dataframe(pd.DataFrame({"feature": np.arange(1000), "Label": ["BENIGN"] * 500 + ["ATTACK"] * 500}), spec, file_sha256="a" * 64)
        rows = [{"step": i, "z": 0.0, "z_thresh_eff": 1.0, "is_surprise": False} for i in range(800)]
        receipt = {"execution_profile": "cpu_engineering", "code_sha256": "b" * 64, "effective_config": {**EXECUTION_PROFILES["cpu_engineering"], "thermo_enabled": True}}
        from sentinel_runner.metrics import evaluate_frozen_predictions
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "request.json"
            job = root / "rd-123456789abc-12345678"
            atomic_json(request, {"schema": "eidos.sentinel-runner.request.v0.2", "lockDigest": lock_digest(spec), "spec": spec.to_dict()})
            def evaluate(*args):
                self.assertTrue((job / "engine_trace.jsonl").is_file())
                self.assertEqual(json.loads((job / "status.json").read_text())["status"], "EVALUATING_FROZEN_PREDICTIONS")
                return evaluate_frozen_predictions(*args)
            with patch("sentinel_runner.job.prepare_kaggle_dataset", return_value=prepared), patch("sentinel_runner.job.run_full_engine", return_value=({"step_rows": rows}, receipt)), patch("sentinel_runner.job.evaluate_frozen_predictions", side_effect=evaluate):
                self.assertEqual(run_job(request, job), 0)
            status = json.loads((job / "status.json").read_text())
            self.assertEqual(status["status"], "COMPLETED_ENGINEERING")
            self.assertEqual(status["gatesAdvanced"], 0)
            self.assertEqual(status["metrics"]["prediction_trace_sha256"], sha256_file(job / "engine_trace.jsonl"))
            self.assertTrue(all((job / name).is_file() for name in status["artifacts"]))

    def test_external_failure_logs_still_require_operator_auth(self):
        from sentinel_runner.api import get_artifact
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"EIDOS_JOB_ROOT": directory, "EIDOS_RUNNER_TOKEN": "test-only"}):
            job_id = "rd-123456789abc-12345678"
            root = Path(directory) / job_id
            root.mkdir()
            (root / "runner.log").write_text("failure diagnostic")
            with self.assertRaises(HTTPException) as rejected:
                get_artifact(job_id, "runner.log", None)
            self.assertEqual(rejected.exception.status_code, 401)
            for invalid in ("test-only", "Basic test-only", "Bearer different", "Bearer λ"):
                with self.assertRaises(HTTPException) as rejected:
                    get_artifact(job_id, "runner.log", invalid)
                self.assertEqual(rejected.exception.status_code, 401)
            self.assertEqual(str(get_artifact(job_id, "runner.log", "Bearer test-only").path), str(root / "runner.log"))
            with self.assertRaises(HTTPException):
                get_artifact(job_id, "../request.json", "Bearer test-only")
