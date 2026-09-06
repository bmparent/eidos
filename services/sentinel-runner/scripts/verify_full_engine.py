"""Reproduce the Lab's synthetic reference through the actual Torch engine.

From services/sentinel-runner:
  python scripts/verify_full_engine.py --output /tmp/eidos-reference
No download, operator credential, or paid compute is used.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from sentinel_runner.dataset import prepare_dataframe
from sentinel_runner.diagnostics import summarize_engine
from sentinel_runner.engine_bridge import run_full_engine
from sentinel_runner.job import atomic_json
from sentinel_runner.metrics import evaluate_frozen_predictions
from sentinel_runner.spec import ExperimentSpec, lock_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["cpu_engineering", "cpu_mechanisms"], default="cpu_engineering")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    t = np.arange(1000, dtype=np.float64)
    rng = np.random.RandomState(41)
    values = np.column_stack([np.sin(t / (11 + 7 * i)) + 0.03 * rng.randn(len(t)) for i in range(8)])
    values[420:580] += 3 * np.sin(np.arange(160)[:, None] / 3) + 2
    frame = pd.DataFrame(values, columns=[f"sensor_{i}" for i in range(8)])
    frame["Label"] = np.where((t >= 420) & (t < 580), "ATTACK", "BENIGN")
    payload = frame.to_csv(index=False, lineterminator="\n").encode()
    input_hash = hashlib.sha256(payload).hexdigest()
    spec = ExperimentSpec.from_dict({
        "schema": "eidos.sentinel-lab.experiment.v0.2", "evidenceClass": "REAL_DATA_ENGINEERING",
        # This is a local generated fixture; it is never downloaded from Kaggle.
        "dataset": {"provider": "kaggle", "ref": "eidos/synthetic-verification", "version": 1, "file": "synthetic.csv", "expectedSha256": input_hash},
        "dataContract": {"labelColumn": "Label", "negativeLabels": ["BENIGN"], "orderMode": "source", "excludedColumns": [], "featureColumns": [], "maxRows": 1000},
        "split": {"calibration": 0.2, "evaluation": 0.6, "sealedHoldout": 0.2},
        "engine": {"version": "0.4.7.02", "features": 64, "seed": 0, "configProfile": "cicids_webattacks", "executionProfile": args.profile},
        "protocol": {"labelPolicy": "sealed_until_prediction_freeze", "normalization": "calibration_only_zscore", "projection": "seeded_gaussian_or_pad", "heldoutPolicy": "exclude_from_engineering_run", "proofVerdict": "BLOCKED_RESOURCE_BEFORE_HELDOUT"},
    })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "synthetic.csv").write_bytes(payload)
    prepared = prepare_dataframe(frame, spec, file_sha256=input_hash, source_path="generated synthetic fixture")
    result, receipt = run_full_engine(prepared, spec, args.output / "engine")
    metrics, trace = evaluate_frozen_predictions(result["step_rows"], prepared)
    diagnostics = summarize_engine(result["step_rows"], receipt, result.get("lab_geometry"))
    assert metrics["evaluation_rows_scored"] == 600
    assert not metrics["heldout_evaluated"]
    assert prepared.frames.shape == (800, 64)
    artifact = {
        "source": "Recorded synthetic reference; not a user's run or real-world dataset",
        "reproduce": "services/sentinel-runner/scripts/verify_full_engine.py --profile " + args.profile,
        "input_sha256": input_hash, "lock_digest": lock_digest(spec),
        "engine_code_sha256": receipt["code_sha256"], "seed": 0,
        "input_rows": 1000, "engine_rows": 800, "evaluation_rows": 600, "heldout_rows": 200,
        "gates_advanced": 0, "diagnostics": diagnostics,
    }
    atomic_json(args.output / "reference.json", artifact)
    atomic_json(args.output / "metrics.json", metrics)
    atomic_json(args.output / "engine_receipt.json", receipt)
    print(json.dumps({"profile": args.profile, "evaluation_rows": metrics["evaluation_rows_scored"], "reference": str(args.output / "reference.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
