import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from eidos_brain.proof import run_proof


def test_s1_generator_determinism():
    x1, y1, w1 = run_proof.gen_backdoor(1000, 7)
    x2, y2, w2 = run_proof.gen_backdoor(1000, 7)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert w1 == w2


def test_s6_has_no_labels():
    _, y, _ = run_proof.gen_noise_thrash(0, 1024)
    assert y.sum() == 0


def test_value_metric_finite():
    y = np.zeros(100, dtype=int)
    alerts = np.zeros(100, dtype=int)
    m = run_proof.compute_metrics("strict_joint_value", "zscore", y, alerts, 1.0, False, False)
    assert math.isfinite(m.value)


def test_zstd_skip_behavior():
    _, status, reason = run_proof._bytes_ratio(np.zeros(8), "zstd")
    if status == "skipped":
        assert reason == "zstandard package unavailable"


def test_artifact_contract(tmp_path: Path):
    out = tmp_path / "smoke"
    run_proof.main.__wrapped__ if hasattr(run_proof.main, "__wrapped__") else None
    cmd = [sys.executable, "-m", "eidos_brain.proof.run_proof", "--suite", "smoke", "--seeds", "0", "--frames", "1000", "--out", str(out)]
    subprocess.run(cmd, check=True)
    for name in ["manifest.json", "summary.csv", "benchmark_report.md", "theorem_status.md"]:
        assert (out / name).exists()


def test_discovery_card_replay(tmp_path: Path):
    out = tmp_path / "smoke2"
    subprocess.run([sys.executable, "-m", "eidos_brain.proof.run_proof", "--suite", "smoke", "--seeds", "0", "--frames", "1000", "--out", str(out)], check=True)
    card = json.loads(next((out / "discovery_cards").glob("*.json")).read_text())
    assert "replay_command" in card
    replay = json.loads(next((out / "replay_logs").glob("*.json")).read_text())
    assert replay["status"] in {"success", "failure"}


def test_counterexample_output(tmp_path: Path):
    out = tmp_path / "cex"
    subprocess.run([sys.executable, "-m", "eidos_brain.proof.run_proof", "--suite", "counterexamples", "--seeds", "0", "--frames", "1000", "--out", str(out)], check=True)
    report_files = list((out / "counterexamples" / "nuisance_subspace_anomaly").glob("report_seed_*.json"))
    assert report_files
