from __future__ import annotations

import csv
import math
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sentinel_v3_controlled_runner_smoke(tmp_path):
    out_dir = tmp_path / "eidos_sentinel_v3_smoke"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_sentinel_v3_controlled.py",
            "--reservoirs",
            "64",
            "128",
            "--features",
            "16",
            "--warmup",
            "200",
            "--frames-per-regime",
            "300",
            "--seed",
            "42",
            "--out",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    required = [
        "summary_v3_patch.csv",
        "summary_v3_patch_full.json",
        "run_manifest_v3_patch.json",
        "per_regime_summary_v3_patch.csv",
        "acceptance_v3_patch.csv",
        "drive_manifest.json",
    ]
    for name in required:
        assert (out_dir / name).exists(), name
    assert (out_dir / "eidos_v3_steps_1400_reservoir_64.csv").exists()
    assert (out_dir / "eidos_v3_steps_1400_reservoir_128.csv").exists()

    with (out_dir / "summary_v3_patch.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        for value in row.values():
            if value in {"True", "False", "", "GREEN", "AMBER", "RED"} or value.endswith(".csv"):
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            assert math.isfinite(numeric)

    with (out_dir / "acceptance_v3_patch.csv").open(newline="", encoding="utf-8") as handle:
        acceptance = list(csv.DictReader(handle))
    assert acceptance
    expected_columns = {
        "normal_false_alert_rate",
        "abnormal_alert_rate",
        "frozen_red_rate",
        "noise_alert_rate",
        "backdoor_alert_rate",
        "finite_residual_stats",
        "pass_all",
    }
    assert expected_columns.issubset(acceptance[0].keys())
    assert any(row["pass_all"] == "True" for row in acceptance)
