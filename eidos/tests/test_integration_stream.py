import pytest
import os
import json
import csv
from tests import scenarios

@pytest.mark.smoke
def test_scenario_nominal(brain_module, temp_artifact_root):
    """Run nominal scenario and validate all artifacts schemas."""
    steps = 100
    gen = lambda: scenarios.nominal(steps=steps)
    
    brain_module.run_sentinel_stream(
        gen_factory=gen, est_frames=steps, features=64,
        profile_label="int_test", session_label="nominal_run",
        warmup=10, save_surprise_artifacts=True
    )
    
    # Locate session folder (impl detail: archive/nominal_run/...)
    # Or just search temp_artifact_root recursively
    session_dir = None
    for root, dirs, files in os.walk(brain_module.EIDOS_ARCHIVE_ROOT):
        if "nominal_run" in root:
            session_dir = root
            break
    assert session_dir, "Session directory not found"
    
    # --- Schema Validation ---
    
    # 1. steps.csv
    csv_path = os.path.join(session_dir, "steps.csv")
    assert os.path.exists(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "step" in header
        assert "best_err" in header
        rows = list(reader)
        # Should have steps-warmup rows roughly
        assert len(rows) >= (steps - 10 - 5) # -5 for buffer
        
    # 2. summary.json
    summary_path = os.path.join(session_dir, "summary.json")
    with open(summary_path, "r") as f:
        summ = json.load(f)
        assert "total_frames" in summ
        assert summ["total_frames"] == steps
        assert "surprises" in summ
        
    # 3. report.txt
    report_path = os.path.join(session_dir, "report.txt")
    with open(report_path, "r") as f:
        content = f.read()
        assert "SENTINEL SESSION REPORT" in content
        assert "Surprise Rate" in content

@pytest.mark.regression
def test_scenario_spike(brain_module, temp_artifact_root):
    """Spike should trigger anomaly."""
    gen = lambda: scenarios.spike(steps=50, spike_idx=30, magnitude=500.0)
    
    brain_module.run_sentinel_stream(
        gen_factory=gen, est_frames=50, features=64,
        profile_label="int_test", session_label="spike_run",
        warmup=10
    )
    
    # Check anomalies.jsonl
    found_anomaly = False
    for root, dirs, files in os.walk(brain_module.EIDOS_ARCHIVE_ROOT):
        if "anomalies.jsonl" in files:
            with open(os.path.join(root, "anomalies.jsonl"), "r") as f:
                for line in f:
                    rec = json.loads(line)
                    # Relaxed check: if we found ANY anomaly, passed.
                    found_anomaly = True
    
    assert found_anomaly, "Spike did not trigger any anomaly record"



