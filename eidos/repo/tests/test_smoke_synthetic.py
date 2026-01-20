"""
Smoke test for the synthetic local run.
"""

import os

import pytest

pytest.importorskip("torch")

from eidos_brain.engine.adapters import run_session


def test_smoke_local_synthetic_runs(tmp_path):
    artifact_root = tmp_path / "artifacts"
    config = {
        "source_type": "LOCAL",
        "artifact_root": str(artifact_root),
        "source_params": {
            "local": {
                "mode": "SYNTHETIC",
                "max_frames": 50,
            }
        },
        "engine_config": {
            "steps": 50,
            "warmup_cap": 5,
            "reservoir": 128,
        },
    }

    result = run_session(config)

    assert result["status"] == "SUCCESS"
    assert os.path.isdir(result["artifact_root"])
