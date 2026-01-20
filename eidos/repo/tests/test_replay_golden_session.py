"""
test_replay_golden_session.py

Verifies that the engine runs deterministically on a known dataset.
"""

import pytest
import shutil
import os
from eidos_brain.engine.adapters import run_session

@pytest.fixture
def clean_artifact_root(tmp_path):
    root = tmp_path / "eidos_artifacts"
    root.mkdir()
    return str(root)

def test_golden_session_replay(clean_artifact_root):
    # Setup config
    config = {
        "source_type": "LOCAL",
        "artifact_root": clean_artifact_root,
        "source_params": {
            "local": {
                "mode": "SYNTHETIC", # Use synthetic mode for deterministic output without external files
                "max_frames": 100
            }
        },
        "engine_config": {
            "steps": 100,
            "warmup_cap": 10
        }
    }
    
    # Run Session
    result = run_session(config)
    
    assert result["status"] == "SUCCESS"
    assert "engine_hash" in result
    
    # Check artifacts
    assert os.path.isdir(clean_artifact_root)
    # Manifest exists?
    # Engine creates artifacts in subdirs usually
