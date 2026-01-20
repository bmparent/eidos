"""
Tests for provenance manifest writer.
"""

import json
import os

from eidos_brain.utils.provenance import write_run_manifest


def test_manifest_writer_no_throw(tmp_path):
    path = write_run_manifest("session-123", {"mode": "synthetic"}, str(tmp_path))
    assert path is not None
    assert os.path.exists(path)


def test_manifest_contains_expected_keys(tmp_path):
    path = write_run_manifest("session-456", {"mode": "synthetic"}, str(tmp_path))
    assert path is not None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("session_id", "engine_hash", "repo_commit", "config_hash", "config", "created_at"):
        assert key in data
