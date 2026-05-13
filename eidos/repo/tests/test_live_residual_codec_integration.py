import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from eidos_brain.engine.adapters import run_session


def test_live_run_emits_residual_codec_tokens_with_real_metrics(tmp_path):
    artifact_root = tmp_path / "artifacts"
    profile = "live_residual_codec"
    result = run_session(
        {
            "source_type": "LOCAL",
            "artifact_root": str(artifact_root),
            "profile_label": profile,
            "source_params": {
                "local": {
                    "mode": "SYNTHETIC",
                    "max_frames": 10,
                }
            },
            "engine_config": {
                "steps": 10,
                "warmup_cap": 2,
                "reservoir": 32,
                "residual_codec_enabled": True,
                "residual_codec_store_prediction": True,
            },
        }
    )

    assert result["status"] == "SUCCESS"
    compression_dir = artifact_root / "compression" / profile
    meta_paths = sorted(compression_dir.glob("*residual_tokens_meta*.json"))
    jsonl_paths = sorted(compression_dir.glob("*residual_tokens_*.jsonl"))

    assert meta_paths
    assert jsonl_paths

    meta = json.loads(meta_paths[-1].read_text(encoding="utf-8"))
    assert meta["prediction_source"] == "live_eidos_consensus_best_pred"
    assert meta["sentinel_source"] == "live_sentinel_analyze"
    assert meta["hdc_source"] == "live_hippocampus_metrics"
    assert meta["reconstruction_rmse"] is not None
    assert meta["reconstruction_rmse"] < 0.05
    assert meta["frames"] == meta["packed_writer"]["tokens_written"]
    assert meta["packed_writer"]["chunks_written"] >= 1
    assert meta["packed_writer"]["bytes_written"] == meta["packed_bytes"]
    assert meta["jsonl_writer"]["tokens_written"] == meta["frames"]

    first_line = jsonl_paths[-1].read_text(encoding="utf-8").splitlines()[0]
    token = json.loads(first_line)
    assert token["payload"]["prediction"]
    assert "surprise_score" in token["sentinel_metrics"]
    assert "familiarity" in token["hdc_metrics"]
