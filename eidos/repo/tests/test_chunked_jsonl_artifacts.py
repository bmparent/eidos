import json

import pytest

pytest.importorskip("torch")

from eidos_brain.engine.adapters import run_session


def test_live_residual_codec_writes_chunked_jsonl_artifact(tmp_path):
    artifact_root = tmp_path / "artifacts"
    profile = "chunked_residual_codec"
    result = run_session(
        {
            "source_type": "LOCAL",
            "artifact_root": str(artifact_root),
            "profile_label": profile,
            "source_params": {
                "local": {
                    "mode": "SYNTHETIC",
                    "max_frames": 12,
                }
            },
            "engine_config": {
                "steps": 12,
                "warmup_cap": 2,
                "reservoir": 32,
                "residual_codec_enabled": True,
                "residual_codec_store_prediction": True,
                "residual_codec_store_jsonl": True,
                "residual_codec_flush_every_tokens": 2,
                "residual_codec_flush_every_bytes": 1_000_000,
            },
        }
    )

    assert result["status"] == "SUCCESS"
    compression_dir = artifact_root / "compression" / profile
    meta_path = sorted(compression_dir.glob("*residual_tokens_meta*.json"))[-1]
    jsonl_path = sorted(compression_dir.glob("*residual_tokens_*.jsonl"))[-1]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    jsonl_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    tokens = [json.loads(line) for line in jsonl_lines]

    assert len(tokens) == meta["frames"]
    assert meta["jsonl_writer"]["tokens_written"] == meta["frames"]
    assert meta["jsonl_writer"]["chunks_written"] >= 6
    assert meta["jsonl_writer"]["bytes_written"] == jsonl_path.stat().st_size
    assert tokens[0]["payload"]["prediction"]
    assert "surprise_score" in tokens[0]["sentinel_metrics"]
