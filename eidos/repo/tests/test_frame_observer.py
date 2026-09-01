import json
from pathlib import Path

import numpy as np
import pytest

from eidos_brain.proof.frame_observer import FrameObserver, observer_from_config, read_capture


def observation(frame_id: int) -> dict:
    return {
        "frame_id": frame_id,
        "source_id": "fixture",
        "frame": [0.1, 0.2],
        "best_pred": [0.0, 0.0],
        "raw_residual": [0.1, 0.2],
        "normalized_error": 0.15,
        "surprise_score": 1.0,
        "surprise_threshold": 1.5,
        "sentinel_status": "GREEN",
        "sentinel_metrics": {},
        "hdc_metrics": {},
        "thermodynamic_metrics": {"enabled": False, "energy": 0.0, "rho": 1.0, "temperature": 0.0, "lambda": 0.99},
        "codec_decision": {"mode": "reference_or_null"},
        "codec_serialized_bytes": 24,
        "source_refs": [{"frame_id": frame_id}],
    }


def test_disabled_observer_creates_no_artifact(tmp_path: Path):
    capture = tmp_path / "capture.jsonl"
    result = observer_from_config(
        {"meaningful_surprise_enabled": False, "meaningful_surprise_observer_path": str(capture)},
        run_id="disabled",
        config_hash="cfg",
        code_commit="commit",
        replay_command="replay",
    )
    assert result is None
    assert not capture.exists()
    assert not capture.parent.joinpath("capture.jsonl.status.json").exists()


def test_observer_writes_hashed_records_and_completes(tmp_path: Path):
    path = tmp_path / "capture.jsonl"
    writer = FrameObserver(path, run_id="r1", config_hash="cfg", code_commit="abc", replay_command="cmd")
    writer.observe(observation(1))
    writer.observe(observation(2))
    status = writer.finalize()
    assert status["status"] == "COMPLETE"
    assert status["records_written"] == 2
    rows = read_capture(path)
    assert [row["frame_id"] for row in rows] == [1, 2]
    assert rows[0]["record_sha256"]


def test_nonfinite_frame_fails_visibly_and_marks_partial(tmp_path: Path):
    path = tmp_path / "capture.jsonl"
    writer = FrameObserver(path, run_id="r1", config_hash="cfg", code_commit="abc", replay_command="cmd")
    bad = observation(1)
    bad["frame"] = [np.nan]
    with pytest.raises(ValueError, match="non-finite"):
        writer.observe(bad)
    status = json.loads(writer.paths.status.read_text(encoding="utf-8"))
    assert status["status"] == "PARTIAL"
    writer.close_partial("test cleanup")


def test_partial_capture_can_resume_with_same_run_id(tmp_path: Path):
    path = tmp_path / "capture.jsonl"
    writer = FrameObserver(path, run_id="r1", config_hash="cfg", code_commit="abc", replay_command="cmd")
    writer.observe(observation(1))
    writer.close_partial("interrupted")
    resumed = FrameObserver(
        path,
        run_id="r1",
        config_hash="cfg",
        code_commit="abc",
        replay_command="cmd",
        resume=True,
    )
    assert resumed.last_frame_id == 1
    resumed.observe(observation(2))
    resumed.finalize()
    assert len(read_capture(path)) == 2


def test_live_capture_reports_authoritative_sources(tmp_path: Path):
    pytest.importorskip("torch")
    from eidos_brain.proof.grand_proof_runner import capture_live_scenario
    from eidos_brain.proof.grand_proof_scenarios import ScenarioConfig, generate_scenario

    scenario = generate_scenario(
        "S0_nominal",
        seed=0,
        config=ScenarioConfig(features=64, warmup_frames=2, scored_frames=8, outcome_horizon=2),
    )
    rows, receipt = capture_live_scenario(
        scenario,
        out_dir=tmp_path / "capture",
        reservoir=32,
        code_commit="test-commit",
        replay_command="test replay",
    )
    assert len(rows) == 8
    assert receipt["prediction_source"] == "live_eidos_consensus_best_pred"
    assert receipt["sentinel_source"] == "live_sentinel_analyze"
    assert receipt["hdc_source"] == "live_hippocampus_metrics"
    assert Path(receipt["engine_log_path"]).is_file()
    assert "SENTINEL SUMMARY" in Path(receipt["engine_log_path"]).read_text(encoding="utf-8")
    assert rows[0]["prediction_source"] == "live_eidos_consensus_best_pred"
    assert "state_flatness" in rows[0]["sentinel_metrics"]
    assert rows[0]["codec_decision"]["mode"]
