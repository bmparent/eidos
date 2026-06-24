import csv, json
from pathlib import Path

from proof.rng_null_proof import (
    BaselineFrequencyTransitionPredictor,
    EidosEngineAdapter,
    ProofPredictor,
    assert_predict_before_reveal,
    binomial_tail,
    eidos_adapter_status,
    run_proof,
    verdict_for,
)


def test_predict_before_reveal_order_enforced():
    p = BaselineFrequencyTransitionPredictor(10)
    pred = p.predict()
    assert pred == 0
    p.learn(3)
    assert assert_predict_before_reveal(p.call_log)


def test_order_violation_fails():
    try:
        assert_predict_before_reveal([("learn", 0), ("predict", 1)])
    except AssertionError:
        return
    raise AssertionError("expected violation")


def test_chance_baseline_binomial():
    assert binomial_tail(10, 100, 0.1) > 0.4
    assert binomial_tail(30, 100, 0.1) < 0.001


def test_suspicious_above_chance_null_verdict_possible():
    assert verdict_for("null", 0.2, 1/256, 1e-12, 1e-10) == "SUSPICIOUS_ABOVE_CHANCE"


def test_eidos_adapter_imports_or_fails_clearly():
    status = eidos_adapter_status()
    assert status["eidos_adapter"].endswith("EidosEngineAdapter")
    assert status["prediction_backed_by"] in {"baseline_frequency_transition", "eidos_brain"}
    assert status["eidos_brain_backed"] is False
    if not status["sentinel_backed"]:
        assert status.get("adapter_error")


def test_predict_before_reveal_still_enforced_for_eidos_adapter():
    p = EidosEngineAdapter(10)
    assert p.predict() == 0
    p.learn(3)
    assert assert_predict_before_reveal(p.call_log)


def test_smoke_run_writes_artifacts(tmp_path):
    out = tmp_path / "rng"
    result = run_proof("smoke", 42, 200, out)
    expected = ["config.lock.json", "git_commit.txt", "environment.txt", "rng_manifest.json", "predictions.csv", "score_summary.csv", "score_summary.md", "sentinel_summary.json", "compression_summary.json", "null_verdict.json"]
    for name in expected:
        assert (out / name).exists(), name
    assert (out / "plots" / "accuracy_over_time.png").exists() or (out / "plots" / "README.md").exists()
    with (out / "predictions.csv").open() as f:
        row = next(csv.DictReader(f))
    assert {"step", "source_name", "target_space", "predicted_value", "actual_value", "correct", "error", "surprise", "sentinel_status", "sentinel_regime", "compression_ratio"} <= set(row)
    verdicts = json.loads((out / "null_verdict.json").read_text())
    assert verdicts["verdicts"]
    assert result["verdicts"]


def test_official_gate_requires_eidos_brain_backed_true(tmp_path):
    result = run_proof("smoke", 42, 50, tmp_path / "rng")
    verdict = json.loads((tmp_path / "rng" / "null_verdict.json").read_text())
    assert verdict["official_proof_ready"] is verdict["eidos_brain_backed"]
    assert result["official_proof_ready"] is result["eidos_brain_backed"]


def test_official_proof_ready_false_for_baseline_prediction(tmp_path):
    result = run_proof("smoke", 42, 50, tmp_path / "rng")
    assert result["prediction_backed_by"] == "baseline_frequency_transition"
    assert result["official_proof_ready"] is False


def test_baseline_predictor_is_not_reported_as_eidos(tmp_path):
    out = tmp_path / "rng"
    run_proof("smoke", 42, 50, out)
    md = (out / "score_summary.md").read_text()
    assert "Sentinel-backed; prediction currently uses the naive online baseline unless `eidos_brain_backed` is true" in md
    with (out / "score_summary.csv").open() as f:
        row = next(csv.DictReader(f))
    assert "adapter_top1_accuracy" in row
    assert "eidos_top1_accuracy" not in row
    assert "baseline_frequency_transition_accuracy" in row
    assert "uniform_chance" in row
    assert "top1_accuracy" in row  # compatibility alias, not the baseline name


def test_proofpredictor_alias_remains_baseline():
    assert ProofPredictor is BaselineFrequencyTransitionPredictor
