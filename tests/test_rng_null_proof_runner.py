import csv, json
from pathlib import Path

from proof.rng_null_proof import ProofPredictor, assert_predict_before_reveal, binomial_tail, run_proof, verdict_for


def test_predict_before_reveal_order_enforced():
    p = ProofPredictor(10)
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


def test_smoke_run_writes_artifacts(tmp_path):
    out = tmp_path / "rng"
    result = run_proof("smoke", 42, 200, out)
    expected = ["config.lock.json", "git_commit.txt", "environment.txt", "rng_manifest.json", "predictions.csv", "score_summary.csv", "score_summary.md", "sentinel_summary.json", "compression_summary.json", "null_verdict.json"]
    for name in expected:
        assert (out / name).exists(), name
    assert (out / "plots" / "accuracy_over_time.png").exists()
    with (out / "predictions.csv").open() as f:
        row = next(csv.DictReader(f))
    assert {"step", "source_name", "target_space", "predicted_value", "actual_value", "correct", "error", "surprise", "sentinel_status"} <= set(row)
    verdicts = json.loads((out / "null_verdict.json").read_text())["verdicts"]
    assert verdicts
    assert result["verdicts"]
