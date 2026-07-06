from tools.build_month1_final_proof_package import Gate, compute_weighted_score, fmt, status_credit


def test_status_credit_is_explicit_and_conservative():
    assert status_credit("passed", 10) == 10
    assert status_credit("partial", 10) == 5
    assert status_credit("evidence_exists", 10) == 5
    assert status_credit("missing", 10) == 0
    assert status_credit("blocked", 10) == 0
    assert status_credit("unknown", 10) == 0


def test_compute_weighted_score_uses_declared_status_credit():
    gates = [
        Gate("passed gate", 15, "passed", ["a"], "done"),
        Gate("partial gate", 10, "partial", ["b"], "some evidence"),
        Gate("missing gate", 5, "missing", [], "no evidence"),
    ]

    score = compute_weighted_score(gates)

    assert score["earned"] == 20
    assert score["possible"] == 30
    assert "passed=1" in score["formula"]


def test_fmt_preserves_unknown_metrics_as_na():
    assert fmt(None) == "NA"
    assert fmt("") == "NA"
    assert fmt("NA") == "NA"
    assert fmt(1.0) == "1"
    assert fmt(1.2345678) == "1.23457"
