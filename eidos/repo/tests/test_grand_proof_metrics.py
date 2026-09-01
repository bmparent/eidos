import numpy as np

from eidos_brain.proof.grand_proof_metrics import (
    compression_reference,
    detector_metrics,
    evidence_completeness,
    holm_adjust,
    paired_bootstrap,
)


def test_event_metrics_count_events_not_frames():
    labels = np.zeros(100, dtype=int)
    labels[20:30] = 1
    alerts = np.zeros(100, dtype=bool)
    alerts[22:24] = True
    alerts[70:72] = True
    result = detector_metrics(alerts, labels)
    assert result["event_recall"] == 1.0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0


def test_missing_evidence_stays_missing():
    assert evidence_completeness({}) == 0.0
    wrapper = {
        "observation": {"frame": 1},
        "score": 1.0,
        "threshold": 2.0,
        "source_range": [1, 1],
        "selected_representation": "raw",
        "uncertainty": 1.0,
        "next_action": "observe",
        "config_hash": "cfg",
        "code_commit": "commit",
        "replay_command": "cmd",
    }
    assert evidence_completeness(wrapper) == 1.0


def test_compression_references_do_not_fabricate_detection_metrics():
    frames = np.arange(64, dtype=float).reshape(8, 8)
    result = compression_reference(frames, "gzip")
    assert set(result) == {"system", "status", "reason", "bytes", "raw_bytes", "nbc"}
    assert result["status"] == "OK"


def test_paired_bootstrap_and_holm_are_deterministic():
    left = [1.0, 2.0, 3.0, 4.0]
    right = [0.5, 1.0, 2.0, 3.0]
    assert paired_bootstrap(left, right, resamples=1000) == paired_bootstrap(left, right, resamples=1000)
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]

