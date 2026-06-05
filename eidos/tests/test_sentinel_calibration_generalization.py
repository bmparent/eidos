import csv
import json
from pathlib import Path

from tools import build_sentinel_calibration_generalization as generalization


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def event(event_id, start, end, classification="fully_benign"):
    return {
        "event_id": event_id,
        "start_frame": start,
        "end_frame": end,
        "severity": "RED",
        "status": "CONFIRMED",
        "component_count": 1,
        "score_detail": {"raw_hit_count": 2, "false_positive_classification": classification},
    }


def write_run(
    root: Path,
    *,
    sample_mode: str,
    frames: int,
    calibrated_events,
    suppressed_events=None,
    attack_windows=None,
    before_fp10k=10.0,
    after_fp10k=0.0,
    recall=1.0,
    precision=1.0,
    f1=1.0,
):
    root.mkdir(parents=True)
    attack_windows = attack_windows if attack_windows is not None else [{"start_frame": 10, "end_frame": 19}]
    suppressed_events = suppressed_events or []
    pre_events = list(calibrated_events) + [event("fp", 0, 0)] if suppressed_events else list(calibrated_events)
    metrics = {
        "sample_mode": sample_mode,
        "frames_processed": frames,
        "confirmation_mode": "balanced",
        "calibration_enabled": True,
        "calibration_version": "sentinel_calibration_v1",
        "proof_raw_event_count": 5,
        "proof_merged_event_count": 2,
        "proof_deduped_event_count": 2,
        "pre_calibration_confirmed_events": len(pre_events),
        "post_calibration_confirmed_events": len(calibrated_events),
        "calibration_suppressed_events": len(suppressed_events),
        "pre_calibration_confirmed_event_metrics": {
            "precision": 0.5 if attack_windows else None,
            "recall": recall if attack_windows else None,
            "f1": 0.666667 if attack_windows else None,
            "false_positives": 1 if suppressed_events else 0,
            "false_positives_per_10k_frames": before_fp10k,
        },
        "calibrated_event_metrics": {
            "precision": precision,
            "recall": recall if attack_windows else None,
            "f1": f1 if attack_windows else None,
            "false_positives": 0,
            "false_positives_per_10k_frames": after_fp10k,
        },
        "precision": precision,
        "recall": recall if attack_windows else None,
        "f1": f1 if attack_windows else None,
        "false_positives": 0,
        "false_positives_per_10k_frames": after_fp10k,
        "runtime_seconds": 2.0,
        "frames_per_second": frames / 2.0,
        "crash_hit_count": 0,
    }
    write_json(root / "labeled_metrics.json", metrics)
    write_json(
        root / "event_summary.json",
        {
            "label_windows": attack_windows,
            "confirmed_events": calibrated_events,
            "confirmed_event_count": len(calibrated_events),
            "pre_calibration_confirmed_events": pre_events,
            "pre_calibration_confirmed_event_count": len(pre_events),
            "post_calibration_confirmed_events": calibrated_events,
            "post_calibration_confirmed_event_count": len(calibrated_events),
            "raw_events": pre_events,
            "raw_event_count": 5,
            "merged_event_count": 2,
            "deduped_event_count": 2,
            "calibration_suppressed_events": suppressed_events,
        },
    )
    write_json(
        root / "event_confirmation_report.json",
        {
            "mode": "balanced",
            "confirmed_events": pre_events,
            "suppressed_events": [],
            "decisions": [],
        },
    )
    write_json(
        root / "precision_ledger.json",
        {
            "incident_card_accounting": {
                "proof_raw_event_count": 5,
                "proof_merged_event_count": 2,
                "proof_deduped_event_count": 2,
                "duplicate_event_count": 0,
                "incident_card_coverage": 1.0,
            },
            "attack_window_diagnostics": [
                {
                    "start_frame": window["start_frame"],
                    "end_frame": window["end_frame"],
                    "label_distribution": window.get("label_distribution", {}),
                }
                for window in attack_windows
            ],
        },
    )
    write_json(
        root / "sentinel_calibration_v1.json",
        {
            "calibration_enabled": True,
            "calibration_version": "sentinel_calibration_v1",
            "counts": {
                "pre_calibration_confirmed_events": len(pre_events),
                "post_calibration_confirmed_events": len(calibrated_events),
                "suppressed_events": len(suppressed_events),
            },
            "before_metrics": metrics["pre_calibration_confirmed_event_metrics"],
            "after_metrics": metrics["calibrated_event_metrics"],
            "suppressed_events": suppressed_events,
            "guardrails": {"passed": True, "checks": {"crash_hits_zero": True}},
        },
    )
    write_json(
        root / "calibrated_precision_ledger.json",
        {
            "before_after_metrics": {
                "before": metrics["pre_calibration_confirmed_event_metrics"],
                "after": metrics["calibrated_event_metrics"],
            },
            "suppressed_events": suppressed_events,
        },
    )
    write_json(
        root / "proof_digest.json",
        {
            "sample_mode": sample_mode,
            "frames_processed": frames,
            "runtime_seconds": 2.0,
            "frames_per_second": frames / 2.0,
            "crash_scan": {"crash_hit_count": 0},
        },
    )
    write_json(
        root / "run_manifest.json",
        {
            "git": {"branch": "test", "commit": "abc123", "dirty": False},
            "device": {
                "selected_device": "cpu",
                "cuda_available": False,
                "cpu_gpu_mode": "cpu",
                "runtime_seconds": 2.0,
                "frames_per_second": frames / 2.0,
            },
            "sentinel_calibration_v1": {"enabled": True, "version": "sentinel_calibration_v1"},
        },
    )
    write_json(root / "crash_scan.json", {"crash_hit_count": 0, "status": "clean"})
    with (root / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["confirmation_mode"])
        writer.writeheader()
        writer.writerow({"confirmation_mode": "balanced"})


def write_acceptance(root: Path):
    package = {
        "recommendation": {
            "decision": "approved",
            "recommended_baseline": "balanced + sentinel_calibration_v1",
        },
        "summary_rows": [
            {
                "sample": "transition",
                "uncalibrated_fp10k": 10.0,
                "pre_calibration_fp10k": 10.0,
                "post_calibration_fp10k": 0.0,
                "post_calibration_recall": 1.0,
                "post_calibration_precision": 1.0,
                "post_calibration_f1": 1.0,
                "attack_window_coverage_pct": 100.0,
                "first_detection_latency_frames": 0.0,
                "missed_attack_windows": 0,
                "crash_hit_count": 0,
                "status": "passed",
            }
        ],
    }
    write_json(root / "calibration_v1_acceptance.json", package)


def fake_mirror(out_dir: Path, _run_id: str, _run_date: str):
    return {
        "drive_copy_attempted": True,
        "drive_copy_success": False,
        "drive_root": "unknown",
        "drive_run_dir": "unknown",
        "files_considered": [path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*") if path.is_file()],
        "files_copied": [],
        "files_skipped": [],
        "reason": "not configured in test",
        "timestamp_utc": "2026-06-04T00:00:00Z",
    }


def test_generalization_report_writes_required_receipts_and_manifest_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(
        generalization.proof_helpers,
        "collect_git_info",
        lambda _repo_root: {"branch": "test", "commit": "abc123", "dirty": False},
    )
    transition = tmp_path / "runs" / "transition"
    natural = tmp_path / "runs" / "natural_attack"
    normal = tmp_path / "runs" / "normal_only"
    suppressed = [
        {
            "event_id": "fp",
            "start_frame": 0,
            "end_frame": 0,
            "raw_severity": "RED",
            "raw_status": "CONFIRMED",
            "reason_code": "fully_benign_context",
            "nearest_attack_window_distance": 10,
            "labels_around_event": [{"frame": 0, "OriginalLabel": "BENIGN", "EidosProofLabel": "BENIGN"}],
            "suppression_could_affect_recall": False,
            "suppression_would_affect_attack_window_coverage": False,
            "raw_evidence_reference": ["raw_event_1"],
        }
    ]
    write_run(
        transition,
        sample_mode="transition",
        frames=100,
        calibrated_events=[event("tp", 10, 12, "overlap_boundary")],
        suppressed_events=suppressed,
    )
    write_run(
        natural,
        sample_mode="natural",
        frames=120,
        calibrated_events=[event("tp", 10, 12, "overlap_boundary")],
        suppressed_events=[],
    )
    write_run(
        normal,
        sample_mode="natural",
        frames=200,
        calibrated_events=[],
        suppressed_events=suppressed,
        attack_windows=[],
        before_fp10k=50.0,
        after_fp10k=0.0,
        recall=None,
        precision=None,
        f1=None,
    )
    write_acceptance(tmp_path / "acceptance")

    args = generalization.parse_args(
        [
            "--run",
            "transition_scale",
            str(transition),
            "--run",
            "natural_attack",
            str(natural),
            "--run",
            "normal_only",
            str(normal),
            "--acceptance-rerun",
            str(tmp_path / "acceptance"),
            "--acceptance-reference",
            str(tmp_path / "acceptance"),
            "--out",
            str(tmp_path / "out"),
        ]
    )

    result = generalization.run(args, repo_root=tmp_path, mirror_to_drive_fn=fake_mirror, write_docs=False)

    report = result["report"]
    assert report["baseline_record"]["baseline_candidate"] == "balanced + sentinel_calibration_v1"
    assert report["recommendation"]["decision"] == "approve"
    assert report["acceptance_comparison"]["matches_reference"] is True
    assert report["suppression_audit"][0]["suppression_reason"] == "fully_benign_context"
    assert report["suppression_audit"][0]["raw_evidence_reference"] == ["raw_event_1"]
    assert report["generalization_matrix"][0]["raw_event_count"] == 5
    assert report["generalization_matrix"][0]["cpu_gpu_mode"] == "cpu"
    assert any(row["case"] == "normal_only" and row["after_fp10k"] == 0.0 for row in report["generalization_matrix"])
    assert any(row["case"] == "natural_attack" and row["recall"] == 1.0 for row in report["generalization_matrix"])
    assert (tmp_path / "out" / "calibration_v1_generalization_report.md").is_file()
    assert (tmp_path / "out" / "calibration_v1_generalization_report.json").is_file()
    assert (tmp_path / "out" / "calibration_v1_generalization_summary.csv").is_file()
    assert (tmp_path / "out" / "generalization_manifest.json").is_file()
