import csv
import json
from pathlib import Path

from tools import build_sentinel_calibration_acceptance as acceptance


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def event(event_id, start, end):
    return {
        "event_id": event_id,
        "start_frame": start,
        "end_frame": end,
        "severity": "RED",
        "component_count": 1,
    }


def write_run(
    root: Path,
    *,
    calibrated: bool,
    confirmed_events,
    fp10k: float,
    precision,
    recall,
    f1,
    false_positives: int,
    before_fp10k=None,
    suppressed_events=None,
    write_drive=True,
):
    root.mkdir(parents=True)
    label_windows = [{"start_frame": 10, "end_frame": 19, "duration": 10, "label_distribution": {"ATTACK": 10}}]
    metrics = {
        "confirmation_mode": "balanced",
        "calibration_enabled": calibrated,
        "calibration_version": "sentinel_calibration_v1" if calibrated else "disabled",
        "sample_mode": "transition",
        "frames_processed": 100,
        "proof_raw_event_count": 5,
        "proof_merged_event_count": 2,
        "proof_deduped_event_count": 2,
        "proof_confirmed_event_count": len(confirmed_events),
        "confirmed_events": len(confirmed_events),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": false_positives,
        "false_positives_per_10k_frames": fp10k,
        "crash_hit_count": 0,
    }
    if calibrated:
        metrics.update(
            {
                "pre_calibration_confirmed_events": len(confirmed_events) + len(suppressed_events or []),
                "post_calibration_confirmed_events": len(confirmed_events),
                "calibration_suppressed_events": len(suppressed_events or []),
                "pre_calibration_confirmed_event_metrics": {
                    "precision": 0.5,
                    "recall": 1.0,
                    "f1": 0.666667,
                    "false_positives_per_10k_frames": before_fp10k,
                },
                "calibrated_event_metrics": {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_positives_per_10k_frames": fp10k,
                },
            }
        )
    artifacts = {
        "labeled_metrics.json": metrics,
        "event_summary.json": {
            "confirmation_mode": "balanced",
            "label_windows": label_windows,
            "confirmed_events": confirmed_events,
            "confirmed_event_count": len(confirmed_events),
            "raw_event_count": 5,
            "merged_event_count": 2,
            "deduped_event_count": 2,
            "calibration_enabled": calibrated,
            "calibration_version": "sentinel_calibration_v1" if calibrated else "disabled",
        },
        "event_confirmation_report.json": {
            "mode": "balanced",
            "confirmed_events": confirmed_events,
            "suppressed_events": [],
            "decisions": [],
        },
        "precision_ledger.json": {
            "incident_card_accounting": {
                "proof_raw_event_count": 5,
                "proof_merged_event_count": 2,
                "proof_deduped_event_count": 2,
                "duplicate_event_count": 3,
                "incident_card_coverage": 1.0,
            },
            "false_positive_events": [],
            "attack_window_diagnostics": [],
        },
        "proof_digest.json": {
            "confirmation_mode": "balanced",
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": false_positives,
            "false_positives_per_10k_frames": fp10k,
            "git_commit": "abc123",
        },
        "run_manifest.json": {
            "git": {"commit": "abc123", "branch": "test", "dirty": False},
            "config": {"config_hash_sha256": "hash"},
            "device": {"selected_device": "cpu", "cuda_available": False},
            "event_confirmation": {"mode": "balanced"},
            "sentinel_calibration_v1": {
                "enabled": calibrated,
                "version": "sentinel_calibration_v1" if calibrated else "disabled",
                "config_hash_sha256": "calibration-hash",
            },
        },
        "crash_scan.json": {"crash_hit_count": 0, "status": "clean", "crash_hit_files": []},
    }
    if calibrated:
        artifacts["sentinel_calibration_v1.json"] = {
            "calibration_enabled": True,
            "calibration_version": "sentinel_calibration_v1",
            "config_hash_sha256": "calibration-hash",
            "counts": {
                "pre_calibration_confirmed_events": len(confirmed_events) + len(suppressed_events or []),
                "post_calibration_confirmed_events": len(confirmed_events),
                "suppressed_events": len(suppressed_events or []),
            },
            "before_metrics": {
                "precision": 0.5,
                "recall": 1.0,
                "f1": 0.666667,
                "false_positives_per_10k_frames": before_fp10k,
            },
            "after_metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives_per_10k_frames": fp10k,
            },
            "suppressed_events": suppressed_events or [],
            "guardrails": {
                "passed": True,
                "checks": {
                    "transition_attack_window_coverage_100": True,
                    "transition_first_detection_latency_zero": True,
                    "transition_missed_attack_windows_zero": True,
                },
            },
        }
        artifacts["calibrated_precision_ledger.json"] = {
            "calibration_enabled": True,
            "calibration_version": "sentinel_calibration_v1",
            "before_after_metrics": {
                "before": {
                    "precision": 0.5,
                    "recall": 1.0,
                    "f1": 0.666667,
                    "false_positives_per_10k_frames": before_fp10k,
                },
                "after": {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_positives_per_10k_frames": fp10k,
                },
            },
            "suppressed_events": suppressed_events or [],
        }
    for name, payload in artifacts.items():
        write_json(root / name, payload)
    with (root / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["confirmation_mode"])
        writer.writeheader()
        writer.writerow({"confirmation_mode": "balanced"})
    if write_drive:
        write_json(
            root / "drive_manifest.json",
            {
                "drive_copy_attempted": True,
                "drive_copy_success": False,
                "drive_root": "unknown",
                "drive_run_dir": "unknown",
                "reason": "not configured in test",
                "timestamp_utc": "2026-06-04T00:00:00Z",
            },
        )


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


def test_acceptance_package_approves_calibrated_baseline_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        acceptance.proof_helpers,
        "collect_git_info",
        lambda _repo_root: {"branch": "test", "commit": "abc123", "dirty": False},
    )
    uncal = tmp_path / "runs" / "uncal"
    cal = tmp_path / "runs" / "cal"
    write_run(
        uncal,
        calibrated=False,
        confirmed_events=[event("fp", 0, 0), event("tp", 10, 12)],
        fp10k=100.0,
        precision=0.5,
        recall=1.0,
        f1=0.666667,
        false_positives=1,
    )
    write_run(
        cal,
        calibrated=True,
        confirmed_events=[event("tp", 10, 12)],
        fp10k=0.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        false_positives=0,
        before_fp10k=100.0,
        suppressed_events=[{"event_id": "fp", "start_frame": 0, "end_frame": 0, "reason_code": "fully_benign_context"}],
    )
    args = acceptance.parse_args(["--pair", "transition2k", str(uncal), str(cal), "--out", str(tmp_path / "out")])

    result = acceptance.run(args, repo_root=tmp_path, mirror_to_drive_fn=fake_mirror, write_docs=False)

    package = result["package"]
    assert package["recommendation"]["decision"] == "approved"
    assert package["recommendation"]["recommended_baseline"] == "balanced + sentinel_calibration_v1"
    assert package["summary_rows"][0]["raw_pre_post_metrics_visible"] is True
    assert package["guardrail_rows"][0]["transition_first_detection_latency_zero"] is True
    assert (tmp_path / "out" / "calibration_v1_acceptance.md").is_file()
    assert (tmp_path / "out" / "acceptance_manifest.json").is_file()


def test_acceptance_declines_when_pre_calibration_fp10k_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        acceptance.proof_helpers,
        "collect_git_info",
        lambda _repo_root: {"branch": "test", "commit": "abc123", "dirty": False},
    )
    uncal = tmp_path / "runs" / "uncal"
    cal = tmp_path / "runs" / "cal"
    write_run(
        uncal,
        calibrated=False,
        confirmed_events=[event("fp", 0, 0), event("tp", 10, 12)],
        fp10k=100.0,
        precision=0.5,
        recall=1.0,
        f1=0.666667,
        false_positives=1,
    )
    write_run(
        cal,
        calibrated=True,
        confirmed_events=[event("tp", 10, 12)],
        fp10k=0.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        false_positives=0,
        before_fp10k=None,
        suppressed_events=[{"event_id": "fp", "start_frame": 0, "end_frame": 0, "reason_code": "fully_benign_context"}],
    )
    args = acceptance.parse_args(["--pair", "transition2k", str(uncal), str(cal), "--out", str(tmp_path / "out")])

    result = acceptance.run(args, repo_root=tmp_path, mirror_to_drive_fn=fake_mirror, write_docs=False)

    assert result["package"]["recommendation"]["decision"] == "declined"
    assert result["package"]["guardrail_rows"][0]["raw_pre_post_metrics_visible"] is False


def test_acceptance_declines_when_drive_status_is_not_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        acceptance.proof_helpers,
        "collect_git_info",
        lambda _repo_root: {"branch": "test", "commit": "abc123", "dirty": False},
    )
    uncal = tmp_path / "runs" / "uncal"
    cal = tmp_path / "runs" / "cal"
    write_run(
        uncal,
        calibrated=False,
        confirmed_events=[event("fp", 0, 0), event("tp", 10, 12)],
        fp10k=100.0,
        precision=0.5,
        recall=1.0,
        f1=0.666667,
        false_positives=1,
    )
    write_run(
        cal,
        calibrated=True,
        confirmed_events=[event("tp", 10, 12)],
        fp10k=0.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        false_positives=0,
        before_fp10k=100.0,
        suppressed_events=[{"event_id": "fp", "start_frame": 0, "end_frame": 0, "reason_code": "fully_benign_context"}],
        write_drive=False,
    )
    args = acceptance.parse_args(["--pair", "transition2k", str(uncal), str(cal), "--out", str(tmp_path / "out")])

    result = acceptance.run(args, repo_root=tmp_path, mirror_to_drive_fn=fake_mirror, write_docs=False)

    assert result["package"]["recommendation"]["decision"] == "declined"
    assert result["package"]["guardrail_rows"][0]["drive_status_explicit"] is False
