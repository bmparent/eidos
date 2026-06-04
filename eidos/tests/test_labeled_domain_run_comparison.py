import csv
import hashlib
import json
from pathlib import Path

from tools import compare_labeled_domain_runs as compare_runs


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_hashes(root: Path):
    hashes = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[path.relative_to(root).as_posix()] = digest
    return hashes


def fake_drive_manifest(out_dir: Path, _run_id: str, _run_date: str):
    return {
        "drive_copy_attempted": True,
        "drive_copy_success": False,
        "drive_root": "unknown",
        "drive_run_dir": "unknown",
        "files_considered": [path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*") if path.is_file()],
        "files_copied": [],
        "files_skipped": [],
        "reason": "not configured in test",
        "timestamp_utc": "2026-06-02T00:00:00Z",
    }


def event(event_id, start, end, classification=None, component_count=1):
    payload = {
        "event_id": event_id,
        "start_frame": start,
        "end_frame": end,
        "component_count": component_count,
        "severity": "RED",
    }
    if classification:
        payload["score_detail"] = {
            "false_positive_classification": classification,
            "start_frame": start,
            "end_frame": end,
        }
    return payload


def write_fake_run(
    root: Path,
    *,
    mode: str,
    precision: float,
    recall: float,
    f1: float,
    false_positives: int,
    fp10k: float,
    confirmed_events,
    suppressed_events=None,
    omit=(),
):
    root.mkdir(parents=True)
    label_windows = [{"start_frame": 10, "end_frame": 19, "duration": 10, "label_distribution": {"ATTACK": 10}}]
    metrics = {
        "confirmation_mode": mode,
        "sample_mode": "transition",
        "frames_processed": 100,
        "proof_raw_event_count": 5,
        "proof_merged_event_count": 2,
        "proof_deduped_event_count": 2,
        "proof_confirmed_event_count": len(confirmed_events),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": false_positives,
        "false_positives_per_10k_frames": fp10k,
        "duplicate_event_count": 3,
        "incident_card_count": 2,
        "incident_card_coverage": 1.0,
        "eidos_compression_ratio": 2.5,
        "external_compression_baselines": {
            "best_baseline": "lzma",
            "best_baseline_compression_ratio": 1.2,
        },
        "runtime_seconds": 2.0,
        "frames_per_second": 50.0,
        "crash_hit_count": 0,
    }
    artifacts = {
        "labeled_metrics.json": metrics,
        "event_summary.json": {
            "confirmation_mode": mode,
            "label_windows": label_windows,
            "confirmed_events": confirmed_events,
            "confirmed_event_count": len(confirmed_events),
            "raw_event_count": 5,
            "merged_event_count": 2,
            "deduped_event_count": 2,
        },
        "event_confirmation_report.json": {
            "mode": mode,
            "confirmed_events": confirmed_events,
            "suppressed_events": suppressed_events or [],
            "decisions": [],
        },
        "precision_ledger.json": {
            "incident_card_accounting": {
                "proof_raw_event_count": 5,
                "proof_merged_event_count": 2,
                "proof_deduped_event_count": 2,
                "duplicate_event_count": 3,
                "incident_card_coverage": 1.0,
                "incident_card_coverage_detail": {
                    "incident_cards_written": 2,
                    "proof_deduped_events": 2,
                    "coverage_ratio": 1.0,
                },
            },
            "false_positive_events": [
                {"classification": "fully_benign", "view": "raw", "start_frame": 0, "end_frame": 0}
            ],
            "attack_window_diagnostics": [],
        },
        "proof_digest.json": {
            "confirmation_mode": mode,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": false_positives,
            "false_positives_per_10k_frames": fp10k,
            "git_commit": f"{mode}-commit",
            "eidos_compression_ratio": 2.5,
        },
        "run_manifest.json": {
            "git": {"commit": f"{mode}-commit", "branch": "test", "dirty": False},
            "config": {"config_hash_sha256": f"{mode}-hash"},
            "device": {
                "selected_device": "cpu",
                "cuda_available": False,
                "runtime_seconds": 2.0,
                "frames_per_second": 50.0,
            },
            "event_confirmation": {"mode": mode},
        },
        "crash_scan.json": {
            "crash_hit_count": 0,
            "status": "clean",
            "crash_hit_files": [],
        },
        "benchmark_summary.json": {"mode": mode},
    }
    for name, payload in artifacts.items():
        if name not in omit:
            write_json(root / name, payload)
    if "benchmark_summary.csv" not in omit:
        with (root / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["confirmation_mode", "best_baseline", "best_baseline_compression_ratio"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "confirmation_mode": mode,
                    "best_baseline": "lzma",
                    "best_baseline_compression_ratio": "1.2",
                }
            )
    return root


def test_loads_multiple_fake_run_folders(tmp_path):
    balanced = write_fake_run(
        tmp_path / "runs" / "balanced",
        mode="balanced",
        precision=1.0,
        recall=1.0,
        f1=1.0,
        false_positives=0,
        fp10k=0.0,
        confirmed_events=[event("tp", 10, 12)],
    )
    off = write_fake_run(
        tmp_path / "runs" / "off",
        mode="off",
        precision=0.5,
        recall=1.0,
        f1=0.666,
        false_positives=1,
        fp10k=100.0,
        confirmed_events=[event("fp", 0, 0, "fully_benign"), event("tp", 10, 12)],
    )

    loaded = [compare_runs.load_run(balanced, repo_root=tmp_path), compare_runs.load_run(off, repo_root=tmp_path)]

    assert [item.comparison_row["confirmation_mode"] for item in loaded] == ["balanced", "off"]
    assert loaded[0].comparison_row["attack_window_coverage_pct"] == 100.0
    assert loaded[1].false_positive_taxonomy == {"fully_benign": 1}


def test_missing_optional_artifacts_are_reported_and_outputs_are_created(tmp_path, monkeypatch):
    monkeypatch.setattr(
        compare_runs.proof_helpers,
        "collect_git_info",
        lambda _repo_root: {"branch": "test", "commit": "abc123", "dirty": False},
    )
    complete = write_fake_run(
        tmp_path / "runs" / "complete",
        mode="balanced",
        precision=1.0,
        recall=1.0,
        f1=1.0,
        false_positives=0,
        fp10k=0.0,
        confirmed_events=[event("tp", 10, 12)],
    )
    missing = write_fake_run(
        tmp_path / "runs" / "missing",
        mode="low_noise",
        precision=1.0,
        recall=1.0,
        f1=1.0,
        false_positives=0,
        fp10k=0.0,
        confirmed_events=[event("tp", 10, 12)],
        omit=("proof_digest.json", "benchmark_summary.csv", "benchmark_summary.json", "crash_scan.json"),
    )
    out_dir = tmp_path / "artifacts" / "comparison"
    args = compare_runs.parse_args(
        [
            "--runs",
            str(complete),
            str(missing),
            "--out",
            str(out_dir),
            "--recommendation-policy",
            "balanced_f1",
        ]
    )

    result = compare_runs.run(args, repo_root=tmp_path, mirror_to_drive_fn=fake_drive_manifest, write_docs=False)

    assert result["recommendation"]["recommended_mode"] == "balanced"
    for name in compare_runs.COMPARISON_FILES:
        assert (out_dir / name).is_file()
    assert (out_dir / "comparison_manifest.json").is_file()
    assert (out_dir / "drive_manifest.json").is_file()
    report = (out_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert "proof_digest.json;crash_scan.json;benchmark_summary.csv;benchmark_summary.json" in report


def test_recommendation_policy_behavior():
    rows = [
        {
            "confirmation_mode": "off",
            "precision": 0.5,
            "recall": 1.0,
            "f1": 0.666,
            "false_positives_per_10k_frames": 100.0,
            "attack_window_coverage_pct": 100.0,
        },
        {
            "confirmation_mode": "low_noise",
            "precision": 1.0,
            "recall": 0.6,
            "f1": 0.75,
            "false_positives_per_10k_frames": 0.0,
            "attack_window_coverage_pct": 100.0,
        },
        {
            "confirmation_mode": "balanced",
            "precision": 0.95,
            "recall": 0.9,
            "f1": 0.924,
            "false_positives_per_10k_frames": 5.0,
            "attack_window_coverage_pct": 100.0,
        },
        {
            "confirmation_mode": "high_recall",
            "precision": 0.8,
            "recall": 1.0,
            "f1": 0.889,
            "false_positives_per_10k_frames": 20.0,
            "attack_window_coverage_pct": 100.0,
        },
    ]

    assert compare_runs.recommend_mode(rows, "precision_first")["recommended_mode"] == "low_noise"
    assert compare_runs.recommend_mode(rows, "balanced_f1")["recommended_mode"] == "balanced"
    assert compare_runs.recommend_mode(rows, "recall_first")["recommended_mode"] == "high_recall"


def test_comparison_does_not_mutate_source_run_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        compare_runs.proof_helpers,
        "collect_git_info",
        lambda _repo_root: {"branch": "test", "commit": "abc123", "dirty": False},
    )
    run_a = write_fake_run(
        tmp_path / "runs" / "a",
        mode="balanced",
        precision=1.0,
        recall=1.0,
        f1=1.0,
        false_positives=0,
        fp10k=0.0,
        confirmed_events=[event("tp", 10, 12)],
    )
    run_b = write_fake_run(
        tmp_path / "runs" / "b",
        mode="off",
        precision=0.5,
        recall=1.0,
        f1=0.666,
        false_positives=1,
        fp10k=100.0,
        confirmed_events=[event("fp", 0, 0, "fully_benign"), event("tp", 10, 12)],
    )
    before = {run_a: file_hashes(run_a), run_b: file_hashes(run_b)}
    args = compare_runs.parse_args(
        [
            "--runs",
            str(run_a),
            str(run_b),
            "--out",
            str(tmp_path / "artifacts" / "comparison"),
            "--recommendation-policy",
            "balanced_f1",
        ]
    )

    compare_runs.run(args, repo_root=tmp_path, mirror_to_drive_fn=fake_drive_manifest, write_docs=False)

    assert before == {run_a: file_hashes(run_a), run_b: file_hashes(run_b)}
