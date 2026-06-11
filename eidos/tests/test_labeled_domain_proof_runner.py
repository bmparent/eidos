import csv
import json
from pathlib import Path

import numpy as np

from proof import event_confirmation
from proof import sentinel_calibration_v1
from tools import run_labeled_domain_proof


class FakeProjector:
    def __init__(self, features, seed=0):
        self.features = features

    def to_dim(self, row):
        out = np.zeros(self.features, dtype=float)
        width = min(len(row), self.features)
        out[:width] = row[:width]
        return out


class FakeEngine:
    ENGINE_VERSION = "fake-labeled-engine"
    EIDOS_BRAIN_CONFIG = {"domain": "generic", "sigma_k": 2.0}
    AutoProjector = FakeProjector

    def run_stream_once(self, gen_factory, est_frames, features, profile_label, session_label, cfg_overrides, **kwargs):
        rows = []
        for idx, (event, _meta) in enumerate(gen_factory()):
            attack = bool(event.get("attack"))
            z = 5.4 if attack else 0.4
            rows.append(
                {
                    "step": idx,
                    "z": z,
                    "z_thresh_eff": 2.5,
                    "ratio": 1.5 + idx / 100.0,
                    "status": "RED" if attack else "GREEN",
                    "dominance": 0.9 if attack else 0.1,
                    "state_entropy": 0.2 if attack else 0.8,
                }
            )
        return {
            "summary": {"frames_processed": len(rows), "ratio": rows[-1]["ratio"] if rows else None},
            "step_rows": rows,
            "report_text": "fake labeled proof",
        }


def write_fixture(path: Path) -> None:
    rows = [
        ("10.0.0.1-10.0.0.2", "10.0.0.1", "10.0.0.2", "80", "6", "10", "2", "1", "120", "BENIGN"),
        ("10.0.0.1-10.0.0.2", "10.0.0.1", "10.0.0.2", "80", "6", "12", "2", "1", "140", "BENIGN"),
        ("10.0.0.3-10.0.0.4", "10.0.0.3", "10.0.0.4", "443", "6", "80", "9", "3", "900", "Web Attack - Brute Force"),
        ("10.0.0.3-10.0.0.4", "10.0.0.3", "10.0.0.4", "443", "6", "84", "10", "4", "920", "Web Attack - Brute Force"),
        ("10.0.0.3-10.0.0.4", "10.0.0.3", "10.0.0.4", "443", "6", "86", "11", "4", "950", "Web Attack - Brute Force"),
        ("10.0.0.5-10.0.0.6", "10.0.0.5", "10.0.0.6", "80", "6", "11", "2", "1", "130", "BENIGN"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Flow ID",
                "Source IP",
                "Destination IP",
                "Destination Port",
                "Protocol",
                "Flow Duration",
                "Total Fwd Packets",
                "Total Backward Packets",
                "Flow Bytes/s",
                "Label",
            ]
        )
        writer.writerows(rows)


def test_load_labeled_dataset_maps_cicids_labels_and_features(tmp_path):
    fixture = tmp_path / "cicids_fixture.csv"
    write_fixture(fixture)

    dataset = run_labeled_domain_proof.load_labeled_dataset(
        dataset="cicids_webattacks",
        file_path=fixture,
        label_column="Label",
        attack_labels=["Web Attack - Brute Force"],
        normalize_non_benign_as=None,
        sample_mode="natural",
        frames=6,
        max_rows=None,
        engine=FakeEngine(),
        features=8,
        seed=42,
        repo_root=tmp_path,
    )

    assert dataset.frames.shape == (6, 8)
    assert dataset.labels.tolist() == [0, 0, 1, 1, 1, 0]
    assert dataset.label_distribution["BENIGN"] == 3
    assert dataset.normalized_label_distribution["ATTACK"] == 3
    assert "Flow Duration" in dataset.feature_columns
    assert dataset.events[2]["attack"] is True
    assert dataset.events[2]["OriginalLabel"] == "Web Attack - Brute Force"
    assert dataset.events[2]["EidosProofLabel"] == "ATTACK"
    assert dataset.events[2]["src_ip"] == "10.0.0.3"


def test_run_writes_labeled_domain_artifacts_with_fake_engine(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    fixture = repo_root / "fixture.csv"
    write_fixture(fixture)
    engine_path = repo_root / "fake_engine.py"
    engine_path.write_text("# fake engine\n", encoding="utf-8")
    out_dir = repo_root / "artifacts" / "cicids_webattacks_proof_test"

    monkeypatch.setattr(
        run_labeled_domain_proof.proof_helpers,
        "collect_git_info",
        lambda _repo_root: {
            "branch": "test-branch",
            "commit": "abc123",
            "dirty": False,
            "errors": [],
            "status_short": "",
        },
    )
    def fake_write_environment(path, _repo_root):
        path.write_text("environment\n", encoding="utf-8")
        return {"pytest": "fake"}

    monkeypatch.setattr(run_labeled_domain_proof, "write_environment", fake_write_environment)

    args = run_labeled_domain_proof.parse_args(
        [
            "--dataset",
            "cicids_webattacks",
            "--file",
            str(fixture),
            "--label-column",
            "Label",
            "--attack-labels",
            "Web Attack - Brute Force",
            "--frames",
            "6",
            "--seed",
            "42",
            "--out",
            str(out_dir),
            "--suite",
            "smoke",
        ]
    )

    result = run_labeled_domain_proof.run(
        args,
        repo_root=repo_root,
        load_engine_fn=lambda _out_dir, _repo_root: (FakeEngine(), engine_path),
        mirror_to_drive_fn=lambda _out_dir, _run_id, _run_date: {
            "drive_copy_attempted": True,
            "drive_copy_success": False,
            "drive_root": "unknown",
            "drive_run_dir": "unknown",
            "files_considered": [],
            "files_copied": [],
            "files_skipped": [],
            "reason": "not configured in test",
            "timestamp_utc": "2026-05-31T00:00:00Z",
        },
    )

    assert result.exit_code == 0
    for name in (
        "config.json",
        "run_manifest.json",
        "labeled_metrics.json",
        "labeled_metrics.md",
        "benchmark_summary.csv",
        "benchmark_summary.md",
        "event_summary.json",
        "precision_ledger.json",
        "precision_ledger.md",
        "event_confirmation_report.json",
        "event_confirmation_report.md",
        "sentinel_calibration_v1.json",
        "sentinel_calibration_v1.md",
        "sentinel_calibration_report.json",
        "sentinel_calibration_report.md",
        "calibrated_precision_ledger.json",
        "calibrated_precision_ledger.md",
        "candidate_funnel_report.json",
        "candidate_funnel_report.md",
        "confirmation_profile_sweep.json",
        "confirmation_profile_sweep.csv",
        "confirmation_profile_sweep.md",
        "proof_digest.json",
        "proof_digest.md",
        "engine_reopen_gate.json",
        "engine_reopen_gate.md",
        "crash_scan.json",
        "environment.txt",
        "drive_manifest.json",
        "codex_journal.md",
        "plain_language_test_analysis.md",
    ):
        assert (out_dir / name).is_file()
    assert (out_dir / "incident_cards").is_dir()

    metrics = json.loads((out_dir / "labeled_metrics.json").read_text(encoding="utf-8"))
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    digest = json.loads((out_dir / "proof_digest.json").read_text(encoding="utf-8"))
    assert metrics["frames_processed"] == 6
    assert metrics["candidate_events"] >= 1
    assert metrics["confirmed_events"] >= 1
    assert metrics["proof_raw_event_count"] >= 1
    assert metrics["proof_merged_event_count"] >= 1
    assert metrics["proof_deduped_event_count"] >= 1
    assert metrics["proof_confirmed_event_count"] >= 1
    assert {"raw", "merged", "deduped", "confirmed", "calibrated"} <= set(metrics["event_view_metrics"])
    assert metrics["sentinel_calibration_mode"] == "off"
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["crash_hit_count"] == 0
    assert metrics["normalized_label_distribution"] == {"ATTACK": 3, "BENIGN": 3}
    assert config["core_behavior"]["sentinel_thresholds_changed"] is False
    assert digest["clean"] is True
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["precision_ledger_json"] == "precision_ledger.json"
    assert manifest["outputs"]["event_confirmation_report_json"] == "event_confirmation_report.json"
    assert manifest["outputs"]["sentinel_calibration_v1_json"] == "sentinel_calibration_v1.json"
    assert manifest["outputs"]["sentinel_calibration_report_json"] == "sentinel_calibration_report.json"
    assert manifest["outputs"]["calibrated_precision_ledger_json"] == "calibrated_precision_ledger.json"
    assert manifest["outputs"]["engine_reopen_gate_json"] == "engine_reopen_gate.json"
    assert manifest["outputs"]["candidate_funnel_report_json"] == "candidate_funnel_report.json"
    assert manifest["outputs"]["confirmation_profile_sweep_csv"] == "confirmation_profile_sweep.csv"
    assert manifest["event_confirmation"]["mode"] == "balanced"
    assert manifest["sentinel_calibration_v1"]["enabled"] is False
    assert "tracked_dirty" in manifest
    assert "untracked_generated_files" in manifest
    assert "untracked_non_generated_files" in manifest
    assert "git_dirty_reason" in manifest
    assert manifest["device"]["selected_device"] in {"cpu", "cuda"}
    ledger = json.loads((out_dir / "precision_ledger.json").read_text(encoding="utf-8"))
    assert {"raw", "merged", "deduped", "calibrated"} <= set(ledger["precision_lift_summary"])
    assert "calibrated_events" in ledger
    confirmation_report = json.loads((out_dir / "event_confirmation_report.json").read_text(encoding="utf-8"))
    assert confirmation_report["mode"] == "balanced"
    assert "precision_lift_summary" in confirmation_report
    drive_manifest = json.loads((out_dir / "drive_manifest.json").read_text(encoding="utf-8"))
    assert drive_manifest["drive_copy_attempted"] is True
    assert drive_manifest["drive_copy_success"] is False
    assert drive_manifest["reason"] == "not configured in test"
    calibration_report = json.loads((out_dir / "sentinel_calibration_v1.json").read_text(encoding="utf-8"))
    assert calibration_report["calibration_enabled"] is False
    assert calibration_report["core_behavior_boundary"]["sentinel_thresholds_changed"] is False
    sentinel_report = json.loads((out_dir / "sentinel_calibration_report.json").read_text(encoding="utf-8"))
    assert sentinel_report["mode"] == "off"
    assert {"raw", "merged", "deduped", "calibrated"} <= set(sentinel_report["raw_vs_calibrated"])
    gate = json.loads((out_dir / "engine_reopen_gate.json").read_text(encoding="utf-8"))
    assert gate["verdict"] in {
        "BLOCKED",
        "CALIBRATION_ONLY",
        "CALIBRATION_ONLY_NEEDS_TUNING",
        "READY_FOR_NARROW_CORE_EXPERIMENT",
    }
    assert {"pytest", "labeled_proof", "crash_scan", "raw_visibility", "core_untouched"} <= set(gate["checks"])
    funnel = json.loads((out_dir / "candidate_funnel_report.json").read_text(encoding="utf-8"))
    assert [stage["stage"] for stage in funnel["stages"]] == [
        "raw_candidates",
        "merged_events",
        "deduped_events",
        "confirmed_events",
        "calibrated_confirmed_events",
    ]
    sweep = json.loads((out_dir / "confirmation_profile_sweep.json").read_text(encoding="utf-8"))
    assert sweep["profiles"][0]["profile"] == "balanced"


def test_calibration_config_serialization_is_conservative():
    args = run_labeled_domain_proof.parse_args(
        [
            "--dataset",
            "cicids_webattacks",
            "--file",
            "tests/fixtures/cicids_webattacks_tiny.csv",
            "--label-column",
            "Label",
            "--frames",
            "10",
            "--seed",
            "42",
            "--suite",
            "smoke",
            "--confirmation-mode",
            "balanced",
            "--calibration-enabled",
        ]
    )

    config = run_labeled_domain_proof.build_calibration_config(args)
    payload = sentinel_calibration_v1.config_to_dict(config)

    assert payload["calibration_enabled"] is True
    assert payload["calibration_version"] == "sentinel_calibration_v1"
    assert payload["confirmation_mode_baseline"] == "balanced"
    assert payload["suppress_duplicate_noise"] is True
    assert payload["suppress_fully_benign_pressure"] is True
    assert payload["core_behavior_boundary"]["reservoir_dynamics_changed"] is False
    assert len(sentinel_calibration_v1.config_hash(config)) == 64


def test_sentinel_calibration_mode_alias_controls_calibration_defaults():
    off_args = run_labeled_domain_proof.parse_args(
        [
            "--dataset",
            "cicids_webattacks",
            "--file",
            "tests/fixtures/cicids_webattacks_tiny.csv",
            "--label-column",
            "Label",
            "--frames",
            "10",
            "--seed",
            "42",
            "--suite",
            "smoke",
            "--sentinel-calibration-mode",
            "off",
        ]
    )
    high_recall_args = run_labeled_domain_proof.parse_args(
        [
            "--dataset",
            "cicids_webattacks",
            "--file",
            "tests/fixtures/cicids_webattacks_tiny.csv",
            "--label-column",
            "Label",
            "--frames",
            "10",
            "--seed",
            "42",
            "--suite",
            "smoke",
            "--sentinel-calibration-mode",
            "high_recall",
        ]
    )

    assert off_args.calibration_enabled is False
    assert off_args.confirmation_mode == "off"
    assert high_recall_args.calibration_enabled is True
    assert high_recall_args.confirmation_mode == "high_recall"


def test_calibration_suppression_reason_accounting_and_raw_visibility():
    confirmed = [
        {
            "event_id": "benign_fp",
            "start_frame": 5,
            "end_frame": 6,
            "component_count": 1,
            "score_detail": {"raw_hit_count": 2, "false_positive_classification": "fully_benign"},
        },
        {
            "event_id": "attack_hit",
            "start_frame": 20,
            "end_frame": 22,
            "component_count": 2,
            "score_detail": {"raw_hit_count": 2, "false_positive_classification": "overlap_boundary"},
        },
    ]
    raw_labels = ["BENIGN"] * 20 + ["Web Attack - Brute Force"] * 10
    proof_labels = ["BENIGN"] * 20 + ["ATTACK"] * 10
    config = sentinel_calibration_v1.SentinelCalibrationConfig(calibration_enabled=True)

    report = sentinel_calibration_v1.apply_calibration(
        confirmed_events=confirmed,
        raw_event_count=4,
        merged_event_count=2,
        deduped_event_count=2,
        attack_windows=[{"start_frame": 20, "end_frame": 29}],
        raw_labels=raw_labels,
        proof_labels=proof_labels,
        frames_processed=30,
        config=config,
        sample_mode="transition",
        crash_hit_count=0,
    )

    assert report["guardrails"]["passed"] is True
    assert report["counts"]["raw_engine_card_events"] == 4
    assert report["counts"]["pre_calibration_confirmed_events"] == 2
    assert report["counts"]["post_calibration_confirmed_events"] == 1
    assert report["suppressed_events"][0]["reason_code"] == "fully_benign_context"
    assert report["suppressed_events"][0]["suppression_would_affect_attack_window_coverage"] is False
    assert report["suppressed_events"][0]["suppression_could_affect_recall"] is False
    assert report["suppressed_events"][0]["labels_around_event"]
    assert "raw_evidence_reference" in report["suppressed_events"][0]
    assert report["before_metrics"]["false_positives"] == 1
    assert report["after_metrics"]["false_positives"] == 0


def test_calibration_attack_window_guard_cannot_suppress_first_in_window_detection():
    confirmed = [
        {
            "event_id": "first_attack_hit",
            "start_frame": 20,
            "end_frame": 20,
            "component_count": 1,
            "score_detail": {"raw_hit_count": 1, "false_positive_classification": "overlap_boundary"},
        }
    ]
    raw_labels = ["BENIGN"] * 20 + ["Web Attack - XSS"] * 10
    proof_labels = ["BENIGN"] * 20 + ["ATTACK"] * 10
    config = sentinel_calibration_v1.SentinelCalibrationConfig(calibration_enabled=True)

    report = sentinel_calibration_v1.apply_calibration(
        confirmed_events=confirmed,
        raw_event_count=1,
        merged_event_count=1,
        deduped_event_count=1,
        attack_windows=[{"start_frame": 20, "end_frame": 29}],
        raw_labels=raw_labels,
        proof_labels=proof_labels,
        frames_processed=30,
        config=config,
        sample_mode="transition",
        crash_hit_count=0,
    )

    assert report["suppressed_events"] == []
    assert report["after_metrics"]["recall"] == 1.0
    assert report["attack_window_summary_after"]["attack_window_coverage_pct"] == 100.0
    assert report["attack_window_summary_after"]["first_detection_latency_frames"] == 0


def test_calibration_benign_only_pressure_reason():
    config = sentinel_calibration_v1.SentinelCalibrationConfig(calibration_enabled=True)

    report = sentinel_calibration_v1.apply_calibration(
        confirmed_events=[{"event_id": "natural_fp", "start_frame": 5, "end_frame": 5}],
        raw_event_count=1,
        merged_event_count=1,
        deduped_event_count=1,
        attack_windows=[],
        raw_labels=["BENIGN"] * 10,
        proof_labels=["BENIGN"] * 10,
        frames_processed=10,
        config=config,
        sample_mode="natural",
        crash_hit_count=0,
    )

    assert report["suppressed_events"][0]["reason_code"] == "benign_only_pressure"
    assert report["after_metrics"]["false_positives"] == 0


def test_attack_label_parser_accepts_single_comma_and_repeated_values():
    parsed = run_labeled_domain_proof.parse_attack_labels(
        [
            ["ATTACK"],
            ["Web Attack - Brute Force, Web Attack - XSS"],
            ["Web Attack - Sql Injection"],
        ]
    )

    assert parsed == [
        "ATTACK",
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - Sql Injection",
    ]


def test_raw_cicids_replacement_character_label_matches_configured_attack_label():
    raw_label = "Web Attack \ufffd Brute Force"

    assert (
        run_labeled_domain_proof.normalize_proof_label(raw_label, ["Web Attack - Brute Force"])
        == "ATTACK"
    )
    assert (
        run_labeled_domain_proof.normalize_proof_label("Web Attack \ufffd XSS", ["Web Attack - XSS"])
        == "ATTACK"
    )
    assert (
        run_labeled_domain_proof.normalize_proof_label(
            "Web Attack \ufffd Sql Injection",
            ["Web Attack - Sql Injection"],
        )
        == "ATTACK"
    )


def test_normalize_non_benign_as_attack_maps_unconfigured_labels():
    assert (
        run_labeled_domain_proof.normalize_proof_label(
            "DoS Hulk",
            [],
            normalize_non_benign_as="ATTACK",
        )
        == "ATTACK"
    )
    assert (
        run_labeled_domain_proof.normalize_proof_label(
            "BENIGN",
            [],
            normalize_non_benign_as="ATTACK",
        )
        == "BENIGN"
    )


def test_balanced_sample_creation_shuffles_seeded_binary_sample(tmp_path):
    fixture = tmp_path / "cicids_fixture.csv"
    write_fixture(fixture)

    dataset = run_labeled_domain_proof.load_labeled_dataset(
        dataset="cicids_webattacks",
        file_path=fixture,
        label_column="Label",
        attack_labels=["Web Attack - Brute Force"],
        normalize_non_benign_as=None,
        sample_mode="balanced",
        frames=4,
        max_rows=None,
        engine=FakeEngine(),
        features=4,
        seed=7,
        repo_root=tmp_path,
    )

    assert dataset.sample_receipt["mode"] == "balanced"
    assert dataset.sample_receipt["selected_row_counts"] == {"BENIGN": 2, "ATTACK": 2, "total": 4}
    assert dataset.sample_receipt["order_preserved"] is False
    assert sorted(dataset.proof_labels) == ["ATTACK", "ATTACK", "BENIGN", "BENIGN"]


def test_transition_sample_creation_preserves_benign_then_attack_order(tmp_path):
    fixture = tmp_path / "cicids_fixture.csv"
    write_fixture(fixture)

    dataset = run_labeled_domain_proof.load_labeled_dataset(
        dataset="cicids_webattacks",
        file_path=fixture,
        label_column="Label",
        attack_labels=["Web Attack - Brute Force"],
        normalize_non_benign_as=None,
        sample_mode="transition",
        frames=4,
        max_rows=None,
        engine=FakeEngine(),
        features=4,
        seed=7,
        repo_root=tmp_path,
    )

    assert dataset.proof_labels == ["BENIGN", "BENIGN", "ATTACK", "ATTACK"]
    assert dataset.sample_receipt["order_preserved"] is True
    assert dataset.sample_receipt["first_attack_row"]["sample_frame"] == 2
    assert dataset.sample_receipt["transition_boundary"] == {"benign_end_frame": 1, "attack_start_frame": 2}


def test_natural_sample_creation_replays_original_order(tmp_path):
    fixture = tmp_path / "cicids_fixture.csv"
    write_fixture(fixture)

    dataset = run_labeled_domain_proof.load_labeled_dataset(
        dataset="cicids_webattacks",
        file_path=fixture,
        label_column="Label",
        attack_labels=["Web Attack - Brute Force"],
        normalize_non_benign_as=None,
        sample_mode="natural",
        frames=4,
        max_rows=None,
        engine=FakeEngine(),
        features=4,
        seed=7,
        repo_root=tmp_path,
    )

    assert dataset.proof_labels == ["BENIGN", "BENIGN", "ATTACK", "ATTACK"]
    assert dataset.source_row_indices == [0, 1, 2, 3]
    assert dataset.sample_receipt["order_preserved"] is True


def test_balanced_blocks_preserves_order_inside_blocks(tmp_path):
    fixture = tmp_path / "cicids_fixture.csv"
    write_fixture(fixture)

    dataset = run_labeled_domain_proof.load_labeled_dataset(
        dataset="cicids_webattacks",
        file_path=fixture,
        label_column="Label",
        attack_labels=["Web Attack - Brute Force"],
        normalize_non_benign_as=None,
        sample_mode="balanced_blocks",
        frames=4,
        max_rows=None,
        engine=FakeEngine(),
        features=4,
        seed=7,
        repo_root=tmp_path,
    )

    receipt = dataset.sample_receipt
    assert receipt["mode"] == "balanced_blocks"
    assert receipt["block_count"] == 2
    assert receipt["benign_block_count"] == 1
    assert receipt["attack_block_count"] == 1
    for block in receipt["blocks"]:
        block_indices = [
            event["source_row_index"]
            for event in dataset.events[block["sample_start_frame"] : block["sample_end_frame"] + 1]
        ]
        assert block_indices == sorted(block_indices)
    assert sorted(dataset.proof_labels) == ["ATTACK", "ATTACK", "BENIGN", "BENIGN"]


def test_natural_attack_windows_selects_context_and_preserves_source_order(tmp_path):
    fixture = tmp_path / "cicids_fixture.csv"
    write_fixture(fixture)

    dataset = run_labeled_domain_proof.load_labeled_dataset(
        dataset="cicids_webattacks",
        file_path=fixture,
        label_column="Label",
        attack_labels=["Web Attack - Brute Force"],
        normalize_non_benign_as=None,
        sample_mode="natural_attack_windows",
        frames=5,
        natural_window_pre=1,
        natural_window_post=1,
        natural_window_max_windows=1,
        max_rows=None,
        engine=FakeEngine(),
        features=4,
        seed=7,
        repo_root=tmp_path,
    )

    assert dataset.source_row_indices == [1, 2, 3, 4, 5]
    assert dataset.sample_receipt["mode"] == "natural_attack_windows"
    assert dataset.sample_receipt["attack_window_slice_count"] == 1
    assert dataset.sample_receipt["window_slices"][0]["pre_context_rows"] == 1
    assert dataset.sample_receipt["window_slices"][0]["post_context_rows"] == 1
    assert dataset.sample_receipt["order_preserved"] is True


def test_event_merging_and_duplicate_collapse():
    raw_events = [
        {"event_id": "a", "start_frame": 10, "end_frame": 12, "source": "engine_card", "severity": "AMBER"},
        {"event_id": "b", "start_frame": 20, "end_frame": 22, "source": "sentinel_confirmed", "severity": "RED"},
        {"event_id": "c", "start_frame": 80, "end_frame": 81, "source": "engine_card", "severity": "AMBER"},
    ]

    merged = run_labeled_domain_proof.merge_detection_events(raw_events, merge_gap=10)
    deduped = run_labeled_domain_proof.dedupe_detection_events(merged)

    assert len(merged) == 2
    assert merged[0]["start_frame"] == 10
    assert merged[0]["end_frame"] == 22
    assert merged[0]["component_count"] == 2
    assert len(deduped) == 2
    assert deduped[0]["source"] == "proof_merged"


def test_event_confirmation_mode_off_preserves_raw_decision_view():
    raw_events = [
        {"event_id": "raw_1", "start_frame": 5, "end_frame": 5, "source": "engine_card", "severity": "AMBER"},
        {"event_id": "raw_2", "start_frame": 20, "end_frame": 22, "source": "sentinel_confirmed", "severity": "RED"},
    ]
    merged = run_labeled_domain_proof.merge_detection_events(raw_events, merge_gap=10)
    deduped = run_labeled_domain_proof.dedupe_detection_events(merged)

    report = event_confirmation.apply_confirmation(
        raw_events=raw_events,
        merged_events=merged,
        deduped_events=deduped,
        label_windows=[{"start_frame": 20, "end_frame": 30}],
        raw_labels=["BENIGN"] * 20 + ["Web Attack - Brute Force"] * 11,
        proof_labels=["BENIGN"] * 20 + ["ATTACK"] * 11,
        frames_processed=31,
        mode="off",
    )

    assert report["confirmed_event_count"] == len(raw_events)
    assert report["candidate_event_count"] == len(raw_events)
    assert {decision["reason_codes"][0] for decision in report["decisions"]} == {"confirmed_mode_off"}


def test_duplicate_raw_events_collapse_into_one_confirmed_event():
    raw_events = [
        {"event_id": "a", "start_frame": 100, "end_frame": 100, "source": "engine_card", "severity": "RED"},
        {"event_id": "b", "start_frame": 101, "end_frame": 101, "source": "engine_card", "severity": "RED"},
        {"event_id": "c", "start_frame": 102, "end_frame": 102, "source": "sentinel_confirmed", "severity": "RED"},
    ]
    merged = run_labeled_domain_proof.merge_detection_events(raw_events, merge_gap=5)
    deduped = run_labeled_domain_proof.dedupe_detection_events(merged)

    report = event_confirmation.apply_confirmation(
        raw_events=raw_events,
        merged_events=merged,
        deduped_events=deduped,
        label_windows=[{"start_frame": 100, "end_frame": 110}],
        raw_labels=["BENIGN"] * 100 + ["Web Attack - XSS"] * 11,
        proof_labels=["BENIGN"] * 100 + ["ATTACK"] * 11,
        frames_processed=111,
        mode="balanced",
    )

    assert report["candidate_event_count"] == 1
    assert report["confirmed_event_count"] == 1
    assert len(report["confirmed_events"][0]["source_event_refs"]) == 3
    assert "confirmed_attack_overlap" in report["confirmed_events"][0]["reason_codes"]


def test_single_benign_spike_is_suppressed_in_low_noise_mode():
    raw_events = [
        {"event_id": "benign_spike", "start_frame": 7, "end_frame": 7, "source": "engine_card", "severity": "AMBER"},
    ]
    merged = run_labeled_domain_proof.merge_detection_events(raw_events, merge_gap=0)
    deduped = run_labeled_domain_proof.dedupe_detection_events(merged)

    report = event_confirmation.apply_confirmation(
        raw_events=raw_events,
        merged_events=merged,
        deduped_events=deduped,
        label_windows=[],
        raw_labels=["BENIGN"] * 20,
        proof_labels=["BENIGN"] * 20,
        frames_processed=20,
        mode="low_noise",
    )

    assert report["confirmed_event_count"] == 0
    assert report["suppressed_event_count"] == 1
    reasons = report["suppressed_events"][0]["reason_codes"]
    assert "suppressed_single_frame_spike" in reasons
    assert "suppressed_low_evidence_fully_benign" in reasons


def test_attack_overlapping_event_remains_confirmed_in_high_recall_mode():
    raw_events = [
        {"event_id": "attack_hit", "start_frame": 50, "end_frame": 50, "source": "engine_card", "severity": "AMBER"},
    ]
    merged = run_labeled_domain_proof.merge_detection_events(raw_events, merge_gap=0)
    deduped = run_labeled_domain_proof.dedupe_detection_events(merged)

    report = event_confirmation.apply_confirmation(
        raw_events=raw_events,
        merged_events=merged,
        deduped_events=deduped,
        label_windows=[{"start_frame": 45, "end_frame": 55}],
        raw_labels=["BENIGN"] * 45 + ["Web Attack - Sql Injection"] * 11,
        proof_labels=["BENIGN"] * 45 + ["ATTACK"] * 11,
        frames_processed=56,
        mode="high_recall",
    )

    assert report["confirmed_event_count"] == 1
    assert "confirmed_attack_overlap" in report["confirmed_events"][0]["reason_codes"]


def test_recall_guarded_confirms_strong_short_proof_event_without_labels():
    raw_events = [
        {"event_id": "strong_short", "start_frame": 20, "end_frame": 20, "source": "engine_card", "severity": "RED"},
    ]
    merged = run_labeled_domain_proof.merge_detection_events(raw_events, merge_gap=0)
    deduped = run_labeled_domain_proof.dedupe_detection_events(merged)

    balanced = event_confirmation.apply_confirmation(
        raw_events=raw_events,
        merged_events=merged,
        deduped_events=deduped,
        label_windows=[],
        raw_labels=["BENIGN"] * 40,
        proof_labels=["BENIGN"] * 40,
        frames_processed=40,
        mode="balanced",
    )
    guarded = event_confirmation.apply_confirmation(
        raw_events=raw_events,
        merged_events=merged,
        deduped_events=deduped,
        label_windows=[],
        raw_labels=["BENIGN"] * 40,
        proof_labels=["BENIGN"] * 40,
        frames_processed=40,
        mode="recall_guarded",
    )

    assert balanced["confirmed_event_count"] == 0
    assert guarded["confirmed_event_count"] == 1
    assert guarded["confirmed_events"][0]["confirmation_mode"] == "recall_guarded"


def test_balanced_mode_reports_precision_lift_summary():
    raw_events = [
        {"event_id": "fp", "start_frame": 5, "end_frame": 5, "source": "engine_card", "severity": "AMBER"},
        {"event_id": "tp", "start_frame": 50, "end_frame": 52, "source": "sentinel_confirmed", "severity": "RED"},
    ]
    label_windows = [{"start_frame": 50, "end_frame": 60}]
    merged = run_labeled_domain_proof.merge_detection_events(raw_events, merge_gap=3)
    deduped = run_labeled_domain_proof.dedupe_detection_events(merged)
    report = event_confirmation.apply_confirmation(
        raw_events=raw_events,
        merged_events=merged,
        deduped_events=deduped,
        label_windows=label_windows,
        raw_labels=["BENIGN"] * 50 + ["Web Attack - Brute Force"] * 11,
        proof_labels=["BENIGN"] * 50 + ["ATTACK"] * 11,
        frames_processed=61,
        mode="balanced",
    )
    confirmed = report["confirmed_events"]
    view_metrics = {
        "raw": run_labeled_domain_proof.metrics_for_event_view(raw_events, label_windows, 61),
        "merged": run_labeled_domain_proof.metrics_for_event_view(merged, label_windows, 61),
        "deduped": run_labeled_domain_proof.metrics_for_event_view(deduped, label_windows, 61),
        "confirmed": run_labeled_domain_proof.metrics_for_event_view(confirmed, label_windows, 61),
    }

    enriched = event_confirmation.add_metric_summary(report, view_metrics)

    assert "precision_lift_summary" in enriched
    assert enriched["precision_lift_summary"]["raw_to_confirmed_false_positive_reduction_count"] >= 1


def test_missing_optional_confirmation_fields_do_not_crash():
    raw_events = [
        {"event_id": "minimal", "start_frame": 2, "end_frame": 4, "source": "engine_card"},
    ]
    merged = run_labeled_domain_proof.merge_detection_events(raw_events, merge_gap=0)
    deduped = run_labeled_domain_proof.dedupe_detection_events(merged)

    report = event_confirmation.apply_confirmation(
        raw_events=raw_events,
        merged_events=merged,
        deduped_events=deduped,
        label_windows=[],
        raw_labels=["BENIGN"] * 10,
        proof_labels=["BENIGN"] * 10,
        frames_processed=10,
        mode="balanced",
    )

    assert report["decisions"]
    assert "peak_z" in report["decisions"][0]["missing_evidence_fields"]
    assert "top_driver_consistency" in report["decisions"][0]["missing_evidence_fields"]


def test_event_confirmation_report_markdown_is_written(tmp_path):
    report = {
        "mode": "balanced",
        "thresholds": event_confirmation.get_thresholds("balanced").__dict__,
        "raw_event_count": 1,
        "merged_event_count": 1,
        "deduped_event_count": 1,
        "candidate_event_count": 1,
        "confirmed_event_count": 0,
        "suppressed_event_count": 1,
        "event_view_metrics": {
            "raw": {"event_count": 1},
            "merged": {"event_count": 1},
            "deduped": {"event_count": 1},
            "confirmed": {"event_count": 0},
        },
        "precision_lift_summary": {"false_positive_reduction_count": 1},
        "decisions": [{"reason_codes": ["suppressed_single_frame_spike"]}],
        "examples": {"confirmed_events": [], "suppressed_events": [{"candidate_id": "c", "start_frame": 1, "end_frame": 1, "confirmation_score": 0.5, "reason_codes": ["suppressed_single_frame_spike"]}]},
    }

    path = tmp_path / "event_confirmation_report.md"
    event_confirmation.write_report_md(path, report)

    assert path.is_file()
    assert "Event Confirmation Report" in path.read_text(encoding="utf-8")


def test_false_positive_classification_uses_attack_window_distance():
    windows = [{"start_frame": 100, "end_frame": 199}]

    pre = {"start_frame": 90, "end_frame": 95, "component_count": 1}
    post = {"start_frame": 205, "end_frame": 210, "component_count": 1}
    benign = {"start_frame": 20, "end_frame": 25, "component_count": 1}
    duplicate = {"start_frame": 20, "end_frame": 25, "component_count": 3}

    assert run_labeled_domain_proof.classify_false_positive(pre, windows, 10) == "pre_attack_near_transition"
    assert run_labeled_domain_proof.classify_false_positive(post, windows, 10) == "post_attack_near_transition"
    assert run_labeled_domain_proof.classify_false_positive(benign, windows, 10) == "fully_benign"
    assert run_labeled_domain_proof.classify_false_positive(duplicate, windows, 10) == "likely_duplicate_noise"


def test_attack_window_detection_latency_and_coverage():
    windows = [{"start_frame": 10, "end_frame": 19, "label_distribution": {"ATTACK": 10}}]
    raw_events = [{"event_id": "hit", "start_frame": 12, "end_frame": 14, "source": "sentinel_confirmed"}]

    diagnostics = run_labeled_domain_proof.attack_window_diagnostics(windows, raw_events)

    assert diagnostics[0]["first_detection_frame"] == 12
    assert diagnostics[0]["detection_latency"] == 2
    assert diagnostics[0]["detections_inside_window"] == 1
    assert diagnostics[0]["coverage_percentage"] == 30.0
    assert diagnostics[0]["missed"] is False


def test_cpu_only_device_receipt_does_not_require_cuda():
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    class FakeVersion:
        cuda = None

    class FakeTorch:
        __version__ = "0.test"
        cuda = FakeCuda()
        version = FakeVersion()

    receipt = run_labeled_domain_proof.collect_device_receipt(
        runtime_seconds=2.0,
        frames_processed=10,
        torch_module=FakeTorch(),
    )

    assert receipt["torch_installed"] is True
    assert receipt["cuda_available"] is False
    assert receipt["selected_device"] == "cpu"
    assert receipt["cpu_gpu_mode"] == "cpu"
    assert receipt["cpu_fallback_used"] is True
    assert receipt["frames_per_second"] == 5.0


def test_git_hygiene_receipt_classifies_generated_and_source_dirty_paths(tmp_path):
    out_dir = tmp_path / "artifacts" / "cicids_webattacks_proof_test"
    git_info = {
        "status_short": "\n".join(
            [
                " M tools/run_labeled_domain_proof.py",
                "?? artifacts/cicids_webattacks_proof_test/sample.csv",
                "?? notes.txt",
            ]
        )
    }

    receipt = run_labeled_domain_proof.git_hygiene_receipt(git_info, out_dir, tmp_path)

    assert receipt["tracked_dirty"] == ["tools/run_labeled_domain_proof.py"]
    assert receipt["untracked_generated_files"] == ["artifacts/cicids_webattacks_proof_test/sample.csv"]
    assert receipt["untracked_non_generated_files"] == ["notes.txt"]
    assert "tracked dirty" in receipt["git_dirty_reason"]
