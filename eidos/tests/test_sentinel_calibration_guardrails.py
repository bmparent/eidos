import csv
import json
from pathlib import Path

from tools import build_sentinel_calibration_guardrails as guardrails


def write_labeled_fixture(path: Path, benign: int = 4, attack: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Flow ID",
                "Source IP",
                "Destination IP",
                "Destination Port",
                "Protocol",
                "Flow Duration",
                "Total Fwd Packets",
                "Label",
            ],
        )
        writer.writeheader()
        for idx in range(benign):
            writer.writerow(
                {
                    "Flow ID": f"b-{idx}",
                    "Source IP": "10.0.0.1",
                    "Destination IP": "10.0.0.2",
                    "Destination Port": "80",
                    "Protocol": "6",
                    "Flow Duration": str(10 + idx),
                    "Total Fwd Packets": "2",
                    "Label": "BENIGN",
                }
            )
        for idx in range(attack):
            writer.writerow(
                {
                    "Flow ID": f"a-{idx}",
                    "Source IP": "10.0.0.3",
                    "Destination IP": "10.0.0.4",
                    "Destination Port": "443",
                    "Protocol": "6",
                    "Flow Duration": str(80 + idx),
                    "Total Fwd Packets": "9",
                    "Label": "Web Attack - Brute Force",
                }
            )


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_leg_plan_records_large_skips_when_only_tiny_fixture_exists(tmp_path):
    dataset = tmp_path / "tiny.csv"
    normal = tmp_path / "normal.csv"
    write_labeled_fixture(dataset, benign=8, attack=4)
    inventory = guardrails.inspect_labeled_csv(dataset, repo_root=tmp_path)
    normal_inventory = guardrails.write_normal_only_fixture(
        source=dataset,
        target=normal,
        label_column="Label",
        repo_root=tmp_path,
    )

    plans = guardrails.build_leg_plan(
        dataset_file=dataset,
        normal_only_file=normal,
        inventory=inventory,
        normal_inventory=normal_inventory,
        cuda={"cuda_available": False},
    )
    by_name = {plan.name: plan for plan in plans}

    assert by_name["tiny_fixture_smoke"].should_run is True
    assert by_name["natural_attack_replay_cpu"].should_run is True
    assert by_name["normal_only_negative_control"].should_run is True
    assert "requires at least 125 benign" in by_name["balanced_250_cpu"].skip_reason
    assert "requires at least 500 benign" in by_name["transition_1k_cpu"].skip_reason
    assert "CUDA unavailable" in by_name["gpu_10k_optional"].skip_reason


def test_dataset_availability_receipt_records_fixture_limits(tmp_path):
    dataset = tmp_path / "cicids_webattacks_tiny.csv"
    normal = tmp_path / "normal.csv"
    out_dir = tmp_path / "artifacts" / "guardrails"
    write_labeled_fixture(dataset, benign=8, attack=4)
    inventory = guardrails.inspect_labeled_csv(dataset, repo_root=tmp_path)
    normal_inventory = guardrails.write_normal_only_fixture(
        source=dataset,
        target=normal,
        label_column="Label",
        repo_root=tmp_path,
    )

    receipt = guardrails.write_dataset_availability_receipt(
        out_dir=out_dir,
        dataset_file=dataset,
        inventory=inventory,
        normal_inventory=normal_inventory,
        label_column="Label",
        attack_labels=("Web Attack - Brute Force",),
        repo_root=tmp_path,
    )

    assert receipt["row_count"] == 12
    assert receipt["available_benign_count"] == 8
    assert receipt["available_attack_count"] == 4
    assert receipt["balanced_250_feasible"] is False
    assert receipt["transition_1k_feasible"] is False
    assert receipt["natural_replay_feasible"] is True
    assert receipt["normal_only_negative_control_feasible"] is True
    assert receipt["final_dataset_verdict"] == "TINY_FIXTURE_ONLY"
    assert (out_dir / "dataset_availability_receipt.json").exists()


def test_allowlist_drive_copy_skips_engine_artifacts(tmp_path):
    out_dir = tmp_path / "artifacts" / "sentinel_calibration_guardrails_2026_06_30"
    run_dir = out_dir / "runs" / "tiny_fixture_smoke" / "off"
    drive_root = tmp_path / "drive"
    for rel in (
        "calibration_guardrail_matrix.json",
        "calibration_guardrail_matrix.md",
        "profile_comparison.csv",
        "attack_window_guardrails.json",
        "false_positive_guardrails.json",
        "dataset_availability_receipt.json",
        "core_touch_policy.json",
        "proof_logic_meaning.md",
        "codex_journal.md",
        "plain_language_test_analysis.md",
    ):
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / rel).write_text("{}", encoding="utf-8")
    for rel in guardrails.PER_RUN_DRIVE_ALLOWLIST:
        (run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / rel).write_text("{}", encoding="utf-8")
    engine_artifact = run_dir / "engine_artifacts" / "reservoir_checkpoint.pt"
    engine_artifact.parent.mkdir(parents=True, exist_ok=True)
    engine_artifact.write_text("large", encoding="utf-8")

    manifest = guardrails.mirror_guardrail_allowlist_to_drive(
        out_dir=out_dir,
        run_date="2026-06-30",
        drive_root=drive_root,
    )

    drive_run_dir = drive_root / "Eidos_Brain_Proof_Phase" / "2026-06-30" / out_dir.name
    assert manifest["copy_status"] == "copied"
    assert (drive_run_dir / "calibration_guardrail_matrix.json").exists()
    assert (drive_run_dir / "runs" / "tiny_fixture_smoke" / "off" / "run_manifest.json").exists()
    assert not (drive_run_dir / "runs" / "tiny_fixture_smoke" / "off" / "engine_artifacts" / "reservoir_checkpoint.pt").exists()
    assert any(item["path"] == "runs/*/*/engine_artifacts/**" for item in manifest["skipped_intentionally"])


def test_summarize_run_holds_when_calibration_collapses_attack_visibility(tmp_path):
    run_dir = tmp_path / "run"
    view = {
        "event_count": 1,
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "false_positives_per_10k_frames": 0.0,
    }
    write_json(
        run_dir / "labeled_metrics.json",
        {
            "sample_mode": "transition",
            "frames_processed": 8,
            "event_view_metrics": {
                "raw": view,
                "merged": view,
                "deduped": view,
                "confirmed": view,
                "calibrated": {**view, "event_count": 0, "true_positives": 0, "recall": 0.0},
            },
            "pre_calibration_confirmed_event_metrics": view,
            "calibrated_event_metrics": {**view, "event_count": 0, "true_positives": 0, "recall": 0.0},
            "recall": 1.0,
            "incident_card_count": 1,
        },
    )
    write_json(
        run_dir / "sentinel_calibration_v1.json",
        {
            "attack_window_summary_before": {
                "attack_window_count": 1,
                "detected_attack_windows": 1,
                "attack_window_coverage_pct": 100.0,
                "missed_attack_windows": 0,
            },
            "attack_window_summary_after": {
                "attack_window_count": 1,
                "detected_attack_windows": 0,
                "attack_window_coverage_pct": 0.0,
                "missed_attack_windows": 1,
            },
            "suppressed_reason_counts": {"fully_benign_context": 1},
        },
    )
    write_json(run_dir / "precision_ledger.json", {"false_positive_events": []})
    write_json(run_dir / "crash_scan.json", {"crash_hit_count": 0})
    write_json(run_dir / "drive_manifest.json", {"drive_copy_success": True})
    leg = guardrails.LegPlan(
        name="tiny_fixture_smoke",
        requested_leg="tiny fixture smoke",
        sample_mode="transition",
        frames=8,
        dataset_file=Path("fixture.csv"),
    )

    row, attack_row, fp_row = guardrails.summarize_run(
        leg=leg,
        profile="balanced",
        run_dir=run_dir,
        command_result={"returncode": 0},
        repo_root=tmp_path,
    )

    assert row["raw_visibility_intact"] is True
    assert row["attack_visibility_collapsed"] is True
    assert row["verdict"] == "HOLD"
    assert attack_row["verdict"] == "HOLD"
    assert fp_row["suppressed_reason_counts"] == {"fully_benign_context": 1}


def test_summarize_run_fails_when_raw_view_is_hidden(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "labeled_metrics.json",
        {
            "sample_mode": "natural",
            "frames_processed": 8,
            "event_view_metrics": {"calibrated": {"event_count": 0}},
        },
    )
    write_json(run_dir / "sentinel_calibration_v1.json", {})
    write_json(run_dir / "precision_ledger.json", {})
    write_json(run_dir / "crash_scan.json", {"crash_hit_count": 0})
    write_json(run_dir / "drive_manifest.json", {"drive_copy_success": True})
    leg = guardrails.LegPlan(
        name="normal_only_negative_control",
        requested_leg="normal-only negative control",
        sample_mode="natural",
        frames=8,
        dataset_file=Path("fixture.csv"),
    )

    row, _, _ = guardrails.summarize_run(
        leg=leg,
        profile="low_noise",
        run_dir=run_dir,
        command_result={"returncode": 0},
        repo_root=tmp_path,
    )

    assert row["raw_visibility_intact"] is False
    assert row["verdict"] == "FAIL"
