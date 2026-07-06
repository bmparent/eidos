import csv
from pathlib import Path

from tools import build_sentinel_guardrail_scale_matrix as scale


def write_webattack_csv(path: Path, *, benign: int = 600, brute: int = 500, xss: int = 0, sql: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [" Destination Port", " Flow Duration", " Total Fwd Packets", " Label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(benign):
            writer.writerow(
                {
                    " Destination Port": "80",
                    " Flow Duration": str(100 + idx),
                    " Total Fwd Packets": "2",
                    " Label": "BENIGN",
                }
            )
        for idx in range(brute):
            writer.writerow(
                {
                    " Destination Port": "443",
                    " Flow Duration": str(1000 + idx),
                    " Total Fwd Packets": "8",
                    " Label": "Web Attack - Brute Force",
                }
            )
        for idx in range(xss):
            writer.writerow(
                {
                    " Destination Port": "443",
                    " Flow Duration": str(2000 + idx),
                    " Total Fwd Packets": "9",
                    " Label": "Web Attack - XSS",
                }
            )
        for idx in range(sql):
            writer.writerow(
                {
                    " Destination Port": "443",
                    " Flow Duration": str(3000 + idx),
                    " Total Fwd Packets": "10",
                    " Label": "Web Attack - Sql Injection",
                }
            )


def test_inspect_csv_dataset_resolves_cicids_leading_label_header(tmp_path):
    dataset = tmp_path / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    write_webattack_csv(dataset, benign=600, brute=500, xss=10, sql=5)

    candidate = scale.inspect_csv_dataset(dataset, repo_root=tmp_path)

    assert candidate.exists is True
    assert candidate.label_column == " Label"
    assert candidate.row_count == 1115
    assert candidate.benign_count == 600
    assert candidate.attack_count == 515
    assert candidate.usable_for_balanced_250 is True
    assert candidate.usable_for_transition_1k is True
    assert candidate.first_attack_row == {
        "row_index_zero_based": 600,
        "csv_line_number": 602,
        "label": "Web Attack - Brute Force",
    }


def test_discovery_selects_larger_labeled_webattacks_csv(tmp_path):
    tiny = tmp_path / "tests" / "fixtures" / "cicids_webattacks_tiny.csv"
    write_webattack_csv(tiny, benign=8, brute=4)
    larger = tmp_path / "artifacts" / "cicids_webattacks_samples" / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    write_webattack_csv(larger, benign=700, brute=600)

    candidates = scale.discover_datasets(
        search_roots=[tmp_path / "tests" / "fixtures", tmp_path / "artifacts"],
        repo_root=tmp_path,
    )
    selected = scale.choose_larger_dataset(candidates)

    assert selected is not None
    assert selected.row_count == 1300
    assert selected.usable_for_transition_1k is True
    assert selected.path.endswith("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")


def test_normal_only_fixture_preserves_resolved_label_column(tmp_path):
    dataset = tmp_path / "data" / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    normal = tmp_path / "out" / "normal_only.csv"
    write_webattack_csv(dataset, benign=7, brute=3)

    candidate = scale.write_normal_only_fixture(
        source=dataset,
        target=normal,
        requested_label_column="Label",
        repo_root=tmp_path,
    )

    assert normal.exists()
    assert candidate.label_column == " Label"
    assert candidate.row_count == 7
    assert candidate.benign_count == 7
    assert candidate.attack_count == 0


def test_scale_leg_plan_marks_larger_cpu_legs_available_and_gpu_skip(tmp_path):
    tiny_path = tmp_path / "tiny.csv"
    large_path = tmp_path / "large.csv"
    normal_path = tmp_path / "normal.csv"
    write_webattack_csv(tiny_path, benign=8, brute=4)
    write_webattack_csv(large_path, benign=700, brute=600)
    write_webattack_csv(normal_path, benign=700, brute=0)
    tiny = scale.inspect_csv_dataset(tiny_path, repo_root=tmp_path)
    selected = scale.inspect_csv_dataset(large_path, repo_root=tmp_path)
    normal = scale.inspect_csv_dataset(normal_path, repo_root=tmp_path)

    plans = scale.build_scale_leg_plan(
        selected=selected,
        normal_candidate=normal,
        tiny_candidate=tiny,
        cuda={"cuda_available": False},
        natural_frames=13000,
        normal_frames=250,
        repo_root=tmp_path,
    )
    by_name = {plan.name: plan for plan in plans}

    assert by_name["tiny_fixture_smoke"].should_run is True
    assert by_name["balanced_250_cpu"].should_run is True
    assert by_name["transition_1k_cpu"].should_run is True
    assert by_name["natural_larger_replay_cpu"].should_run is True
    assert by_name["natural_larger_replay_cpu"].sample_mode == "natural_attack_windows"
    assert by_name["natural_larger_replay_cpu"].natural_window_pre == 250
    assert by_name["natural_larger_replay_cpu"].natural_window_post == 250
    assert by_name["normal_only_negative_control"].should_run is True
    assert by_name["normal_only_negative_control"].frames == 250
    assert by_name["gpu_10k_optional"].should_run is False
    assert "CUDA unavailable" in by_name["gpu_10k_optional"].skip_reason


def test_final_verdict_holds_when_normal_only_fp_exceeds_target(tmp_path):
    dataset = tmp_path / "large.csv"
    write_webattack_csv(dataset, benign=700, brute=600)
    selected = scale.inspect_csv_dataset(dataset, repo_root=tmp_path)

    verdict = scale.final_verdict(
        selected_dataset=selected,
        rows=[
            {
                "run_name": "normal_only_negative_control",
                "profile": "strict",
                "run_status": "completed",
                "verdict": "ROW_HOLD_FP",
            }
        ],
        skipped_legs=[],
        core_policy={"passed": True},
        branch_pushed=False,
    )

    assert verdict == "SCALE_HOLD_FALSE_POSITIVES"


def test_before_after_delta_rows_compare_off_current_and_strict():
    rows = [
        {
            "run_name": "normal_only_negative_control",
            "profile": "off",
            "fp_per_10k_benign_frames": 388.889,
            "precision": 0.0,
            "recall": None,
            "f1": None,
            "attack_window_coverage": None,
            "calibrated_events": 35,
            "suppressed_events": 0,
        },
        {
            "run_name": "normal_only_negative_control",
            "profile": "low_noise",
            "fp_per_10k_benign_frames": 33.3333,
            "precision": 0.0,
            "recall": None,
            "f1": None,
            "attack_window_coverage": None,
            "calibrated_events": 3,
            "suppressed_events": 4,
        },
        {
            "run_name": "normal_only_negative_control",
            "profile": "strict",
            "fp_per_10k_benign_frames": 4.0,
            "precision": 0.0,
            "recall": None,
            "f1": None,
            "attack_window_coverage": None,
            "calibrated_events": 1,
            "suppressed_events": 6,
        },
    ]

    delta_rows = scale.before_after_delta_rows(rows)
    by_comparison = {row["comparison"]: row for row in delta_rows}

    assert by_comparison["off_to_current_calibrated"]["delta_fp_per_10k"] < 0
    assert by_comparison["current_calibrated_to_tuned_calibrated"]["to_profile"] == "strict"
    assert by_comparison["current_calibrated_to_tuned_calibrated"]["target_status"] == "stretch_pass"
