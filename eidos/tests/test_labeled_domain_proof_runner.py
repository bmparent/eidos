import csv
import json
from pathlib import Path

import numpy as np

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
        "proof_digest.json",
        "proof_digest.md",
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
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["crash_hit_count"] == 0
    assert metrics["normalized_label_distribution"] == {"ATTACK": 3, "BENIGN": 3}
    assert config["core_behavior"]["sentinel_thresholds_changed"] is False
    assert digest["clean"] is True
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["precision_ledger_json"] == "precision_ledger.json"
    assert "tracked_dirty" in manifest
    assert "untracked_generated_files" in manifest
    assert "untracked_non_generated_files" in manifest
    assert "git_dirty_reason" in manifest
    assert manifest["device"]["selected_device"] in {"cpu", "cuda"}
    ledger = json.loads((out_dir / "precision_ledger.json").read_text(encoding="utf-8"))
    assert {"raw", "merged", "deduped"} <= set(ledger["precision_lift_summary"])


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
