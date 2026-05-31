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
        max_rows=None,
        engine=FakeEngine(),
        features=8,
        seed=42,
        repo_root=tmp_path,
    )

    assert dataset.frames.shape == (6, 8)
    assert dataset.labels.tolist() == [0, 0, 1, 1, 1, 0]
    assert dataset.label_distribution["BENIGN"] == 3
    assert "Flow Duration" in dataset.feature_columns
    assert dataset.events[2]["attack"] is True
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
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["crash_hit_count"] == 0
    assert config["core_behavior"]["sentinel_thresholds_changed"] is False
    assert digest["clean"] is True
