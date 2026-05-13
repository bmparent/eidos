import csv
import json
from pathlib import Path

from tools import run_proof_baseline


class FakeEngine:
    ENGINE_VERSION = "test-engine"
    EIDOS_BRAIN_CONFIG = {
        "reservoir": 8,
        "sigma_k": 2.0,
        "target_surprise": 0.05,
    }


def test_run_writes_required_baseline_artifacts_with_seed_frames_and_out_dir(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    engine_path = repo_root / "fake_engine.py"
    engine_path.write_text("# fake engine for manifest hashing\n", encoding="utf-8")
    out_arg = Path("artifacts/proof_baseline_2026_05")
    out_dir = repo_root / out_arg

    def fake_load_engine(_out_dir, _repo_root):
        return FakeEngine(), engine_path

    def fake_run_scenarios(_engine, args, _out_dir, _repo_root):
        scenario_dir = _out_dir / "scenarios" / "synthetic_smoke"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        (scenario_dir / "scenario_manifest.json").write_text("{}", encoding="utf-8")
        rows = [
            {
                "suite": args.suite,
                "scenario": "synthetic_smoke",
                "seed": args.seed,
                "frames": args.frames,
                "status": "passed",
                "eidos_compression_ratio": 1.25,
                "baseline_compression_ratio": "",
                "anomaly_recall": "",
                "anomaly_precision": "",
                "anomaly_f1": "",
                "false_positives": "",
                "false_positive_rate": "",
                "anomaly_preservation": "",
                "runtime_seconds": 0.01,
                "frames_per_second": 4800.0,
                "notes": "no labels",
            }
        ]
        return rows, [], ["synthetic_smoke"]

    def fake_pytest(_args, _out_dir, _repo_root):
        run_proof_baseline.minimal_junit_xml(_out_dir / "pytest_results.xml", "fake pytest")
        return run_proof_baseline.PytestResult(
            command="python -m pytest -m smoke --junitxml artifacts/proof_baseline_2026_05/pytest_results.xml",
            returncode=0,
            status="passed",
            reason="fake pytest completed",
        )

    def fake_drive(_out_dir, _run_id, _run_date):
        return {
            "drive_copy_attempted": True,
            "drive_copy_success": False,
            "drive_root": "unknown",
            "drive_run_dir": "unknown",
            "files_considered": [],
            "files_copied": [],
            "files_skipped": [],
            "reason": "not configured in test",
            "timestamp_utc": "2026-05-12T00:00:00Z",
        }

    monkeypatch.setattr(run_proof_baseline, "collect_git_info", lambda _repo_root: {
        "branch": "test-branch",
        "commit": "abc123",
        "dirty": False,
        "errors": [],
        "status_short": "",
    })
    monkeypatch.setattr(run_proof_baseline, "collect_environment", lambda _repo_root: ("environment\n", {"pytest": "fake"}))

    args = run_proof_baseline.parse_args(
        [
            "--suite",
            "smoke",
            "--seed",
            "42",
            "--frames",
            "96",
            "--out",
            str(out_arg),
        ]
    )

    result = run_proof_baseline.run(
        args,
        repo_root=repo_root,
        load_engine_fn=fake_load_engine,
        run_scenarios_fn=fake_run_scenarios,
        run_pytest_fn=fake_pytest,
        mirror_to_drive_fn=fake_drive,
        write_docs_fn=lambda **_: None,
    )

    assert result == 0
    for name in (
        "config.json",
        "benchmark_summary.csv",
        "benchmark_summary.md",
        "pytest_results.xml",
        "environment.txt",
        "git_commit.txt",
        "run_manifest.json",
    ):
        assert (out_dir / name).is_file()
    assert (out_dir / "scenarios").is_dir()
    assert (out_dir / "plots").is_dir()
    assert (out_dir / "plots" / "README.md").is_file()

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"]["seed"] == 42
    assert manifest["benchmark"]["frames"] == 96
    assert manifest["benchmark"]["suite"] == "smoke"
    assert manifest["config"]["config_hash_sha256"] == config["config_hash_sha256"]
    assert manifest["outputs"]["benchmark_summary_csv"] == "benchmark_summary.csv"

    md = (out_dir / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "Seed: `42`" in md
    assert "Frames: `96`" in md
    assert "synthetic_smoke" in md

    with (out_dir / "benchmark_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["scenario"] == "synthetic_smoke"
    assert rows[0]["seed"] == "42"
    assert rows[0]["frames"] == "96"
