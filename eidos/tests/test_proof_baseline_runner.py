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
        "drive_manifest.json",
        "event_summary.json",
        "proof_digest.json",
        "proof_digest.md",
    ):
        assert (out_dir / name).is_file()
    assert (out_dir / "scenarios").is_dir()
    assert (out_dir / "plots").is_dir()
    assert (out_dir / "incident_cards").is_dir()
    assert (out_dir / "logs").is_dir()
    assert (out_dir / "plots" / "README.md").is_file()

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    event_summary = json.loads((out_dir / "event_summary.json").read_text(encoding="utf-8"))
    digest = json.loads((out_dir / "proof_digest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"]["seed"] == 42
    assert manifest["benchmark"]["frames"] == 96
    assert manifest["benchmark"]["suite"] == "smoke"
    assert manifest["config"]["config_hash_sha256"] == config["config_hash_sha256"]
    assert manifest["outputs"]["benchmark_summary_csv"] == "benchmark_summary.csv"
    assert manifest["outputs"]["event_summary_json"] == "event_summary.json"
    assert manifest["outputs"]["proof_digest_json"] == "proof_digest.json"
    assert event_summary["aggregate"]["normal_only_false_positives"] <= 5
    assert digest["crash_scan"]["crash_hit_count"] == 0
    assert digest["clean"] is True

    md = (out_dir / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "Seed: `42`" in md
    assert "Frames: `96`" in md
    assert "synthetic_smoke" in md
    assert "Sentinel false-positive control" in md
    assert "Compression baselines" in md

    with (out_dir / "benchmark_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["scenario"] == "synthetic_smoke"
    assert rows[0]["seed"] == "42"
    assert rows[0]["frames"] == "96"
    assert rows[0]["normal_only_false_positives"] == "0"


def test_resolve_out_dir_accepts_repo_prefixed_relative_path_from_git_parent(tmp_path, monkeypatch):
    repo_root = tmp_path / "eidos"
    repo_root.mkdir()
    monkeypatch.chdir(tmp_path)

    out_dir = run_proof_baseline.resolve_out_dir(Path("eidos/artifacts/readiness"), repo_root=repo_root)

    assert out_dir == repo_root / "artifacts" / "readiness"


def test_compression_baselines_for_frames_records_required_and_optional_status():
    frames = run_proof_baseline.np.arange(24, dtype=float).reshape(6, 4)

    result = run_proof_baseline.compression_baselines_for_frames(frames)
    by_name = {item["name"]: item for item in result["baselines"]}

    assert result["raw_bytes"] == frames.size * 8
    assert by_name["raw"]["compression_ratio"] == 1.0
    assert by_name["zlib"]["compressed_bytes"] > 0
    assert by_name["zlib"]["compression_ratio"] > 0
    assert by_name["lzma"]["compressed_bytes"] > 0
    assert "zstd" in by_name
    assert "lz4" in by_name
    assert "delta_zlib" in by_name
    assert result["best_baseline"]


def test_crash_scan_and_digest_record_crash_hits(tmp_path):
    out_dir = tmp_path / "proof"
    out_dir.mkdir()
    (out_dir / "engine_output.log").write_text("CRASH IN INCIDENT LOGIC\n", encoding="utf-8")
    (out_dir / "notes.txt").write_text(
        "can't convert cuda\nTraceback\nRuntimeError\nValueError\nNaN\nInf\nInformation only\n",
        encoding="utf-8",
    )

    scan = run_proof_baseline.scan_crash_strings(out_dir)

    assert scan["status"] == "not_clean"
    assert scan["crash_hit_count"] == 7
    assert scan["patterns"] == [
        "CRASH IN INCIDENT LOGIC",
        "can't convert cuda",
        "Traceback",
        "RuntimeError",
        "ValueError",
        "NaN",
        "Inf",
    ]
    assert {item["path"] for item in scan["crash_hit_files"]} == {"engine_output.log", "notes.txt"}


def test_crash_scan_separates_known_nonfatal_nan_telemetry(tmp_path):
    out_dir = tmp_path / "proof"
    out_dir.mkdir()
    (out_dir / "engine_output.log").write_text(
        "Frame 2000 | HIPP bank=INCIDENT sim=NaN chi=0.000 write=0\n",
        encoding="utf-8",
    )
    (out_dir / "crash_scan.json").write_text(
        '{"patterns": ["Traceback", "NaN"], "note": "receipt metadata"}',
        encoding="utf-8",
    )

    scan = run_proof_baseline.scan_crash_strings(out_dir)

    assert scan["status"] == "clean"
    assert scan["crash_hit_count"] == 0
    assert scan["warning_hit_count"] == 1
    assert scan["warning_hit_files"][0]["path"] == "engine_output.log"


def test_run_returns_nonzero_when_pytest_fails(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    engine_path = repo_root / "fake_engine.py"
    engine_path.write_text("# fake engine for manifest hashing\n", encoding="utf-8")

    def fake_load_engine(_out_dir, _repo_root):
        return FakeEngine(), engine_path

    def fake_run_scenarios(_engine, args, _out_dir, _repo_root):
        return [
            {
                "suite": args.suite,
                "scenario": "synthetic_smoke",
                "seed": args.seed,
                "frames": args.frames,
                "status": "passed",
                "normal_only_false_positives": "",
                "notes": "",
            }
        ], [], ["synthetic_smoke"]

    def fake_pytest(_args, _out_dir, _repo_root):
        run_proof_baseline.minimal_junit_xml(_out_dir / "pytest_results.xml", "fake failure", status="failure")
        return run_proof_baseline.PytestResult(
            command="python -m pytest -m smoke",
            returncode=1,
            status="failed",
            reason="fake failure",
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
            "artifacts/readiness_failure",
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

    assert result == 1


def test_discover_drive_root_prefers_writable_env_dir(tmp_path, monkeypatch):
    drive_root = tmp_path / "configured-drive"
    monkeypatch.setenv("EIDOS_PROOF_DRIVE_DIR", str(drive_root))

    root, reason = run_proof_baseline.discover_drive_root(
        colab_candidates_override=[],
        local_candidates_override=[],
    )

    assert root == drive_root
    assert "EIDOS_PROOF_DRIVE_DIR" in reason
    assert drive_root.is_dir()


def test_discover_drive_root_accepts_colab_my_drive_path(tmp_path, monkeypatch):
    monkeypatch.delenv("EIDOS_PROOF_DRIVE_DIR", raising=False)
    monkeypatch.delenv("EIDOS_ARTIFACT_ROOT", raising=False)
    colab_root = tmp_path / "content" / "drive" / "MyDrive"
    colab_root.mkdir(parents=True)

    root, reason = run_proof_baseline.discover_drive_root(
        colab_candidates_override=[colab_root],
        local_candidates_override=[],
    )

    assert root == colab_root
    assert "Colab Drive root" in reason


def test_discover_drive_root_skips_unconfigured_plain_local_google_drive(tmp_path, monkeypatch):
    monkeypatch.delenv("EIDOS_PROOF_DRIVE_DIR", raising=False)
    monkeypatch.delenv("EIDOS_ARTIFACT_ROOT", raising=False)
    local_root = tmp_path / "Google Drive"
    local_root.mkdir()
    monkeypatch.setattr(
        run_proof_baseline,
        "local_google_drive_candidates",
        lambda: [local_root],
    )

    root, reason = run_proof_baseline.discover_drive_root(
        colab_candidates_override=[],
    )

    assert root is None
    assert "local Google Drive auto-discovery skipped" in reason


def test_mirror_to_drive_copies_artifact_tree_to_proof_phase(tmp_path, monkeypatch):
    drive_root = tmp_path / "configured-drive"
    monkeypatch.setenv("EIDOS_PROOF_DRIVE_DIR", str(drive_root))
    monkeypatch.delenv("EIDOS_ARTIFACT_ROOT", raising=False)
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    (out_dir / "config.json").write_text("{}", encoding="utf-8")
    (out_dir / "logs").mkdir()
    (out_dir / "logs" / "run.txt").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(
        run_proof_baseline,
        "colab_drive_candidates",
        lambda: [],
    )

    manifest = run_proof_baseline.mirror_to_drive(out_dir, "run_1", "2026-05-23")

    drive_run_dir = drive_root / "Eidos_Brain_Proof_Phase" / "2026-05-23" / "run_1"
    assert manifest["drive_copy_success"] is True
    assert manifest["drive_root"] == str(drive_root)
    assert manifest["drive_run_dir"] == str(drive_run_dir)
    assert (drive_run_dir / "config.json").is_file()
    assert (drive_run_dir / "logs" / "run.txt").is_file()
