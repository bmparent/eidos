#!/usr/bin/env python3
"""Run minimal smoke tests for the Eidos demo pipeline."""
import json
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1] / "repo"


def _insert_repo_path(repo_root: Path) -> None:
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))


def _find_latest_session(archive_root: Path) -> Path:
    if not archive_root.exists():
        raise FileNotFoundError(f"Archive root not found: {archive_root}")
    sessions = [p for p in archive_root.iterdir() if p.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"No sessions under {archive_root}")
    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions[0]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _assert_steps_csv(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    required_cols = {
        "step",
        "is_surprise",
        "best_err",
        "z",
        "z_thresh_eff",
        "ema_err",
        "sigma",
    }
    cols = set(header.split(","))
    missing = required_cols - cols
    if missing:
        raise AssertionError(f"steps.csv missing columns: {sorted(missing)}")


def _assert_anomaly_receipt(path: Path) -> None:
    found = False
    for rec in _load_jsonl(path):
        found = True
        required = [
            "session_id",
            "config_hash",
            "synaptic_hash_initial",
            "synaptic_hash_current",
            "attrib",
            "fingerprint_topk",
            "baseline",
            "evidence_paths",
        ]
        missing = [key for key in required if key not in rec]
        if missing:
            raise AssertionError(f"Anomaly missing receipt fields: {missing}")
    if not found:
        raise AssertionError("anomalies.jsonl was empty")


def _assert_summary(path: Path) -> None:
    summary = _load_json(path)
    for key in ("synaptic_hash_initial", "synaptic_hash_final", "continuity_hash"):
        if key not in summary:
            raise AssertionError(f"summary.json missing {key}")


def _assert_compression_meta(artifact_root: Path) -> None:
    compression_dir = artifact_root / "compression"
    if not compression_dir.exists():
        raise AssertionError("compression artifacts directory missing")
    metas = list(compression_dir.rglob("*bicameral_stream_meta*.json"))
    if not metas:
        raise AssertionError("compression meta artifact missing")


def _run_session(config: dict) -> None:
    from eidos_brain.engine.adapters import run_session

    result = run_session(config)
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Engine run failed: {result}")


def main() -> None:
    repo_root = _repo_root()
    _insert_repo_path(repo_root)

    demo_data = repo_root / "demo_data"
    archive_target = demo_data / "sample_folder"
    if not archive_target.exists():
        raise FileNotFoundError(f"Expected archive folder missing: {archive_target}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base_out = Path.cwd() / "smoke_artifacts" / stamp

    kaggle_config = {
        "source_type": "KAGGLE",
        "artifact_root": str(base_out / "kaggle"),
        "source_params": {
            "kaggle": {
                "dataset_id": str(demo_data),
                "file_name": "sample.csv",
                "max_rows": 200,
                "use_kagglehub": False,
            }
        },
        "engine_config": {
            "steps": 400,
            "warmup_cap": 50,
        },
    }

    local_config = {
        "source_type": "LOCAL",
        "artifact_root": str(base_out / "local"),
        "source_params": {
            "local": {
                "mode": "ARCHIVE",
                "target": str(archive_target),
                "max_frames": 400,
                "max_lines_per_file": 200,
                "snippet_chars": 120,
            }
        },
        "engine_config": {
            "steps": 400,
            "warmup_cap": 50,
        },
    }

    for label, config in (("kaggle", kaggle_config), ("local", local_config)):
        print(f"Running smoke test for {label}...")
        _run_session(config)

        artifact_root = Path(config["artifact_root"]).resolve()
        archive_root = artifact_root / "eidos_brain_archive"
        session_dir = _find_latest_session(archive_root)

        steps_csv = session_dir / "steps.csv"
        anomalies_jsonl = session_dir / "anomalies.jsonl"
        summary_json = session_dir / "summary.json"

        if not steps_csv.exists():
            raise AssertionError(f"steps.csv missing: {steps_csv}")
        if not anomalies_jsonl.exists():
            raise AssertionError(f"anomalies.jsonl missing: {anomalies_jsonl}")
        if not summary_json.exists():
            raise AssertionError(f"summary.json missing: {summary_json}")

        _assert_steps_csv(steps_csv)
        _assert_anomaly_receipt(anomalies_jsonl)
        _assert_summary(summary_json)
        _assert_compression_meta(artifact_root)

        print(f"{label} smoke test passed: {session_dir}")

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
