from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from .dataset import prepare_kaggle_dataset, sha256_file
from .engine_bridge import run_full_engine
from .metrics import evaluate_frozen_predictions
from .spec import ExperimentEnvelope, PROOF_VERDICT


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def artifact_digests(job_dir: Path) -> Dict[str, Dict[str, Any]]:
    result = {}
    for path in sorted(job_dir.rglob("*")):
        if not path.is_file() or path.name in {"status.json", "run_manifest.json"} or "input" in path.parts:
            continue
        relative = path.relative_to(job_dir).as_posix()
        result[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def update_status(job_dir: Path, status: str, **extra: Any) -> None:
    atomic_json(job_dir / "status.json", {
        "schema": "eidos.sentinel-runner.status.v0.2",
        "jobId": job_dir.name,
        "status": status,
        "updatedAt": utc_now(),
        "evidenceClass": "REAL_DATA_ENGINEERING",
        "proofVerdict": PROOF_VERDICT,
        "gatesAdvanced": 0,
        **extra,
    })


def run_job(request_path: Path, job_dir: Path) -> int:
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        envelope = ExperimentEnvelope.from_dict(request)
        update_status(job_dir, "PREPARING_DATASET", lockDigest=envelope.lock_digest)
        prepared = prepare_kaggle_dataset(envelope.spec, job_dir / "input")
        atomic_json(job_dir / "dataset_receipt.json", prepared.receipt)

        update_status(job_dir, "RUNNING_FULL_ENGINE", lockDigest=envelope.lock_digest, rowsSentToEngine=int(prepared.frames.shape[0]))
        results, engine_receipt = run_full_engine(prepared, envelope.spec, job_dir / "engine_artifacts")
        step_rows = list(results.get("step_rows") or [])
        update_status(job_dir, "EVALUATING_FROZEN_PREDICTIONS", lockDigest=envelope.lock_digest)
        metrics, trace_rows = evaluate_frozen_predictions(step_rows, prepared)
        atomic_json(job_dir / "metrics.json", metrics)
        write_jsonl(job_dir / "evaluation_trace.jsonl", trace_rows)

        manifest = {
            "schema": "eidos.sentinel-runner.manifest.v0.2",
            "job_id": job_dir.name,
            "completed_at": utc_now(),
            "lock_digest": envelope.lock_digest,
            "spec": envelope.spec.to_dict(),
            "evidence_class": "REAL_DATA_ENGINEERING",
            "proof": {"verdict": PROOF_VERDICT, "gates_advanced": 0, "heldout_evaluated": False},
            "engine": engine_receipt,
            "dataset_receipt_sha256": sha256_file(job_dir / "dataset_receipt.json"),
            "metrics_sha256": sha256_file(job_dir / "metrics.json"),
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "artifacts": artifact_digests(job_dir),
        }
        atomic_json(job_dir / "run_manifest.json", manifest)
        update_status(
            job_dir,
            "COMPLETED_ENGINEERING",
            lockDigest=envelope.lock_digest,
            metrics=metrics,
            artifacts=["run_manifest.json", "dataset_receipt.json", "metrics.json", "evaluation_trace.jsonl"],
        )
        return 0
    except Exception as exc:
        (job_dir / "failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        update_status(
            job_dir,
            "FAILED",
            error=type(exc).__name__,
            detail=str(exc),
            artifacts=["runner.log", "failure_traceback.log"],
        )
        return 1
