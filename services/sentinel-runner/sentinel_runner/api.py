from __future__ import annotations

import hmac
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from .engine_bridge import discover_engine_path
from .job import atomic_json, update_status
from .spec import ExperimentEnvelope, PROOF_VERDICT


JOB_ID = re.compile(r"^rd-[a-f0-9]{12}-[a-f0-9]{8}$")
app = FastAPI(title="Eidos Sentinel Runner", version="0.2.0")


def job_root() -> Path:
    return Path(os.environ.get("EIDOS_JOB_ROOT", "artifacts/sentinel-runner")).resolve()


def authorize(authorization: Optional[str]) -> None:
    expected = os.environ.get("EIDOS_RUNNER_TOKEN", "")
    supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not expected or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


def resolve_job(job_id: str) -> Path:
    if not JOB_ID.fullmatch(job_id):
        raise HTTPException(status_code=400, detail="invalid job id")
    path = job_root() / job_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="job not found")
    return path


def active_job_count(root: Path) -> int:
    active = {"QUEUED", "PREPARING_DATASET", "RUNNING_FULL_ENGINE", "EVALUATING_FROZEN_PREDICTIONS"}
    count = 0
    for status_path in root.glob("rd-*/status.json"):
        try:
            if json.loads(status_path.read_text(encoding="utf-8")).get("status") in active:
                count += 1
        except (OSError, ValueError):
            continue
    return count


@app.get("/healthz")
def healthz(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    authorize(authorization)
    try:
        engine_path = discover_engine_path()
        engine_ready = True
        engine_name = engine_path.name
    except FileNotFoundError:
        engine_ready = False
        engine_name = None
    return {
        "ok": engine_ready and bool(os.environ.get("KAGGLE_API_TOKEN")),
        "engineReady": engine_ready,
        "engineModule": engine_name,
        "kaggleCredentialsConfigured": bool(os.environ.get("KAGGLE_API_TOKEN")),
        "evidenceClass": "REAL_DATA_ENGINEERING",
        "proofVerdict": PROOF_VERDICT,
    }


@app.post("/v1/experiments", status_code=202)
def create_experiment(payload: Dict[str, Any], authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    authorize(authorization)
    try:
        envelope = ExperimentEnvelope.from_dict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    root = job_root()
    root.mkdir(parents=True, exist_ok=True)
    max_concurrent = max(1, int(os.environ.get("EIDOS_MAX_CONCURRENT_JOBS", "1")))
    if active_job_count(root) >= max_concurrent:
        raise HTTPException(status_code=429, detail="runner capacity occupied; retry after the active job completes")
    job_id = f"rd-{envelope.lock_digest[:12]}-{uuid.uuid4().hex[:8]}"
    directory = root / job_id
    directory.mkdir(parents=False, exist_ok=False)
    request_path = directory / "request.json"
    atomic_json(request_path, payload)
    update_status(directory, "QUEUED", lockDigest=envelope.lock_digest)
    log_handle = (directory / "runner.log").open("ab")
    try:
        subprocess.Popen(
            [sys.executable, "-m", "sentinel_runner.cli", "--request", str(request_path), "--job-dir", str(directory)],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as exc:
        update_status(directory, "FAILED", error=type(exc).__name__, detail="job process could not start")
        raise HTTPException(status_code=500, detail="job process could not start") from exc
    finally:
        log_handle.close()
    return {
        "jobId": job_id,
        "status": "QUEUED",
        "statusUrl": f"/v1/experiments/{job_id}",
        "evidenceClass": "REAL_DATA_ENGINEERING",
        "proofVerdict": PROOF_VERDICT,
    }


@app.get("/v1/experiments/{job_id}")
def get_experiment(job_id: str, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    authorize(authorization)
    path = resolve_job(job_id) / "status.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="job status not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/v1/experiments/{job_id}/artifacts/{artifact_name}")
def get_artifact(job_id: str, artifact_name: str, authorization: Optional[str] = Header(default=None)) -> FileResponse:
    authorize(authorization)
    allowed = {"run_manifest.json", "dataset_receipt.json", "metrics.json", "evaluation_trace.jsonl"}
    if artifact_name not in allowed:
        raise HTTPException(status_code=404, detail="artifact not exposed")
    path = resolve_job(job_id) / artifact_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path)
