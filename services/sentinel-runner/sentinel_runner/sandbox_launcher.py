from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROOF_VERDICT = "BLOCKED_RESOURCE_BEFORE_HELDOUT"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def status(job_dir: Path, state: str, **extra: Any) -> None:
    atomic_json(job_dir / "status.json", {
        "schema": "eidos.sentinel-runner.status.v0.2",
        "jobId": job_dir.name,
        "status": state,
        "updatedAt": utc_now(),
        "evidenceClass": "REAL_DATA_ENGINEERING",
        "proofVerdict": PROOF_VERDICT,
        "gatesAdvanced": 0,
        "executionBackend": "sandbox",
        **extra,
    })


def command(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    job_dir = Path(args.job_dir).resolve()
    request_path = Path(args.request).resolve()
    runner_root = repo_root / "services" / "sentinel-runner"
    venv = repo_root / ".eidos-runner-venv"
    log_path = job_dir / "runner.log"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    lock_digest = str(request.get("lockDigest", ""))
    try:
        status(job_dir, "BOOTSTRAPPING_RUNTIME", lockDigest=lock_digest)
        with log_path.open("a", encoding="utf-8") as log:
            if not (venv / "bin" / "python").is_file():
                subprocess.run([sys.executable, "-m", "venv", str(venv)], cwd=repo_root, check=True, stdout=log, stderr=subprocess.STDOUT)
                subprocess.run(
                    [str(venv / "bin" / "python"), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", str(runner_root)],
                    cwd=repo_root,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join((str(runner_root), str(repo_root / "eidos" / "repo" / "src"), environment.get("PYTHONPATH", "")))
            result = subprocess.run(
                [
                    str(venv / "bin" / "python"),
                    "-m",
                    "sentinel_runner.cli",
                    "--request",
                    str(request_path),
                    "--job-dir",
                    str(job_dir),
                ],
                cwd=repo_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return int(result.returncode)
    except Exception as exc:
        (job_dir / "bootstrap_failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        status(job_dir, "FAILED", lockDigest=lock_digest, error=type(exc).__name__, detail="Sandbox runtime bootstrap failed; inspect runner.log and the bootstrap traceback.")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap one isolated Sentinel Lab Sandbox job")
    parser.add_argument("--request", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    raise SystemExit(command(parser.parse_args()))


if __name__ == "__main__":
    main()
