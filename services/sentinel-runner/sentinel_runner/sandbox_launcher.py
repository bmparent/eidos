from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROOF_VERDICT = "BLOCKED_RESOURCE_BEFORE_HELDOUT"
CPU_TORCH_VERSION = "2.14.0"


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
    job_dir.mkdir(parents=True, exist_ok=True)
    lock_digest = ""
    bootstrap_step = "read_request"
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        lock_digest = str(request.get("lockDigest", ""))
        status(job_dir, "BOOTSTRAPPING_RUNTIME", lockDigest=lock_digest)
        with log_path.open("a", encoding="utf-8") as log:
            bootstrap_step = "locate_uv"
            uv = shutil.which("uv")
            if not uv:
                raise RuntimeError("The managed Sandbox image does not expose the required uv installer.")
            if not (venv / "bin" / "python").is_file():
                bootstrap_step = "create_virtual_environment"
                subprocess.run(
                    [uv, "venv", "--python", sys.executable, "--no-python-downloads", str(venv)],
                    cwd=repo_root,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            # A Python executable alone does not prove a previous install finished.
            bootstrap_step = "install_cpu_runtime"
            subprocess.run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(venv / "bin" / "python"),
                    "--no-cache",
                    "--torch-backend",
                    "cpu",
                    str(runner_root),
                ],
                cwd=repo_root,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            bootstrap_step = "verify_cpu_runtime"
            subprocess.run(
                [
                    str(venv / "bin" / "python"),
                    "-c",
                    (
                        "import torch; "
                        f"assert torch.__version__.split('+')[0] == '{CPU_TORCH_VERSION}'; "
                        "assert torch.version.cuda is None; "
                        "print(f'torch={torch.__version__} backend=cpu')"
                    ),
                ],
                cwd=repo_root,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join((str(runner_root), str(repo_root / "eidos" / "repo" / "src"), environment.get("PYTHONPATH", "")))
            bootstrap_step = "execute_engine_job"
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
        receipt = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
        expected = "COMPLETED_ENGINEERING" if result.returncode == 0 else "FAILED"
        if receipt.get("status") != expected:
            raise RuntimeError(f"Engine process exited {result.returncode} without the expected terminal receipt")
        return int(result.returncode)
    except Exception as exc:
        (job_dir / "bootstrap_failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        return_code = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else None
        exit_detail = f" with exit code {return_code}" if return_code is not None else ""
        status(
            job_dir,
            "FAILED",
            lockDigest=lock_digest,
            error="SANDBOX_BOOTSTRAP_FAILED",
            detail=f"Sandbox bootstrap step {bootstrap_step!r} failed{exit_detail}.",
            artifacts=[name for name in ("source_receipt.json", "runner.log", "bootstrap_failure_traceback.log") if (job_dir / name).is_file()],
        )
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap one isolated Sentinel Lab Sandbox job")
    parser.add_argument("--request", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    raise SystemExit(command(parser.parse_args()))


if __name__ == "__main__":
    main()
