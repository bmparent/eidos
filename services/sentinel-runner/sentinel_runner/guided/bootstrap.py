"""Pinned isolated Sandbox installation; invoked from the verified repo root."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "runner.log").open("a", encoding="utf-8")
    def run(command):
        subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    try:
        run([sys.executable, "-m", "pip", "install", "uv"])
        run([sys.executable, "-m", "uv", "pip", "install", "--python", sys.executable, "--torch-backend", "cpu",
             "-e", "services/sentinel-runner[guided]"])
        run([sys.executable, "-m", "sentinel_runner.guided.cli", "--request", args.request, "--out", args.out])
    except Exception:
        if not (out / "status.json").exists():
            (out / "status.json").write_text(json.dumps({"status": "failed", "error": "RUNTIME_BOOTSTRAP_FAILED",
                "detail": "The isolated runtime could not start; inspect the authenticated diagnostic receipt."}))
        log.write(traceback.format_exc())
        return 1
    finally:
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
