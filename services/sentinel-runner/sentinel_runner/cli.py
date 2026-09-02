from __future__ import annotations

import argparse
from pathlib import Path

from .job import run_job


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one isolated Sentinel Lab real-data experiment")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    return run_job(args.request.resolve(), args.job_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
