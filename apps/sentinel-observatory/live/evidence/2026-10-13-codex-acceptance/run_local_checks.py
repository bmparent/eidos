"""Capture provider-free Observatory regressions from the repository root.

This runner never reads credentials, calls OpenAI, or changes hosted state.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path.cwd().resolve()
OUT = ROOT / "apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance"
LIVE = "apps/sentinel-observatory/live/"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def capture(name: str, argv: list[str], stdout_name: str | None = None) -> dict:
    started = utc_now()
    clock = time.monotonic()
    try:
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=180)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
        reason = None
    except subprocess.TimeoutExpired as exc:
        code, stdout, stderr = None, exc.stdout or b"", exc.stderr or b""
        reason = "Local check exceeded its 180-second bound; no pass is claimed."
    stdout_file = stdout_name or f"{name}.stdout.log"
    (OUT / stdout_file).write_bytes(stdout)
    (OUT / f"{name}.stderr.log").write_bytes(stderr)
    receipt = {
        "name": name,
        "argv": argv,
        "cwd": ".",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "duration_seconds": round(time.monotonic() - clock, 3),
        "exit_code": code,
        "status": "passed" if code == 0 else "failed" if code is not None else "blocked",
        "reason": reason,
        "stdout": stdout_file,
        "stderr": f"{name}.stderr.log",
        "provider_calls": 0,
        "evidence_level": "local_regression",
    }
    if stdout_file.endswith(".xml") and code == 0:
        try:
            tree = ET.fromstring(stdout)
            receipt["test_cases"] = len(tree.findall(".//testcase"))
            receipt["failures"] = len(tree.findall(".//failure"))
            receipt["errors"] = len(tree.findall(".//error"))
            receipt["skipped"] = len(tree.findall(".//skipped"))
        except ET.ParseError:
            receipt["junit_parse_error"] = True
            receipt["status"] = "failed"
    return receipt


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    OUT = parser.parse_args().out.resolve()
    if not OUT.is_relative_to(ROOT):
        raise SystemExit("Keep proof receipts inside this repository.")
    actual = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip()
    if Path(actual).resolve() != ROOT:
        raise SystemExit("Run this command from the repository root.")
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (OUT / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    node = subprocess.check_output(["node", "--version"], text=True).strip()
    (OUT / "environment.txt").write_text(
        f"Python: {sys.version}\nNode: {node}\nPlatform: {platform.platform()}\n"
        "Credentials: not read or captured\nProvider calls: none\n", encoding="utf-8"
    )
    checks = [
        ("node-regressions", ["node", "--test", "--test-reporter=junit"] + [
            LIVE + name for name in (
                "collector.test.mjs", "brain.test.mjs", "browser-interpreter.test.mjs",
                "security-events.test.mjs", "hosted-sensor.test.mjs", "sol-interpreter.test.mjs"
            )
        ], "node-tests.xml"),
        ("python-regressions", [sys.executable, "-m", "unittest", "discover", "-s", LIVE.rstrip("/"), "-p", "test_*.py", "-v"], None),
        ("brain-parity", ["node", LIVE + "verify_brain_parity.mjs"], None),
        ("app-syntax", ["node", "--check", LIVE + "web/app.js"], None),
        ("network-panel-syntax", ["node", "--check", LIVE + "web/network-panel.mjs"], None),
        ("sol-syntax", ["node", "--check", LIVE + "sol-interpreter.mjs"], None),
        ("whitespace", ["git", "diff", "--check"], None),
    ]
    receipts = []
    for name, argv, stdout_name in checks:
        receipt = capture(name, argv, stdout_name)
        receipts.append(receipt)
        (OUT / "local-checks.json").write_text(json.dumps({
            "product_head": commit, "provider_calls": 0, "core_behavior_changed": False,
            "checks": receipts,
        }, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"check": name, "status": receipt["status"], "test_cases": receipt.get("test_cases")}), flush=True)
    return 0 if all(item["status"] == "passed" for item in receipts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
