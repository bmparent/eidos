"""Process boundary for parser, causal Torch analysis, semantic retrieval and telemetry."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import sys
import traceback
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ..job import atomic_json
from .ingestion import MAX_BYTES, parse, plain, confirm


def execute(request: dict, directory: Path):
    operation = request["operation"]
    if operation == "parse":
        encoded = request.get("content", "")
        if len(encoded) > MAX_BYTES * 1.34 + 8:
            raise ValueError("UPLOAD_SIZE")
        data = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(data).hexdigest() != request["sha256"]:
            raise ValueError("INPUT_CHECKSUM_MISMATCH")
        return parse(data, request["filename"], request.get("source"))
    dataset = request.get("dataset")
    if operation == "confirm":
        return confirm(dataset, request["mapping"])
    if operation == "analyze":
        if dataset["kind"] == "documents":
            from .semantic import analyze
            return analyze(dataset, directory)
        from .causal import temporal, unordered
        mapping = confirm(dataset, request["mapping"])
        if mapping["mode"] == "temporal":
            return temporal(dataset, mapping, request.get("options", {}), directory)
        return unordered(dataset, mapping, directory)
    if operation == "retrieve":
        from .semantic import retrieve
        return retrieve(dataset, request["result"], request["question"], directory)
    if operation == "telemetry":
        from .telemetry import process
        return process(request, directory)
    raise ValueError("UNSUPPORTED_OPERATION")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Imported engine may set this for its artifact writers; never let it choose Drive.
    os.environ["EIDOS_ARTIFACT_ROOT"] = str(args.out)
    status = {"schema": "eidos.guided-job.v1", "status": "running", "updatedAt": datetime.now(timezone.utc).isoformat()}
    atomic_json(args.out / "status.json", status)
    finished = threading.Event()
    def heartbeat():
        while not finished.wait(2):
            if (args.out / "cancel.requested").exists():
                atomic_json(args.out / "status.json", {**status, "status": "cancelled"})
                os._exit(2)
            atomic_json(args.out / "heartbeat.json", {"time": datetime.now(timezone.utc).isoformat()})
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        if args.request.stat().st_size > 24_000_000:
            raise ValueError("REQUEST_SIZE")
        request = json.loads(args.request.read_text(encoding="utf-8-sig"))
        result = plain(execute(request, args.out))
        result["receipt"] = {"requestSha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
                             "python": platform.python_version(), "operation": request["operation"],
                             "sourceCommit": request.get("sourceCommit"), "processIsolation": True,
                             "completedAt": datetime.now(timezone.utc).isoformat()}
        serialized = json.dumps(result, allow_nan=False).encode()
        if len(serialized) > 16_000_000:
            raise ValueError("RESULT_SIZE: reduce input rows, features or passages.")
        atomic_json(args.out / "result.json", result)
        status.update(status="completed", resultSha256=hashlib.sha256((args.out / "result.json").read_bytes()).hexdigest())
    except Exception as exc:
        (args.out / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        # Source-derived exception text is bounded; no environment or secrets logged.
        status.update(status="failed", error=type(exc).__name__, detail=str(exc)[:600])
    finished.set()
    status["updatedAt"] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.out / "status.json", status)
    callback = os.environ.get("EIDOS_CALLBACK_URL")
    token = os.environ.get("EIDOS_CALLBACK_TOKEN")
    if callback and token:
        import urllib.request
        try:
            request = urllib.request.Request(callback, data=b"{}", headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=30) as response:
                response.read(4096)
        except Exception:
            # Snapshot and checksum remain intact for status reconciliation.
            (args.out / "callback-status.json").write_text(json.dumps({"delivered": False, "recovery": "poll the stable job identity"}))
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
