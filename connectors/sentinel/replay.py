"""Bounded JSONL replay; no installation, secret logging, or unrelated collection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("artifacts/sentinel-replay-state.json"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100 or not 1 <= args.attempts <= 10:
        raise SystemExit("Batch size 1–100 and attempts 1–10 required.")
    url = os.environ.get("EIDOS_INGEST_URL", "")
    key = os.environ.get("EIDOS_INGEST_KEY", "")
    if not key or not (url.startswith("https://") or url.startswith("http://127.0.0.1:")):
        raise SystemExit("Set EIDOS_INGEST_URL and EIDOS_INGEST_KEY in your environment; never pass secrets in arguments.")
    data = args.file.read_bytes()
    file_hash = hashlib.sha256(data).hexdigest()
    lines = [line for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    state = json.loads(args.state.read_text()) if args.state.exists() else {"fileSha256": file_hash, "offset": 0}
    if state["fileSha256"] != file_hash:
        raise SystemExit("Input changed. Use a new --state path; do not overwrite the previous replay cursor.")
    offset = state["offset"]
    while offset < len(lines):
        batch = []
        for index, line in enumerate(lines[offset:offset + args.batch_size], offset):
            event = json.loads(line)
            event.setdefault("id", f"replay:{file_hash[:16]}:{index + 1}")
            batch.append(event)
        for attempt in range(args.attempts):
            request = urllib.request.Request(url, data=json.dumps({"events": batch}).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=330) as response:
                    receipt = json.load(response)
                if receipt["accepted"] + receipt["duplicates"] != len(batch):
                    raise RuntimeError("The acknowledgement does not account for the entire batch.")
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 502, 503, 504) or attempt + 1 == args.attempts:
                    raise SystemExit(f"Replay paused at offset {offset}, HTTP {exc.code}. Cursor preserved; rerun after fixing the connection.")
                time.sleep(min(30, 2 ** attempt))
            except (urllib.error.URLError, TimeoutError):
                if attempt + 1 == args.attempts:
                    raise SystemExit(f"Replay paused at offset {offset}. Cursor preserved; retry the same input.")
                time.sleep(min(30, 2 ** attempt))
        offset += len(batch)
        state = {"fileSha256": file_hash, "offset": offset, "lastReceipt": receipt}
        args.state.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.state.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(args.state)
        print(json.dumps({"acknowledged": offset, "total": len(lines), "accepted": receipt["accepted"], "duplicates": receipt["duplicates"]}))


if __name__ == "__main__":
    main()
