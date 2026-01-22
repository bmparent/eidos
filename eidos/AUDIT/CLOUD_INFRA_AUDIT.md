# Cloud / Infra Audit

## Docker / Dependencies
- No Dockerfile in repo root; deployment likely external.
- Requirements are implicit in source (torch, numpy, pandas, kagglehub, websockets).

## Cloud Run / API
- FastAPI entrypoint: `repo/src/eidos_brain/service/api.py`.
- Daemon entrypoint: `repo/src/eidos_brain/service/daemon.py`.

## Idempotency / Retry
- No explicit retry/backoff for network fetches (kagglehub, stream).
- Stream connections use basic timeouts; no reconnection loops.

## Timeouts
- Stream URL timeout configured via `STREAM_URL_TIMEOUT` (connect/read tuple).
- No explicit timeout in daemon loop beyond config.

## GCS / Storage
- `HiveStore` abstraction supports GCS upload, but retry/backoff is not implemented.

## Findings
- **MEDIUM**: No explicit retry/backoff for network downloads (kagglehub/streams).
  - Location: stream and kaggle handlers in `eidos_v0_4_7_02.py`.
  - Repro: Simulate transient network errors; run stops without retry.
  - Minimal diff: add retry wrapper with bounded exponential backoff.
