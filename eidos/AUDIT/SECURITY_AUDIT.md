# Security Audit

## Secrets
- Gemini API key loaded from env or `colab_secrets.py`.
- No secrets committed in repo.

## Deserialization / Input Safety
- Archive walker reads files directly (including binary) with size caps.
- JSON/CSV parsing does not sandbox; malicious content could cause memory pressure.

## Network Access
- Stream ingestion accepts arbitrary URLs/IPs and headers; no allowlist.
- WebSocket and TCP streams accept raw content; only parse-safe guards.

## Findings
- **HIGH**: Live stream endpoints are unrestricted; potential SSRF in managed deployments.
  - Location: STREAM configuration in engine.
  - Repro: set STREAM_URL to internal network service.
  - Minimal diff: enforce allowlist or disable in production.

- **MEDIUM**: No explicit secrets redaction in logs or anomaly receipts.
  - Location: `SessionRecorder.record_anomaly()` writes snippets to disk.
  - Repro: ingest secrets in text stream; receipts will persist them.
  - Minimal diff: add regex redaction pass on snippets.
