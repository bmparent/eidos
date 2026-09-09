# Sentinel scoped telemetry connector

Create a source in Sentinel â†’ Monitors, select a named numeric measurement and its unit, and save the one-time ingest key in your secret environment. Mark validation sources as synthetic. The key can only ingest into that source; it cannot read datasets, artifacts or other sources. Keys expire in 30 days and can be revoked immediately from Monitors. No global operator token is distributed.

## OpenTelemetry

The package uses OpenTelemetry Contrib Collector **0.160.0**. Obtain the appropriate official release from [Collector releases](https://github.com/open-telemetry/opentelemetry-collector-releases/releases/tag/v0.160.0) and verify its release checksum. Configuration follows the [Collector configuration reference](https://opentelemetry.io/docs/collector/configuration/) and the [persistent sending queue documentation](https://github.com/open-telemetry/opentelemetry-collector/blob/main/exporter/exporterhelper/README.md).

Set `EIDOS_INGEST_URL` to the exact source URL displayed by Monitors, without `/v1/metrics`, and set `EIDOS_INGEST_KEY` in the process environment or deployment's secret store. Do not put either secret-bearing value into version control. The configured source measurement must match the OTLP metric name exactly. The adapter accepts finite gauge/sum numeric points over OTLP **HTTP JSON**. Protobuf, traces, histograms, exemplars and arbitrary metric sets are outside this release.

From the checkout root, using the downloaded collector executable:

```text
otelcol-contrib validate --config connectors/sentinel/otel-collector.yaml
otelcol-contrib --config connectors/sentinel/otel-collector.yaml
```

The second command starts collection and should be run only on an authorized host. This implementation does not install or start a collector on Brent's machine. The receiver binds only to loopback ports 4317/4318. Configure an authorized application's exporter to that receiver. If an application is remote, explicitly configure its collector networking; a pasted website URL is not permission or a mechanism to observe that process.

The disk-backed queue is `./sentinel-collector-state`; use a persistent, private writable directory for a deployed collector. The package batches up to 100 points, limits collector memory to 128 MiB, uses one sending consumer, retains up to 100 queued requests, and retries transient server failures with bounded backoff. If the disk queue or server buffer fills, pause the source and investigate; finite queues are not a no-loss guarantee. Keep the state directory across collector restarts. Protect it as original telemetry data.

## Replayable JSONL

Each JSONL line is an object with `id`, `eventTime`, and finite numeric `value`, plus optional bounded `attributes`. IDs must be stable for retries. Event time is ISO 8601 with `Z` or an explicit offset. Future timestamps more than five minutes ahead are rejected. Original attributes, source identity, event time and arrival time remain evidence.

Set the same environment variables and run from the repository root:

```text
python connectors/sentinel/replay.py --file connectors/sentinel/sample.jsonl --state artifacts/sentinel-replay-state.json --batch-size 60
```

`sample.jsonl` contains explicitly synthetic service measurements, including an injected failure. The cursor advances atomically only after acknowledgement. Rerunning with the same file/cursor resumes after acknowledged lines; restarting from an earlier cursor resends identical event IDs and is safely deduplicated. Reusing an ID with different contents returns 409 and rolls back the entire batch. Keep credentials in the environment; neither logs nor cursor contain them. Do not edit a source file mid-replay; changed file hashes are rejected. Local tests may use an HTTP loopback endpoint; deployed endpoints must be HTTPS.

## Server envelope and recovery

- Up to four sources per owner; 100 events per request; 4 KB per event; 1,000 buffered events/source; 10,000 newly accepted events/source/day.
- Processing uses the shared one-worker admission slot and at most **four batches/source per rolling day**, each at most 1,000 events and 15 minutes. This is a bounded pilot budget, not a 24-hour high-rate monitoring guarantee. Automatic processing begins when at least 24 events are buffered. After the daily compute budget, arrivals remain buffered until capacity returns; 429 instructs the collector to retain/retry its data.
- Arrival-order offsets are contiguous. Duplicate IDs do not allocate offsets. A batch with an offset gap cannot silently skip data.
- The first 24 accepted, increasing event-time values are warmup. Late/equal/out-of-order timestamps are retained with a distinct outcome and do not rewind model state. Gaps greater than five minutes, stale arrivals, warmup and resets are displayed separately.
- The next-event readout is committed before the next accepted event. Its time is unknown at issuance; it is not a promised 60-second forecast. Dataset analysis provides explicit elapsed-time horizons.
- The durable checkpoint includes reservoir state, readout, inverse covariance, baseline history, issuance, normalization and offset. A crash resumes from the last committed checkpoint; uncommitted work may be recomputed. Compare-and-set prevents an older batch from overwriting newer state. Event receipts are retained separately.
- Reset requires a written reason, archives the previous checkpoint and restarts warmup after the current processed offset. It does not erase input events or historical incident evidence.

Monitors â†’ Refresh health exposes buffered count, processed offset, warmup remaining, late-event count, gaps and recent event evidence. `receiving` describes arrivals, not normality or safety. A collector disconnect should become `stale`; an observed gap describes missing time, not proof that no event occurred.

Local and preview receipts, including collector validation, replay/restart, deduplication, backpressure and revocation, are indexed in `artifacts/sentinel-guided-20260908/`. Physical installation on a particular host is a later owner action: download the verified collector, set that source's two environment variables, and run the documented command on the authorized host.
