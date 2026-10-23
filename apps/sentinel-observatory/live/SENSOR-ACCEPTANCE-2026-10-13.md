# Attached native sensor acceptance · 2026-10-13

The requested sensor is attached to the private hosted Observatory. Its source is actual Observatory API traffic and the outbound HEAD checks users request. It is an application-edge sensor; no Suricata/Zeek capture agent or provider firewall feed has been installed or discovered on an external monitored network.

## Exact running candidate

- App: https://sentinel-evidence-observatory.bmparent.chatgpt.site
- Sites source `501f4fdd79ff04008257148744d31a7cac5d21e8`.
- Sites deployment `appgdep_6ace9781d36c8191a50c8a4acfaff764`, succeeded2026-10-13T20:41:44Z; environment revision4.
- D1 binding `OBSERVATORY_DB`, observed live table `observatory_sensor_events`; migration0001 applied without sample seeds.
- Node collector source `796b9a6e19739b88717d8c02e944293201ad9039`, READY deployment `dpl_7Vu6pBHQPhFuSBii8F2Fc59bRsa6` on protected immutable `eidos-sentinel-idvlsumqq-1brentbm-1876s-projects.vercel.app`.
- Worker SHA256 `eab7ba71abe0fb8fd3a6b34bc3e85ad388e9810edc6e534a1edeeed29c9bb491`.

## Actual hosted observations

Three deliberately requested verification checks produced six durable sensor records: two accepted API requests, two actual outbound HEAD receipts, one rejected API request and one distinct security-control enforcement record. All verification records are marked functional checks, using a server-credential-bound test marker; arbitrary browser declarations cannot enable this marker.

- Literal `https://1.1.1.1/` produced actual HEAD301, public peer matched destination, TLS verified, and Node socket counters measured176 bytes read/291 bytes written for that connection. Redirect remained unfollowed.
- `https://eidos-works.com/` returned actual HEAD200.
- The deliberately supplied loopback URL was denied with400; the recorded `non_public_target` rule action is`deny`. This is a functional control test, not an attack or malicious visitor.
- Unauthenticated access to the sensor feed was rejected by the existing private access policy.
- Repeated feed reads retained the same six records; polling/static/capability requests do not create traffic events.
- The exact hosted snapshot normalized to six evidence-linked native events, one observed control detection and zero confirmed-attack assertions. The native topology distinguishes symbolic service components from actual public endpoints. Preflights with no HEAD create no outbound traffic arc.

Actual receipts are in `evidence/2026-10-13-attached-sensor/`. No provider tokens or client addresses are stored there. Provider-record normalization and DOM tests are distinct from pixel rendering or physical-device observations.

## Persistence, privacy and interpretation

The sensor retains at most1000 records and exposes a seven-day window. Expired rows are purged on subsequent reads/writes; there is no unattended TTL-cleanup claim. Storage/stream operations have bounded deadlines and failures show unavailable. No schema is created during requests. Source-event age remains separate from a successful feed refresh.

Captured metadata excludes bodies, auth/cookies, user agents, incoming client IPs, query strings and target paths. The source does not see traffic outside this Observatory, target internals, LAN packets, provider firewall incidents or access denials that occur before Sites dispatch.

Native duration baselines separate request direction and public endpoint; fewer than12 relevant observations remain provisional. Known controls are distinguished from operational errors and experimental numerical changes. Qwen sees safe status/duration/counter/control enums, not addresses or raw request content. Native model-controller tests pass; no fresh native browser-GPU inference result is claimed.

## Validation

-29 hosted collector tests including network-metadata allowlisting.
-13 Node collector tests and strict TypeScript; actual provider deployment READY.
-35 native sensor/adapter/model tests, including real SQLite schema/retention, stream deadlines, source provenance and model input boundaries.
-17 network-panel DOM regressions plus the endpoint/model contract passed, covering native auto-read, no auto-resume after pause, stale source events, symbolic graph, request/check/control filters, functional-check labels and preflight visibility.
-Independent review findings fixed: physical expiry cleanup on read, stalled response processing, no preflight traffic arcs and declared functional-check labels.

Source and API flows are verified. Pixel rendering remains unverified. This release supplies real observable traffic and security-control evidence within its stated application-edge scope; it does not establish intrusion-detection accuracy or broad network coverage.
