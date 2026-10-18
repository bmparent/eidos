# Acceptance matrix — 18 October 2026

| Gate | Status | Evidence or gap |
|---|---|---|
| factual_integrity | partial | Controlled status/stage/citation assertions pass; the original final cases were exposed by failures and are now regression cases. |
| known_failure_stages | passed_local | 72/72 synthetic controlled cases; real local sockets plus labeled DNS/blackhole simulations. |
| security_and_tenant_isolation | partial | Public-boundary tests pass. Tenant-scoped records, model budgets, exports and notification authorization are not implemented; two real identities remain untested. |
| continuous_operation | not_implemented | Foreground-only; no durable schedule, job leases, seven-day pilot or second independent observer. |
| scheduling_target | not_tested | 99.5% is a proposed qualification target; eligible slots and terminal receipts do not exist yet. |
| incident_behavior | not_implemented | Existing change episodes are not a durable incident/recovery/maintenance state machine. |
| notifications | not_implemented | No durable outbox, controlled sink or provider/inbox delivery receipt. |
| explanation_value | blocked | Deterministic single-arm blinded packet supplied. Genuine corpus Sol-ranked arm and independent human scores are absent; no usefulness percentage. |
| detector_comparison | passed_local | 72 synthetic streams, 11520 frames, fixed thresholds, same-stream gate/EWMA/CUSUM comparisons and whole-stream bootstrap CIs; no superiority. |
| ux | passed_local | Desktop 1440x1000 and mobile 390x844: connect, stage rendering, pause, export, reload/history, GET fallback, keyboard filter focus, reduced motion, no overflow or console errors. Physical device not tested. |
| operations | partial | Build, production dependency separation and local persistence/replay verified. Capacity soak, migration/backup restore/rollback execution remain untested. |
| economics | partial | Local adapter runtime/RSS captured with loaded-host caveats. Actual provider/storage/notification cost is NA; no pilot bill exists. |
