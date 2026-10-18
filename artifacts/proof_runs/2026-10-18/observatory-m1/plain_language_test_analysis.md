# Observatory first milestone — measured engineering result

Runtime code 4a9bd8b9da652a1d6587412316e18d97d35d76f8; proof assembly commit 4f680d0afb299195ae59283679005e81a8f37df9. Original continuation c184ae06819e7ad326f141f378b2d83f9b91152c, stacked on PR77.

Controlled corpus: 72/72 across 12 families. Clean checkout: 151 Node tests passed. Python: 31 passed, one Windows symlink privilege skip. Numerical parity: 954 frames / 41022 field comparisons. Next build and production fixture separation passed. Local browser: 2 viewports. Fresh hosted checks: True.

| Method | TP / FP / FN | Precision | Recall | False incidents / benign target-day | Mean delay s |
|---|---:|---:|---:|---:|---:|
| gate | 24 / 12 / 2 | 0.667 | 0.923 | 1.414 | 0.0 |
| sentinel_plus_gate | 24 / 12 / 2 | 0.667 | 0.923 | 1.414 | 0.0 |
| ewma | 26 / 12 / 0 | 0.684 | 1.000 | 1.414 | 69.2 |
| cusum | 26 / 14 / 0 | 0.650 | 1.000 | 1.650 | 69.2 |

All four methods retain failures, false incidents, delays and stream-cluster CIs in comparison.json. Sentinel plus gate matched the gate alone. These results do not justify superiority or a production algorithm change. Runtime measurements came from a loaded Windows host and are not cloud cost estimates.

## Failures retained

The first corpus run passed 70/72: real GET truncation/abort ordering and retained-status validation exposed two defects. Those failures remain in corpus-attempt-1. The repaired regression run and clean checkout pass 72/72. A parallel unit run exposed too-short local fixture deadlines on a heavily loaded Windows host plus a bad test source path; deadlines now allow real TLS setup and focused checks run serially. No production timeout was increased. Browser attempts retain the old loaded-server timing failure, a selector collision, missing favicon and cross-viewport rate-limit collision. Final browser checks use distinct logical targets and pass without exempting console errors. The first detector runner missed an import; its partial output and failure log remain. Packaging initially failed on npm shim resolution and GNU tar interpreting a Windows drive colon as a remote archive. The direct validated Node build and TAR_OPTIONS=--force-local recover the supported source workflow. Git checkout line-ending conversion changed the recorded capture hash; the exact Git blob was restored and is pinned by .gitattributes. No failing receipt was deleted.

## Commercial gates

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

## Proof Logic + Meaning

### Goal reached
First collector/fault-harness engineering milestone: locally passed (72/72); private deployment status: succeeded. Commercial qualification remains partial. This does not advance Lab G0–G6.

### Previous state
PR79 had known TLS cleanup loss, incomplete later-failure evidence, one-address sampling, ambiguous HEAD 405/501 coverage and no locked controlled corpus. Version 16 remained foreground-only.

### Technical logic utilized
Capture negotiated TLS and certificate/network metadata before socket cleanup, preserve partial stages, validate all DNS answers before numeric pinning, try at most three validated addresses per method/hop under one shared deadline, and retain every attempted outcome separately from the selected HEAD response. GET fallback is explicit and keeps the original HEAD status. The existing optimistic SQLite/D1 checkpoint write and duplicate evidence identity support retain → explain → export/replay. Unknown TLS/peer values remain null. No private fixture allowance enters production.

### Math / scoring logic
Total deadline D = start + min(requested_timeout, 12000 ms). Attempt budget B_i = (D - now) / remaining_addresses; attempt deadline never exceeds D. Each successful TCP/TLS timing is measured from its own socket events.

The unchanged operational gate uses preceding values: m = median(history); MAD = median(abs(x_i - m)); T = max(5 ms, 0.25 abs(m), 6 * 1.4826 * MAD). gate = abs(x_t - m) > T. Sentinel operational flag = raw_Sentinel_change AND gate. No current label enters either rule.

precision = TP/(TP+FP); recall = TP/(TP+FN); false incidents per benign target-day = FP/(benign scored frames * 300/86400). Whole streams, not correlated frames, are bootstrap units (2000 resamples). Warmup/reset and missing frames are unscored; reported missing_fraction combines these and is not collector coverage. Recovery numbers are alarm-end scoring, not a verified incident state machine. Undefined metrics are null.

### Philosophical meaning
Restraint before alarm; honesty before optimization; reproducibility is truth that can be revisited. The comparison publishes false incidents and a negative result.

### Why this is better
Actual later-stage failures retain the successful stages needed for the next check. One failed address no longer hides a later measured response. Operators can distinguish unsupported HEAD, HTTP 408, connection timeout, incomplete redirects and missing collector evidence, with retained receipts and visible limits.

### How this moves Eidos closer to the north-star goal
The north-star claim is a self-monitoring streaming intelligence codec that learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors internal state and emits readable incident receipts. This milestone strengthens reproducible measurement and explanation. It does not prove compression, full-engine generalization, security detection or unattended incident operation.

### Evidence
artifacts/proof_runs/2026-10-18/observatory-m1/clean-corpus/run_manifest.json and raw_cases.jsonl; artifacts/proof_runs/2026-10-18/observatory-m1/detectors-final/comparison.json, manifest.json, streams.jsonl and stream_scores.jsonl; clean-node-tests.txt; python-tests-final.txt; brain-parity.txt; production-trace.json; artifacts/proof_runs/2026-10-18/observatory-m1/browser-final/browser-receipts.json and screenshots; hosted/receipt.json when present; release.json; drive_manifest.json.

### Remaining uncertainty
Synthetic faults are not live pilot observations. The original corpus evaluation cases are consumed regression cases after repair. Model/provider simulations are not genuine Sol usefulness evidence. Two tenants, independent observers, durable scheduling, outbox delivery, backup/restore, load capacity, seven-day coverage, measured commercial cost and physical-device behavior remain unproven. No Sentinel superiority was observed.

Final full Node suite including capacity release: 152 passed; the separate clean-checkout suite passed 151.
