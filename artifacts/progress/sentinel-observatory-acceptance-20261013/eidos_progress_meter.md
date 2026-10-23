# Sentinel Observatory proof progress

Scoped app acceptance: 100/100, calculated from the explicit accepted gates below. Overall Brain proof readiness: **unknown**; no project-wide gate audit was performed.

| Gate | Weight | Status | Receipt |
| --- | ---: | --- | --- |
| Secure dedicated credential and private deployment | 15 | passed | [secret-binding.json](../../../apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/secret-binding.json) |
| Actual accepted exact-model provider response | 25 | passed | [actual-sol-provider-receipt.json](../../../apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/actual-sol-provider-receipt.json) |
| Authoritative citations and durable cache | 15 | passed | [authoritative-input-comparison.json](../../../apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/authoritative-input-comparison.json) |
| Safe controls and provider-failure regressions | 15 | passed | [hosted-concurrency-cooldown.json](../../../apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/hosted-concurrency-cooldown.json) |
| Authenticated desktop/mobile rendered acceptance | 20 | passed | [browser-checks.json](../../../apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/browser-checks.json) |
| Reproducible proof package and journal | 10 | passed | [run_manifest.json](../../../apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/run_manifest.json) |

Precision, recall, false-positive rates and compression value remain unknown. Raw native controls remain visible in the run manifest.

## Proof Logic + Meaning

### Goal reached
**passed**: finite private Observatory credential activation, actual manual Sol execution, authoritative cache/control checks, and authenticated rendered desktop/mobile acceptance. This is the attached task's app gate.

### Previous state
The deployed app had no OpenAI key, no actual Sol receipt, and no rendered browser acceptance. Feed updates and graph activation replaced focused controls. Two Python checks relied on a tiny wall-clock window or unavailable Windows symlink privilege.

### Technical logic utilized
The server resolves a retained evidence ID, sanitizes bounded scalar metadata, reserves one durable attempt, calls exact `gpt-6.1-sol`, and validates the completed provider identity, usage and eligible structured ranking. Code rebuilds facts and citations from the authoritative record. The linked HTTP ID and canonical edge ID share the same cache. Polling and normal selection are read-only with respect to model jobs. The UI preserves the pause/resume button's DOM identity and restores graph, source and evidence-row focus by stable record keys after rendering.

### Math / scoring logic
`accepted = completed_provider_response AND exact_model AND valid_usage AND eligible_ranking AND authoritative_input_ids`.
`maximum_new_attempts = min(20, floor(0.50 / 0.05)) = 10 per UTC day`.
`observed_reservations = 2 * 0.05 = $0.10`; repeat requests for the first evidence created no second job.
Canonical parity uses `abs(actual-reference) <= 1e-7 + 1e-8 * abs(reference)` with exact categorical agreement: 954 frames and 41,022 comparisons passed. No Sentinel scoring or threshold changed. Precision, recall and false-positive rates are `null`: no labeled security truth set was evaluated.

### Philosophical meaning
Explanation before automation; restraint before alarm. Reproducibility is truth that can be revisited. A successful model call is separated from the truth of a security diagnosis.

### Why this is better
Before: credential and rendered acceptance were pending; focus vanished during feed refresh or graph activation. After: real provider IDs and usage exist, cached aliases reuse them, live duplicate/cooldown requests reject safely, and graph, record and source controls keep keyboard focus through selection or a real feed refresh. Initial failures, their passing followups, and the remaining OS skip stay visible.

### How this moves Eidos closer to the north-star goal
Eidos Brain is a self-monitoring streaming intelligence codec. It learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors its own internal state, and emits human-readable incident receipts. This milestone strengthens reproducible operation and evidence-linked explanations. It does not evaluate streaming learning quality, compression value, anomaly-preservation quality, or superiority to other detectors/compressors.

### Evidence
`actual-sol-provider-receipt.json`, `final-version-cache-readback.json`, `authoritative-input-comparison.json`, `hosted-concurrency-cooldown.json`, `hosted-negative-after-key.json`, `final-external-state.json`, `browser-checks.json`, both final UI exports, screenshots, `local-focus-followup/node-tests.xml`, Python logs, parity receipt, `deployment-sol-configured.json`, and `drive_manifest.json`.

### Remaining uncertainty
Ranking usefulness and detection quality remain unproven. No labeled-domain, compression, GPU, physical-phone or alternate-browser acceptance is claimed. Provider outages and daily-dollar exhaustion were tested through isolated fixtures rather than deliberately damaging credentials or spending the remaining budget. Real OS symlink creation is skipped on this Windows host (WinError 1314). Relative/directory rejection and symlink-metadata rejection before open pass. Mounted Drive readback does not independently establish cloud synchronization. Eido remains an evaluation candidate.
