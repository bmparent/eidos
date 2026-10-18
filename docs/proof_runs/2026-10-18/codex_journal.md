# Codex Journal — 2026-10-18

## What happened today

Reconciled PR79, PR77, owner-private Site version 16 and its immutable Node collector; isolated all work and repaired the first collector milestone.

## What was accomplished

Controlled retained evidence, socket repair, bounded alternates, explicit GET coverage, locked corpus, causal detector comparisons and browser replay acceptance.

## Tests and commands run

- `node --test --test-concurrency=1 apps/sentinel-observatory/live/*.test.mjs`
- `python -m unittest discover -s apps/sentinel-observatory/live -p 'test_*.py' -v`
- `node apps/sentinel-observatory/live/fault-harness/freeze.mjs`
- `node apps/sentinel-observatory/live/fault-harness/run.mjs --out artifacts/proof_runs/2026-10-18/observatory-m1/corpus`
- `node apps/sentinel-observatory/live/fault-harness/compare-detectors.mjs --out artifacts/proof_runs/2026-10-18/observatory-m1/detectors`
- `node apps/sentinel-observatory/live/verify_brain_parity.mjs`
- `npm --prefix apps/sentinel-lab ci --no-audit --no-fund`
- `npm --prefix apps/sentinel-lab run build`
- `node apps/sentinel-observatory/live/fault-harness/serve.mjs`

See command-specific receipts for exact outputs and pass/fail. Browser commands also accept --playwright-module and --browser for an existing runtime.

## Problems encountered

The first corpus run passed 70/72: real GET truncation/abort ordering and retained-status validation exposed two defects. Those failures remain in corpus-attempt-1. The repaired regression run and clean checkout pass 72/72. A parallel unit run exposed too-short local fixture deadlines on a heavily loaded Windows host plus a bad test source path; deadlines now allow real TLS setup and focused checks run serially. No production timeout was increased. Browser attempts retain the old loaded-server timing failure, a selector collision, missing favicon and cross-viewport rate-limit collision. Final browser checks use distinct logical targets and pass without exempting console errors. The first detector runner missed an import; its partial output and failure log remain. Packaging initially failed on npm shim resolution and GNU tar interpreting a Windows drive colon as a remote archive. The direct validated Node build and TAR_OPTIONS=--force-local recover the supported source workflow. Git checkout line-ending conversion changed the recorded capture hash; the exact Git blob was restored and is pinned by .gitattributes. No failing receipt was deleted.

## What changed

Collector/relay/presentation, test-only harness, portable Worker packaging and proof reports.

## What did not change

Reservoir dynamics, RLS, Sentinel thresholds, target-metrics-v2 operational policy, Lab core/research gates, original recorded capture, private audience and other Works releases were preserved. No billing, new integration, paid plan or notification was enabled.

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

## Artifacts generated

C:/Users/bmpar/codex-worktrees/observatory-fault-harness-20261018/artifacts/proof_runs/2026-10-18/observatory-m1

## Google Drive archive status

See drive_manifest.json. A checksum-verified mounted Drive copy does not independently establish remote synchronization.

## Thoughts on improvement

Use a new untouched qualification corpus after repairs; obtain independently scored explanations; implement tenant- and observer-scoped durable jobs before unattended pilots.

## Where to improve next

One PR for scoped ownership migration and observer-specific baselines, followed by durable reservation/lease/outbox work.

## Anything that stands out

The richer engine supplied no detector advantage on these timing generators. Publish that negative result and retain the simplest validated seam for a separately reviewed product-default decision.

## End-of-task summary

1. Files changed: collector, relay, presentation, Worker builder, test harness and proof docs.
2. Core behavior changed: no. Requested transport behavior repaired; GET fallback remains opt-in.
3. Tests: 151 Node; 31 Python passed/one privilege skip; two browser sizes; build/parity/trace checks.
4. Repo-root commands: above and per-run manifests.
5. Artifacts: measured cases, raw streams, scores, logs, manifests, screenshots, review packet.
6. Plain-language analysis: written.
7. Journal: written.
8. Drive: drive_manifest.json, including skipped files/reasons.
9. Limits: commercial matrix remains open.
10. Follow-ups not implemented: tenant scopes, scheduler, two observers, incidents/outbox and seven-day pilot.
11. Proof Logic + Meaning: written.
12. Math: deadline, median/MAD gate, event precision/recall, bootstrap.
13. Philosophy: restraint, honesty and revisitable truth.
14. Better state: successful partial stages retained, bounded alternatives and reproducible failure receipts.
15. North-star: strengthens reproducible measurement and readable explanations.
16. Evidence: named above.
17. Unproven: compression, generalization, Sol usefulness, commercial operation and physical devices.

Final full Node suite: 152 passed. Drive root discovery uses EIDOS_PROOF_DRIVE_DIR then EIDOS_ARTIFACT_ROOT. Configured root: G:\My Drive; final destination is Eidos_Brain_Proof_Phase/2026-10-18/observatory-m1 beneath the selected root. Actual copied files, skipped files, checksums and result are recorded in drive_manifest.json.

One fresh manual gpt-6.1-sol provider check used 736 input / 24 output tokens and its repeat was cached with the same provider receipt. Existing server credentials and hard budget were reused; no new key or automatic loop. This proves execution/cache behavior, not usefulness. Exact owner-private request was one retained evidence ID, evt_395a8240465a41e3a4684fd43c418152, sent to /api/interpret. See model-provider-check/manual-sol-verification.json.
