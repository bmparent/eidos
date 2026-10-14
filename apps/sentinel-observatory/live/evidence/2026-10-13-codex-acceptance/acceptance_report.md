# Sentinel Observatory — private Sol acceptance

Result: **passed within the finite private-app acceptance scope**.

App: https://sentinel-evidence-observatory.bmparent.chatgpt.site
Draft PR: https://github.com/bmparent/eidos/pull/73

## Exact accepted deployment

- Product runtime: `702d6f67f8c2b7bdb9bac83db7d007c878369b86`.
- Sites source: `bb2bc77c7d9eaef67d6ff4833781e3c8517771d8`.
- Deployment: `appgdep_6acec8be2d8081918792a2e7a6042ed1`; succeeded at environment revision **5**.
- Saved version: `appgprj_6ace6f7f4a5481918adb34a5cb366de3~appgver_6f82db7dbf20819192d8525457d0f205`.
- Worker SHA-256: `c57a2cf25de9c73d05535ec7619d79f6a0b79685042ae0c5043e0611791147c5`.
- Existing `OBSERVATORY_DB`, collector secrets and owner-only policy revision 1 were preserved. No external viewers were added. Applied migrations and the recorded capture were preserved.
- Final live JavaScript assets match the normalized inspected source. `/recorded` preserves the original recorded HTML; Cloudflare adds its challenge-platform script before the closing body. The original source file is unchanged, and no hosting protection was disabled. See `live-source-verification.json` and `source-invariants.json`.

## Actual provider and control acceptance

The key named **Sentinel Observatory** was created for **Personal / eidos_brain** through the connected OpenAI Developers encrypted-key workflow and bound only as server secret `OPENAI_API_KEY`. The picker did not appear; the owner selected the exact project through the connector's target list and an explicit chat choice. No API-key file or plaintext key log was produced.

The authenticated UI manually requested `evt_fe7cd21a06dd4dd694dce191b6d21ea1`. The server resolved canonical `edge_add9ea4f6d3c429d8c9a1202e50fe31d` and accepted a completed `gpt-6.1-sol` response: `resp_08c8f5f69e4799b9016acec99fd21087d291d1c2ab9830e892`, 390 input / 54 output / 444 total tokens, low reasoning, `model_ran:true`, `output_accepted:true`. All seven facts and both conditional explanations cite the retained canonical record. Its HTTP 301, 41.98 ms, verified TLS and 176/291 socket counters match the saved comparison receipt. No cause or attack was invented.

Repeating the same evidence in the UI and selecting its native alias returned the cached result. The job ledger remained at one job until a separately declared concurrency test intentionally created the second attempt. That test returned 200 for its completed provider response, **409** for the concurrent duplicate, and **429** for different evidence during cooldown. Two completed jobs reserve **$0.10** in total; this is not an invoice. The actual first cache gate passed without a second provider call.

With the real key configured, unknown IDs returned 404, imported IDs 400, extra client observations 400, and cross-origin requests 403. Configured polling and ordinary record selection left the model-job table empty before the first manual invocation. Provider errors, redirects, incomplete/malformed output, wrong model, timeout/cancellation, global request count and daily-dollar guards pass isolated regression tests with fixture credentials; live credentials were never deleted or altered for failure testing.

## Rendered browser acceptance and repairs

Authenticated owner Chromium browser checks passed at **1440x900** and **390x844**: connect/pause/resume, retained paused history after reload, measured evidence field, graph panning and endpoint selection, request/check/control filters, selected-evidence interpretation, cache and missing-key fallback, source age versus feed freshness, citations, normalized inspector, JSON downloads, dialog Escape and keyboard focus. Relevant application console warnings/errors were absent. The endpoint and sensor feed were left paused; the temporary viewport override was reset.

Fixed: sensor polling replaced the pause/resume button and dropped focus; it now retains DOM identity across real freshness updates. Collapsed mobile line breaks no longer join words/sentences. The deadline test now verifies watchdog scheduling/expiration deterministically. Symlink tests separate ordinary-path checks, a real-OS permission skip, and rejection of symlink metadata before opening.

## Validation and preserved failures

- 115 Node tests passed; Node JUnit XML retained.
- Python: 30 run, 29 passed, one explicit skip. Real OS symlink creation is skipped on this Windows host (WinError 1314). Relative/directory rejection and symlink-metadata rejection before open pass.
- Canonical numerical parity: 954 frames / 41,022 comparisons passed.
- Syntax, whitespace and exact compiled Worker ESM validation passed.
- The initial Python deadline and symlink failures are retained in the top-level local logs; the successful rerun is in `local-after-fixes/`.
- Packaging first selected unavailable WSL Bash (`HCS_E_CONNECTION_TIMEOUT`). The same supported Sites source helper succeeded using installed Git Bash and local archive handling. A short-lived source credential was renewed; no hosting audience or database migration was changed to work around this.
- The native table viewer truncated `result_json`; the full validated receipt was saved through authoritative cached API readback. The table ledger independently confirms job count, status, reservations and tokens.

Repo-root commands:

```text
node --test --test-reporter=junit apps/sentinel-observatory/live/collector.test.mjs apps/sentinel-observatory/live/brain.test.mjs apps/sentinel-observatory/live/browser-interpreter.test.mjs apps/sentinel-observatory/live/security-events.test.mjs apps/sentinel-observatory/live/hosted-sensor.test.mjs apps/sentinel-observatory/live/sol-interpreter.test.mjs
python -m unittest discover -s apps/sentinel-observatory/live -p 'test_*.py' -v
node apps/sentinel-observatory/live/verify_brain_parity.mjs
python apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/run_local_checks.py --out apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/local-after-fixes
node --check apps/sentinel-observatory/live/web/network-panel.mjs
git diff --check
```

Actual Sites source packaging used the supported `site-workflow.mjs` helper, exact Worker generation and `scripts/validate-artifact.mjs`. The source helper returned the pushed source SHA and unchanged archive used for private deployment. No manual PYTHONPATH changes were used.

## Evidence boundaries and metrics

The attached sensor records this Observatory's incoming requests, outbound HEAD checks and known control enforcement. Functional-check labels from retained records remain visible. The final UI export has 110 normalized source records with counts `{"collector": 30, "policy_event": 22, "request": 58}`. These counts are not confirmed attacks or false-positive labels. No new demo incidents or simulated live measurements were added. Raw, merged and calibrated security-quality metrics are unknown where no ground truth exists; see `run_manifest.json` for explicit `null` values and reasons.

The unchanged literal-IP path still pins the public address and verifies TLS. DNS-name Worker checks cannot pin subsequent DNS resolution. The native sensor does not capture target visitors, LAN packets or provider firewall events. Broader intrusion evidence requires authorized Suricata/Zeek or application logs. Eido remains an evaluation candidate, not an accepted loaded checkpoint.

## Artifact locations and archive

Local: `apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/`.
Journal and analysis: `docs/proof_runs/2026-10-13/`.
Scoped visual progress: `artifacts/progress/sentinel-observatory-acceptance-20261013/`.
Drive disposition and checksums: `drive_manifest.json`. The mirror copies to a new dated run directory when the configured mounted Drive is writable. Cloud synchronization is a separate, unverified state.

## Proof Logic + Meaning

### Goal reached
**passed**: finite private Observatory credential activation, actual manual Sol execution, authoritative cache/control checks, and authenticated rendered desktop/mobile acceptance. This is the attached task's app gate.

### Previous state
The deployed app had no OpenAI key, no actual Sol receipt, and no rendered browser acceptance. The live pause control was replaced by each feed update and lost keyboard focus. Two Python checks relied on a tiny wall-clock window or unavailable Windows symlink privilege.

### Technical logic utilized
The server resolves a retained evidence ID, sanitizes bounded scalar metadata, reserves one durable attempt, calls exact `gpt-6.1-sol`, and validates the completed provider identity, usage and eligible structured ranking. Code rebuilds facts and citations from the authoritative record. The linked HTTP ID and canonical edge ID share the same cache. Polling and normal selection are read-only with respect to model jobs. The UI fix preserves the pause/resume button's DOM identity while updating freshness text.

### Math / scoring logic
`accepted = completed_provider_response AND exact_model AND valid_usage AND eligible_ranking AND authoritative_input_ids`.
`maximum_new_attempts = min(20, floor(0.50 / 0.05)) = 10 per UTC day`.
`observed_reservations = 2 * 0.05 = $0.10`; repeat requests for the first evidence created no second job.
Canonical parity uses `abs(actual-reference) <= 1e-7 + 1e-8 * abs(reference)` with exact categorical agreement: 954 frames and 41,022 comparisons passed. No Sentinel scoring or threshold changed. Precision, recall and false-positive rates are `null`: no labeled security truth set was evaluated.

### Philosophical meaning
Explanation before automation; restraint before alarm. Reproducibility is truth that can be revisited. A successful model call is separated from the truth of a security diagnosis.

### Why this is better
Before: credential and rendered acceptance were pending; focus vanished during feed refresh. After: real provider IDs and usage exist, cached aliases reuse them, live duplicate/cooldown requests reject safely, and the focused control survives an observed real feed update. Initial test failures and the remaining OS skip remain visible.

### How this moves Eidos closer to the north-star goal
Eidos Brain is a self-monitoring streaming intelligence codec. It learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors its own internal state, and emits human-readable incident receipts. This milestone strengthens reproducible operation and evidence-linked explanations. It does not evaluate streaming learning quality, compression value, anomaly-preservation quality, or superiority to other detectors/compressors.

### Evidence
`actual-sol-provider-receipt.json`, `authoritative-input-comparison.json`, `hosted-concurrency-cooldown.json`, `hosted-negative-after-key.json`, `final-external-state.json`, `browser-checks.json`, both final UI exports, screenshots, `local-after-fixes/node-tests.xml`, Python logs, parity receipt, `deployment-sol-configured.json`, and `drive_manifest.json`.

### Remaining uncertainty
Ranking usefulness and detection quality remain unproven. No labeled-domain, compression, GPU, physical-phone or alternate-browser acceptance is claimed. Provider outages and daily-dollar exhaustion were tested through isolated fixtures rather than deliberately damaging credentials or spending the remaining budget. Real OS symlink creation is skipped on this Windows host (WinError 1314). Relative/directory rejection and symlink-metadata rejection before open pass. Mounted Drive readback does not independently establish cloud synchronization. Eido remains an evaluation candidate.


## Recommended next step

Review conditional ranking usefulness against an explicitly labeled, authorized evidence set before making any detector-quality or broader Brain value claim. No additional feature or scientific expansion was implemented.


## Final Google Drive disposition

Artifacts copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-10-13\sentinel-observatory-sol-acceptance-20261013` with SHA-256 readback; cloud synchronization remains unverified.
