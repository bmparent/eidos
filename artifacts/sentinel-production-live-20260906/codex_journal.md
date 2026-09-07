# Codex Journal — September 6–7, 2026

## What happened today
# Sentinel Lab production audit — September 6–7, 2026

Status: **VERIFIED_ENGINEERING_WORKFLOW**. Live job **rd-8a14916b3ea7-696b7427** reached **COMPLETED_ENGINEERING**. Diagnostic: 9ca66d3f-2ed2-427c-940d-0eb971a94b01. Authenticated reconnect and full snapshot verification closed on September 7. The September 6 reload restored the same job and lock; after that tab closed, September 7 reconnect used the saved job ID in a fresh tab.

## Released changes
PR #43 (https://github.com/bmparent/eidos/pull/43) released shared SQL admission, lease recovery, stable client retry intent, missing-metric handling and browser regression checks. PR #44 (https://github.com/bmparent/eidos/pull/44) released bounded authenticated verification of every immutable snapshot artifact. PR #45 (https://github.com/bmparent/eidos/pull/45) published the documentation-only closeout. Production is READY at 771076ab2b2c0487d9abe4d19522cde7fa6e6b3a, deployment dpl_E4h7W4qLAgn5sd3ptPh4ihzAjPYy. Execution source was 31b1a895182b35a1c376a3e532ade9d3107aa13d; authenticated verification used 9f4fc52ef79d36b9e268e124413afc1ae2f1f857. The engine was not rerun for either follow-up.

## Live result
Pinned CICIDS2017 version 3, WebAttacks-Thursday-no-metadata.parquet; standard CPU profile, seed 0, 1,000 rows. The SHA-256, lock and four source hashes match. 200 calibration + 600 evaluation rows entered the label-free stream; 200 holdout rows stayed excluded. Seven original downloads succeeded via the production UI. All 25 declared immutable hashes and sizes match. The same job reconnected after reload; the verification receipt confirms a resumed retrieval session ended stopped.

The displayed result matched 600/600 scored predictions: TP 0, FP 301, TN 299, FN 0; FPR 50.17%. Recall, AUC, average precision and detection delay are unavailable on this all-benign slice. The observatory switched from its recorded example to this job's diagnostics.

## Validation and reproducibility
46 app tests and 26 runner tests passed. Merge-commit CI, TypeScript and production build passed. Controlled browser tests covered launch/progress/failure/download/reload, missing metrics, keyboard access, mobile overflow and reduced motion. Remote admission testing used 12 independent clients without allocating production compute. New receipt/status downloads still reject unauthenticated requests with HTTP 401.

Repo-root commands:

`npm.cmd test --prefix apps/sentinel-lab`

`npm.cmd run lint --prefix apps/sentinel-lab`

`npm.cmd run build --prefix apps/sentinel-lab`

`npm.cmd run qa:browser --prefix apps/sentinel-lab` (controlled suite in prior audit)

`node apps/sentinel-lab/scripts/verify-live-evidence.mjs artifacts/sentinel-production-live-20260906`

## Changed files and artifacts
Implementation and tests are listed in PRs #43/#44. New follow-up files: lib/experiments/artifact-verifier.ts, sandbox.ts, tests/artifact-verifier.test.ts, tests/sandbox-lifecycle.test.ts, scripts/verify-live-evidence.mjs and docs/audit-2026-09-06-live.md under apps/sentinel-lab. Journals and plain-language analysis are under docs/proof_runs/2026-09-06. Core model behavior, profiles, thresholds, labels and splits did not change. The local evidence directory is C:\Users\bmpar\codex-worktrees\sentinel-audit-20260906\artifacts\sentinel-production-live-20260906. Earlier audit artifacts, unrelated dirty worktrees and Eidos Works were preserved.

## Drive archive
See drive_manifest.json for the configured Drive root, exact mirror path, file list and verified hashes. The mirror is separate from the earlier audit package. No credentials or raw dataset/held-out input files are included.

## Proof Logic + Meaning

### Goal reached
Authenticated launch, completion, reload/reconnect, full immutable verification and stopped-session retrieval passed for one bounded production job. Shared admission and retries passed controlled tests against an actual remote validation database.

### Previous state
The workflow was supported by local execution and mocks but had no authenticated successful production job. Advisory list/count admission could race and retries lacked durable shared identity.

### Technical logic utilized
A primary SQL write transaction conditionally admits a reservation only below capacity; unique retry identity returns the same job. Leases recover abandoned reservations and fence stale allocators. Source discovery verifies a pinned Git commit before the launcher starts. Normalization uses calibration rows only, labels stay outside engine input, predictions freeze before evaluation, and held-out rows never enter the engine. Snapshot retrieval is explicit and cleanup precedes the verification receipt.

### Math / scoring logic
Admission invariant: occupied reservations <= configured capacity; 12 competing clients at capacity 1 produced 1 admission and 11 rejections, with 0 duplicate jobs on 12 retries. Integrity: SHA256(actual bytes) and byte count equal each manifest entry. Evaluation coverage = 600/600, with score, threshold and alert correspondence for every frozen prediction. FPR = FP/(FP+TN) = 301/600 = 50.1667%; TP=0, FP=301, TN=299, FN=0. Recall and AUC are null because positive count is zero. Raw row metrics remain visible; event-merged/deduplicated/calibrated metrics are NA because this task did not compute those categories. Scientific readiness score is null because no scientific gates were evaluated.

### Philosophical meaning
Reproducibility is truth that can be revisited; honest accounting comes before optimization. A successful execution cannot erase false positives.

### Why this is better
The evidence now links an actual production allocation to its source, dataset, frozen predictions, evaluation and internal diagnostics. Shared admission closes a demonstrated race and stable retries prevent accidental duplicate compute.

### North-star connection
This strengthens reproducible operation and inspection of internal state in the self-monitoring streaming intelligence codec. It does not establish compression advantage, useful anomaly detection or held-out generalization.

### Evidence
job-receipt.json, source_receipt.json, run_manifest.json, dataset_receipt.json, metrics.json, engine_trace.jsonl, evaluation_trace.jsonl, engine_diagnostics.json, live-evidence-verification.json, production-deployment.json, merge-ci.json, shared-admission-verification.json and browser-qa.json. artifact_verification.json records all 25 hashes and stopped provider status.

### Remaining uncertainty
- Only benign evaluation examples: 301 false positives / 600 benign rows; recall and ROC AUC cannot be estimated.
- Zero scientific proof gates advanced; overall scientific readiness is unknown. Held-out data remains excluded.
- Observed live stages were runtime bootstrapping and completed engineering. Intermediate stages passed between observations.
- Active-download preservation, retryable outages, crashes and expiry were verified with controlled regressions, not deliberately induced in production.
- Shared admission covers experiment jobs; a global cap including all short retrieval VMs and the optional external runner backend remains outside this qualification.
- Provider snapshots have seven-day expiry. Local and Drive receipts preserve evidence beyond that lifecycle.
- The optional browser CI workflow patch was not applied because the GitHub credential lacked workflow scope; the rendered suite passed locally.
- Automatic approval review blocked optional deletion of three QA browser profiles; they were left in place.


## Next step
Keep detection-quality and held-out research gates separate from this engineering success. No additional research experiment was implemented.


## End-of-task summary
Files changed: PRs #43/#44 and closeout evidence. Core behavior unchanged. Tests and commands, artifacts, local path, analysis, mathematical logic, philosophical meaning, evidence and limits are above. Drive status is in drive_manifest.json. No unrelated feature or proof expansion was implemented.
