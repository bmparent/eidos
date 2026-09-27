# Production workflow audit follow-up

Baseline inspected: `4d17d8bac43d2beb235f4abcc2adbf271781be51` (PR #42). Vercel production was READY on that exact commit at the start of this audit. Work uses an isolated checkout and preserves the original dirty Eidos checkout and all Eidos Works work.

## Changes

- Strict shared admission on the existing libSQL backend, with atomic capacity reservation, deadline and owner fencing, conservative provider reconciliation, and durable retry identities. No plan, concurrency or engine-profile expansion.
- Browser launch intent persists before dispatch; reloading or retrying a lost response reuses its key and settings. Operator secrets remain only in page memory. Success responses include the diagnostic ID.
- Missing, undefined and non-finite metric values render as N/A. Artifact errors retain the useful server explanation. Retrieval cleanup failures are surfaced rather than silently reported as success.
- Source-receipt validation checks native absolute paths, allowing the same real Git fixture tests on Windows and Linux while retaining exact commit and file verification.

## Validation evidence

The initial local pass revealed Windows-only fixture path validation and transient database-file cleanup issues. Both are repaired and retained in the initial log. The following pass completed all 43 app tests (16 JavaScript + 27 TypeScript).

The separate remote validation database test used 12 independent libSQL clients: one admission, 11 capacity rejections, and 12 retries of the same job without another allocation. Abandoned-reservation recovery and stale-owner fencing passed. An uncertain allocation remained occupied past its deadline until explicit reconciliation. No Sandbox compute was allocated by this controlled test.

Sanitized receipts, final checks, browser evidence, deployment identity and access limitations are collected in the repo-local `artifacts/sentinel-production-audit-20260906/` folder and the final audit JSON. Release and live-execution results must be read from those final receipts, not inferred from the checks above.

## Access findings

GitHub CLI and the connected Vercel read tools work. The old local release API credential was rejected by Vercel (`403`, `not_found`, `User not found`). The authenticated Vercel dashboard confirms `EIDOS_OPERATOR_TOKEN` exists in Production as a non-revealable secret. The existing local release environment contains the database configuration but no Lab operator credential. Credentials were not printed, replaced or committed.

## Scientific interpretation

No new detection-quality result is claimed here. The previous local pinned slice had 600 benign evaluation examples, no positive examples, 301 false positives and 299 true negatives: `FPR = 301 / (301 + 299) = 0.5016667`. Recall and ROC AUC are undefined. Successful execution does not establish useful detection or generalization.

## Proof Logic + Meaning

The gate pursued is reliable, authenticated production experiment execution and retrieval; shared admission is a supporting engineering gate. The old release repaired startup and lifecycle handling, but capacity counting remained advisory and a lost response could trigger a distinct job. Atomic `count + conditional insert` enforces `occupied <= configured capacity` among admitted jobs. A stable request key maps retries to one job, while the owner/phase/deadline comparison fences stale allocators. Runtime metrics now preserve unknown values rather than crashing the results UI.

This is restraint before execution and truth that can be revisited. It strengthens reproducible operation and understandable receipts for the self-monitoring streaming intelligence codec. The frozen 20/60/20 split, calibration-only preprocessing, exclusion of labels from engine inputs, prediction freeze before evaluation and excluded holdout remain unchanged. `gates_advanced = 0` throughout.

The source tests, lifecycle tests and remote admission receipt support only their stated engineering claims. An authenticated live job and downloaded artifact checks are required to establish the production workflow. Snapshot reads, temporary provider failures and terminal receipts have controlled regression coverage; that coverage is not a substitute for observing those operations against the live provider. Optional external-runner distributed concurrency and held-out scientific proof remain unqualified.

## Authenticated live follow-up at 23:49 UTC

Brent supplied operator access in the production password field. Job rd-8a14916b3ea7-696b7427 completed on release 31b1a895182b35a1c376a3e532ade9d3107aa13d. Seven authenticated downloads were saved under artifacts/sentinel-production-live-20260906. Their exposed immutable hashes, four Git source hashes, locked dataset split and 600 frozen-prediction correspondences passed `node apps/sentinel-lab/scripts/verify-live-evidence.mjs`.

A read-only authenticated receipt now verifies all declared immutable files inside a completed snapshot, with path and byte limits and provider status read after cleanup. No core engine, thresholds, labels or split changed. App tests passed: 46 total. Type checking and the production build passed. Existing runner checks are unchanged and run in release CI. Full live verification of the new receipt and reload/reconnect remains pending deployment at this journal checkpoint.

### Proof Logic + Meaning

The execution gate passed where it was previously access-blocked. SHA-256 and byte-count equality bind evidence to files; 600/600 prediction correspondences bind evaluation to the frozen trace. FPR = 301/600 = 50.17%; recall and AUC are null because no positive examples exist. This is reproducibility and honest accounting, not a useful-detector claim. It strengthens reproducible execution and inspection of internal state in Eidos's streaming codec north star. Held-out data remains excluded and gates advanced remain zero. See apps/sentinel-lab/docs/audit-2026-09-06-live.md for detailed logic and limits. The live Drive supplement will be copied after final retrieval; the earlier audit mirror remains preserved.
