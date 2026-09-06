# Codex Journal — 2026-09-06

## What happened today
Continued the Sentinel Lab production audit from PR #42 in an isolated worktree. Production initially matched 4d17d8bac43d2beb235f4abcc2adbf271781be51. Original dirty worktrees, engine artifacts, Eidos Works code and existing Drive files were preserved.

## What was accomplished
Strict shared Sandbox job admission, stable retry identity, stale-owner fencing, conservative reservation recovery, safe missing metrics and explicit retrieval cleanup failures. Regression checks passed locally and in Linux CI. Browser fixtures cover the operator lifecycle and accessibility. The authenticated live run remains blocked on the existing operator credential.

## Tests and commands run
From the repository root:
- `npm ci --prefix apps/sentinel-lab --no-audit --no-fund` — passed.
- `npm test --prefix apps/sentinel-lab` — final pass: 43 tests. Initial failures are retained in app-tests.log; final results are in app-tests-final.log.
- `npm run lint --prefix apps/sentinel-lab` — type issues repaired; final TypeScript validation passed inside the production build and Linux CI.
- `npm run build --prefix apps/sentinel-lab` — passed.
- `node apps/sentinel-lab/node_modules/tsx/dist/cli.mjs --tsconfig apps/sentinel-lab/tsconfig.json apps/sentinel-lab/scripts/verify-admission.ts` — passed against the existing separate validation database, with its pre-existing environment loaded privately. Twelve competing clients, one admission, eleven capacity rejections; twelve retries, zero duplicate jobs. No Sandbox allocation.
- `npm run qa:browser --prefix apps/sentinel-lab` — passed with the installed Chromium headless executable specified by CHROME_BIN. Desktop 1440x1000, mobile 390x844; screenshots, console checks, keyboard, reduced motion and controlled lifecycle fixtures.
- `node apps/sentinel-lab/scripts/verify-production-access.mjs` — public settings lock passed; dispatch, status and artifact access returned 401 without credentials. This is not successful engine execution.
- `gh run view 34066469815 --repo bmparent/eidos --log` — captured passing app and runner CI. Runner: Python 3.14.7, Torch 2.14.0+cpu, 26 tests, both supported bounded CPU profiles. Existing CI uses its established app/service working directories. A repo-root workflow improvement was preserved as an optional patch because the OAuth credential lacks workflow scope.

## Problems encountered
Windows source receipt checks assumed slash-prefixed paths; native absolute-path validation now supports the real local fixture without weakening Linux checks. Database fixture cleanup needed bounded retry. Browser selectors used rendered capitalization instead of actual text and were corrected. Three owned temporary browser profiles stayed locked; later cleanup was rejected by automatic approval review as blocked by policy. No further cleanup was attempted.

The old Vercel API release credential returned 403/not_found. The connected read tools and authenticated dashboard remain available. EIDOS_OPERATOR_TOKEN exists as a non-revealable production secret and is absent from the available local environment. No secret was exposed or replaced. Local Python has Torch 2.6.0+cpu; production-compatible runner validation used the passing fresh CI environment instead of altering the host runtime.

## What changed
Narrow control-plane admission, retry recovery, result rendering, source validation, regression scripts and documentation. The release PR is https://github.com/bmparent/eidos/pull/43. Exact production identity is recorded after deployment in the local release receipt.

## What did not change
Core model behavior was untouched: reservoir/RLS, profiles, thresholds, anomaly policy, labels, ordering, normalization policy, splits, compression and held-out boundaries. No concurrency increase, upgraded plan or full-capacity profile. Zero proof gates advanced.

## Proof Logic + Meaning
- Goal reached: shared-admission engineering gate passed; full authenticated production workflow remains access-blocked.
- Previous state: independent requests could count the same spare capacity, and a retry could launch a distinct job.
- Technical logic: primary write transaction for count and conditional insertion; unique retry hash and job identity; owner/phase/deadline compare-and-update before allocation; provider confirmation before releasing an uncertain allocation.
- Math: admit iff occupied < capacity, with occupied=count(RESERVED+ALLOCATING+RUNNING), inside the same transaction. Repeated key implies the same job. The 20/60/20 causal partition and prediction freeze remain fixed. Proof gates advanced=0.
- Philosophical meaning: restraint before execution; reproducibility is truth that can be revisited.
- Why better: enforceable job admission and recoverable retry intent replace advisory counting and ambiguous retries; unknown metrics no longer crash the result view.
- North-star connection: strengthens reproducible operation and understandable receipts for the self-monitoring streaming intelligence codec; does not establish detector quality.
- Evidence: shared-admission-verification.json, app-tests-final.log, ci-34066469815.log, browser/browser-qa.json, production-access-verification.json, audit-receipt.json.
- Remaining uncertainty: live allocation, full-engine completion, stopped-session retrieval and immutable artifact hashes require operator access. Controlled fixtures are not live provider proof. External-runner multi-instance admission and global quota over retrieval VMs are not qualified.

The previous local pinned slice had FP=301 and TN=299 over 600 benign evaluation examples: FPR=301/(301+299)=50.17%. There were no positive examples, so recall and ROC AUC were undefined. No new detection-quality measurement was made in this task.

## Artifacts generated
Repo-local: artifacts/sentinel-production-audit-20260906/. Committed compact audit: apps/sentinel-lab/docs/audit-2026-09-06-followup.json and .md. Local receipts include logs, pinned settings, remote admission results, browser screenshots, release identity, manifest, analysis and this journal. Actual live job artifacts are absent because authentication is blocked; their absence is explicit.

## Google Drive archive status
Configured root G:/My Drive is available. The final artifact package is mirrored to Eidos_Brain_Proof_Phase/2026-09-06/sentinel-production-audit-20260906/ with file hashes. Final copied files, skipped files and verification outcome are in drive_manifest.json; a missing or failed receipt must not be interpreted as a successful copy. Existing Drive files are preserved.

## Thoughts on improvement
Obtain operator access through the existing password field, run the pinned 1,000-row experiment once, and verify source, prediction and artifact receipts. Do not infer execution quality from settings readiness.

## Where to improve next
The remaining acceptance check is the authenticated live run. External-runner admission and optional CI artifact upload are separate follow-ups, not implemented here.

## Anything that stands out
Successful orchestration does not establish a useful detector. False positives remain visible and the held-out protocol stays closed.

## End-of-task summary
1. Files changed: control-plane admission, retries, metric formatting, source path validation, tests, audit scripts and docs in PR #43.
2. Core behavior changed: no.
3. Tests: 43 app; 26 runner in CI; both CPU profile fixtures; remote admission; rendered browser fixtures.
4. Repo-root commands: recorded above. Existing CI service-relative commands are retained because workflow updates are unauthorized by the credential.
5. Artifacts: logs, JSON receipts, screenshots, manifests and release identity.
6. Plain-language analysis: written alongside this journal.
7. Journal: this file.
8. Drive: see verified final drive_manifest.json.
9. Limitations: operator-authenticated production run and live artifact hashes unavailable.
10. Follow-ups: live acceptance; optional CI workflow patch and external-runner admission are unimplemented.
11. Proof Logic + Meaning: written.
12. Math/logic: atomic capacity inequality, unique retry identity, fenced state transition; fixed causal split.
13. Philosophy: restraint and revisitable evidence.
14. Improvement: enforced shared capacity and meaningful retry recovery.
15. North-star: reproducible execution and explainable receipts.
16. Evidence files: listed above.
17. Unproven: detection utility, held-out generalization and authenticated production lifecycle.
