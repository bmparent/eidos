

<!-- sentinel-observatory-sol-acceptance-20261013 -->
# Codex Journal — 2026-10-13

## What happened today
Continued the existing private Observatory and draft PR #73. Completed dedicated key activation, actual Sol/cache/control checks and authenticated desktop/mobile acceptance.

## What was accomplished
Saved real provider and authoritative evidence receipts; verified safe concurrency/cooldown/rejections; repaired feed-control, graph-node, source and evidence-row focus plus mobile spacing; preserved numerical behavior, recorded capture, owner-only access and applied migrations.

## Tests and commands run
115 Node tests passed. Python ran 30: 29 passed and one explicit Windows symlink skip. Numerical parity passed 954 frames / 41,022 comparisons. Rendered 1440x900 and 390x844 checks, Worker ESM validation, syntax and whitespace passed.

```text
node --test --test-reporter=junit apps/sentinel-observatory/live/collector.test.mjs apps/sentinel-observatory/live/brain.test.mjs apps/sentinel-observatory/live/browser-interpreter.test.mjs apps/sentinel-observatory/live/security-events.test.mjs apps/sentinel-observatory/live/hosted-sensor.test.mjs apps/sentinel-observatory/live/sol-interpreter.test.mjs
python -m unittest discover -s apps/sentinel-observatory/live -p 'test_*.py' -v
node apps/sentinel-observatory/live/verify_brain_parity.mjs
python apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/run_local_checks.py --out apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/local-after-fixes
python apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/run_local_checks.py --out apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/local-focus-followup
node --check apps/sentinel-observatory/live/web/network-panel.mjs
git diff --check
```

## Problems encountered
The secure picker did not surface; the owner explicitly selected Personal / eidos_brain from connector-discovered targets. Initial Python failures were retained and repaired as test portability issues. WSL packaging failed; installed Git Bash completed the supported packaging helper. The native table viewer truncated a result JSON value; authoritative cached API readback supplied the complete sanitized receipt. Browser selector waits sometimes expired before asynchronous output completed; the existing operation was inspected rather than reissued as a new provider attempt. The final service readback initially omitted the required Bearer prefix and Origin header; platform authentication and the application same-origin control rejected those requests. Correctly formed service access returned the existing cached receipt; no security settings changed.

## What changed
`web/network-panel.mjs`, `web/index.html`, two local boundary-test files, scoped `.gitattributes` and `.gitignore` receipt policies, audit/handoff/README, this run's receipts and dated proof documentation/progress artifacts. Runtime `OPENAI_API_KEY` was added as a secret; environment revision is 5. The narrow log exception keeps every manifested receipt available after a clean checkout.

## What did not change
Core numerical Sentinel behavior, RLS/reservoir/scoring/threshold policy, collector and Sol algorithms, cost controls, applied migrations, owner-only access, recorded evidence and protected relay configuration were untouched. Eido was not loaded or accepted.

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


## Artifacts generated
This run folder includes config, run manifest, provider/deployment/access receipts, original and corrected test logs, Node XML, parity result, browser checks/screenshots/exports, analysis, journal, ledgers and visual progress. The expected-omission reasons are explicit in run_manifest.json.

## Google Drive archive status
See drive_manifest.json for the actual root, dated run directory, copied files, skipped files, checksums and copy result. Only mounted readback is verified; remote cloud synchronization is unknown.

## Thoughts on improvement
Keep application acceptance separate from scientific detection and compression proof. Keep the one OS-permission skip visible.

## Where to improve next
A narrowly scoped, labeled usefulness review of the conditional rankings.

## Anything that stands out
Zero model jobs existed after configured polling and ordinary selection. The first cached UI repeat kept one job. The separate real concurrency test produced a second completed job and safe 409/429 responses. Two jobs reserve $0.10; actual billing is unknown.

## End-of-task summary
1. Files changed: narrow UI/test fixes, audit/handoff/README, receipt byte policy, proof artifacts and dated docs.
2. Core behavior changed: no.
3. Tests added/skipped: split symlink coverage plus metadata rejection; one real-OS symlink skip (WinError 1314).
4. Repo-root commands run: listed above; exact argv/logs in local-checks.json.
5. Artifacts generated: manifests, logs/XML, provider/cache/control/deployment receipts, screenshots/exports, analysis, journal and progress/logic ledgers.
6. Local artifact folder: apps/sentinel-observatory/live/evidence/2026-10-13-codex-acceptance/.
7. Drive status: exact final copy result and file list in drive_manifest.json; remote sync unknown.
8. Plain-language analysis: written locally and included in the mirror set.
9. Codex journal: written locally and included in the mirror set.
10. Known limitations: one OS symlink skip; no physical-phone, alternate-browser, detector-quality or compression-value claim.
11. Follow-up not implemented: labeled ranking usefulness review and broader scientific proof.
12. Proof Logic + Meaning: written in audit, analysis, journal and logic ledger.
13. Specific math: acceptance conjunction, per-evidence deduplication, UTC reservation cap and canonical parity tolerance.
14. Philosophical meaning: explanation before automation; restraint before alarm; revisitable truth.
15. Why better: real execution/caching receipts and rendered acceptance replace previously pending gates; focus survives polling.
16. North-star contribution: reproducible operation and evidence-linked explanations.
17. Evidence: provider receipt, authoritative input comparison, browser-checks, exports/screenshots, local logs/parity, final-external-state and deployment receipt.
18. Remaining uncertainty: ranking usefulness, security ground truth, compression benefit, actual billed spend and remote cloud synchronization.


## Final Google Drive disposition

Artifacts copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-10-13\sentinel-observatory-sol-acceptance-20261013` with SHA-256 readback; cloud synchronization remains unverified.
