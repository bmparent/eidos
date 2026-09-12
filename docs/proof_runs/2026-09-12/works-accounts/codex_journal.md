# Codex Journal — 2026-09-12

## What happened today
Reconciled the held Works account backend with the current portable editor in an isolated worktree.

## What was accomplished
Controlled owner, session, recovery, image validation, concurrent-save, account-switch and database-binding checks passed. Source/vendor parity verified.

## Tests and commands run
See plain_language_test_analysis.md and captured logs for commands and results.

## Problems encountered
External dependency junction prevents default Turbopack build; webpack passed. Hosted provider acceptance is still blocked. Earlier browser harness navigation/selector failures were corrected and retained externally.

## What changed
Only Works adapter, portable vendor sources, additive migration/configuration, tests and this evidence package. Frontend reconciliation is a separate reviewable branch.

## What did not change
No research engine, reservoir/RLS, thresholds, incidents, compression, benchmark inputs or existing proof archives changed.

## Proof Logic + Meaning

### Goal reached
Partial: reproducible Works account reconciliation passes controlled owner-isolation, password/session, delayed-response and database-binding checks. Hosted acceptance remains blocked.

### Previous state
The held account branches could not safely be applied over the newer editor. Advanced document storage and delayed responses needed explicit acceptance boundaries.

### Technical logic utilized
Reused existing identity and session handlers. Owner equality and expected-head compare-and-swap protect cloud writes; transactional quotas limit admission. An account-change nonce invalidates pending UI work across tabs, while the server remains the authorization authority. Strict authoring schemas and real image decoding protect stored assets. Exported portable sources are byte-compared with their vendor copies.

### Math / scoring logic
Authorized save = authenticated owner equals stored owner AND expected owner equals session owner AND expected revision equals current head. Two concurrent writes against one head must produce one success and one conflict (200,409). No research readiness percentage is inferred from these software tests.

### Philosophical meaning
Reproducibility is truth that can be revisited; owner isolation makes receipts trustworthy.

### Why this is better
Before: incompatible held branches and possible stale account responses. After: separate reviewable stack, explicit default-off advanced cloud writes, local state preserved across actual account switches, and reproducible validation receipts.

### How this moves Eidos closer to the north-star goal
This improves the reliability of the Works hosting surface and evidence trail. It does not establish learning, compression, anomaly preservation or detector performance for the Eidos Brain north-star claim.

### Evidence
See run_manifest.json, source_vendor_parity.json, accounts-controlled-accepted.json, accounts-switch-final.json and the captured test/build logs in artifacts/works_accounts_20260912/.

### Remaining uncertainty
Hosted two-account recovery and email delivery, existing Google consent, Stripe TEST and physical-device checks remain unproven. Research/model/core behavior was not changed.

## Artifacts generated
artifacts/works_accounts_20260912/: manifests, source/vendor hashes, local API/browser results, test/build logs and this journal.

## Google Drive archive status
See drive_manifest.json for the actual configured-path copy result.

## Thoughts on improvement
Keep hosted provider proof separate from controlled unit and browser proof.

## Where to improve next
Complete hosted owner-controlled confirmation, recovery, Google linking and Stripe TEST gates with existing authorized resources.

## Anything that stands out
Cross-tab session changes are workspace invalidation events, even when the document itself did not change.

## End-of-task summary
Files: Works backend/vendor/tests/config and evidence. Core behavior unchanged. Tests and commands captured above. Local artifacts, analysis, journal and Proof Logic + Meaning written. Owner equality, compare-and-swap and epoch invalidation explain the results. This improves reproducibility and trustworthy state handling; it does not establish research claims. Hosted providers and physical devices remain unproven. Drive result is recorded separately.
