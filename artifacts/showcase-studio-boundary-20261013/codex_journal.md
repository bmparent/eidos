# Codex Journal — 2026-10-13

## What happened today

Created a separate Works shared/backend candidate from delivery PR72 exact952e68ff, preserving every existing worktree. Added an explicit local-studio cloud boundary and paired an isolated protected Preview with the new showcase site origin.

## What was accomplished

Schema5, future numeric versions and disguised legacy documents carrying studios now return409 before old validation/persistence. The response explains local storage and JSON backup. Legacy behavior is preserved. Three new negative tests and seven actual hosted API checks passed.

## Tests and commands run

`npm --prefix apps/sentinel-lab test`:76passed(16MJS+60TypeScript). `npm --prefix apps/sentinel-lab run lint`:passed. `node --check` passed for both new Preview helpers. Exact helper invocations are in the implementation report; ignored configuration supplies credentials without printing values.

## Problems encountered

The first hosted harness expected provider401/403, but unauthenticated Vercel correctly redirected302 to its sign-in. The failed assumption and actual result are preserved. API deployment target preview is represented by an omitted target/null; the explicit preview string was rejected before deployment. The corrected redeploy remained Preview. CLI environment upserts were slow, so final public disabled-feature overrides were also confirmed through native branch-only tools.

## What changed

Shared studio guard, save/HTTP entry validation order, three tests, branch-only Preview tools, logs and acceptance documentation.

## What did not change

Core Brain/Sentinel numerical behavior, schema1 ownership/revision rules, purchased archives/rights, original preview branches, Production and GitHub secrets were unchanged. Schema5 account saving is not enabled.

## Proof Logic + Meaning

Goal: local and hosted format-safety gate passed. Previously an older validator had no explicit studio-field boundary. Logic: local_only=(schema>=5 OR own studios field), then409 before database access. Tests prove database_touches=0 and input_after=input_before; existing concurrent-save/purchase tests prove preserved legacy behavior. This is honesty before convenience. It strengthens auditable state and reproducibility around the north-star codec without proving learning, compression or detector quality. Evidence: app-tests.log,typecheck.log,hosted-boundary-acceptance.json,deployment.json,preview-environment.json. Authenticated schema5 ownership/reopen, real email/payment/provider flows and scientific readiness remain unproven and gated.

## Artifacts generated

Repo-local artifacts/showcase-studio-boundary-20261013 contains logs, acceptance/Preview/deployment receipts, journal/analysis and hash inventory. No secret values were saved.

## Google Drive archive status

Exact configured mounted destination and checksum result are in drive_manifest.json. Mounted copy is distinct from remote cloud synchronization.

## Thoughts on improvement

Keep client and server format activation paired. Local backup remains useful while hosted owner acceptance is unavailable.

## Where to improve next

Run two-account schema5 ownership/reopen/asset tests only after the format contract is explicitly implemented and authorized for activation.

## Anything that stands out

A friendly cloud refusal prevents data loss more reliably than silently accepting an incomplete document.

## End-of-task summary

1.Files:guard,save/route calls,tests,Preview helpers,receipts/docs.
2.Core behavior:no.
3.Tests:three new negatives;76appcases and tsc passed;seven hosted checks passed.
4.Commands:report contains repo-root invocation list.
5.Artifacts:local receipts/logs/hashes.
6.Plain-language analysis:written.
7.Journal:written.
8.Drive:manifest records copy/checksums.
9.Limits:schema5 cloud/provider acceptance not enabled.
10.Follow-up:not implementing cloud activation or production release.
11.ProofLogic+Meaning:written.
12.Math:format predicate,zeroDBtouches,input equality.
13.Meaning:honesty before convenience.
14.Improvement:explicit refusal protects local data.
15.North star:auditable state and reproducibility.
16.Evidence:exact source/deployment/tests/Preview manifests.
17.Uncertainty:hosted owner/reopen/provider/scientific claims.
