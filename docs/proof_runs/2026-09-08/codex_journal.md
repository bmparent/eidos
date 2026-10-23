# Codex Journal — 2026-09-08

## What happened today

Completed the authorized coordinated member release using isolated worktrees. The source, schema, email connection, browser checks and production maintenance were checked directly. See the release handoff for exact SHAs, deployment IDs and receipts.

## What was accomplished

Merged backend PR48 and four narrow compatibility/feed fix PRs; deployed both apps; applied additive migrations; verified the studio sender while preserving inbound mail; enabled accounts and full-text delivery after real controlled inbox checks; corrected the production feed failure without relaxing site protections.

## Tests and commands run

The website and Sentinel commands and results are recorded in the Validation section of release-handoff.md and their saved logs. All final automated checks passed. The live human-verification gate is pending.

## Problems encountered

Legacy schema collision; Vercel marketplace Free resource unavailable for an already-enrolled Resend team; browser human challenge; three corrected local harness assumptions; Cloudflare Bot Fight Mode challenge of a backend feed request. A count check was also adjusted to distinguish expired quotas removed by normal maintenance from durable records. Original artifacts were retained.

## What changed

Email-account table naming, server publication feed configuration and validation, deployment/environment/DNS settings, and release documentation.

## What did not change

Research/model/core behavior, experiments, thresholds, calibration, original storefronts, inbound email, existing payment connections and authentication boundaries were not modified.

## Proof Logic + Meaning

**Goal reached:** the operational deployment, compatible schema, sender connection, controlled email and scheduled-job gates passed. Complete customer verification remains partial pending the live human challenge. This is a product release, not an Eidos research proof.

**Previous state:** implementation existed but email was unconfigured; production's legacy member table conflicted with the new schema, and the activated server job could not retrieve the public feed through bot protection.

**Technical logic:** keep legacy and email identities in separate tables; atomically consume hashed email tokens; authorize sessions and scoped agent keys; reveal mentions only through published-source joins; claim durable deliveries and reuse their idempotency keys; select the identical hosted feed without changing customer links.

**Math / accounting:** delivery identity is `digest:member_id:UTC_day`; repeated attempts share one provider key. Observed retry result: two API attempts, one provider ID, one inbox digest. The application reserves at most 90 sign-in/newsletter attempts/day; even 31 full days total 2,790 attempts, below the 3,000/month Free allowance before other account traffic. This is a cap, not a claim about subscriber throughput or delivery guarantees.

**Philosophical meaning:** reproducibility is truth that can be revisited; continuity means preserving existing identities and evidence while adding a new capability.

**Why better:** actual delivery and scheduled execution replace setup assumptions; the release preserves legacy data and has a narrow, tested remedy for a real production failure.

**North-star connection:** this improves the auditable operational surface around Eidos and supports reproducible operator access. It does not establish that Eidos learns streams, compresses predictable behavior, preserves anomalies or outperforms detectors. No research proof score or progress dashboard was changed because no research gate was evaluated.

**Evidence:** the deployment, migration, browser, provider, Gmail, maintenance and configuration receipts named above.

**Remaining uncertainty:** live authenticated customer confirmation is pending; the original storefront is password-protected; long-term deliverability, subscriber volume, revenue and research performance were not measured. No new research metric or percentage is claimed.


## Artifacts generated

Both worktrees: artifacts/members-release-20260908; backend docs/proof_runs/2026-09-08 includes this journal and plain_language_test_analysis.md.

## Google Drive archive status

The mounted copy failed with `ENOSPC` before copying any file. The complete ZIP was then uploaded directly through the Drive connector; metadata readback confirmed the intended folder, name and byte count. Remote archive: https://drive.google.com/file/d/15F62XqrJaFKoYgNp7Wiq4KzmcyCF0dS9/view?usp=drivesdk. All artifacts also remain local. No existing content was removed. Remote checksum was not exposed by the connector; local ZIP integrity and SHA-256 were checked.

Configured root is G:\My Drive. The archive operation and per-file hash checks are recorded in drive_manifest.json. Destination: Eidos_Brain_Proof_Phase/2026-09-08/members-release-20260908. Do not infer remote cloud synchronization solely from a successful mounted-drive copy.

## Thoughts on improvement

Keep production jobs in the release gate whenever previously disabled functionality is activated. Provider credentials and a successful local test alone did not reveal the custom-domain feed challenge.

## Where to improve next

Complete the requested live customer confirmation when the human challenge is cleared. Keep the existing hourly workflow; no duplicate automation is needed.

## Anything that stands out

Both real release blockers were integration mismatches, not research-engine issues. Preserving separate databases and baseline configuration made the fixes narrow.

## End-of-task summary

Files changed, commands, core-behavior boundary, evidence, math, meaning and remaining uncertainty are recorded above and in release-handoff.md. No research benchmark, metric, proof score or dashboard was generated. Drive status is explicit in drive_manifest.json.


## Sentinel guided implementation

Separate task entry: [Sentinel guided codex journal.md](sentinel_guided_codex_journal.md). The existing Works release entry above is preserved. All four product paths are implemented; full hosted compute remains blocked by Vercel HTTP 402. Local integration and partial hosted receipts, frozen qualification failures, source hashes and Drive status are documented in that entry.
