# Plain-language release test analysis — 2026-09-08

The configured Google Drive copy failed because the mounted drive reported no free space (`ENOSPC`). The complete artifact set remains local. No existing files were deleted to make room.

The release makes free member accounts and optional complete-article emails available on Eidos Works. The important test was whether a customer could receive real mail and whether the actual hourly server job could run, while the existing store, inquiry and Lab systems remained intact.

The website and backend checks passed. Real sign-in and full-article messages reached the authorized inbox. The same article was retried after a simulated lost response; the provider returned the same message ID and the inbox received one digest. Unsubscribe, private reading lists and moderated mentions passed in the separate validation database. The production hourly job passed after its feed source was moved to the identical static Pages artifact.

An activated account page is not proof of a completed live login. That final browser gate is waiting for the human Cloudflare verification. The private original InkSoft catalog could not be inspected. No research or revenue claim follows from these release tests.

All logs, screenshots and sanitized receipts are local under artifacts/members-release-20260908. The configured Drive archive and hash verification are recorded by drive_manifest.json; mounted copy success does not establish remote synchronization.

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
