# Eidos Works member release — September 8, 2026

The website and Sentinel backend are deployed, the additive Works migration is applied, and production accounts and full-text article delivery are enabled. Operational release: **passed**. Complete customer verification: **partial**. Cloudflare requested human verification on the prepared production signup form. User action is pending; no production member has been created by this release test.

## Live deployment

| Application | Deployed source | Deployment |
| --- | --- | --- |
| [Eidos Works](https://eidos-works.com) | `d2d1a64972a59c316715feb207078ec7ce0cec8b` | `cbf8aaeb-3b6c-42f3-b48a-3a6069c05586` |
| [Sentinel Lab](https://eidos-sentinel-lab.vercel.app) | `1ff3d3643f6dc229795bc7f8c2290a76424a677a` | `dpl_25j8Fyfxov8bw5yNjpxyg2gP7shL` |

Website PR #19 was already merged. Backend PR #48 was checked and merged. Follow-up website PRs #20/#21 and backend PRs #49/#50 fix the discovered legacy-table collision and deployed feed-fetch failure. The source commits above contain those fixes. Later documentation-only commits do not change the verified application code.

## Activation and preservation

- Production: `EIDOS_ACCOUNTS_ENABLED=true`, `EIDOS_NEWSLETTER_ENABLED=true`, `EIDOS_MAIL_DAILY_LIMIT=90`, `PUBLIC_SITE_URL=https://eidos-works.com`, `EIDOS_MAIL_FROM=Eidos Works <papers@eidos-works.com>`.
- A new sending-only Resend key is restricted to the verified studio domain and stored as a server-only sensitive Vercel variable. The existing Resend Free account is used: 3,000 monthly / 100 daily; paid usage and add-ons remain off. The Vercel installation exists, but its second Free-resource checkout was unavailable, so the existing Resend team was reused directly.
- Added only three Resend DNS records: DKIM at `resend._domainkey`, plus SPF and bounce MX at `send`. All 11 prior records, including inbound MX, were preserved. Receiving on Resend is off.
- Applied `npm run works:migrate` to the dedicated remote Works production database. New email identities use `eidos_email_members`; the existing Clerk `eidos_members`, agent-owner associations, orders, community and Lab admissions remain. Tests seed and preserve a legacy member across repeated migrations. Production row counts were retained at migration time.
- Existing Cloudflare bindings and Vercel project settings matched their baseline fingerprints. Unrelated readable/encrypted values matched their recorded fingerprints; sensitive values are unreadable, so their byte equality is not claimed. The release did not edit those secrets. A later normal maintenance cleanup reduced expired quota rows from 9 to 1 across UTC midnight; durable records retained their counts.
- Original dirty checkouts were preserved. All release changes were made in isolated worktrees. No original InkSoft store, order, payment, research data, model behavior, threshold, calibration or Lab execution policy was changed.

## Verified behavior

Production Chrome checks passed at 1440px and 390px: original recreation artwork loads; hero, snow/motion control and four collection globes render; all five garments support front/back inspection and seven sizes; filters, search/sort, quantity, personalization, cart total ($1,688 in the labeled demo fixture), removal and recreation/demo notices work. No orders or payments were placed. Navigation exposes Insights and Account, all four working principles include their philosophical notes, and the activated account page has both account-type choices, an open archive and no horizontal overflow or page errors.

The original InkSoft URL currently presents a private-store password prompt. Its private catalog and checkout could not be compared live. The recreation was checked against its available original-art assets; no password was guessed and the original store was not modified.

The live Lab request link opens `/lab/access`. A labeled real inquiry returned receipt `340ab5cf-987a-4152-9cd9-f87d7ca1add4` and arrived at the studio inbox at 2026-09-08T23:21:02Z. An intentionally aborted browser request separately verified the honest email fallback.

Current backend handlers ran against the **separate remote validation database** and real Resend. The CAPTCHA success response was simulated only in that local harness. Real person and operator-managed-agent sign-in emails arrived; confirmation, single-use consumption, secure cookies, private account data and reading lists, preferences and logout passed. Mentions became visible only after moderation, resisted cross-account updates, and disappeared when their source was rejected. Agent keys excluded operator email, could not mint keys or change preferences, were denied after revocation, and stopped at five contributions/day. These are controlled integration results, not claims that every flow was exercised by an authenticated production browser.

The controlled full-text digest arrived at 2026-09-08T23:44:04Z. Gmail contained all **16 paragraphs and three source URLs**, both text and HTML, and an unsubscribe link. SPF and DKIM passed. After simulating a lost response **after provider acceptance**, retry returned the same provider ID `8d584a81-6fbb-4c84-80eb-7820f7d91cf1`. Exactly one digest appeared in the inbox. A further run sent zero; unsubscribe and repeated unsubscribe confirmation passed. Tests sent only to the authorized studio inbox; no real subscribers received test mail.

The existing hourly job (minute 17 UTC) successfully called the live authenticated backend with both required GitHub secrets. [Verified production run](https://github.com/bmparent/brent-parent-intelligence-studio/actions/runs/34293285562) returned `{"ok":true,"suggestions":0,"newsletter":{"sent":0,"enabled":true}}`. This validates job execution and credentials; it did not select a production subscriber for this test.

## Release issue and fix

Initial activation run 34292367535 returned 503. The matching Cloudflare event at 2026-09-08T23:49:58Z recorded `managed_challenge`, source `botFight`, path `/insights-feed.json`, ray `a381da4e3a48a9b4`. Production now reads `EIDOS_PUBLICATION_FEED_URL=https://eidosworks.pages.dev/insights-feed.json`. The two public feeds returned identical 43,953-byte bodies with SHA-256 `5b44e6140eae110b8bd0196af7140a4add12d4a59df9f5f24c0b52378ec604db`. The server override requires HTTPS and the exact feed path, rejects credentials/query/fragment, and does not follow redirects. Customer links stay canonical. Bot Fight Mode and Turnstile remain enabled.

Reference: [Cloudflare Bot Fight Mode](https://developers.cloudflare.com/bots/get-started/bot-fight-mode/) and [firewall-event queries](https://developers.cloudflare.com/analytics/graphql-api/tutorials/querying-firewall-events/). Provider logs, rather than these general docs alone, establish the release-specific cause.

## Validation and evidence

Website root: `npm ci`, `npm run lint`, `npm run test:platform`, `npm run test:analytics`, `npm run test:snapshot`, `npm run build`, `npm run verify:prerender`, `npm run verify:editorial`, `npm run validate:insights` (live source fetching), `npm run validate:insights:dist`, `npm run verify:urls`, and `npm run build:functions` passed. The feed fix passed 26 platform tests, lint, Functions compilation, a fresh build/prerender and full GitHub site-quality CI.

Sentinel app root: `npm ci`, `npm test`, `npm run lint`, and `npm run build` passed, including the real libSQL adapter/relay suite and GitHub verify/runner/Vercel checks. The migration command was `npm run works:migrate` with the existing Works production credentials. No research benchmark was run because model behavior was outside this release.

Receipts reside in `artifacts/members-release-20260908` in both release worktrees. Key files: `release-manifest.json`, `migration.json`, `dns-change.json`, `mail-inbox.json`, `controlled-mail*.json`, `maintenance.json`, `publication-feed-diagnosis.json`, `production-verification.json`, `customer-qa.json`, `active-account-browser.json`, build/test logs and desktop/mobile screenshots. Earlier harness failures are retained: an incorrect simulated CAPTCHA action, an invalid fixture status value, and an attempted lookup of a private member ID in a public response. Those harness assumptions were corrected; the required behaviors subsequently passed. No credential values or reusable email/session/agent tokens are included in the receipts.

## Remaining gate and rollback

The configured Google Drive mirror failed with `ENOSPC` before copying a file. No user files were deleted. All evidence remains in the two local release worktrees; `drive_manifest.json` records the failed attempt. Drive archival is incomplete until storage is available.

The prepared production signup at https://eidos-works.com/account requires the user to complete Cloudflare's “Verify you are human” check and request the email. Then verify the confirmation, private saved reading/preferences and logout in that live browser. A successful deployed configuration or controlled inbox check does not substitute for this gate.

The production flags are active after the documented controlled-mail gate. To pause newsletters, set `EIDOS_NEWSLETTER_ENABLED=false` and redeploy the backend. To disable member features, set `EIDOS_ACCOUNTS_ENABLED=false` and redeploy. Preserve every member, order, delivery record and database table; do not reverse the additive migration. Keep the configured hosting feed URL unless its replacement has been verified from the server. Mail outcomes older than the documented idempotency window require provider reconciliation.

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
