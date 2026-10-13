# Paid template release review — October 13, 2026

Brent authorized reviewing all prepared templates and releasing them for sale. The
review found a genuine code blocker: the existing collection rejects LIVE keys,
LIVE Checkout Sessions and LIVE payment events. The production collection URL
also currently renders the site's 404. The existing four-family preview remains
separate from production.

## Reviewed products

| Product | Current developer version | Current WordPress version | Review result |
| --- | --- | --- | --- |
| Switchboard | 1.0.2 | 1.0.2 | Current ZIPs rebuild unchanged; service selector observed working in hosted Chrome |
| Tideglass | 1.0.1 | 1.0.2 | Current ZIPs rebuild unchanged; evening scene/caption observed in hosted Chrome |
| Matter | 1.0.1 | 1.0.2 | Current ZIPs rebuild unchanged; close material inspection observed in hosted Chrome |
| Nightjar | 1.0.4 | 1.0.2 | Current ZIPs rebuild unchanged; Ideas poster reveals the correct event in hosted Chrome |
| Cinematic Starter | Existing product | Not a WordPress theme | Existing $29 source, terms and original paid archive retained; backend regression passes |
| Spoke & Bench | Private draft 0.1.0 | Not implemented | Its 57,025-byte ZIP passes CRC and all 16 source hashes; no commerce integration or rendered acceptance |

The four-family offers retain draft $99 developer / $129 WordPress amounts.
Spoke & Bench retains its separate unpublished section/landing/site/personalized
proposals. No price, payment, private archive, or purchased version was changed.
Squarespace 7.1 and GoDaddy Websites + Marketing remain unverified Switchboard
pilots, with no purchasable package. Other experimental portfolio demonstrations
are not being reclassified as paid templates.

The eight current buyer archives still contain candidate/draft license and
support notices. Before publication, finalize the notices in new immutable
versions, rebuild and recheck those ZIPs, and retain every earlier archive. Do not
rewrite bytes already associated with a version or prior purchase.

## Prepared repair

The continuation branch is `codex/template-release-readiness-20261013` in both
repositories. It starts at site `d0657f0fa645f877596176a57904b78525cb90a8` and
backend `f9f8cd30839d36127786b7b7826e6c44de92d2c7`, the current PR102/PR71 heads.
It is a source preparation candidate, with LIVE activation absent by default.

- An independent `EIDOS_TEMPLATE_LIVE_ENABLED=true` flag permits LIVE checkout
  only with TEST disabled, the canonical HTTPS origin, a LIVE Stripe key,
  configured database/private archives/webhook/mail/rate key and real Turnstile
  keys. The TEST candidate's dummy verification keys cannot open LIVE purchases.
- `EIDOS_TEMPLATE_STRIPE_ACCOUNT_ID` identifies the intended merchant. Before a
  LIVE order or session is created, `/v1/account` must match that ID and report
  both charges and payouts enabled. The connected merchant observed through the
  Stripe connector is Eidos Works, `acct_1Rj03gKC8pRG5Tr9`; its connection is not
  proof that a deployment's key targets it.
- Checkout response IDs and webhook modes must match the stored session. Wrong
  modes, prices and merchants fail closed. LIVE requests use the skill-prescribed
  `2026-08-26.dahlia` version and integration identifier; existing TEST requests
  retain their pinned `2025-06-30.basil` contract. Actual provider acceptance of
  the new LIVE request remains untested.
- Receipts and recovery use the order's session mode for their labels. Paid
  fulfillment, recovery and refund/dispute revocation continue if new sales are
  disabled. Original order version/hash/price snapshots remain intact.
- The storefront accepts verified LIVE catalog responses, uses Buy/secure
  checkout wording for available editions, and retains terms/challenge and
  exact price/version requirements. TEST/default/unavailable states remain
  closed. The browser suite now exercises both catalog modes using explicit
  fixtures, without making a provider transaction.

Only the four changed shared commerce files are exported to the backend. The
other existing vendored differences are preserved. No migration or production
configuration is applied by this continuation.

## Evidence and limits

| Check | Observation | State |
| --- | --- | --- |
| Current package integrity and source boundaries | 803 checks pass; all eight frozen ZIP hashes unchanged | Pass locally |
| Joined manifest and storefront guide contracts | Six manifest and six storefront tests pass | Pass locally |
| Backend behavior | 16 MJS + 56 TypeScript tests pass, including five new controlled LIVE cases | Pass locally |
| Site platform and existing commerce | 35 tests pass | Pass locally |
| Site/worker verification | Lint/typecheck, client/SSR build, prerender, shop/editorial/URL checks and Functions compilation pass; existing Wellway warning remains | Pass locally |
| Backend typecheck | No errors | Pass locally |
| Actual hosted original four-family preview | Meaningful pages and the four signature interactions observed in cloud Chrome; no payment submitted | Pass for that existing preview only |
| New-source rendered LIVE catalog behavior | Dedicated three-engine workflow added; results must be read from its exact head | Pending CI |
| Local browser | Runtime browser download failed; existing local Chrome could not launch | Blocked locally |
| Spoke & Bench | CRC, byte count and 16 manifest hashes match; no new browser/commerce acceptance | Package verified only |
| Scoped preview Resend binding | Metadata has a production-only sensitive key and no branch/generic Preview key; values not read, copied or broadened | Blocked |
| Real delivery and signed event acknowledgement | No new delivered inbox receipt/recovery or genuine paid-event HTTP acknowledgement observed | Blocked |
| Authenticated purchases | Accounts remain disabled in the template preview; delivered-login two-member proof remains open | Blocked if enabling member purchasing |
| WordPress visual installations | Historical native API/browser evidence retains its recorded limits, including the 6.6 supplemental font-weight crash; no fresh minimum-version visual editor acceptance | Open |
| Hosted builder installations | No authorized editor/plan/publication proof for the two pilots | Blocked |

Local controlled LIVE fixtures call no real Stripe, Turnstile or Resend service.
They establish behavior under simulated responses, not a LIVE sale, merchant
receipt, received email or hosted buyer. Prior provider evidence belongs to its
original exact sources and is not reassigned to this continuation.

## Release decision and next action

**Production NO-GO.** Authorization is present; the remaining constraints are
functional/configuration and evidence gaps, not a request for another release
permission. Keep PR102/PR71 and this continuation draft until the relevant gates
are met. Do not silently expose paid ZIPs to work around delivery.

First configure a separate sending-only `RESEND_API_KEY` for the verified
`eidos-works.com` sender in Vercel project `eidos-sentinel-lab`, Preview target,
branch `codex/paid-templates-20261010`. Redeploy the isolated paired backend, then
observe real guest TEST receipt/recovery, expiry/single-use behavior, and genuine
paid-event HTTP acknowledgement on the final source. If member purchasing is
enabled, additionally stage accepted member verification and complete normal
delivered sign-in with two owner-controlled inboxes.

Developer editions may be considered as a clearly scoped first sale release
after their commerce and final-package gates pass. WordPress and each builder
edition require their own accepted installation boundaries. Spoke & Bench also
needs rendered/editor/download acceptance and a separate catalog/fulfillment
integration before it can be sold.

Before enabling production, apply the additive template schema to the confirmed
production database, include every current and retained private archive in the
exact backend bundle, bind the intended LIVE merchant and genuine production
webhook/verification/mail settings, and verify the staged paired deployment.
Do not enable unrelated accounts, AI, authoring, subscriptions or community flags.

The final source read observes site main `5e67573c5bec3723daa6d47b9ad0d1d58f45883e`
(the community-loading repair) and canonical backend
`dpl_BDUhZ7W9Akz58SDaVXJ2bXVF4rNQ` / `4c746ff`. The Cloudflare deployment identity still needs its connected-host readback before
a release; the repository main SHA is not being asserted as deployment proof.
Preserve the newer community fix when integrating or reverting the template
candidate. After any eventual LIVE sale,
disable new template checkout while retaining compatible fulfillment handlers,
schema, keys, order rows and all sold archives; reverting to TEST-only fulfillment
would strand LIVE buyers. No rollback or release was performed in this review.
