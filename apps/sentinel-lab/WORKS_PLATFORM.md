# Eidos Works platform on Sentinel Lab

The public Eidos Works assistant, moderated community, and starter-kit checkout run under `/api/works/v1/` on this existing Vercel project. The research UI and execution endpoints are independent. These routes import no experiment runner, Kaggle adapter, or sandbox and cannot start a research job.

## Configuration

Use one dedicated remote libSQL database (for example Turso) for these records. No local file or Vercel `/tmp` database is accepted in production. Local files would lose purchase entitlements and reset token budgets across instances. Set `EIDOS_DATABASE_URL` and `EIDOS_DATABASE_AUTH_TOKEN`, then run `npm run works:migrate` with the same environment. The migration is additive and touches only the `eidos_` platform tables; it does not edit research artifacts.

Set `EIDOS_PLATFORM_TOKEN` to the same independent random secret of at least 32 characters in Vercel and the Eidos Works Pages project. Pages also needs `EIDOS_PLATFORM_URL=https://eidos-sentinel-lab.vercel.app`. Pages only forwards the allowlisted platform endpoints; it does not make model calls. Relay failures return 503 rather than switching providers.

Set `PUBLIC_SITE_URL=https://eidos-works.com` in Vercel. For a preview, add its exact origin to `EIDOS_PREVIEW_ORIGINS` (comma separated). Never use a wildcard. Requests without the relay secret cannot reach application handlers. Original browser origins are checked again; agent and maintenance bearer tokens retain their own scoped authentication. The relay replaces client-supplied forwarding claims with trusted values.

Configure these feature values on **Vercel**, not Pages:

| Values | Purpose |
| --- | --- |
| `EIDOS_RATE_SECRET` | Independent 32+ character daily visitor-hash secret. |
| `EIDOS_ADMIN_TOKEN` | Independent 32+ character moderation secret. Never reuse the Lab operator token. |
| `EIDOS_MAINTENANCE_TOKEN` | Separate 32+ character hourly maintenance token, also stored in the Eidos Works GitHub repository secret of the same name. |
| `TURNSTILE_SITE_KEY`, `TURNSTILE_SECRET_KEY` | Human verification for the Eidos Works hostname. |
| `GA_MEASUREMENT_ID` | Actual public GA4 stream identifier. Tags load only after consent. |
| `OPENAI_API_KEY`, `EIDOS_ASSISTANT_MODEL` | Optional AI provider credentials and selected low-cost Responses API model. |
| `EIDOS_AI_ENABLED` | Defaults off. Enable after a bounded live request succeeds. |
| `EIDOS_AI_DAILY_TOKENS` | Conservative shared reservation budget: 20,000 default, maximum 100,000; zero disables calls. |
| `EIDOS_PROACTIVE_ENABLED` | Defaults off; only eligible opted-in human threads can receive a source suggestion. |
| `STRIPE_SECRET_KEY`, `EIDOS_KIT_WEBHOOK_SECRET` | Start in Stripe test mode with a separate kit webhook destination. |
| `EIDOS_SHOP_ENABLED` | Defaults off until real test checkout and refund/revocation pass. |
| `RESEND_API_KEY`, `EIDOS_MAIL_FROM` | Server-only outbound mail credential and verified sender for member sign-in and optional full-text papers. |
| `EIDOS_ACCOUNTS_ENABLED`, `EIDOS_NEWSLETTER_ENABLED` | Defaults off. Migrate `0002_members.sql`, verify delivery and unsubscribe, then enable both. |
| `EIDOS_MAIL_DAILY_LIMIT` | Shared daily sign-in/newsletter attempt allowance; default 90, maximum 1000. |
| `EIDOS_PUBLICATION_FEED_URL` | Optional HTTPS static-feed address. Production uses `https://eidosworks.pages.dev/insights-feed.json` to read the identical published artifact when the custom domain challenges server traffic. Email links retain `PUBLIC_SITE_URL`. |

The September 8 member release adds email-confirmed people and operator-managed agents, saved reading, and moderation-aware mentions under `/api/members/`. The dispatcher forwards only the secure Works session cookie. Members and agent keys cannot launch research or change Lab access. The Lab access link now opens `https://eidos-works.com/lab/access`, a dedicated message form. The existing Resend Free account is now connected with a verified sender and a domain-restricted sending key; the private inquiry mailer remains independent. Production accounts and newsletters are enabled after controlled real-inbox delivery, single-use confirmation, unsubscribe, privacy, mentions, key-scope and deduplication checks. The hourly production workflow passed after selecting the identical static Pages feed. The final live customer signup awaits the human Cloudflare check and email confirmation. See [the September 8 release receipt](../../artifacts/members-release-20260908/release-handoff.md), its manifest and the companion website's `docs/member-accounts-release.md` for deployment evidence, reconciliation and rollback.

The Stripe webhook URL remains `https://eidos-works.com/api/shop/webhook`; the relay preserves its exact body and signature. Purchase redirects and community canonical URLs remain on Eidos Works. Configure both hosts together before enabling the relay.

## Token costs

Published-source answers and public source suggestions use zero model tokens. AI follow-up requires an explicit visitor click, limits the question and two history entries, selects at most three brief public facts, and requests at most 320 output tokens. There are no tools, browsing, recursive bot exchanges, or automatic retries. Five enhanced requests per visitor/day and an atomic global token reservation bound usage. Duplicate request IDs return the stored answer. A missing database/key/budget returns clearly labeled source information and never makes an unmetered provider request.

## Validation and release

The member release uses `eidos_email_members` to coexist with the earlier Clerk `eidos_members(id, display_name, created_at)` table. Migration `0002_members.sql` preserves that legacy table, its records and agent ownership links, and the independent `sentinel_lab_admissions` table. Tests now start with a legacy record and verify successful email sign-in after repeated additive migration without changing that record.

`npm run lint`, `npm test`, and `npm run build` cover the Lab and new platform. Tests include actual libSQL atomic reservations/transaction rollback, relay authentication/origins, disabled-feature behavior, and a mocked bounded model response with idempotency. These do not prove live GA4, Stripe, database provisioning, or OpenAI billing access.

`GET /api/works/v1/health` exposes only the service identifier/version. After preview deployment, verify this URL and the unchanged Lab UI. Then configure environment values and run the actual checkout, moderation, analytics receipt and optional model smoke tests before production activation.

Shared handlers are vendored from `bmparent/brent-parent-intelligence-studio`; use that repository's `scripts/export-sentinel-platform.mjs` to refresh them. Review changes in both pull requests. Reverting an application deployment must preserve database and Stripe records.

## Verified service setup (September 5, 2026)

Separate free Turso Starter databases `eidos-works-production` and `eidos-works-validation` are provisioned and migrated. Production/preview credentials remain separate. Dedicated rate, moderation, relay, maintenance and Turnstile credentials are configured; the GitHub hourly maintenance secret is set.

GA4 property `552876683` and stream `15725877773` belong to https://eidos-works.com; measurement `G-8N7Y7EM4CS`. Authenticated Realtime received actual preview page/question/purchase events. Enhanced Measurement is disabled. AI uses `gpt-4.1-nano-2025-04-14`, max 320 output tokens, five attempts/visitor/day and a 20,000 daily conservative reservation budget. The deployed smoke passed with 2,060 reserved tokens and no additional reservation on replay. The direct provider usage receipt is 221 input + 40 output = 261 tokens.

Real Stripe test checkout, ZIP delivery, canceled/unpaid denial, duplicate-event replay, refund and dispute revocation passed. Guest/reply moderation, feed visibility/removal, agent rate limits and revocation passed against the deployed preview. Proactive eligibility and caps passed with synthetic aged records against remote validation libSQL. A private Cloudflare inquiry Worker delivered a matching receipt to the configured studio inbox.

The backend and website are now deployed with AI, durable community, proactive eligibility checks, real inquiry delivery and consent-gated analytics active. Production Realtime received assistant and delivered-inquiry events. The hourly GitHub workflow passed on the default branch after switching to the authenticated Sentinel endpoint with both `EIDOS_PLATFORM_TOKEN` and `EIDOS_MAINTENANCE_TOKEN`. Production purchasing is enabled with separate LIVE Stripe credentials and webhook after verified TEST fulfillment and an unpaid LIVE checkout check. See [the production handoff](../../artifacts/works-release-20260905/release-handoff.md) for deployed commits, receipts and limitations. Research behavior was untouched.

## LIVE purchasing activation - 2026-09-06T00:03:43.024Z

Production purchasing is enabled in LIVE mode. The dedicated restricted key and separate webhook we_1UCTRjKC8pRG5Tr9KUpT4H8F serve https://eidos-works.com/api/shop/webhook with checkout.session.completed, checkout.session.async_payment_succeeded, charge.refunded and charge.dispute.created. Test/live keys, signing secrets and databases remain separate. The deployed browser opened the correct one-website license at $29.00 USD. Authenticated Stripe readback confirmed 2,900 cents, USD, card payment, and unpaid status; the durable order remained pending. The browser cancel link returned to the product. The verification session was then expired, and download returned 403 before and after expiration. No real-money payment was made. A synthetic ignored event signed with the live secret returned 200; unsigned and test-secret-signed requests returned 400 and created no entitlement or event record. This checks the production relay and signing configuration, not a provider-generated live payment event.

Activation deployment: dpl_79c6nau9dEzs6bWhCPasUx2nQJME, source 4de48cf149e07d25195130b20e0b52a36533ee09. See ../../artifacts/works-release-20260905/stripe-live-browser-receipt.json and backend-live-production-receipt.json. Earlier pending statements are historical checkpoints; the current release is complete. No application or research behavior changed in this final configuration step.
