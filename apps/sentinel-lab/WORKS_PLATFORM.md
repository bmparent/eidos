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

The Stripe webhook URL remains `https://eidos-works.com/api/shop/webhook`; the relay preserves its exact body and signature. Purchase redirects and community canonical URLs remain on Eidos Works. Configure both hosts together before enabling the relay.

## Token costs

Published-source answers and public source suggestions use zero model tokens. AI follow-up requires an explicit visitor click, limits the question and two history entries, selects at most three brief public facts, and requests at most 320 output tokens. There are no tools, browsing, recursive bot exchanges, or automatic retries. Five enhanced requests per visitor/day and an atomic global token reservation bound usage. Duplicate request IDs return the stored answer. A missing database/key/budget returns clearly labeled source information and never makes an unmetered provider request.

## Validation and release

`npm run lint`, `npm test`, and `npm run build` cover the Lab and new platform. Tests include actual libSQL atomic reservations/transaction rollback, relay authentication/origins, disabled-feature behavior, and a mocked bounded model response with idempotency. These do not prove live GA4, Stripe, database provisioning, or OpenAI billing access.

`GET /api/works/v1/health` exposes only the service identifier/version. After preview deployment, verify this URL and the unchanged Lab UI. Then configure environment values and run the actual checkout, moderation, analytics receipt and optional model smoke tests before production activation.

Shared handlers are vendored from `bmparent/brent-parent-intelligence-studio`; use that repository's `scripts/export-sentinel-platform.mjs` to refresh them. Review changes in both pull requests. Reverting an application deployment must preserve database and Stripe records.

## Verified service setup (September 5, 2026)

Separate free Turso Starter databases `eidos-works-production` and `eidos-works-validation` are provisioned and migrated. Production/preview credentials remain separate. Dedicated rate, moderation, relay, maintenance and Turnstile credentials are configured; the GitHub hourly maintenance secret is set.

GA4 property `552876683` and stream `15725877773` belong to https://eidos-works.com; measurement `G-8N7Y7EM4CS`. Authenticated Realtime received actual preview page/question/purchase events. Enhanced Measurement is disabled. AI uses `gpt-4.1-nano-2025-04-14`, max 320 output tokens, five attempts/visitor/day and a 20,000 daily conservative reservation budget. The deployed smoke passed with 2,060 reserved tokens and no additional reservation on replay. The direct provider usage receipt is 221 input + 40 output = 261 tokens.

Real Stripe test checkout, ZIP delivery, canceled/unpaid denial, duplicate-event replay, refund and dispute revocation passed. Guest/reply moderation, feed visibility/removal, agent rate limits and revocation passed against the deployed preview. Proactive eligibility and caps passed with synthetic aged records against remote validation libSQL. A private Cloudflare inquiry Worker delivered a matching receipt to the configured studio inbox.

The backend and website are now deployed with AI, durable community, proactive eligibility checks, real inquiry delivery and consent-gated analytics active. Production Realtime received assistant and delivered-inquiry events. The hourly GitHub workflow passed on the default branch after switching to the authenticated Sentinel endpoint with both `EIDOS_PLATFORM_TOKEN` and `EIDOS_MAINTENANCE_TOKEN`. Production purchasing remains disabled pending Stripe live-key email verification. See [the production handoff](../../artifacts/works-release-20260905/release-handoff.md) for deployed commits, receipts, remaining action and limitations. Research behavior was untouched.
