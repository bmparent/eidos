# Eidos Works production handoff - 2026-09-05

The website and backend are deployed. All eight platform release gates passed, including separate LIVE Stripe activation. Production purchasing is enabled; no real-money purchase was made during verification.

## Production and merged changes
- Website: https://eidos-works.com; Cloudflare Pages project eidosworks, deployment e3082c39-08b0-4175-8366-3bb674bce1c8, source c9e35f43c5c4c8abd90a4a11ffdceea49d603af9.
- Backend: https://eidos-sentinel-lab.vercel.app; Vercel activation deployment dpl_79c6nau9dEzs6bWhCPasUx2nQJME, source 4de48cf149e07d25195130b20e0b52a36533ee09. Later documentation-only merges may create equivalent deployments without changing the verified application code.
- Merged: website PR 6, backend PR 37, maintenance fix PR 7, Contact spacing fix PR 8, and documentation PRs 9 (website) and 39 (backend). Merged commit IDs are in release-state.json.
- Rollback: Cloudflare eeddf874-7035-4718-8fe7-4c734aed486c; Vercel dpl_DR6RwDA5gsvhLWi8GcRXJ9PrJCZF. Preserve platform databases and payment records during rollback.
- Original dirty checkouts were preserved. Snapshot remains independently gated. The research interface still shows BLOCKED_RESOURCE_BEFORE_HELDOUT and zero advanced gates.

## Active configuration
Separate free Turso production and validation databases are migrated. Community, moderation, agent access, source answers, optional AI, inquiry delivery and consent-gated analytics are active. The hourly workflow (17 minutes past each hour) passed from the default branch in run 33997382547 with ok=true and suggestions=0. It requires independent platform and maintenance secrets.

GA4 account 386114675, property 552876683 (Eidos Works), stream 15725877773, measurement G-8N7Y7EM4CS. Enhanced Measurement is off. Realtime confirmed production page views, assistant events and generate_lead; the lead's page_location was https://eidos-works.com/contact/. generate_lead and purchase are key events, with no invented monetary lead value. Traffic type is registered as an event-scoped custom dimension. Preview QA uses traffic_type=qa; production QA in an ordinary browser can classify as human and must not be interpreted as organic conversions.

AI model: gpt-4.1-nano-2025-04-14. Explicit visitor activation, maximum 320 output tokens, five attempts per visitor/day, 20,000 conservative reserved tokens/day, one call and no retries. Three real smoke calls were made across direct, preview and production checks. The direct response reported 221 input/40 output/261 total tokens. Production storage records one request and 2,006 reserved tokens. Source answers use zero model tokens; duplicate and failure paths retain the documented bounds.

## Stripe and delivery evidence
Stripe TEST checkout paid 2,900 cents USD and received the correct 6,012-byte ZIP containing index.html, styles.css, script.js, README.md and LICENSE.txt. SHA-256: c5723e4908f8aff386f50ec41c59dfa673a8e4280c46f2eed2399e30ee5f7988. Canceled/unpaid checkout, a real test refund and a separate test dispute each received download 403. An API-retrieved event locally re-signed with the test webhook secret was idempotent; this was not a provider resend. Required async-success handling is subscribed and covered by automated tests; an asynchronous payment method was not used for the real card checkout.

The production inquiry test reached the configured studio inbox (Gmail message 1a073c1c396c6d57) with the exact test label; SPF and DKIM passed. The site success message alone was not treated as proof. Product fulfillment is an on-screen download; automatic purchase-email delivery is not claimed.

## Live activation completed
Production purchasing is enabled in LIVE mode. The dedicated restricted key and separate webhook we_1UCTRjKC8pRG5Tr9KUpT4H8F serve https://eidos-works.com/api/shop/webhook with checkout.session.completed, checkout.session.async_payment_succeeded, charge.refunded and charge.dispute.created. Test/live keys, signing secrets and databases remain separate. The deployed browser opened the correct one-website license at $29.00 USD. Authenticated Stripe readback confirmed 2,900 cents, USD, card payment, and unpaid status; the durable order remained pending. The browser cancel link returned to the product. The verification session was then expired, and download returned 403 before and after expiration. No real-money payment was made. A synthetic ignored event signed with the live secret returned 200; unsigned and test-secret-signed requests returned 400 and created no entitlement or event record. This checks the production relay and signing configuration, not a provider-generated live payment event.

No release blocker remains. No paid subscription was selected.

## Validation
- Backend: npm --prefix apps/sentinel-lab run works:migrate; npm --prefix apps/sentinel-lab test; npm --prefix apps/sentinel-lab run lint; npm --prefix apps/sentinel-lab run build.
- Separate validation migration: node --env-file=apps/sentinel-lab/.env.validation.local apps/sentinel-lab/scripts/migrate-works.mjs.
- Remote storage: node --env-file=apps/sentinel-lab/.env.local --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/verify-works-storage.ts.
- Proactive fixtures: node --env-file=apps/sentinel-lab/.env.validation.local --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/verify-works-proactive.ts.
- Website: npm run test:platform; npm run test:analytics; npm run test:snapshot; npm run lint; npm run build; npm run verify:prerender; npm run verify:editorial; npm run validate:insights:dist; npm run verify:urls; npm run build:functions.
- Final CSS check: npm run build and npm run verify:prerender, followed by browser measurement on local build and production.
- Deployment: node node_modules/wrangler/bin/wrangler.js pages deploy dist --project-name eidosworks --branch main --commit-hash c9e35f43c5c4c8abd90a4a11ffdceea49d603af9 --commit-dirty=false.
All commands ran from the relevant repository root. All listed checks passed. Lab tests: 21. Website platform tests: 17, analytics: one, plus Snapshot smoke. No extra algorithm tests were added for the CSS-only follow-up; browser checks, build, prerender and CI cover it.

## Changed files and artifacts
Website changes include relay runtime handling, consent/navigation tracking, inquiry route and private email Worker, deterministic kit generation, reply acknowledgement, the hourly workflow, Contact spacing, focused tests and release docs. Backend adds remote service verification scripts and documentation; research/model code is unchanged.

Local evidence: artifacts/works-release-20260905 in the backend checkout and artifacts/release-20260905 in the website checkout. The backend folder includes the journal, plain-language analysis, proof ledger, progress SVG/HTML, JSON receipts, raw test logs and test ZIP. Journals/analysis also live under docs/proof_runs/2026-09-05. Drive destination: G:/My Drive/Eidos_Brain_Proof_Phase/2026-09-05/works-release-20260905. See drive_manifest.json for the copied file list and verified hashes.

## Proof Logic + Meaning

### Goal reached
All eight platform release gates passed. The site, backend, live checkout configuration and unpaid-access protection are deployed and verified. Payment fulfillment, refunds and disputes were exercised using real Stripe TEST transactions; no live revenue claim is made.

### Previous state
Local checks existed, but durable production storage, hosted provider behavior, real payment fulfillment, delivery, and analytics receipt had not been established. The relay failed in the actual Workers runtime.

### Technical logic utilized
Remote libSQL transactions enforce atomic quotas and durable event/entitlement writes. Independent service credentials protect the Pages-to-Sentinel relay, moderation and maintenance. The AI path reserves budget before one explicit provider call. Signed Stripe events grant or revoke receipt-hash entitlements. Review controls public community visibility. The private inquiry Worker fixes the destination and acknowledges provider acceptance. Scheduled maintenance now calls the authenticated backend directly after GitHub requests through the site returned 403.

### Math / scoring logic
Quota acceptance requires used + requested <= limit. Thirty concurrent reservations of 100 against a 750 cap accepted seven, reserving 700. AI reservation = UTF8_bytes(provider_payload) + 512 + 320. Preview reserved 2,060 and its duplicate added zero; the production browser call reserved 2,006. The direct provider reported 221 input + 40 output = 261 tokens. Proactive suggestions satisfy daily_total <= 10. Payment entitlement requires paid status, matching product/order/session metadata, currency USD, and amount 2,900 cents. Research readiness is null because no research gates were evaluated.

### Philosophical meaning
Reproducibility is truth that can be revisited. Restraint before automation means that consent, moderation, quotas, revocation and explicit limits govern the deployed system.

### Why this is better
Real hosted receipts now support the release. A Workers-specific redirect failure, lost reply acknowledgement, GitHub maintenance failure and mobile Contact overlap were corrected and verified. A browser redirect is distinguished from a paid entitlement; a delivered test inquiry is distinguished from a customer lead.

### How this moves Eidos closer to the north-star goal
Eidos Brain is a self-monitoring streaming intelligence codec. It learns live streams, compresses predictable behavior, preserves meaningful anomalies, monitors its own internal state, and emits human-readable incident receipts. This platform milestone strengthens reproducible operation and auditable receipts around that work. It does not prove stream learning, compression, anomaly preservation or detector superiority. Core research behavior was unchanged.

### Evidence
See stripe-live-browser-receipt.json, stripe-live-configuration.json, backend-live-production-receipt.json, production-http-receipt.json, production-browser-receipt.json, backend-production-receipt.json, maintenance-production-receipt.json, production-inquiry-receipt.json, ga4-realtime-receipt.json, storage-smoke.json, ai-provider-smoke.json, ai-deployed-smoke.json, community-deployed-smoke.json, agent-deployed-smoke.json, proactive-storage-smoke.json, stripe-deployed-smoke.json and the final test/build logs.

### Remaining uncertainty
LIVE purchasing is configured and an unpaid production checkout was verified. All paid verification transactions were TEST mode; no live payment or revenue was created. Proactive time eligibility used synthetic aged fixtures against remote validation storage. No natural 24-hour wait, physical mobile-device test, GPU benchmark, compression metric or held-out proof was performed. Bot classification is heuristic. Mounted Drive copies are hash-verified; cloud synchronization completion is not independently verified.
