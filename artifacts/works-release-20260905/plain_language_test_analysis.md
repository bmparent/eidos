# Eidos Works service validation — 2026-09-05

The hosted preview now connects the cinematic website to Sentinel, remote storage, opt-in AI, moderated community and Stripe test fulfillment. Both free Turso allocations were provisioned after explicit owner acceptance of the service terms. The inquiry test reached the configured studio Gmail inbox with a matching receipt. GA4 Realtime received actual page/question/purchase events for the validation preview. Paid test checkout returned the promised five-file ZIP; cancel/unpaid, refund and dispute paths denied downloads. No real-money test purchase was made.

The authoritative website project is Cloudflare Pages `eidosworks`, direct-upload branch `main`. Its original rollback deployment is `eeddf874-7035-4718-8fe7-4c734aed486c`. The Lab rollback deployment is `dpl_DR6RwDA5gsvhLWi8GcRXJ9PrJCZF`, research commit `6bb0b980349f94c55a6ca1ca9665737570c0fd01`. Original dirty worktrees were preserved; this work uses isolated branches for website PR 6 and backend PR 37.

## What passed and what failed

The service checks listed above passed. The original `redirect: error` option failed in the actual Workers runtime; supported manual redirects plus explicit 3xx rejection fixed it. A Miniflare regression test now proves the runtime accepts the relay. An early runtime redirect mock test failed because Miniflare's mock fetch adapter followed a redirect; redirect rejection is covered by the focused relay unit test, while the runtime test covers supported options. A browser reply acknowledgement was overwritten by the next Turnstile callback and is now retained, with a regression test. Early CLI/API authentication attempts failed; authorized OAuth/provider UI configuration resolved service access. Stripe's live-key verification email still requires its browser confirmation; Chrome reported an extension popup blocking the mail tab.

## Configuration

GA4: account 386114675, Eidos Works property 552876683, production stream 15725877773 at https://eidos-works.com, measurement G-8N7Y7EM4CS. Enhanced Measurement off. AI: gpt-4.1-nano-2025-04-14, output cap 320, five attempts/visitor/day, 20,000 reserved tokens/day, one call, no retries. Production AI and proactive flags are configured true for the next deployment; production shop stays false until its separate live key/webhook is ready. The maintenance repository secret is set for the hourly default-branch workflow.

## Tests and commands run

- `npm --prefix apps/sentinel-lab run works:migrate` — passed against production Turso.
- `node --env-file=apps/sentinel-lab/.env.validation.local apps/sentinel-lab/scripts/migrate-works.mjs` — passed against separate validation Turso.
- `node --env-file=apps/sentinel-lab/.env.local --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/verify-works-storage.ts` — passed.
- `node --env-file=apps/sentinel-lab/.env.validation.local --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/verify-works-proactive.ts` — passed; owned fixtures removed.
- `npm --prefix apps/sentinel-lab run lint` — passed.
- `npm --prefix apps/sentinel-lab test` — 21 passed (17 existing research/UI tests, four platform tests).
- `npm --prefix apps/sentinel-lab run build` — passed.
- Companion website: `npm run test:platform` — 17 passed; `npm run test:analytics` — one passed; Snapshot smoke, lint, build, prerender, editorial, Insights dist, URLs, Functions build — passed.

## Proof Logic + Meaning

### Goal reached
The platform service-validation gate passed. The overall public release is partial until merged deployments and final production checks are recorded.

### Previous state
The existing implementation had passing local tests but no verified durable production allocation, analytics stream receipt, provider delivery, or completed hosted payment flow. The deployed relay failed in Cloudflare despite passing Node mocks.

### Technical logic utilized
A dedicated remote libSQL adapter uses atomic quota updates and transactional payment/event writes. Separate credentials and databases isolate production from validation. Cloudflare transport preserves body/signatures, rejects redirects, and authenticates to Sentinel. Explicit AI activation reserves a conservative token bound before one provider call. Signed Stripe events grant or revoke a receipt-hash entitlement; browser redirects grant nothing. Human-reviewed publication controls community pages and feeds. The inquiry Worker fixes its destination and acknowledges only provider acceptance.

### Math / scoring logic
Quota acceptance is atomic: accept only when used + requested <= limit. Thirty concurrent requests of 100 against a 750 limit accepted seven (700 reserved). AI reservation = UTF8_bytes(provider_payload) + 512 + 320; the deployed smoke reserved 2,060, and its duplicate added zero. Direct provider usage was 221 input + 40 output = 261 tokens. Public suggestions satisfy daily_total <= 10; nine fixtures were accepted after one earlier suggestion, and the second pass added zero. Payment requires paid status, correct order/session/product metadata, USD and 2,900 cents. No overall Eidos research-readiness percentage is computed because this platform task does not evaluate the research gates.

### Philosophical meaning
Reproducibility means truth that can be revisited. Restraint before automation means quotas, author consent, review and revocation remain operational constraints, not documentation promises.

### Why this is better
The evidence now includes real hosted services and durable state rather than only mocks. A Workers-specific failure was reproduced and corrected. Paid, unpaid, duplicated, refunded and disputed paths have distinct observed outcomes. The reply confirmation remains visible after verification refresh.

### How this moves Eidos closer to the north-star goal
This milestone strengthens reproducible operation and human-readable receipts around the public Sentinel platform. It does not establish that Eidos Brain learns or compresses streams better, preserves anomalies, or beats other detectors/compressors. Core reservoir, RLS, thresholds, Sentinel labels, research executor and experiment artifacts were not changed.

### Evidence
`storage-smoke.json`, `ai-provider-smoke.json`, `ai-deployed-smoke.json`, `community-deployed-smoke.json`, `agent-deployed-smoke.json`, `proactive-storage-smoke.json`, `stripe-deployed-smoke.json`, `inquiry-service-smoke.json`, `ga4-realtime-receipt.json`, `tests-final.log`, `lint-final.log`, and `build-final.log` in `artifacts/works-release-20260905/`. Website runtime and browser-form regression tests are in the companion PR.

### Remaining uncertainty
Final production deployment, live Stripe credential activation and production browser checks remain pending at this checkpoint. Stripe receipts are test-mode simulations, not revenue. Proactive time eligibility used synthetic aged fixtures against remote validation storage, not a natural 24-hour user wait. Traffic classification by user agent is heuristic. No research benchmark, GPU run, compression ratio, anomaly metric, or held-out proof was performed.

## Local and Drive artifacts

Receipts are saved in `artifacts/works-release-20260905/`; this analysis and journal are also in `docs/proof_runs/2026-09-05/`. `drive_manifest.json` records the actual mirror result. Keys, bearer receipts and database tokens are excluded.

## Next step

Complete live Stripe verification, merge verified PRs, deploy backend before the Pages relay, check production and update this checkpoint with deployment receipts. Research performance remains outside this release's claims.
