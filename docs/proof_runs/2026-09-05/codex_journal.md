# Codex Journal — 2026-09-05

## What happened today

Completed the Eidos Works hosted service validation described in the companion plain-language analysis. The production rollout remains pending at this checkpoint.

## What was accomplished

Dedicated free durable databases, secret separation, real opt-in AI, moderated guest/reply/agent checks, revoked keys, quota caps, real test payments and revocations, actual email delivery and GA4 receipt.

## Tests and commands run

- `npm --prefix apps/sentinel-lab run works:migrate` — passed against production Turso.
- `node --env-file=apps/sentinel-lab/.env.validation.local apps/sentinel-lab/scripts/migrate-works.mjs` — passed against separate validation Turso.
- `node --env-file=apps/sentinel-lab/.env.local --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/verify-works-storage.ts` — passed.
- `node --env-file=apps/sentinel-lab/.env.validation.local --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/verify-works-proactive.ts` — passed; owned fixtures removed.
- `npm --prefix apps/sentinel-lab run lint` — passed.
- `npm --prefix apps/sentinel-lab test` — 21 passed (17 existing research/UI tests, four platform tests).
- `npm --prefix apps/sentinel-lab run build` — passed.
- Companion website: `npm run test:platform` — 17 passed; `npm run test:analytics` — one passed; Snapshot smoke, lint, build, prerender, editorial, Insights dist, URLs, Functions build — passed.

## Problems encountered

Cloudflare rejected a Node-supported redirect option. The real runtime regression is fixed. Reply verification replaced the saved-message acknowledgement; fixed and tested. Initial provider auth issues were resolved. Live Stripe key creation is awaiting its requested email verification; a Chrome extension popup blocked automation on the Gmail tab.

## What changed

Backend adds explicit remote-storage/provider/proactive verification scripts and a normalized schema-hash description. The companion website fixes relay/runtime behavior, consent/navigation tracking, deterministic ZIP newlines, inquiry delivery and reply acknowledgement.

## What did not change

Core research/model behavior was untouched: no reservoir, RLS, surprise, threshold, compression, experiment executor or research artifact changes.

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

## Artifacts generated

Local `artifacts/works-release-20260905/` contains JSON receipts, logs, downloaded test ZIP, this journal, analysis, proof ledger and a static progress dashboard. Companion website artifacts are under `artifacts/release-20260905/`.

## Google Drive archive status

See `drive_manifest.json` for the mounted root, destination, copied files, hashes and any failure. Copy is attempted to the configured mounted Drive, never required for test success.

## Thoughts on improvement

Hosted runtime smoke checks caught a real problem that pure Node mocks missed. Keep one small Workers runtime test and separate test/live payment stores. Reuse this evidence instead of repeatedly paying for provider smoke calls.

## Where to improve next

Finish the production rollout and verify the operator-facing handoff against its actual deployment IDs.

## Anything that stands out

A successful browser redirect cannot grant a purchase. Webhook receipts and durable entitlements provide the proof. All demonstrated purchases were test mode and do not show revenue.

## End-of-task summary

1. Files changed: verification scripts, vendor hash documentation, reports and companion website runtime/UX changes.
2. Core behavior changed: no.
3. Tests: 21 Lab, 17 website platform, one analytics, Snapshot smoke and build checks passed.
4. Commands: listed above; executed from repository roots.
5. Artifacts: JSON receipts, logs, ZIP integrity, reports, ledger and dashboard.
6. Plain-language analysis: written.
7. Journal: written.
8. Google Drive: actual result in drive_manifest.json.
9. Limits: production rollout and live key verification pending at this checkpoint.
10. Unimplemented follow-ups: no research algorithm/benchmark expansion.
11. Proof Logic + Meaning: written.
12. Math: quota inequality, token reservation and payment predicates included.
13. Philosophy: reproducibility and restraint before automation.
14. Improvement: hosted evidence and durable storage replace assumptions.
15. North-star connection: reproducible operation and incident/transaction receipts only.
16. Evidence files: listed in the proof section.
17. Uncertainty: no research-performance or real-revenue claim.
