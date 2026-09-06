# Codex Journal - 2026-09-05

## What happened today
Completed the service validation and deployed the backend before the cinematic website. Production AI, community, inquiry delivery, GA4 and hourly maintenance are verified. LIVE purchasing is now enabled after owner verification, a separate live webhook, deployment and unpaid checkout checks.

## What was accomplished
Merged PRs 6 and 37 plus maintenance PR 7 and Contact spacing PR 8. Provisioned isolated free databases after explicit owner terms approval. Saved real hosted receipts and verified production layouts, routing, indexing and the preserved research interface.

## Tests and commands run
- Backend: npm --prefix apps/sentinel-lab run works:migrate; npm --prefix apps/sentinel-lab test; npm --prefix apps/sentinel-lab run lint; npm --prefix apps/sentinel-lab run build.
- Separate validation migration: node --env-file=apps/sentinel-lab/.env.validation.local apps/sentinel-lab/scripts/migrate-works.mjs.
- Remote storage: node --env-file=apps/sentinel-lab/.env.local --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/verify-works-storage.ts.
- Proactive fixtures: node --env-file=apps/sentinel-lab/.env.validation.local --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/verify-works-proactive.ts.
- Website: npm run test:platform; npm run test:analytics; npm run test:snapshot; npm run lint; npm run build; npm run verify:prerender; npm run verify:editorial; npm run validate:insights:dist; npm run verify:urls; npm run build:functions.
- Final CSS check: npm run build and npm run verify:prerender, followed by browser measurement on local build and production.
- Deployment: node node_modules/wrangler/bin/wrangler.js pages deploy dist --project-name eidosworks --branch main --commit-hash c9e35f43c5c4c8abd90a4a11ffdceea49d603af9 --commit-dirty=false.
All commands ran from the relevant repository root. All listed checks passed. Lab tests: 21. Website platform tests: 17, analytics: one, plus Snapshot smoke. No extra algorithm tests were added for the CSS-only follow-up; browser checks, build, prerender and CI cover it.

## Problems encountered
Workers rejected redirect:error; manual redirect rejection fixed it. A Turnstile callback replaced reply confirmation; state retention fixed it. GitHub requests through the site returned 403; direct authenticated backend scheduling passed. Git CLI lacked workflow scope; the already-authorized GitHub connector successfully updated that one workflow. Contact padding hid its label under the mobile header; shared page padding fixed it. Stripe email verification initially failed across browser sessions. Local Chrome and the owner-completed SMS challenge allowed secure key capture and live activation.

## What changed
Public platform service wiring, request/receipt controls, browser UI fixes, the maintenance workflow, tests and release artifacts. Exact evidence and source references appear in release-handoff.md.

## What did not change
No reservoir, RLS, surprise scoring, Sentinel thresholds/labels, compression, experiment executor, held-out inputs or research artifacts were changed. Original dirty checkouts were preserved.

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

## Artifacts generated
See artifact_manifest.json, release-state.json and release-handoff.md. Both repository-local folders are retained.

## Google Drive archive status
The configured mounted Drive mirror is G:/My Drive/Eidos_Brain_Proof_Phase/2026-09-05/works-release-20260905. drive_manifest.json records copies, hashes and limitations. No secret files are mirrored.

## Thoughts on improvement
Hosted runtime checks found failures that Node-only tests missed. Keep preview/live storage and payment modes separate, and reuse these provider receipts rather than repeating paid smoke calls.

## Where to improve next
Review actual customer outcomes and operating budgets through the existing dashboards. No further release action is required.

## Anything that stands out
Test revenue, delivered QA inquiries and human-looking QA sessions are evidence of plumbing, not business outcomes.

## End-of-task summary
1. Files: platform wiring, focused regression checks, two UI/runtime fixes, workflow and reports.
2. Core research behavior changed: no.
3. Tests: 21 Lab, 17 platform, one analytics, Snapshot smoke and required build checks passed.
4. Repo-root commands: listed above.
5. Artifacts: JSON receipts, logs, test ZIP, reports, manifest, ledger and dashboard.
6. Local folders: artifacts/works-release-20260905 and companion artifacts/release-20260905.
7. Drive: mounted copy and hashes recorded in drive_manifest.json.
8. Plain-language analysis: written.
9. Journal: written.
10. Known limit: no real-money purchase was made; paid fulfillment evidence is from Stripe TEST mode.
11. Follow-up: normal operational review only; no research expansion implemented.
12. Proof Logic + Meaning: written.
13. Math: quota inequality, conservative token reservation, payment predicates.
14. Philosophy: reproducibility and restraint before automation.
15. Improvement: actual deployed service receipts replace assumptions.
16. North-star: reproducible operation and readable receipts; no new research-performance proof.
17. Evidence: named in the proof section and artifact manifest.
18. Remaining uncertainty: no live revenue, natural-time proactive wait, physical-device or held-out claim.

## LIVE activation closeout - 2026-09-06T00:03:43.024Z

Production purchasing is enabled in LIVE mode. The dedicated restricted key and separate webhook we_1UCTRjKC8pRG5Tr9KUpT4H8F serve https://eidos-works.com/api/shop/webhook with checkout.session.completed, checkout.session.async_payment_succeeded, charge.refunded and charge.dispute.created. Test/live keys, signing secrets and databases remain separate. The deployed browser opened the correct one-website license at $29.00 USD. Authenticated Stripe readback confirmed 2,900 cents, USD, card payment, and unpaid status; the durable order remained pending. The browser cancel link returned to the product. The verification session was then expired, and download returned 403 before and after expiration. No real-money payment was made. A synthetic ignored event signed with the live secret returned 200; unsigned and test-secret-signed requests returned 400 and created no entitlement or event record. This checks the production relay and signing configuration, not a provider-generated live payment event.

Validation: node apps/sentinel-lab/.vercel/stripe-live-activate.mjs inspect/configure/preflight; node apps/sentinel-lab/.vercel/live-release-config.mjs inspect/enable/deploy/status; node apps/sentinel-lab/.vercel/verify-live-release.mjs inspect/expire. These private operational helpers run from the repository root and keep credentials in ignored server configuration. Their sanitized outputs are the live JSON receipts. No application code changed during this final activation; the already-tested merged source 4de48cf149e07d25195130b20e0b52a36533ee09 was redeployed with production environment changes.
