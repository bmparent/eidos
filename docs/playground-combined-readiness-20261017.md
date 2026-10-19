# Combined Playground readiness — October 17, 2026

## Scope and provenance

This disabled candidate combines frontend visual source fbee9a7761df9cb848aff8720aa5a17fb37c1b62, AI PR122 source 0b82f72baf03c5758d2ef2314476b7c0ebfe3946 and purchase PR124 source a5f8e34a25b8e78496801cd53afa5a4bf38640bd. Backend combines AI PR81 source 7af331e8117f5dce0c927964c3d249e80f59f997 and purchase PR82 source ab4431493cf963901e6013230c32dedf8e684736. Original commits remain in merge ancestry. Tutorial PR125 is outside this integration.

Backend automatic deployment is explicitly disabled for integrate/playground-readiness-20261017 before branch creation. No deployed migration, TEST pin, activation flag, credential, provider spend or public-main change is part of this candidate.

## Combined verification

Local: 120 Playground tests, 37 platform tests and template suites pass; frontend lint/typecheck/client+SSR build/Functions compilation pass (one existing Wellway lint warning). Backend 20 MJS and 141 TS tests, typecheck and optimized build pass with commerce disabled. Browser acceptance runs in draft PR CI; exact remote heads and run results are recorded in the paired PR descriptions.

Twenty shared AI/payment contracts including migration 0008 match byte-for-byte after LF normalization. The separate 18-file delivery/AI operation source graph is refreshed and verified, inventory SHA-256 98e369dbec16cfebe58eb2c7540616317b5abe875eed45e5dd434d89bf07ba4b. Legacy cloud renderer is unchanged. Migration preservation/idempotency/rollback checks run in disposable fixtures only.

## Actual candidate offer for owner approval

Server source functions/_shared/platform/templateCatalog.ts sets one-time USD 99 developer editions:
- switchboard-developer-v1, version 1.0.3
- tideglass-developer-v1, version 1.0.2
- matter-developer-v1, version 1.0.2

These are the three current Playground starters. nightjar-developer-v1 version 1.0.5 also exists at USD 99 in the wider catalog, but is not a current starter. WordPress editions cost USD 129 and are not eligible for approved-page attachment. Existing USD 29 Cinematic Starter is separate. Source pricing is not proof of owner approval or deployed LIVE availability.

The requested offer decision is whether to sell the three developer editions above at those prices with the reviewed local-page package attached alongside the immutable original developer archive. Confirm the published terms before activation: candidate /shop/template-terms proposes one commercial website including a client website, no template redistribution, 30 days installation help, and 90 days defect corrections for the purchased major version. It links existing /terms and billing support. Those terms promise investigation and replacement or an appropriate resolution for failed download, duplicate charge or defects, preserving applicable rights; they state no refund window or blanket no-refund rule. Owner approval must resolve any desired additional refund conditions before activation; this integration makes no new legal commitment. Hosting, domain, third-party setup and ongoing services are separate; no recurring Eidos fee. Refund/revocation stops future retrieval, not already downloaded files. No funded customer AI allowance is included.

## Next disabled hosted rehearsal (separate release approval)

1. Integrate/review the paired draft and confirm exact-head CI. Freeze source pair and preserve current TEST deployment and its archived evidence.
2. Obtain permission to deploy an isolated disabled rehearsal and apply additive migration 0008 to the intended Works database. Take/verify appropriate existing recovery coverage first. Do not copy or relabel TEST rows. Keep all sales flags absent/false and AI dispatch disabled.
3. Verify canonical production routing and expected merchant account binding read-only, database identity/schema, private current/retained archive hashes, and actual sender/challenge/rate-limit bindings without printing secret values. Credential creation/transmission or persistent binding changes require their own user handoff/confirmation.
4. Confirm public config reports sales/AI unavailable; attempted freezes/checkouts fail closed; existing authorized paid-read access remains available on an appropriate fixture or separately approved rehearsal data. Check signed-webhook rejection and idempotency with permitted controlled fixtures, not a new customer transaction.
5. Resolve the historical TEST acknowledgement gap by inspecting recorded checkout-event delivery specifically to endpoint we_1UQym7KC8pRG5Tr9lcIO9qKu in Stripe TEST dashboard. Capture HTTP 2xx for that endpoint. Do not repeat the already-passed USD99 TEST purchase/refund. Aggregate pending_webhooks and app-paid state do not establish that endpoint's acknowledgement.

## Final production gates

Canonical PUBLIC_SITE_URL=https://eidos-works.com; EIDOS_PLAYGROUND_DELIVERY_MODE=live; exact approved EIDOS_TEMPLATE_VERIFIED_EDITIONS; expected merchant acct_1Rj03gKC8pRG5Tr9. Readiness also requires Works DB/archive storage, LIVE Stripe binding, signed webhook secret, >=32-character rate secret, Resend sender/key and real Turnstile. TEST flags and local bypass must be off. EIDOS_TEMPLATE_LIVE_ENABLED, EIDOS_SHOP_ENABLED and EIDOS_PLAYGROUND_DELIVERY_LIVE_ENABLED remain absent/false until explicit activation approval.

Existing read-only merchant evidence (charges/payouts enabled, no due requirements) does not prove deployed credential binding. An actual LIVE charge requires separate exact purchase authorization and a validation plan covering merchant/mode/total, status, inbox, both archive hashes, owner-only recovery and monitoring. Rollback disables new sales while retaining paid-read code and all LIVE tables/order history.
