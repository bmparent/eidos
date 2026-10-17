# Playground purchase readiness — October 17, 2026

## Source and isolation

Frontend starting source: `9352f4bd440227d677633d52c182e1b80515c25d` (pending PR #111). Backend payment evidence source: `6f7a29350ed8837ba86ce5d26603bfd674ef2c0e` (PR #78). Neither the canonical production site nor the connected TEST deployment is changed by this work. This source fix is independent of AI activation and visual-editor changes.

## New safety correction

Checkout refreshes authenticated capabilities before dispatch. Previously a refreshed server price could differ from the price displayed when terms were accepted. The client now compares reviewed and refreshed owner, product, edition, version, archive identity, currency and price. A changed price stops before checkout dispatch, replaces displayed details, clears acceptance and challenge, and requires another explicit acceptance. Both increases and decreases require review. Unavailable or mismatched editions still fail closed. This guard does not invent a new price or change the server-owned catalog. The final Stripe checkout remains the payment total review; this client check is not a server-side immutable quote contract.

## What is already established

The backend's `docs/playground-connected-test-20261015.md` records the genuine $99 USD TEST purchase, two owned-account isolation, exact approved and original archive bytes, actual receipt and recovery inbox arrival, single-use recovery, $1 TEST allowance, genuine refund and access revocation. The accepted-mail race correction preserves a provider-accepted receipt. These are historical hosted receipts on their recorded revisions, not newly rerun transactions and not revenue. The refunded allowance is zero/frozen, and no LIVE AI allowance is promised.

A fresh October 17 read-only Stripe account retrieval confirms Eidos Works account `acct_1Rj03gKC8pRG5Tr9`, business URL `https://eidos-works.com`, charges and payouts enabled, no currently due requirements, and no disabled reason. The connector exposes only LIVE context. This is merchant readiness, not proof of the deployed credential, pricing approval, or TEST webhook acknowledgement.

## Paired implementation now prepared, still disabled

Ordinary template purchases already had guarded LIVE mode. This continuation adds the approved-page attachment path without changing deployed flags or credential bindings:

- `0008_project_delivery_live.sql` adds separate LIVE snapshot, chunk and order tables with `CHECK(mode='live')`. The original TEST tables, migrations, constraints and rows stay untouched. No backfill or relabeling occurs.
- Server reads/writes select hardcoded table names by validated mode; canonical origin, LIVE key shape and expected merchant identity are required for LIVE mode. New freezes/checkouts additionally require `EIDOS_PLAYGROUND_DELIVERY_LIVE_ENABLED=true` and all existing `templateLiveReady` checks. The new flag has no enabled default.
- Signed payment requires the stored binding's mode, session mode, account ownership, exact frozen hashes, version, amount/currency and verified merchant. Ambiguous duplicate attachment bindings fail closed.
- New-sales shutdown does not remove existing paid recovery/download access. Refund/dispute and late-payment replay retain the existing revocation boundary.
- Client accepts explicit TEST or LIVE capabilities, requires production verification for LIVE, presents the correct payment label, and separates browser attempts by mode. Price/edition/mode changes require renewed review. Paid recovery does not depend on new-sales capability or current catalog.
- Backend environment adapter forwards only the new named activation field. No environment value or secret was set.

Local tests use disposable SQLite/libsql databases and intercepted provider responses. They are not LIVE payment acceptance. Applying migration 0008 to a real database and enabling the flag require a separate release decision. Runtime rollback should disable new sales while retaining the current paid-read code; reverting to old TEST-only code after LIVE orders exist would interrupt access. Do not drop LIVE tables or discard paid order history as a rollback step.

## User-facing purchase path and existing terms

Prepare frozen revision → choose matching parent developer edition → inspect server price/version → approve the reviewed local page → confirm authenticated purchase availability → save exact reviewed snapshot → read existing edition terms → complete verification → Stripe checkout → explicitly check server status → download original and approved page separately. A return redirect alone grants nothing. Hosting, domains, ongoing services, forms and third-party services remain separate. Existing `/shop/template-terms` refers to the approved `/terms` refund policy and billing support; refund/revocation prevents future retrieval and cannot reclaim downloaded files. This patch makes no new license, refund, support-duration or LIVE price commitment.

## Remaining external evidence and owner decisions

1. In the existing Stripe TEST dashboard, inspect endpoint `we_1UQym7KC8pRG5Tr9lcIO9qKu` for the recorded genuine checkout's event and capture that endpoint's HTTP 2xx delivery acknowledgement. The event's aggregate pending-webhook count and app-paid state do not identify an individual endpoint acknowledgement. Do not create another payment or refund to obtain this evidence. The available connector cannot read the TEST context.
2. Approve the final production purchase terms/prices and whether approved-page attachment is part of that offer. Existing $99 TEST evidence alone is not a LIVE commercial decision.
3. After the paired implementation receives exact-head CI and hosted acceptance, explicitly approve production binding/deployment. Verify exact LIVE account binding without revealing credentials; preserve rollback and disabled new-sales controls. AI funding/activation remains separate.

## Production configuration/verification packet

Before enabling new approved-page purchases, verify the selected production backend has the intended dedicated Works database, all immutable current/retained private archives and migration 0008, and the paired relay pins exact reviewed code. These are release checks, not operations performed by this change.

The canonical `PUBLIC_SITE_URL` must be `https://eidos-works.com`; delivery mode `live`; the independent `EIDOS_PLAYGROUND_DELIVERY_LIVE_ENABLED` flag must remain absent/false until approved. The existing template LIVE flag, shop flag, verified-edition list, expected Stripe merchant ID, LIVE credential, signed webhook binding, real Turnstile configuration, rate secret, mail sender and production mail binding all must satisfy existing readiness checks. TEST flags, local bypass and public TEST challenge keys cannot enable LIVE checkout. Do not copy Preview-only mail alias behavior into Production. No new AI allowance is implied.

Before release, inspect exact endpoint acknowledgement for the recorded TEST event and run exact-head browser fixtures. Production activation still needs a separately authorized configuration/deployment decision and a permitted validation plan; no LIVE charge is authorized here. A first actual LIVE transaction, if later authorized, must verify mode, merchant, total, return/status, inbox delivery, both archive hashes, owned recovery and operational monitoring. Existing TEST evidence cannot be relabeled as that proof.

## New verification

- Initial independent price-consent patch: 19 focused tests and 111 full Playground tests passed
- Expanded mode-aware Playground suite: 117 passed, 0 failed, including controlled LIVE checkout/payment, owner isolation, disabled sales with paid recovery, refund/replay and migration preservation
- Backend suite: 20 MJS + 134 TypeScript tests passed; optimized backend build passed with commerce disabled (private-archive activation gate not exercised)
- Frontend TypeScript and production client/SSR build passed; lint has zero errors and one pre-existing Wellway fast-refresh warning
- Backend typecheck passed
- Independent read-only review found the retained-session price edge; immutable server pricing plus explicit matching review and consent reset now close it. Reviewer found no remaining blocking mode/auth/recovery issue
- New 390px controlled browser scenario checks updated displayed price, cleared terms, no checkout until renewed consent, then exactly one intercepted checkout; not executed here because the browser executable is absent and this environment has prior browser-launch restrictions. Exact-head CI browser verification remains pending
- Additive migration/idempotency and rollback checks execute only inside disposable in-memory fixtures; original TEST rows/chunks remain byte-identical
- No provider charge/refund, deployed schema mutation, feature-flag change, credential operation, TEST repin or deployment performed
