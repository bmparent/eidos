# Playground provider security review and narrow repairs — 2026-10-14

Two source findings were repaired. Focused validation passed **47 tests, 0 failures**, and backend TypeScript checking passed. All provider transports were controlled fixtures. No real key, counting request, generation request, payment or hosted account was used.

## Finding 1: operator budget changes could bypass the preliminary $5 check

Before the repair, the adapter checked the configured budget before asynchronous work. The reservation and dispatch statements checked the current configured budget against committed cost, but did not independently enforce the authorized $5 ceiling. A concurrent operator increase could therefore allow an already-admitted request to proceed under a larger budget.

The repair defines the real test ceiling in the ledger and checks it in operator configuration and inside the atomic SQL predicates for reservation, dispatch and input-count admission. It applies to real `owner-approved` rate cards. Historical controlled fixture budgets remain supported. It does not release or erase existing submitted/settled/unresolved cost.

The regression simulates an operator increase after a preliminary admission/reservation. A new reservation is rejected and an already-reserved dispatch receives no dispatch token. It also verifies that synthetic fixture budgets still work.

## Finding 2: outbound counting preceded reference/account validation and had no durable claim or quota

Before the repair, malformed action references or members without billing setup could trigger an upstream count before the eventual quote rejection. Concurrent duplicate references could also perform multiple count requests. This was quota exposure rather than a spend bypass under an evidenced zero-fee counting contract; the actual billing gate remains unresolved and disabled.

The repaired order is:

1. Validate the typed context and idempotency reference; load or create the owner-bound quote using existing ledger validation. A missing billing account is rejected before outbound I/O.
2. Atomically claim the count against that quote, checking its state, expiration/month, source-payment status, frozen account, available funds, monthly cap, current policy, real $5 ceiling and remaining provider commitment headroom.
3. In the same atomic claim, require fewer than **20 attempts per owner per rolling hour**, **60 attempts globally per rolling hour** and **200 lifetime attempts** in the existing TEST ledger. These limits are shared across projects, models, workers and processes using that database. Failed and uncertain attempts still count; month rollover never resets the lifetime limit.
4. Persist the count client UUID, request SHA-256 and full serialized request before calling the counting endpoint. Only the winning claim may send. A failed or uncertain attempt is retained and never retried with that action reference.
5. Store a successful full-input proof and completed count receipt together in one database transaction. Generation still uses the exact frozen bytes and requires its separate atomic money reservation and one-time dispatch claim.

Count admission does not reserve or debit customer money. A denied count may leave an unreserved quoted-action receipt; it grants no entitlement, creates no provider commitment and sends no upstream request. Existing quote storage/retention policy is unchanged by this repair.

The counting preflight is still permitted only after `verified-zero` is supported by actual primary/billing evidence. Its unresolved fee is not assigned an invented zero cost by these tests. If counting is billable, a reviewed explicit fee/reservation contract must replace this gate before real counting starts.

## Regression coverage

- Invalid reference and missing billing setup cause zero count calls.
- Zero monthly cap, insufficient funds, frozen accounts, refunded funding and exhausted provider budget cause zero count calls.
- Twelve concurrent identical quote attempts perform one durable count request.
- Two independent libSQL connections cannot double-claim the same count.
- A failed connection retains the count UUID, unknown provider identity/token count and non-retryable failed state.
- Owner/hour, global/hour and lifetime quota boundaries block before upstream I/O; switching models does not reset them.
- The transport fixture verifies the durable claim and frozen request exist before every outbound count.
- Prior dedicated-key/global-kill, public readiness, authenticated owner, exact customer maximum, generation deduplication, unknown liability, provider identity, cache/reasoning billing, overrun and refund regressions remain passing.

## Commands and evidence

From the backend repository root:

```text
npm --prefix apps/sentinel-lab run test:playground-ai
npm --prefix apps/sentinel-lab run lint
```

- `focused-security-tests.txt`: **47 passed**, comprising 25 accounting and 22 provider/API tests.
- `backend-security-typecheck.txt`: **passed**.
- `source-manifest.json`: SHA-256 of the working source; parent finalization records the eventual commit/deployment.
- `mirror-parity.json`: paired ledger/adapter/schema/migration/API bytes.

The full backend suite, compiled parity and manual-only preview deployment are being finalized separately by the root task. Earlier source/deployment receipts remain historical. Windows libSQL retains synthetic temporary database handles until process exit in some cross-connection tests; this is logged and does not skip the assertions.

## Remaining limits

No real counting fee, provider quality, API account access, hosted authentication, TEST commerce funding or production readiness is accepted here. All real transport remains off. The $5 guarantee covers this adapter's reviewed provider contract and the same persistent TEST ledger; starting another live-spend database would create a separate ledger and is not an authorized test procedure. Provider behavior outside its declared token/tier contract remains an overrun/unknown-liability incident that pauses the route; customer charges remain capped.
