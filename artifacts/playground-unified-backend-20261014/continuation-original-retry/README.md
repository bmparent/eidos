# Frozen original-order checkout retry — 2026-10-14

An existing pending original order could send the latest catalog version to the payment fixture after a dispatch failure, while its immutable order retained the prior version. A matching signed paid acknowledgement then rejected that inconsistent checkout. A missing or replaced historical archive also did not prevent dispatch.

The retry now reads product/edition/version and price from its existing order. It verifies the retained original archive and the SHA-256 of its actual bytes before dispatch or returning a cached session. An existing original attempt needs its retained archive; it does not require the latest catalog archive to exist. Product, edition, account ownership, terms, mode, rate/challenge checks and terminal order state still apply. An owner/attempt conflict rejects before attachment preparation. Attached-page validation remains independent.

Both `templateShop.ts` mirrors contain the same change. No original archive was edited. No real payment, email, or provider call was made: these tests replace outbound transport with controlled fixtures and reject unknown network destinations.

## Validation receipts

- `before-tests.txt`: both new focused regression tests failed before the fix. The first sent version 1.0.3 for frozen 1.0.2; the second dispatched despite missing historical bytes.
- `after-focused-tests.txt`: both focused regressions passed after the fix.
- `after-template-commerce-tests.txt`: complete backend template commerce suite, 23 tests passed.
- `after-backend-delivery-tests.txt`: four backend delivery tests passed using the package script that supplies their existing app-relative fixture cwd.
- `after-commerce-tests.txt`: retained failed invocation from repository root. All 23 commerce tests passed, but four unrelated delivery tests could not find app-relative fixtures from that cwd. The corrected package-script run is recorded separately; this was a command/cwd issue, not a hidden product failure.

The positive regression starts with a failed fixture dispatch, models a frozen historical 1.0.2 order at 8900 cents while the current server catalog remains 1.0.3 at 9900 cents, and makes only the historical archive available. Retrying creates one session for the same order; metadata and description keep 1.0.2 and 8900 cents. The fixture signs the payment event using the exact dispatched metadata, verifies it twice without duplicate effects, and downloads the original historical bytes, hash and filename.

The negative regression rejects missing retained archives, changed declared hashes, and changed bytes that falsely claim the stored hash. It keeps the order pending with no session and makes zero successful checkout dispatches.

## Related frontend receipts

`artifacts/playground-unified/continuation-recovery/commerce-attachment-tests.txt` records all seven attachment tests passing. An older test expected a rejected owner/attempt conflict to create an empty attachment table; that incidental side effect was removed by earlier validation, and the test now requires no attachment schema mutation.

`commerce-snapshot-recovery-tests.txt` records 18 snapshot/recovery tests passing. Existing provenance wording is now independent of purchase status: the same frozen ZIP remains the reviewed artifact before and after purchase, entitlement comes from a separate verified order receipt, and licensed original archives/notices are delivered separately. These are source/fixture checks; no hosted account, inbox, real TEST payment, physical-device or unfamiliar-user acceptance is claimed.

## Remaining work

The root task must refresh/export the isolated renderer graph containing the already changed provenance source, validate the complete combined candidate, update source/deployment pins and the final handoff, and mirror the new evidence folder. The original release/payment/provider gates remain separate.
