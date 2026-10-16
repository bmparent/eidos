# Playground hard-cap accounting continuation — 2026-10-14

This is a source and local fixture audit of all 14 required cap scenarios in the original handoff. It does not accept provider integration, prices, a hosted AI proposal, payment delivery, inbox delivery or physical devices. Provider credentials and Brent's total USD provider budget remain unanswered. No provider adapter, payment request, mail request, commit or deployment was added by this work.

## Changes and before evidence

Four related accounting defects were reproduced and repaired in the exact frontend/backend copies of `functions/_shared/platform/playgroundAi.ts`:

1. A second reconciliation could replace an already recorded provider response/request identity and reduce a pessimistic liability with unrelated evidence. First identity is now claimed atomically and all later evidence must match both IDs and the dispatch token. The negative single-client and independent-client tests failed on the original code (`before-tests.log`). Matching late evidence still resolves actual provider expense and never revives a waived customer charge.
2. Usage outside the supported meter, including a very large safe integer, a fractional token count, missing usage and unsupported call count, threw before pausing dispatch. `extreme-usage-audit-probe.json` and `before-unmeterable-tests.log` retain the failure. The failure transaction now pauses the mode, retains the original held or pessimistic liability and records reconciliation pending. It does not invent an actual expense. A subsequent invalid report preserves a prior customer waiver; valid matching evidence may settle the actual amount without automatically unpausing the route.
3. Even when provider cost could be metered, retail multiplication could overflow before `Math.min` applied the customer's approved maximum. `before-retail-overflow-tests.log` reproduces that failure. The customer pricing input is now capped at the frozen provider maximum before multiplication, while actual provider expense remains uncapped and above-bound evidence pauses the route.
4. Result serialization/size failure could abort before known provider expense and an above-bound route pause were committed. `before-invalid-result-tests.log` preserves the original failure. Oversized, circular, BigInt and otherwise non-JSON results now cause a customer technical waiver while known provider cost settles and required pausing proceeds. Tests cover both ordinary and above-bound usage and ensure later replacement-result replay cannot revive the charge. A future trusted adapter still owns operation/schema validity; the ledger independently enforces serializable bounded result storage.

Only the two accounting module copies and `apps/sentinel-lab/tests/works-playground-ai.test.ts` are code changes from this work. Schema, provider transport, external configuration, checkout, prices and model selection are unchanged.

## Source map for the 14 scenarios

Paths below are relative to the backend repository. `ledger` means `apps/sentinel-lab/lib/works/vendor/functions/_shared/platform/playgroundAi.ts`; `tests` means `apps/sentinel-lab/tests/works-playground-ai.test.ts`. Line references describe this uncommitted continuation; source hashes are recorded in `receipt.json`.

| # | Required invariant | Source mechanism and local evidence | Acceptance boundary |
|---|---|---|---|
| 1 | Concurrent $0.60 actions with $1 remaining admit one | Ledger:196 performs one conditional write for both customer and provider reservations; tests:65 and tests:388 cover same-client and independent-client contention. Rejected actions have no synthetic dispatch. | Real provider execution is not enabled. |
| 2 | Replay produces one reservation, dispatch and settlement | Ledger:172 freezes owner/idempotency/fingerprint; ledger:221 claims a one-time dispatch token; ledger:240 fixes immutable reconciliation identity. Tests:74, tests:239 and tests:493 cover replay and conflicting evidence. Migration `0005_playground_ai.sql`:48 also enforces unique action keys/provider responses. | Counter is a controlled dispatch fixture, not a provider acknowledgement. |
| 3 | $50 balance/$10 cap; top-ups do not raise cap | Ledger:96 stores the explicit cap separately from grants; ledger:196 includes all retail commitments. Tests:84 consumes included then purchased value and rejects after top-up. | No LIVE allowance, cap or price is approved. |
| 4 | Shared owner across projects; cross-account denial | Ledger:157 scopes action reads by owner and mode; ledger:196 scopes balances/period usage by owner. Tests:96 varies document, tests a signed second owner and rejects forged owner fields. `api/playground/ai.ts`:9 derives the member from authentication. | This proves server identity behavior locally, not physical second-device acceptance. |
| 5 | Reject forged totals/model/tools and unbounded expansion | Ledger:46 limits the rate card to one call, no retry/subagent/paid tools and bounded tokens/runtime; ledger:172 validates scope and freezes the server card. Tests:113 rejects forged fields, excess context, invalid prices and unsupported routes. Tests:182 also rejects unusable meter evidence after dispatch. | Actual SDK retries, framing, reasoning usage and provider rate bounds need a separately accepted adapter. |
| 6 | Release below maximum; never charge above it | Ledger:264–308 meters and caps customer pricing while recording known provider cost separately. Tests:125, tests:136 and tests:182 cover normal, above-bound, retail overflow and unmeterable paths. | Unknown expense is explicitly retained as a hold/pessimistic reserve; it is not claimed to be measured actual expense. |
| 7 | Technical failures restore allowance; provider cost remains | Ledger:282–304 waives unusable proposals while storing provider expense; tests:148 and tests:157 show customer funds restored but provider budget still consumed. Tests:281 covers later waiver after a refund. | Provider output validation and real failure billing are not accepted by these accounting fixtures. |
| 8 | Unknown submission retains upstream liability; no blind retry | Ledger:221 only issues one token; ledger:310 marks unknown; ledger:317 releases only proven pre-dispatch actions; ledger:324 preserves upstream maximum at the 24-hour customer waiver. Tests:182, tests:221, tests:239 and tests:493 include missing/invalid usage, crash/timeout, late evidence and identity races. | The 24-hour policy is tested with a supplied clock. Hosted maintenance cadence and provider reconciliation retrieval remain unproven. |
| 9 | Duplicate/unpaid/out-of-order events cannot duplicate grants | Ledger:335 requires signed commerce evidence; ledger:347 binds immutable merchant/order/owner/SKU/version/price/mode; migration:20 is unique per order/kind. Tests:263 and `works-templates.test.ts`:88 test duplicate grants, wrong merchant/mode, unpaid evidence, retry after mail failure and late paid replay after refund. | Stripe and mail responses are fixtures. No real payment or inbox claim follows. |
| 10 | Cap edits preserve existing commitments | Ledger:123 exposes the effective floor and pending lower limit; ledger:196 atomically admits against the requested cap; accepted existing commitments remain valid. Tests:319 and tests:451 exercise reduction with admission, dispatch and settlement, including two independent connections. | The tests cover both possible admission/reduction winners rather than asserting a scheduler order. Hosted distributed acceptance remains separate. |
| 11 | Month reset cannot duplicate balance or misassign late cost | Ledger:33 uses server UTC; ledger:123 filters retail usage by frozen action period; migration:57 keeps prepaid availability across all periods; provider commitments are lifetime for the mode. Tests:330 and tests:344 cover scarce and ample provider budget, simultaneous old/new-month jobs and late settlement. | No client clock can choose the period; calendar behavior is local deterministic evidence. |
| 12 | Aggregate provider budget stops all customer work | Ledger:110 and ledger:196 sum all mode actions, independent of customer/model; ledger:221 rechecks at dispatch. Tests:368, tests:388 and tests:407 cover competing customers and budget reductions around admission/dispatch. Unsupported tools are disabled by the rate card. | Mixed approved model execution is not enabled or accepted. Manual editing/download acceptance belongs to separate browser receipts, not these ledger tests. |
| 13 | Refund/chargeback/replay cannot create reusable free balance | Ledger:366 freezes affected accounts; ledger:196 and ledger:221 also reject revoked source orders. Tests:263 and tests:281 cover duplicate reversal, mixed consumed/held grants, pre-dispatch cancellation, deadline waiver, late evidence and replay. Existing template tests cover signed dispute/refund revocation. TEST funding cannot grant LIVE. | Released values after a refund can remain visible in a frozen account, but are not reusable; there is no unfreeze endpoint here. No surprise charge is created. |
| 14 | Outage/insufficient balance preserves manual work and avoids fallback | Ledger:71 fails closed on disabled/invalid policy; ledger:196 denies insufficient funds; ledger:264 pauses unmeterable usage. Tests:65, tests:148, tests:182 and tests:368 cover denial, failure expense and disabled dispatch. `api/playground/ai.ts`:28 exposes no generation route. | A real provider outage, no-downgrade adapter behavior and hosted manual-work continuity still need separate acceptance after authorization. |

## Validation and meaning

The first three fixes passed 19 focused tests and a full backend run of 104 passed/1 skipped. The added mixed-refund, ample-budget month-boundary and multi-connection cap-race coverage then passed 22/22 (`final-expanded-focused-tests.log`) without requiring a behavior change. The final result-storage fix and all coverage pass 23/23 in `final-result-focused-tests.log`. Final full-suite/typecheck outcomes and exact hashes are in `receipt.json`; older logs are retained as scoped intermediate evidence, not represented as final source runs. Do not add focused totals to the full-suite total or call a fixture a hosted/provider result. The Windows file-symlink test is skipped because Developer Mode/privilege is unavailable.

Exact repo-root commands:

```text
node --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs --test apps/sentinel-lab/tests/works-playground-ai.test.ts
npm --prefix apps/sentinel-lab test
npm --prefix apps/sentinel-lab run lint
```

Frontend repo-root scoped check:

```text
npx eslint functions/_shared/platform/playgroundAi.ts
```

The accounting logic is:

```text
customer_commitment(owner, period) = held approved maxima + settled included/purchased charges
available_balance(owner) = grants - revoked amounts - all-period customer holds/charges
provider_commitment(mode) = held provider maxima + settled actual costs + unresolved pessimistic costs
admission: customer_commitment + Q <= requested_limit
           available_balance >= Q
           provider_commitment + provider_maximum <= owner_budget
customer_charge = min(approved_maximum, retail_rate(min(actual_provider_cost, frozen_provider_maximum)))
```

The capped input to customer pricing is equivalent for a monotone positive rate card and prevents overflow before the customer cap is applied. Provider expense remains independent. Unknown usage keeps its existing liability and stops dispatch; it never becomes free budget because a parser rejected it. The principle is that uncertainty must remain an obligation, and later evidence must belong to the same recorded request.

The three additional coverage gaps did not justify new product behavior. Remaining external gates are the unanswered key choice and total USD provider budget, current accepted provider pricing, a bounded adapter, hosted authentication/persistence/concurrency, an actual authorized proposal with durable provider usage, maintenance cadence, and separate payment/inbox/device acceptance. The accounting module itself still provides no provider transport. No additional confirmed accounting defect was identified in this bounded audit; this is not an exhaustive formal proof.

## Artifact and archive status

All before failures, the probe, after logs, audit and hash receipt are retained in this repository folder. Logs contain synthetic fixture identities and local paths. Windows libSQL may retain synthetic temporary database handles until the test process exits; the tests preserve that diagnostic instead of hiding cleanup failures. The parent task owns the final Drive archive and deployment evidence. This continuation did not copy to Drive or change hosted state.
