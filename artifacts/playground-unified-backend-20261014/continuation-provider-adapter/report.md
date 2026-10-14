# Playground provider adapter — controlled source acceptance, 2026-10-14

The server now supports a full-input quote, exact-maximum approval, one provider generation and a validated typed proposal. Controlled tests pass. No real provider request was made. This report does not accept hosted accounts, model quality, TEST payment, mail delivery or production activation.

## What changed

- Added `playgroundAiProvider.ts`, the server-only Responses adapter, mirrored into both checkouts.
- Added quote/generate actions to the existing authenticated Playground API. Billing owner comes from the signed-in member. Browsers cannot supply model, prices, provider budget, tools or another owner.
- Added an immutable full-request preflight table to the existing AI ledger migration. It captures the exact serialized Responses payload, SHA-256, provider-counted full input tokens and count-request identity.
- Extended optional rate/usage categories for cache reads, cache writes and reasoning, preserving historical cards and fixtures. Recorded provider identities cannot be replaced during reconciliation; response-header identity survives malformed response bodies.
- Added dedicated server credential and activation types/environment forwarding. `OPENAI_API_KEY` is never a Playground fallback. Secret values were not read, logged or used.
- Added a separate public `playgroundAiReady` configuration flag so a dedicated Playground key does not depend on the unrelated legacy assistant key. Legacy `aiReady` is preserved. Account readiness remains separate, and the authenticated ledger is the final budget gate. Public readiness performs no database or provider I/O. The adapter also enforces the global `EIDOS_AI_ENABLED` kill switch before count or generation dispatch.
- Added 16 adapter/API regressions and two focused ledger regressions. The existing 23 accounting regressions remain covered.

## Validated behavior

Initial `npm --prefix apps/sentinel-lab run test:playground-ai`: **39 passed, 0 failed**. This includes 25 accounting tests and 14 provider adapter/API tests. Subsequent readiness/global-kill validation: **41 passed, 0 failed**, recorded in `focused-readiness-tests.txt` (25 accounting plus 16 adapter/API tests).

Initial `npm --prefix apps/sentinel-lab test`: **126 passed, 0 failed, 1 skipped** across both test commands, before the subsequent readiness/global-kill patch. Final backend validation is being rerun by the root task and must be cited separately. The skip is the existing Windows symlink privilege limitation; it requires a symlink-capable Linux or authorized Windows host.

Initial `npm --prefix apps/sentinel-lab run lint`: **passed**. Subsequent readiness typechecking is recorded separately in `backend-readiness-typecheck.txt`.

Every provider response in these tests is synthetic. The HTTP success test stubs `globalThis.fetch` and restores it afterward. No real API key, payment, mail service or hosted account was exercised. Temporary Windows libSQL fixture handles may remain locked until the test process exits; this is recorded in the logs.

## Cost and dispatch logic

The hard provider budget is **$5 aggregate across the existing persistent test-mode ledger**, including reserved, submitted, settled and unresolved actions. It does not reset when a model changes or a month rolls over. A higher configured budget is rejected by this adapter. A fresh persistent database would be a separate budget, so operators must reuse the same configured TEST ledger for every test process/model.

Before generation, the full input is counted using `POST /v1/responses/input_tokens`. The exact counted payload is frozen and reused byte-for-byte for generation. A missing, malformed, unsupported, oversized or changed count/payload blocks generation. UTF-8 byte counts are used only for bounded memory, never claimed as a complete token bound.

For each token category, the ledger rounds up to an integer microdollar:

```text
ordinary_input = total_input - cached_input - cache_write_input
provider_cost = ceil(ordinary_input * ordinary_rate / 1e6)
              + ceil(cached_input * cached_rate / 1e6)
              + ceil(cache_write_input * cache_write_rate / 1e6)
              + ceil(total_output * output_rate / 1e6)
provider_maximum = ceil(max_input * highest_input_category_rate / 1e6)
                 + ceil(max_output * output_rate / 1e6)
```

Rates above are microdollars per million tokens. Output usage already contains reasoning and formatting tokens; reasoning is recorded but never added twice. Cache writes replace ordinary input pricing rather than adding another full input fee. Category rounding can conservatively exceed the unrounded per-request price by less than four microdollars; invoice reconciliation remains separate.

The request explicitly selects Standard processing (`service_tier: default`), fixed reasoning effort and a total output-token cap. It has no tools, retries, agent delegation, previous response, background processing or streaming. Explicit prompt-cache mode has no breakpoints, so the reviewed provider contract disables cache reads/writes. Reservation still prices every input token at the more expensive cache-write rate, and reported categories are metered independently.

The reviewed test helper requires an explicit model. It contains Standard short-context rates for GPT-6 Astra and GPT-6.1 Sol and expires on 2026-10-21. With its 32,000 input / 4,000 total-output caps, the maximum provider reservation is $0.60 for Astra and $0.12 for Sol. These are bounded test configurations, not accepted model quality or customer retail pricing. The helper uses a 1:1 test retail ratio with cent rounding; commercial margin remains unaccepted.

The caller must approve the frozen customer maximum exactly. Both customer and provider reservations occur atomically before the single dispatch claim. `X-Client-Request-Id` uses the persisted action UUID. A replay returns the same receipt instead of submitting again. A timeout, crash, missing usage category, different processing tier, unexpected model, tool or subagent retains provider liability. Known invalid/unusable typed output waives the customer while retaining metered provider expense. Above-bound metered expense remains visible, the customer charge stays capped and the route pauses.

`store:false` avoids stored provider response content. It also means automatic response retrieval is unavailable. For an uncertain request, the persisted client request UUID and any real header/body identities support provider/operator reconciliation. No response ID or zero-dollar expense is invented. A customer deadline waiver never erases the upstream commitment.

## Current gates

The user authorized a dedicated Playground key and a **total $5 provider-test budget**. The key is being placed by the owner directly in the protected Vercel Preview branch environment as `EIDOS_PLAYGROUND_OPENAI_API_KEY`. This report does not claim it is present or deployed.

Real transport remains disabled. Reviewed primary documentation establishes accurate counting, schema/role framing, cache pricing and output limits, but **does not establish whether the token-count endpoint is billable**. `EIDOS_PLAYGROUND_INPUT_COUNT_PRICING` remains unset. It must never be set to `verified-zero` without primary documentation or actual billing evidence confirming zero counting fees. Controlled tests intentionally exercise that branch with an in-memory stub; that is not real billing evidence.

The isolated TEST database, real account/member acceptance, verified TEST funding receipt, signed payment acknowledgement, refunds, recovery inbox evidence and provider quality comparison remain pending. There was no production activation or LIVE charge.

## Protected hosted test harness plan

Use the existing authenticated Playground API and the same configured TEST database for every model/process. Do not create a separate in-memory or temporary live-spend ledger. Do not fabricate paid orders, mint synthetic credits in the hosted database, or substitute operator authorization for a real member session.

1. Connect the dedicated key, isolated TEST database and the existing relay/account configuration through their normal protected Preview bindings. Keep unrelated AI, community, proactive work and mail automation disabled.
2. Resolve counting-endpoint billing. If a fee exists, add its explicit pricing and durable reservations to the same aggregate ledger before any real count request. If zero fees are evidenced, record the source/receipt before setting the server gate.
3. Configure the reviewed explicit model rate card and a lifetime provider budget of at most 5,000,000 microdollars using the server/operator configuration helper against this same database. Existing commitments must remain visible. Existing maintenance/operator authentication can reconcile deadlines/reversals, but it currently has no pricing-configuration or credit-minting endpoint.
4. Establish a real member session and verified TEST funding through the existing signed commerce path. A test harness may reuse that authenticated browser/session with owner consent; credentials stay out of command output and artifacts. Synthetic sessions are not real account acceptance.
5. From the authenticated browser, run one narrow task at a time through quote -> displayed maximum -> explicit generate -> typed proposal review -> apply/undo. Capture sanitized action/public-balance receipts, full-payload hash/count metadata and provider request IDs from the operator ledger. Do not store secrets or raw private assets in evidence.
6. Compare Astra and Sol by changing the server-owned card without replacing the database/budget. Old unmatched quotes must expire or be cancelled. Unresolved submitted costs remain committed. Save both usable and failed outputs, timing, actual token categories, rounded cost and remaining aggregate budget. Stop when a reservation cannot fit, or when any route pause/unknown liability occurs.

The existing maintenance endpoint also runs community/newsletter work and deletes expired platform records. It should not be invoked merely as a convenient credential bypass. A narrow existing-operator-authorized configuration command is preferable to expanding that endpoint or enabling unrelated services.

## Primary sources reviewed

- [OpenAI pricing](https://developers.openai.com/api/docs/pricing): Standard short-context category rates and processing tiers.
- [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching): explicit mode without breakpoints; cache-write replacement pricing; `usage.input_tokens_details.cache_write_tokens` and cached counters.
- [Counting tokens](https://developers.openai.com/api/docs/guides/token-counting): exact full-input count including structural framing; local estimation limitations; output usage and output-cap coverage.
- [API overview](https://developers.openai.com/api/reference/overview): durable client request ID and server request identity.
- [Responses multi-agent](https://developers.openai.com/api/docs/guides/responses-multi-agent): delegation requires explicit enablement/beta; this adapter supplies neither.

## Receipts and limits

`focused-tests.txt`, `full-backend-tests.txt`, `backend-typecheck.txt`, `focused-readiness-tests.txt`, `backend-readiness-typecheck.txt` and `source-manifest.json` accompany this report. Source changes were uncommitted when validation ran; hashes identify the working source. The initial logs predate the public-readiness/global-kill patch and are preserved as historical receipts. Parent-task final validation, artifacts/Drive mirroring and protected deployment are handled by the root task after all source changes freeze. This report alone does not claim an updated hosted preview or a cloud Drive sync.
