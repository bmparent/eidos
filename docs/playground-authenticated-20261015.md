# Playground authenticated TEST continuation

The separate protected Preview diagnostic checks the dedicated branch key using only `GET https://api.openai.com/v1/models`. It returns whether the two requested model IDs are present, the provider request ID and frozen source identity. It does not run generation or counting, reveal provider errors or credentials, modify database rows, change the manual preview alias, or activate accounts/commerce. Production returns 404 even if its flag is set. Preview additionally requires `EIDOS_PLAYGROUND_PROVIDER_DIAGNOSTICS=test-auth`, the existing private relay token and exact Playground site origin.

The approved key stays in Vercel as `EIDOS_PLAYGROUND_OPENAI_API_KEY`, scoped to Preview branch `codex/playground-unified-20261014`. The deploy helper inherits it server-side and repeats all other isolated disabled overrides. No local OpenAI key file is needed.

```text
node --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs --test apps/sentinel-lab/tests/playground-provider-check.test.ts
npm --prefix apps/sentinel-lab test
npm --prefix apps/sentinel-lab run lint
npm --prefix apps/sentinel-lab run build
node apps/sentinel-lab/scripts/deploy-playground-auth-check.mjs --cli=<authenticated-vercel-cli-entry> --deploy
```

Receipts go to `artifacts/playground-authenticated-20261015/`. A successful metadata request proves credential authentication and listed model access; inference, actual usage, token-count billing, owner isolation, TEST funding and release acceptance require their own observations.

An existing ignored configuration was found in the retained paid-template backend worktree. A read-only connection to its explicitly selected `EIDOS_TEMPLATES_TEST` validation database succeeded. Its contents remain private. No credentials were read from the Vercel Secret API and no other branch bindings were changed.

## Token-count request repair and remaining billing gate

The [official spending-controller guide](https://developers.openai.com/cookbook/articles/per_run_spending_controller_responses_api#check-the-budget-before-each-request) says to send generation controls such as `max_output_tokens`, `service_tier` and `store` only with the response request. The [count reference](https://developers.openai.com/api/reference/cli/resources/responses/subresources/input_tokens/methods/count) establishes the supported input-bearing fields. The paired adapters now send a count payload containing the identical model, input, instructions, reasoning, schema and tools, without generation-only settings. The durable count claim freezes that exact count payload while the preflight separately freezes/hashes the complete generation payload. Fixture assertions enforce both contracts.

The [token-count guide](https://developers.openai.com/api/docs/guides/token-counting) and [pricing page](https://developers.openai.com/api/docs/pricing) do not explicitly establish a count-endpoint fee. `EIDOS_PLAYGROUND_INPUT_COUNT_PRICING=verified-zero` remains unset. The [provider spend-limit guide](https://developers.openai.com/api/docs/guides/spend-limits) also warns that enforcement can lag; it cannot establish an exact per-test bound.

A separate one-time operator probe is prepared for an explicit owner exception to the handoff's unknown-cost prohibition. It accepts no customer model/input/body. It counts one fixed public Sol sample, with no generation. Before network I/O, one atomic TEST-ledger transaction claims the entire $5 as unknown liability and pauses dispatch. Concurrent requests, failure, restart or policy reconfiguration cannot initiate another count or release that hold. It never asserts that $5 is the actual fee or a documented fee maximum. No code automatically reconciles this uncertainty. It must remain blocked until billing evidence is reviewed.

```text
node --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/prepare-playground-ai-test.ts --configuration=<existing-ignored-test-config>
node --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/prepare-playground-ai-test.ts --configuration=<existing-ignored-test-config> --apply
node apps/sentinel-lab/scripts/deploy-playground-auth-check.mjs --count-probe --configuration=<existing-ignored-test-config> --cli=<authenticated-vercel-cli-entry>
```

The last command is a dry run; `--deploy` creates the protected one-count backend only after the owner authorizes that exception. The public manual preview stays pinned to its earlier database-free deployment.

## Rotation and durable TEST setup

An existing OpenAI key-creation dialog unexpectedly returned a plaintext secret in browser tool output. It was not copied into source, public artifacts or a local env file. The owner confirmed revocation and replacement. The previous protected deployment now receives OpenAI 401, verifying that its captured key is revoked.

The fresh protected Preview `dpl_5PT2sZzgRhHSUditYGHuZ53kAco1`, from source `bff5f9a4484f3293948acf5d2337666559ddb6f3` and bundle `9e4fc2ff8718ee4d8fc2807eba20cda56fcd9fe4f0113702d2c170550f02bfb7`, also receives OpenAI 401. Direct authenticated CLI metadata shows the Production Secret entry was updated at `1792097136789`, while the branch-specific Preview entry retains `1792021933232`. The exact Preview branch replacement is therefore pending owner action. No secret was decrypted, read back, copied from Production or retargeted. Runtime receipts are saved under `artifacts/playground-authenticated-20261015/rotation-v2/`; the original successful authentication receipt remains historical.

The additive TEST-ledger setup passed against the existing approved validation database. Its policy has a $5 aggregate lifetime budget, no commitments, generation disabled and no pause reason. Before/after counts match: four members, fifteen orders, nine template-order records, three projects and zero Playground orders. No customer identity, paid status or funding was manufactured. See `test-ledger-setup.json`.

Current validation: backend 140 passed plus one existing Windows symlink skip; frontend platform 36 passed; backend typecheck and optimized build, frontend Functions compilation, and deployment/verification helper syntax checks passed. The first count-probe fixtures exposed an adapter-result-shape assumption before any provider call. The claim now verifies its durable unique client token rather than relying on batch result metadata, and its concurrency, failure, replay and policy-reconfiguration fixtures pass.

No count or generation request has been submitted. The one-count exception is still awaiting owner approval independently of key rotation. Authenticated funded customer inference, actual usage/cost, two-account hosted isolation, model comparison and production release remain unproven.

## Subsequent authenticated count acceptance

The owner corrected the exact Preview Secret, and a fresh protected deployment (`dpl_8PwhFhCpJvdj5HF7G9sJZyvzv1C4`) authenticated with OpenAI HTTP 200. Both requested model IDs are listed; missing-token and wrong-origin negatives still reject. The earlier 401 observations above remain historical. Evidence is in `rotation-v3/`.

The owner then explicitly approved the one-time count exception. Protected deployment `dpl_EtvmuXCjyknG5Fa9MVJzsfNY31vF` submitted one fixed public Sol count. OpenAI returned **701 input tokens**, HTTP 200, provider request `req_91e73b7e1c5e40d39a9cd846dc247661`. Its frozen 3,586-byte payload includes the schema and all supported input-bearing fields. The request hash, client identity and provider identity match across the HTTP and durable-journal receipts.

Fresh-process readback confirms exactly one diagnostic claim and result, zero generation actions and the entire $5 held with dispatch paused. Actual count cost remains unknown (`null`). The project usage dashboard displayed “No data”; missing usage is not zero-cost evidence. No hold was released and no generation or further count was submitted. This is one Sol count acceptance, not model quality, generation-cost, funded customer, Astra-count or production acceptance.

Repo-root commands for this acceptance:

```text
node scripts/submit_playground_count_probe.mjs --receipt=<protected-count-deployment-receipt> --out=<versioned-count-evidence> --cli=<authenticated-vercel-cli-entry>
node scripts/verify_playground_count_receipt.mjs --configuration=<existing-ignored-test-config> --out=<versioned-count-evidence>
```

The submit helper writes a local start receipt exclusively and refuses blind resubmission; server claims remain independently fenced. The readback helper only reads the approved TEST ledger and never releases liability. See `artifacts/playground-authenticated-20261015/count-v1/RESULT.md` and its JSON/request receipts. No core Brain/Sentinel research behavior changed.

## Conservative count reconciliation and one real generation

The [general pricing rule](https://developers.openai.com/api/docs/pricing) establishes that the Responses API has no separate API charge and tokens use the chosen model's input/output rates. Together with the exact count, this supports retaining **1,753 microdollars ($0.001753)** for all 701 input tokens at Sol's highest reviewed input-category rate, $2.50 per million. This is a conservative interpretation of the general pricing terms, **not an explicit count-endpoint fee confirmation or a settled invoice**. Actual count cost remains `null`. The dashboard's absent data is not evidence of zero cost. Further count calls remain disabled; `EIDOS_PLAYGROUND_INPUT_COUNT_PRICING` stays unset.

The append-only `eidos_pg_ai_diagnostic_costs` table records bounded and settled diagnostic expenses separately. Original claim/result journal entries remain unchanged. Policy snapshots and all three atomic customer admission writes include these liabilities. Reconfiguring the policy cannot create extra budget by omitting the count expense. A later settled receipt takes precedence over a pessimistic bound without deleting either receipt.

The protected `test-generation-one` Preview diagnostic can generate one fixed public Sol sample using the existing exact count. It accepts no customer prompt, model or funding. One SQLite write checks the count identity, payload, pricing reconciliation and shared lifetime budget, then reserves $0.12 before network I/O. Concurrent or repeated calls cannot dispatch again. It uses the same adapter request, typed-operation validation and usage parser as customer proposals. Missing usage retains the maximum and pauses the policy; known technical failure retains provider cost. Customer accounts and all public AI/commerce flags stay disabled. Operator inference does not prove funded customer acceptance or model-comparison quality.

Repo-root commands:

```text
node --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/reconcile-playground-count.ts --configuration=<existing-ignored-test-config> --pricing-evidence=<reviewed-public-pricing-receipt> --out=<versioned-evidence> --apply
node apps/sentinel-lab/scripts/deploy-playground-auth-check.mjs --generation-probe --configuration=<existing-ignored-test-config> --out=<versioned-evidence> --cli=<authenticated-vercel-cli-entry> --deploy
node scripts/submit_playground_generation_probe.mjs --receipt=<generation-deployment-receipt> --out=<versioned-evidence> --cli=<authenticated-vercel-cli-entry>
node --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs apps/sentinel-lab/scripts/verify-playground-generation.ts --configuration=<existing-ignored-test-config> --out=<versioned-evidence>
```

The generation helper writes its local start marker exclusively and cannot blind-retry. Never delete that marker or the fixed server claim. Inspect `generation-v1/` for the actual outcome and lifetime budget rather than treating this procedure as proof that generation passed.

The real protected generation now passed at application source `00f467f00786a2c605dcb55641212af19098442a`, Preview `dpl_4DJ3rXsJvW8LGtaYTxwezHh9z3Bh`. Sol returned a validated headline edit with 701 input and 34 output tokens; every cache/reasoning category is explicitly present and zero. Metered generation expense is $0.001742. With the retained $0.001753 count ceiling, the lifetime commitment is $0.003495 and remaining budget is $4.996505. Fresh-process readback matches request hashes and provider/client identities. Hosted 401/403 negatives, two settled 409 replays and local no-resubmit checks pass without another generation.

The actual returned operation was applied and packaged through the shared renderer, then opened in Chrome at desktop and mobile sizes. The headline renders, navigation expands/collapses, and no relevant console error or horizontal overflow appeared. This is one operator headline edit, not a funded customer flow, Astra comparison or production release. See `generation-v1/RESULT.md` for receipts, math and remaining gates. No API key action is required.

```text
node scripts/verify_playground_generation_guards.mjs --out=<versioned-evidence> --cli=<authenticated-vercel-cli-entry>
node --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs scripts/render_playground_generation.mts --out=<versioned-evidence>
```
