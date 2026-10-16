# Playground connected TEST continuation — 2026-10-15

The connected candidate is separate from the database-free manual editor preview and the canonical production site. Its fixed origin is `https://eidosworks-playground-test-20261015.pages.dev`. It uses the existing approved TEST validation database, all 8 current and 24 retained immutable archives, the verified Stripe TEST merchant, the existing relay credential and the already bound Preview mail secret. No branch-global environment entries or production aliases are changed by the new deployment helper.

The relay now accepts the AI endpoint's 110,000-byte request envelope and allows 105 seconds. The backend allows 120 seconds around the provider's existing 90-second deadline so the accounting transaction can finish. Other relay endpoints keep their existing deadlines and byte limits. The public Cloudflare TEST key contract is restricted to the two explicitly named TEST candidates. Only the new Playground TEST origin can use the member action, and it additionally requires accounts enabled; real Siteverify success is still required.

No new AI provider request was made by this source change. The existing lifetime ledger retains 1,753 microdollars of pessimistic count liability plus 1,742 microdollars of metered Sol generation expense: 3,495 of the same 5,000,000 microdollar authorization. General AI remains disabled while input-count billing and customer acceptance are incomplete. No paid/member record, credit, invoice fee, customer acceptance or production release is fabricated.

## Reproduction

From the backend repository root:

```text
node --test scripts/test_playground_test_candidate.mjs
npm --prefix apps/sentinel-lab test
npm --prefix apps/sentinel-lab run lint
npm --prefix apps/sentinel-lab run build
node scripts/deploy_playground_test_candidate.mjs --configuration=<approved ignored TEST configuration> --cli=<authenticated Vercel CLI>
node scripts/deploy_playground_test_candidate.mjs --configuration=<approved ignored TEST configuration> --cli=<authenticated Vercel CLI> --deploy
node scripts/deploy_playground_test_candidate.mjs --cli=<authenticated Vercel CLI> --inspect
```

The first deployment form only freezes the inventory. The explicit `--deploy` form creates or reuses this candidate's Stripe TEST endpoint, stores its signing secret only in the ignored `.vercel/playground-test.private.json`, and submits a protected Preview bundle. It inherits the already authorized branch mail and Playground provider secret without decrypting either. The script refuses another database, a LIVE key, a wrong merchant, a missing relay, dirty application source or an incomplete immutable archive set.

## Validation and journal

The full application suite and optimized build passed locally. The final deployment/endpoint guards passed 3 tests; the changed TEST challenge was covered in the 24-test template suite. The private presence-only binding diagnostic passed 4 tests and makes no provider request when invoked with its binding-only header. Frontend relay tests passed 5 tests, including acceptance at exactly 110,000 bytes, overflow rejection before transport and the longer AI deadline. The original direct root invocation of `works-platform.test.ts` failed because that older suite resolves migrations against the application working directory; the supported root command `npm --prefix apps/sentinel-lab test` passed without changing the older test behavior. Logs are retained under `artifacts/playground-authenticated-20261015/release-continuation/`.

All 32 immutable archive byte counts and SHA-256 values were restored and checked against the existing frozen receipt; no original archive was modified. This documentation records source preparation, not completed hosted account/payment/mail tests. Their subsequent receipts must establish real sign-in, Stripe signed-event acknowledgement, inbox delivery, exact approved-page download and refund revocation. Model comparison, human review, physical-device checks, LIVE pricing/funding and production activation remain separate gates.

## Hosted state and continuation

### Owner-selected mail name — 2026-10-16 UTC

The owner saved the existing Resend credential as `RESEND_playground_preview`. Metadata inspection found this new entry assigned to Production. Its existing entry was moved to Preview, branch `codex/playground-unified-20261014`, with a metadata-only PATCH containing `target` and `gitBranch`; no secret was decrypted, replaced, printed or supplied. Both OpenAI entries retained identical IDs, update timestamps and targets. The scope receipt is `artifacts/playground-authenticated-20261015/release-continuation/mail-alias-20261016/scope-repair.json`.

The Works adapter and private presence diagnostic now resolve that alias only when `VERCEL_ENV=preview`, `EIDOS_PLAYGROUND_DELIVERY_MODE=test` and `PUBLIC_SITE_URL=https://eidosworks-playground-test-20261015.pages.dev`. An existing nonempty `RESEND_API_KEY` retains priority. Other hosts and Production ignore the alias. Deployment inherits the authorized alias rather than exporting its value. This change does not require revoking either provider key. Hosted presence, provider acceptance and inbox arrival remain separate checks recorded after redeployment.

The mail alias passed hosted presence, both owned-inbox sign-ins, genuine session verification and one-time replay rejection on application `9b3ece16dd7a1485bd23ebc4b3e54559a3a98f8d`. OpenAI model metadata authentication also passed with zero inference/count requests. Connected frontend `146d30b202636382b239dabbaec75301a7b2bbf2` was published by workflow `38836485803`; its exact backend/source/archives and unauthorized/signature boundaries passed. The canonical production deployment remained unchanged.

A genuine $99 Stripe TEST checkout subsequently produced the exact approved-page ZIP and original archive. Both other-account and anonymous downloads were refused. The receipt and requested recovery email arrived in the business inbox; recovery returned the same frozen snapshot and rejected reuse. The initial receipt status nevertheless showed `failed` with a nonempty provider acceptance ID. The receipt sender's catch handler allowed a late failure to overwrite success. The paired fix preserves an accepted receipt against later failures and uses the persisted provider ID to interpret older inconsistent rows without rewriting their raw history. A controlled two-send race plus an older-hook overwrite regression now covers this. Provider acceptance and actual inbox arrival remain distinct evidence.

The acceptance CLI also now respects the public member contract (which intentionally omits database IDs). It validates the real issued session through the account API and a read-only TEST session lookup, without seeding members or sessions. For private one-time input use an interactive terminal with echo disabled; include `--configuration=<approved ignored TEST configuration>` for verification. An initial harness-only expectation of a public member ID failed and is retained in the evidence folder.

The receipt fix is deployed as application `76d07348095405fe601378a0fd37ac2a0cfd85dd`, protected deployment `dpl_HXWoTvRuKkkQ6MAGKy5dMRQ2YfZT`, bundle `4b463bd3fab1131491ff8ef9cd9060ca675021971e8f5e64df1c3ed7f5ebd4d4`. Frontend `fbfd3631d2caac5c274d83ccaf1e7cd0985399c8` is pinned to this exact source; connected TEST publication `38838978059` passed. Hosted recheck reports receipt status `sent` and identical approved/original bytes. Raw historical `failed` status and its real provider acceptance ID remain in the payment evidence, alongside separate inbox observations.

Use `--phase=payment-evidence --configuration=<approved ignored TEST configuration>` to match the signed application receipt to its genuine Stripe TEST event. It records the event's pending webhook count without inferring delivery to one endpoint. The observed event had two pending webhooks; endpoint-specific acknowledgement still needs the owner's Stripe dashboard sign-in. The connected Stripe plugin exposes only this merchant's LIVE context, so it was not used for TEST operations or switched into LIVE mode.

The genuine TEST refund succeeded for 9,900 cents. Hosted status changed to `refunded`, both original and approved-page downloads returned 403, the approved snapshot became revoked, and the included test allowance returned to zero with the account frozen. Replaying the grant after refund returned `granted=false,reversed=true`; the $1 account cap did not change. Final ledger readback has one retained grant, zero customer AI actions and the unchanged 3,495-microdollar lifetime commitment (including the unresolved count-price liability). No new generation or count request was made in this continuation.

### Earlier blocked state (retained history)

Protected backend `dpl_7tit8cwPPHhq1tBEuG3wVRT57a9T` is READY at application source `2a06e12528cb42b00f8dbfdc1219e64c5a719e9f`, bundle `297768a1e45b4bda2acbfcd736266192f1cbfba0638c8e072ebfebb426cc5f46`. Application CI `38829258229` passed on that source. The private runtime check confirms accounts enabled, TEST database selected, sender/rate/Turnstile present and the dedicated Playground key present. **RESEND_API_KEY is absent at runtime**, despite its Preview branch metadata entry. Account readiness consequently remains false. The owner has been asked to privately update only that exact Preview branch mail entry; the OpenAI key needs no replacement.

A fresh hosted provider metadata check also passed on this exact backend: `/v1/models` authenticated with both Sol and Astra listed, zero inference/count requests, and no project/organization response headers. This separately confirms the OpenAI credential still works without spending generation tokens.

The new Stripe TEST endpoint `we_1UQym7KC8pRG5Tr9lcIO9qKu` is disabled until the connected frontend passes its exact runtime/signature checks. Existing endpoints were not modified. The connected TEST frontend has not been published. Its workflow is registered and its validation passed; an early manual run was canceled before deployment after runtime preflight found the missing mail binding. The existing manual frontend was independently rebuilt and deployed at `0e9931a1ea0126b2f4b104e60affe9ff6bebf3e7` with hosted assets, identity and disabled-feature checks passing.

`scripts/accept_playground_hosted.mts` is the prepared acceptance runner. It uses ordinary sign-in links for the two owned inboxes, not inserted member/session rows. Magic-link and recovery tokens enter privately on stdin; secure session/download capabilities remain only in an ignored `.vercel` file. Relay headers and request bodies use temporary ignored files rather than process arguments. Public evidence contains hashed member IDs, statuses, hashes and totals, not tokens.

After the mail binding is saved, redeploy the exact reviewed application and update the frontend pin to that new deployment. Recheck private bindings and account readiness, then manually dispatch `playground-test-preview.yml` on the isolated branch. Only after the published relay passes may this command enable this task's TEST endpoint:

```text
node scripts/toggle_playground_test_webhook.mjs --configuration=<approved ignored TEST configuration> --state=enabled
```

Acceptance phases run from the backend root with the same `--cli` and, for `fund`, `refund` and `ledger`, `--configuration`:

```text
node --import ./apps/sentinel-lab/node_modules/tsx/dist/loader.mjs scripts/accept_playground_hosted.mts --cli=<authenticated CLI> --phase=preflight
```

The ordered phases are `signin`, `verify --slot=business`, `verify --slot=personal`, `freeze`, `checkout`, genuine Stripe TEST payment, `payment`, `fund`, `recover`, `redeem`, `payment`, `refund`, `revocation`, `ledger`. Verification/redeem require the actual matching inbox token on stdin; do not paste it into chat or shell arguments. `fund` calls the existing verified-paid-order grant logic for exactly $1 of TEST included allowance, preserving the owner's chosen $1 monthly cap. It cannot grant without matching signed payment/merchant evidence. No LIVE funding policy is implied. `refund` submits a real TEST refund of only that newly accepted owned paid order. No phase marks paid or manufactures a webhook signature.

These phases are prepared, **not passed** while mail is missing. API acceptance is separate from browser checkout/return, actual receipt/recovery inbox arrival and Stripe endpoint-specific delivery acknowledgement. The broad editor CI retained a WebKit mobile artwork click timeout at `0e9931a`; the identical failed-job rerun passed at run `38829178964`. The first failure log, screenshot and receipt remain retained. No UI source fix or physical Safari acceptance is claimed from that retry.

The authenticated provider cost CSV reports zero for both October 15 key totals, including the known metered generation. It does not establish final settlement or count-specific fees. The original 1,753-microdollar count liability is retained; further counts and customer AI remain disabled. Fresh ledger readback still shows 4 members, 15 orders, 9 template orders, 0 AI actions and 0 AI grants. Delivery tables remain uninitialized until the first genuine authorized freeze, rather than seeded for acceptance.
