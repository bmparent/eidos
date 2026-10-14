# Unified Playground backend candidate — October 14, 2026

This is Eidos Works product work, not a Brain/Sentinel model milestone. No research engine or scientific behavior changed.

## Source reconciliation

The clean PR74 source `37f4633f236bacae14c5b799becc2d2cea5528f4` was isolated in branch `codex/playground-unified-20261014`. PR76 source `16a1926d5bbd33cd52bb0c93ee15dd949b1bc698` adds only the immutable private archive deployment gate. Both are based on PR72 `952e68ff75da6857e25f249bebd96ea65116e294`. They were deliberately merged without conflicts at `01e82ac73a1bcb6a3159e93dccf96172fded4964`. No main, Observatory, other worktree, buyer archive, or existing deployment was overwritten.

## Implementation and limits

- Additive migration `0005_playground_ai.sql` extends the same Works commerce database with owner/mode limits, grants tied to existing paid orders, immutable quotes, action/settlement records, and journal entries. There is no independently authoritative wallet.
- All currency amounts are integer USD microdollars (`1 USD = 1,000,000`), bounded below JavaScript's safe integer limit. Multiplication/division uses `BigInt`. Provider input/output categories round up independently; retail charges round up to whole cents. The only committed rate card is a clearly labeled synthetic test fixture. No real model/pricing acceptance is claimed.
- One conditional SQLite UPDATE atomically claims included/purchased holds, the monthly cap, and the aggregate lifetime provider budget. SQL reads current balances inside the same write statement. Two independent database connections are tested. No browser-supplied owner/model/totals are accepted.
- A dispatch compare-and-set returns a one-time token to exactly one worker. Once claimed, another worker cannot submit again. A crash after that claim is uncertain, even if dispatch might not have happened. The upstream reserve remains held until final usage or pessimistic settlement.
- The current owner provider budget is checked again inside dispatch's atomic SQL write. Lowering it below aggregate commitments immediately blocks reserved work from dispatching. Operator status reports reserved, submitted, settled and unresolved amounts separately, plus configured budget, remaining amount and an explicit over-budget/block state. Previously submitted work remains an obligation; lowering a budget never pretends that it can recall that request or that an undispatched hold was spent.
- Usable proposals are billed from actual metered usage, bounded by the approved maximum; unused holds release. Invalid/unusable results waive customer usage but retain genuine provider cost. A reported overrun pauses the route; unknown expansion retains the maximum and requires reconciliation. Apply/Discard is not a billing operation.
- Monthly periods are UTC calendar months defined by the server. Late settlement stays with the original action period. Prepaid and provider holds span month boundaries. Lower cap requests block new reservations immediately while reporting already committed usage as an effective floor. Top-ups do not write the cap.
- Customer holds expire after 24 hours from dispatch. Settlement and account reads enforce the deadline, and the existing authenticated maintenance handler runs the sweep when TEST billing is configured. Expiry waives the customer and settles the unknown upstream maximum pessimistically; a later receipt cannot re-debit that action. Scheduling and hosted deadline execution remain acceptance dependencies while AI is off.
- Grants require a paid immutable template order, an event-journal-backed signed-payment receipt for the configured merchant, matching owner/SKU/edition/version/USD amount, and matching TEST mode. The actual merchant ID is observed through Stripe's authenticated account endpoint at checkout and at signed payment processing, and bound immutably to the order. Wrong accounts and unsupported Connect events cannot create allowance evidence. Historical unbound purchases retain guest receipt/download/retry behavior but cannot mint new allowance evidence. Each order/kind grants once regardless of grant-rule version changes. Guest receipts cannot mint authenticated-account balance. There is no public mint/top-up endpoint or approved funding rule.
- Refund/reversal source status blocks admission and dispatch atomically even before reconciliation. Reconciliation freezes the account and removes only unspent amounts, preserving committed accounting without negative balances or surprise charges. Reversed grants cannot be replayed.

## HTTP boundary

`/api/playground/ai` is relayed through the same authenticated Works dispatcher. It exposes authenticated TEST balance/status, explicit owner cap edits, and cancellation only. Unknown/forged fields are rejected. Generation, quoting and provider submission are deliberately unavailable publicly. `EIDOS_PLAYGROUND_AI_MODE` is empty in the candidate environment, all public account/AI/commerce flags stay false, and LIVE funding is rejected even by the internal policy function.

The existing schema-5 HTTP409/local-backup cloud boundary remains intact. Separate opt-in delivery storage is described below; it does not change the legacy cloud contract. No hosted persistence, inbox arrival, checkout completion, provider execution, or two-account hosted acceptance is claimed.

## Immutable page delivery candidate

The frontend-authoritative schema-5 renderer graph is exported into `lib/works/project-delivery/`, isolated from the accepted legacy schema-1 model. Its manifest records every transitive source file, UTF-8 LF byte length and SHA256; `.gitattributes` preserves those bytes on Windows and Linux. The backend renderer adapter exports only this separate graph. From the frontend repository root, `node scripts/export-playground-delivery.mjs <backend-app-directory>` verifies parity; `--write` installs the graph. Do not use the full legacy platform exporter for this contract.

`GET/POST /api/playground/delivery` uses the authenticated relay and signed-in owner. The independent `EIDOS_PLAYGROUND_DELIVERY_MODE=test` gate is off by default and refuses both canonical `eidos-works.com` hosts and `eidosworks.pages.dev`. Request bodies are bounded to 1,520,000 bytes, canonical snapshots to 1.5 MB, archives to 6 MB, and each owner to 20 snapshots / 30 MB. One transaction inserts the immutable source and chunked archive only if the owner's current quota permits it. Stable request references replay the same snapshot; changed payloads conflict. SQLite rollback and independent worker connections are tested. Browser-supplied archives, HTML, owner IDs and claimed totals are not accepted as authority.

Additive migrations `0006_project_delivery.sql` and `0007_project_delivery_orders.sql` use the existing Works database. The migration runner discovers numbered files; no hosted migration was run. Checkout can attach a previously frozen page only after the actual parent archive bytes/hash/version and signed-in owner match. The attachment joins the initial order transaction, before any external checkout session. A paid webhook must match that immutable attachment and the observed configured TEST merchant. Original archives and receipt downloads retain their established defaults.

The optional `approved-page` download requires the same signed-in owner, paid TEST order, ready fulfillment, no refund/revocation, and exact stored archive/source hashes. Recovery preserves the binding and rejects an independently revoked payment before consuming its token. Current sales availability, current edition pins and a newly changed original archive do not replace an already stored approved archive. The frontend order fixture verifies that historical-download behavior with mocked provider traffic. The backend full suite separately verifies existing original/guest receipts, signed merchant evidence, cap accounting, refunds, mail retries, and archive retention against real libSQL.

The second preview remains manual-only: no database, no archives, no delivery mode, no accounts, AI, checkout, mail or provider credentials. The source can be reviewed there, but authenticated storage and paid recovery still need the separately authorized TEST persistence/payment acceptance gates.

### Canonical runtime correction

Browser acceptance subsequently exposed a cross-build boundary missing from the initial source-level tests: `pageRuntime.toString()` produces different text after Vite, esbuild and Next transforms. The same valid snapshot therefore generated different `script.js` and ZIP hashes. The exact-byte acceptance gate correctly rejected the package. This differs from the separate `tsx` fixture helper issue; the actual Next runtime contained no `__name` leak and executed correctly in desktop/mobile Chromium.

The corrected graph imports checked-in canonical page/composition/authoring runtime strings from `runtimeBundle.ts`. Original runtime behavior is unchanged. Frontend generation/check commands and cross-build regression tests enforce regeneration from those sources. The backend graph now contains exactly 17 files, inventory `3254b61e995185a2753b98ddf97c378e62c91273b9539d6e5929138e4e55c19f`; only the three superseded copies in the isolated graph were removed after comparison with their previous manifest. Legacy cloud source and frontend original runtime sources remain intact. Export and backend tests reject missing, obsolete, unknown or symlinked graph entries.

The actual optimized Next route module was loaded by plain Node with no TypeScript loader, executed against in-memory SQLite, and forbidden from making outbound requests. Its freeze/read result equals the canonical esbuild reference byte-for-byte: 66,201 bytes, SHA256 `b26d51c04ddee2f6ff2cb421d13edaaeb668e64c8e25ba3cc4209ccceb5956a3`. Backend 100 tests, typecheck and optimized build pass again. Pre-fix input/output ZIPs and mismatch receipts are retained under `delivery/compiled-runtime-before/`; the corrected probe, reference builder and ZIPs are under `delivery/compiled-runtime-after/`. Old stored ZIPs are verified as stored rather than silently recompiled. Browser and protected-preview acceptance for the corrected candidate are recorded in subsequent receipts.

## Validation and receipts

Run from repository root:

```text
npm --prefix apps/sentinel-lab ci
npm --prefix apps/sentinel-lab run test:playground-ai
npm --prefix apps/sentinel-lab test
npm --prefix apps/sentinel-lab run lint
npm --prefix apps/sentinel-lab run build
node --check apps/sentinel-lab/scripts/configure-playground-preview.mjs
```

Controlled accounting coverage maps to all fourteen required cap scenarios in `works-playground-ai.test.ts`, including two separate connection races (admission and owner budget reduction). The focused suite passes 15 tests. The full suite passes 96 tests with one explicit platform skip. It covers shared owner limits, quote idempotency, cap/top-up semantics, forged payloads, technical waivers, overrun pause, unknown outcomes, late settlement, refunds, signed-payment merchant identity and retry requirements, historical receipt preservation and TEST/LIVE isolation. This is source/runtime fixture evidence, not paid-provider evidence. Full suite logs, focused logs, build/typecheck, dependency audits before/after and an acceptance ledger are in `artifacts/playground-unified-backend-20261014/`.

The scoped dependency repair pins Next.js 16.3.8 and refreshes compatible transitive security patches. The prior 16.3.4 pin included a critical next/og issue documented in [GHSA-vcvr-r3jv-pc5j](https://github.com/advisories/GHSA-vcvr-r3jv-pc5j); [Next.js 16.3.8 release](https://github.com/vercel/next.js/releases/tag/v16.3.8) is the selected compatible patch. The refreshed npm audit reports zero vulnerabilities. Build and typecheck pass on 16.3.8. A newly added retry fixture initially expected 502 instead of the established 503 mail failure contract; the failed run is retained, the assertion was corrected, and the full suite was rerun successfully.

The inherited archive suite could not create a Windows file symlink (EPERM). Its new named subtest explicitly records that platform skip; missing, altered and duplicate archive gates still run. Windows libSQL can retain a synthetic temporary database handle until process exit; the test records the exact retained fixture rather than hiding a cleanup issue. No customer data is in that fixture.

## Preview and pairing

Intended frontend origin: `https://eidosworks-playground-20261014.pages.dev`. Backend branch is `codex/playground-unified-20261014`. Its first manual-only preview has no database. A future paired persistence preview may use only the approved isolated `EIDOS_TEMPLATES_TEST` validation database. `configure-playground-preview.mjs` accepts an existing ignored TEST configuration, selects only the required database/relay/protection bindings, and writes only the new branch's Preview variables. Its default is a dry run. Every feature flag is off and all provider/payment/mail credentials are blank. Never deploy under the older commerce environment branch by accident.

The authenticated CLI can inspect the old studio branch's environment metadata, but its sensitive database/relay values are intentionally not returned by the API. The original ignored configuration or a secure supported reuse mechanism is needed. No sensitive-read bypass or unrelated credential reuse was attempted.

The first editor milestone can use `--manual-only` without any database: all database selector/fallback bindings are explicitly blank, a new relay secret is confined to this new branch and the new frontend `EIDOS_PLAYGROUND_PLATFORM_TOKEN` GitHub secret, and all features remain off. This does not prove persistence. The inherited global Preview configuration currently enables shop/AI, so `apps/sentinel-lab/vercel.json` disables automatic Git deployment for this exact branch before it is pushed; manual deployment follows the explicit overrides. This uses the documented [Vercel branch deployment setting](https://vercel.com/docs/project-configuration/git-configuration). Other branches and project-level settings are unchanged.

The manual deployment exposes the exact committed SHA and bundle hash as `runtimeSourceRevision` and `runtimeBundleHash` on authenticated `/api/public-config`. It intentionally leaves the separate commerce `EIDOS_TEMPLATE_SOURCE_REVISION` unset. The manual-only bundle requires a clean committed source tree and sets the runtime identities in its deployment request.

The first protected manual preview is READY at `https://eidos-sentinel-gddofh939-1brentbm-1876s-projects.vercel.app`. Its deployed source is `0e0aabc48f0ca7894b917495b661ef44cd337dce`, bundle SHA256 `77b27dd43cb78e66b83fff84bfba70c532aaf7d3a367d891e19e4b754023a870`, deployment `dpl_GKzpxmZ16TXCry6Pwr6jX5LgB48w`. All 42 isolated environment overrides are recorded by name only and repeated on the deployment request. The new relay key exists only in ignored configuration, branch/deployment environment and frontend GitHub secret `EIDOS_PLAYGROUND_PLATFORM_TOKEN`. An unauthenticated health request redirects to deployment protection. Authenticated runtime/flag/schema proof is delegated to the paired frontend CI using the approved existing project bypass; READY alone is not that acceptance.

The second protected manual preview is READY at `https://eidos-sentinel-44ncm0vx1-1brentbm-1876s-projects.vercel.app`, deployment `dpl_Hb3DmunnkTpknbwhqzpGhtKUbqY1`. Exact source `be549a56f15043f7d189db1bad049b904000c344` and bundle `a7baede78b21805167ca61112e0b541128b60cb0979b94524bbc0d4011037b68` are verified against provider metadata. It adds the frozen 19-file delivery graph with inventory `800edfd6034839506f046ee422260b9dedc55e6e9cb23ef034193a35efe76415` and reviewed order hooks. Final backend tests: 100 passed, one existing Windows symlink skip; four focused delivery tests, typecheck and optimized build passed. Sixteen paired shared contracts and the isolated graph match the frontend. All 43 branch/deployment overrides remain disabled or blank, including delivery mode; database and archive counts remain zero. Unauthenticated health returns 302. The first deployment and its receipt remain retained. Later evidence-only Git commits do not change this deployed source pin; authenticated runtime acceptance remains the frontend CI gate.

Manual commands from repository root, after the source-controlled Git deployment suppression is pushed and branch overrides are configured:

```text
node apps/sentinel-lab/scripts/configure-playground-preview.mjs --manual-only --apply --cli=<authenticated CLI entry>
node apps/sentinel-lab/scripts/bundle-template-preview.mjs --manual-only --environment-branch=codex/playground-unified-20261014 --out=apps/sentinel-lab/.vercel/playground-deployment.private.json
node apps/sentinel-lab/scripts/deploy-playground-preview.mjs --bundle=apps/sentinel-lab/.vercel/playground-deployment.private.json --cli=<authenticated CLI entry> --deploy
node apps/sentinel-lab/scripts/deploy-playground-preview.mjs --cli=<authenticated CLI entry> --inspect
```

Commit the sanitized configuration receipt before the clean-source bundle step. The deploy helper refuses a different branch, changed source, unmatched SHA/bundle, nonmanual payload or incomplete scoped configuration. Manual bundle archive completeness is explicitly `0 / intentionally omitted`: commerce is disabled. The original complete TEST bundle and all original purchased archives remain retained and unaffected; future commerce deployment must still pass the full private archive gate.

`bundle-template-preview.mjs --environment-branch=codex/playground-unified-20261014 --out=<private path>` now explicitly targets this candidate, preserves the historical default for existing callers, and verifies every private immutable archive before producing a complete bundle. The ignored original packages must first be restored with exact hashes. A commerce-disabled UI build has the archive gate inactive; this is distinct from a complete commerce-enabled package build.

Shared source files must be copied to the corresponding frontend `functions/` and `migrations/` paths and listed by its platform exporter: `playgroundAi.ts`, `playgroundAiSchema.ts`, `api/playground/ai.ts`, `0005_playground_ai.sql`, the one-line core environment addition, the conditional maintenance hook, and the small signed-template webhook/receipt hook. Preserve frontend-only source rather than replacing its entire shared tree.

## Remaining release dependencies

Authorized Playground key choice and explicit owner test budget; verified model IDs/access/current uncached pricing; a bounded Responses adapter using the one-time dispatch claim; validated scope-preserving proposal integration; real hosted metered proposal; model comparison; actual hosted two-account billing ownership; verified maintenance cadence; approved TEST funding/grant rules and a verified merchant binding; immutable snapshot/derivative delivery; signed TEST payment/refund and actual inbox receipt/recovery acceptance; exact final preview/runtime identities. Public AI and production sales must remain off until these separate gates pass.

Rollback source is the clean integration merge above. Do not drop additive financial records, reset existing balances, delete paid archives, or reassign historical orders during rollback.
