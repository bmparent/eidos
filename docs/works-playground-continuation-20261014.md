# Unified Playground backend continuation — October 14, 2026

This is the current continuation of the existing draft candidate. Historical build/deployment and browser ZIP receipts remain in `works-playground-unified-20261014.md`; those source pins and counts are not the final continuation source. This work changes the Eidos Works editor/backend contract and proof tooling, without changing Brain/Sentinel research behavior or original purchased archives.

## Changes

- The separate frontend-authoritative delivery/proposal renderer now has 18 files, inventory SHA-256 `19960a2f371b413e041e19773f5f2512efb72f43d32e984ed97d62e5c774471c`. It includes typed element/section/page/brand proposal validation and the unified editor's current Motion/renderer/runtime and provenance changes. The legacy cloud renderer stays separate.
- The bounded Responses adapter uses the dedicated `EIDOS_PLAYGROUND_OPENAI_API_KEY`, fixed reviewed Standard test cards, explicit scope/target, strict one-to-six-operation output, exact full-input preflight capture, durable payload hashes and one dispatch claim. It disables automatic fallback, retries, tools, subagents, streaming/background work and implicit model selection. All models and attempts share the operator's aggregate maximum of $5.
- Exact counting has its own fail-closed gate: live generation requires verified input-count billing. A token reservation is not reported as actual usage or actual provider cost. No real model call or usage/cost acceptance is claimed by the fixture suite.
- The $5 real-provider ceiling is checked inside the atomic reservation, dispatch and count-admission writes against current operator state. Raising a budget concurrently cannot bypass the ceiling. Before any upstream count, the adapter validates the action reference/context and current account funding/cap/reversal state, then commits a durable one-time claim. Failed or uncertain counting does not retry that action. Claims are bounded to 20 per owner/hour, 60 globally/hour and 200 lifetime across models; independent database connections are tested. A two-model, 20-case comparison must respect those windows and report any partial matrix explicitly.
- Provider receipt identities are recorded before complete response parsing and cannot be replaced during reconciliation. Unmeterable/extreme usage, unknown expansion, malformed output and invalid stored results retain upstream obligations, waive unusable customer charges and pause dispatch where required. Matching late receipts may reconcile cost without reviving a waived customer debit.
- Dedicated Playground readiness is independent of the legacy generic-key readiness contract. The global AI enablement gate also applies consistently to quote and generation admission. Public accounts/AI remain off in the manual preview.
- Existing original-order retries use frozen product/edition/version and price, verify retained original archive bytes/hash before dispatch, and can resume even when the current catalog archive is absent. The signed paid fixture and exact historical download pass. Owner/attempt conflicts reject before preparing attachment schema. Original guest and Cinematic Starter contracts remain covered.

## Current acceptance limits

The user approved the dedicated Playground key and a $5 aggregate test budget. Initial metadata-only inspection found a sensitive variable named `eidos_sentinel_lab` scoped to Production, while the required dedicated name was absent. Current primary Vercel [Secret documentation](https://vercel.com/docs/environment-variables/sensitive-environment-variables) forbids renaming a saved Secret, although the general [PATCH schema](https://vercel.com/docs/rest-api/projects/edit-an-environment-variable) makes value optional for supported edits. No rename, scope mutation, value read/decryption or secret copy was attempted.

The owner subsequently saved `EIDOS_PLAYGROUND_OPENAI_API_KEY`. Fresh metadata verifies a sensitive/Secret binding for Preview branch `codex/playground-unified-20261014`, created October 14 at 23:52:13.232 UTC. Metadata also shows a separate same-name Production entry. Neither entry nor the earlier mistaken variable was changed or deleted by this task, and no secret value was read. The key does not need to be committed, stored in a GitHub repository file or made global. Binding metadata does not prove key validity or provider execution; billing/counting policy, hosted owner persistence and operator cap configuration remain gates before live testing.

Metadata also shows all existing `EIDOS_TEMPLATES_TEST_TURSO_DATABASE_URL` / `EIDOS_TEMPLATES_TEST_TURSO_AUTH_TOKEN` pairs are Preview branch-specific; none is global. A different branch cannot automatically inherit the sibling branch pair. No TEST database environment was retargeted/exported, and the manual candidate keeps database bindings blank. Metadata existence does not prove usable credentials.

Authenticated hosted two-account ownership, real TEST checkout/refund/signed acknowledgement, inbox delivery/recovery, actual provider usage/cost, model comparison, unfamiliar-user acceptance and physical devices remain separate gates. Fixtures and local compiled output do not close them. No production activation or LIVE charge is authorized here.

## Validation and evidence

Current receipts are under `artifacts/playground-unified-backend-20261014/continuation-budget-final/`. The final combined suite passed 134 tests with zero failures and one existing Windows symlink skip; the focused provider/accounting suite passed 47, current typecheck passed, and the optimized Next 16.3.8 build passed. Earlier receipts, including `continuation-final/` for intermediate source `472a915`, retain their original source stage and are not final acceptance for the budget/count-admission fixes.

```text
npm --prefix apps/sentinel-lab test
npm --prefix apps/sentinel-lab run lint
npm --prefix apps/sentinel-lab run build
node --check apps/sentinel-lab/scripts/deploy-playground-preview.mjs
node artifacts/playground-unified-backend-20261014/continuation-final/reference.cjs <authoritative-frontend-root>
node artifacts/playground-unified-backend-20261014/continuation-final/probe.cjs artifacts/playground-unified-backend-20261014/continuation-budget-final/compiled-browser-parity/browser-approved.zip --out=artifacts/playground-unified-backend-20261014/continuation-budget-final/compiled-browser-parity --require-match
```

The reference is generated from fresh current public Matter source with deterministic IDs, a synthetic one-pixel PNG, block Motion and mobile image sizing. The actual Next probe uses plain Node, loads the optimized route's real compiled module, freezes/reads through real in-memory SQLite, forbids outbound fetch, parses the exported script and requires exact ZIP equality. Its reference builder is deliberately separate from the tested Next runtime. The final rebuilt Next output matches the current raw Vite browser package byte-for-byte: 128,594 bytes, SHA-256 `044459b53a6dbc32163061bfe49e201ad695ef7f08c3ac3d322ec3f635333bf8`. Its separate current reference also matches at 68,025 bytes, SHA-256 `7036f418606de0f87fab460a13b632237bad849e14c48e7b0881ab081ec672e4`. Both current compiled receipts are in `continuation-budget-final/`; old browser ZIPs remain historical receipts.

## Manual-only preview

The deployment helper repeats all existing isolated disabled/blank overrides and explicitly blanks `EIDOS_PLAYGROUND_OPENAI_API_KEY` and `EIDOS_PLAYGROUND_INPUT_COUNT_PRICING` on the deployment request. A newly saved branch credential therefore cannot activate this manual preview. Accounts, AI, shop, mail, community and proactive features remain disabled; no database or original private archive is bundled. Original complete TEST candidates and all 32 retained paid archives are not modified.

Only the clean committed candidate is bundled, with exact source SHA and bundle hash recorded in deployment metadata and authenticated runtime configuration. Provider READY/protection/source metadata are verified separately from the paired frontend's authenticated runtime/feature checks. The current deployment receipt is kept outside application source; later evidence-only commits do not relabel its exact deployed source pin.

After READY/runtime verification, `node scripts/archive_playground_backend_evidence.mjs --date=2026-10-14` writes fresh source/artifact manifests and mirrors listed public-fixture evidence to a new configured Drive folder with SHA-256 readback. Its `--dry-run` validates the file selection without copying. It refuses an application source change after the committed freeze, symlinks, private/env paths and unapproved ZIP filenames. Each archive receives a new local metadata directory and Drive destination; previous archives are not overwritten. Mounted-folder verification does not establish remote cloud synchronization. No further evidence-only source push is needed for these local/mirrored receipts.
