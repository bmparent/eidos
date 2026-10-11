# Paid templates backend candidate — 2026-10-10

Extended the existing Works ledger with immutable server-owned template/edition/version/archive snapshots, TEST-only checkout, signed webhook fulfillment, expiring retrieval and one-time recovery. Existing Cinematic ZIP bytes remain covered by regression tests. Eidos Brain research and production remain untouched.

Final local validation: 63 tests passed, zero failed/skipped; typecheck passed. Webpack build passed before final additive async-failure/challenge hardening. The default Turbopack build failed on the shared node_modules junction; the remote final source build must pass before hosted acceptance. Eight private packages match the current website packaging receipt. Only four developer editions have installation acceptance; WordPress and builder pilots remain blocked.

Provider setup uses the existing merchant, protected Vercel project, new additive relay bypass and new TEST webhook. The named validation DB migration retained six existing orders. Mail remains blocked because a usable Resend secret has not been supplied by provider readback. Local mocked mail assertions are not received-inbox proof. Cloudflare public TEST keys still call siteverify and are confined to exact candidate host, TEST Stripe and commerce actions.

Rollback: restore prior paired preview or disable the template flag; retain additive schema and sold version archives. No production promotion or live charge is authorized.

Artifacts remain repo-local. Google Drive mirror was not configured for this Works task. Build outputs were measured; automatic execution review rejected .next cleanup and no deletion was performed.

Protected preview dpl_7V42tCLLY3HAwEndSo7VuASKM7GR built the final commerce source c074b4d successfully in 20 seconds. Anonymous API access redirects; authenticated catalog returns four eligible developer editions with exact source revision and blocked CMS/pilots. Hosted price override, unknown receipt and unverified WordPress controls passed. Actual public TEST siteverify + nonmatching recovery returned the generic 200 response. Real checkout/fulfillment and inbox delivery remain separate gates.

Operational helper correction: current Vercel create-deployment API does not accept per-deployment env in its public schema. Helper now upserts variables only for the candidate preview branch, then creates the protected preview; no production variables are targeted.

## Versioned compatibility fixes — 2026-10-11 UTC

WordPress themes and Nightjar developer source advance to 1.0.1. The other three developer packages remain at 1.0.0. Five replaced archives are retained alongside eight current archives; all eight original archive hashes remain present and unchanged. The importer validates explicit package versions and refuses to overwrite immutable paths. The preview uploader verifies and bundles current plus retained archives. A regression demonstrates old paid Nightjar and WordPress snapshots still return their original 1.0.0 bytes, filename, and hash after the catalog advances; an unavailable old archive fails closed instead of returning the current package. Focused tests10/10 and typecheck passed. WordPress stays unavailable; production and Brain research remain untouched.
Final full suite64/64 passed (16 JavaScript +48 TypeScript), with zero failures/skips. Final typecheck passed.
