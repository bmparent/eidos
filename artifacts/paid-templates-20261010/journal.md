# Paid templates backend candidate — 2026-10-10

Extended the existing Works ledger with immutable server-owned template/edition/version/archive snapshots, TEST-only checkout, signed webhook fulfillment, expiring retrieval and one-time recovery. Existing Cinematic ZIP bytes remain covered by regression tests. Eidos Brain research and production remain untouched.

Final local validation: 63 tests passed, zero failed/skipped; typecheck passed. Webpack build passed before final additive async-failure/challenge hardening. The default Turbopack build failed on the shared node_modules junction; the remote final source build must pass before hosted acceptance. Eight private packages match the current website packaging receipt. Only four developer editions have installation acceptance; WordPress and builder pilots remain blocked.

Provider setup uses the existing merchant, protected Vercel project, new additive relay bypass and new TEST webhook. The named validation DB migration retained six existing orders. Mail remains blocked because a usable Resend secret has not been supplied by provider readback. Local mocked mail assertions are not received-inbox proof. Cloudflare public TEST keys still call siteverify and are confined to exact candidate host, TEST Stripe and commerce actions.

Rollback: restore prior paired preview or disable the template flag; retain additive schema and sold version archives. No production promotion or live charge is authorized.

Artifacts remain repo-local. Google Drive mirror was not configured for this Works task. Build outputs were measured; automatic execution review rejected .next cleanup and no deletion was performed.
