# Playground spatial composition follow-up

This branch is based on the held account integration PR #58 (`81f8b2e8c9876e59720712be780bb8b2c17c7785`). It pairs with frontend PR #35, `bmparent/brent-parent-intelligence-studio:codex/playground-spatial-composition-20260910`.

Only the seven portable Playground source/vendor modules and this note change. No research executor, route dispatch, credentials, DB migrations, provider configuration, payment flags or production deployment is modified.

The shared model accepts opt-in schema 3 hero composition and retains v1/v2 behavior until upgrade. Layout intent and image ownership slots survive validation/render/export. The editor runtime is a dependency of the shared renderer but is only emitted when its editor bridge flag is enabled. Ordinary export has no drag handles or editing runtime.

Read the frontend `docs/playground/spatial-composition-20260910.md` for scope and evidence. Sixteen portable Node tests and 17 isolated Chromium checks passed there; these are not a completed backend build/test run, real hosted ownership test or device acceptance. Run the complete backend lint/tests/build and source/vendor parity before paired preview use. Re-run the frontend vendor exporter only from the matching source checkout; do not overwrite unrelated research or alter frozen purchased archives.

Keep this PR draft. The original Google/Resend credential approvals, hosted account acceptance and WebKit Turnstile diagnosis remain outstanding. Public AI/images and live Playground sales remain disabled. Physical iPhone, Stripe TEST and actual authorized InkSoft tests remain separate. No production merge is authorized by this follow-up.
