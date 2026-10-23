# Local studio / account boundary — October 13, 2026

Candidate branch: `codex/showcase-studio-boundary-20261013`, stacked on delivery draft PR72 exact head `952e68ff75da6857e25f249bebd96ea65116e294`. Existing dirty template/delivery worktrees are preserved.

The site adds schema-5 local studios while the backend's accepted cloud contract remains schema 1. The new `studioBoundary.ts` rejects schemaVersion ≥ 5 or any own `studios` field with HTTP409 and an explicit local JSON-backup message. It runs before older validation and before database access in `saveProject`. The HTTP route checks it after same-origin/body validation and before authentication/schema initialization, so an unsupported-format request cannot write tables or discard studio data. No owner data is returned by this format rejection.

Legacy schema-1 saving, immutable revisions, stale-head conflicts, owner checks, asset storage and purchased archives retain their existing contracts. No format migration, renderer/font sync, cloud-format activation, account acceptance or payment release is implied. Production remains unchanged.

Validation from repo root:

```sh
npm --prefix apps/sentinel-lab test
npm --prefix apps/sentinel-lab run lint
```

All 76 app cases passed (16 MJS + 60 TypeScript), including three new cases covering direct and HTTP rejection before any database read, source preservation, disguised schema1+studios input, future versions and legacy delegation. Existing concurrent-save and historical-purchase regressions passed. TypeScript passed. Logs: `artifacts/showcase-studio-boundary-20261013`.

## Proof Logic + Meaning

The format-safety gate passed locally. Before this change, the older backend rejected unknown schema versions generically and lacked an explicit safeguard against studio fields attached to an older schema; older validators may otherwise drop fields they do not know. The logic is `local_only = numeric_schema >= 5 OR has_own_studios_field`; local-only implies HTTP409 before persistence. Three negative tests prove database touches = 0 and input-after = input-before. Legacy acceptance remains tested by the existing suite.

This is honesty before convenience: a local backup must not be mislabeled an account revision. It strengthens reproducibility and user trust around the north-star system's auditable receipts. It does not change or prove learning, anomaly detection, compression, scientific readiness or provider ownership acceptance. Exact source/base/test receipts support this narrow result; paired authenticated schema-5 save/reopen, two-account denial and production release remain unproven and gated.

Rollback is base `952e68ff75da6857e25f249bebd96ea65116e294`; preserve all private archives, orders, entitlements and existing configuration. The paired site should copy only `studioBoundary.ts` byte-for-byte and call the same guard before its current validation; do not wholesale replace the backend vendor tree.

## Exact hosted Preview acceptance

READY protected Preview: https://eidos-sentinel-rkhg0orv8-1brentbm-1876s-projects.vercel.app; deployment `dpl_HJnTcwVPPwj77qebkGZiCoXM9z5L`; product source `3ebbc4d0e4cfd9b72947b097366e8b1b6fe8b573`. Seven actual hosted API checks passed. Schema5, disguised schema1+studios and future schema6 returned409 with JSON-backup instructions. Public configuration has accountsReady=false, aiReady=false and shopReady=false. Invalid relay returned401; an unrelated Pages origin returned403; unauthenticated access redirected302 to vercel.com sign-in. An initial harness expected401/403 only; its precise302 observation and correction are preserved.

Only `codex/showcase-studio-boundary-20261013` branch Preview variables were written. The existing scoped relay, protection bypass and `EIDOS_TEMPLATES_TEST` validation database remain the same; exact database hostname and variable names are in `preview-environment.json`, with no values. All commerce, account, AI, newsletter, proactive and community-studio flags are false. OpenAI/mail/Playground payment credentials are explicitly empty. TEST Stripe key is retained behind disabled commerce. Production, existing branches and GitHub secrets were not changed. Allowed origin is exactly `https://eidosworks-showcase-20261013.pages.dev`; no wildcard was added.

The source-only environment configuration/verification helpers reproduce this scope and preserve credentials in the existing ignored input. CLI64.0.2 was installed into the npm execution cache without changing repository dependencies, using official [Vercel API CLI documentation](https://vercel.com/docs/cli/api). This exact deployed product source is separate from the subsequent documentation/tooling receipt commit. New authenticated cloud-format acceptance and provider delivery remain gated.
