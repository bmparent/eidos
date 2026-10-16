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

The full application suite and optimized build passed locally. The new deployment guards passed 2 tests; the changed TEST challenge was covered in the 24-test template suite. Frontend relay tests passed 5 tests, including acceptance at exactly 110,000 bytes, overflow rejection before transport and the longer AI deadline. The original direct root invocation of `works-platform.test.ts` failed because that older suite resolves migrations against the application working directory; the supported root command `npm --prefix apps/sentinel-lab test` passed without changing the older test behavior. Logs are retained under `artifacts/playground-authenticated-20261015/release-continuation/`.

All 32 immutable archive byte counts and SHA-256 values were restored and checked against the existing frozen receipt; no original archive was modified. This documentation records source preparation, not completed hosted account/payment/mail tests. Their subsequent receipts must establish real sign-in, Stripe signed-event acknowledgement, inbox delivery, exact approved-page download and refund revocation. Model comparison, human review, physical-device checks, LIVE pricing/funding and production activation remain separate gates.
