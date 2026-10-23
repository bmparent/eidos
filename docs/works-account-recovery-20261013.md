# Works account candidate recovery — October 13, 2026

Brent requested starting the Command Center's account and template acceptance priorities. This recovery merges current backend main `4c746ffccf9249c7ddabd58f839ecb5fcece4fd0` into held #67 at `98314f33043f3772642acf64abba3a1db2567752`, preserving both parents and all existing records.

The only merge conflict is the Community thread renderer. Canonical site export preserves current immediate-publication behavior, direct Reply controls, versioned returning-browser assets and branding alongside the held account/owner implementation. Site recovery starts from #98 `9120d53d8f27104a9d1903d64bd19af7a54f93c2` and merges public main `fb74c4cf00defef23ce3ee96af4e51c350c3e807`. Exact pair revisions are recorded in the final PR receipt.

The imported Community test originally assumed a bare moderation bearer token was sufficient. Recovery retains the stricter owner boundary: without an explicit automation allowance it still returns 401 and creates no agent. The in-memory fixture now declares its intended automation authority and verifies both denial and a durable owner-audit entry. No runtime authentication guard is relaxed.

The original frozen preview's canonical Account page loads password, Google and email-link controls. Its immutable Pages hostname displays an account-load error; identity/OAuth acceptance must use the configured canonical validation origin. Actual signup, Google consent, recipient-observed email, two-owner save/reopen and owner-console access remain open. Direct protected-content inspection was unavailable through the Vercel connector, so that attempt is not health evidence.

Keep this draft unmerged and production NO-GO. Retain the existing preview-only database binding and additive schema. This source recovery changes no provider configuration, production alias, LIVE payment, model budget, customer data or archive. Fixture tests, source parity and READY builds do not complete hosted-user acceptance. The frozen starting pair remains a rollback identity; do not delete account or payment records to roll back code.

Local verification completed: all 16 JavaScript checks and 66 TypeScript checks passed across the full run and the focused corrected Community rerun. TypeScript and the Next production build pass. The imported Community fixture was the only initial behavioral failure; it now verifies denied undeclared authority, explicit fixture authority and owner-audit persistence. These results are local evidence, not real hosted identity or inbox acceptance.
