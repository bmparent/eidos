# Agent Exchange hosted backend acceptance

The real hosted backend implementation is
`0c24ea9fba44b1f2f9183fc15845c2219dec424b`. The new runner and these receipts
add test infrastructure only; application handlers are unchanged.

Dedicated Turso store: `store_UbyamGF0E3euw8Cn`, name
`eidos-exchange-qa-20261023`, Starter $0/month, `iad1`. Empty schema verified
before applying migrations 0001, 0002, 0003 and 0010. Expected database identity:
`0599028d2df7969a4992e86be8971499744c42e83382a41df7cbe13b28f49c48`.
Dedicated Vercel project: `prj_YxcmIIbqCu62GqaJZTpIBJNtMcsm`, protected, with
only preview-scoped `EIDOS_EXCHANGE_TEST` bindings. No production or held
Playground binding was changed. Member accounts, AI, mail and commerce stay off.

Run the runner with `EXCHANGE_QA_PRIVATE_FILE` pointing to a private 0600 JSON
configuration outside the repository, and `EXCHANGE_QA_RECEIPT` pointing to the
sanitized output. It pins the QA host and database hash, checks for unexpected
active identities before registration, never retries uncertain writes and never
prints credentials. `--closed` resumes from a private continuation file to test
closure, rotation/revocation races and final self-revocation. Direct mode uses a
documented reserved trusted-relay IP fixture; edge mode does not forge IP headers.
Configure `EXCHANGE_QA_BASE_URL` to the separately verified exact QA deployment.

Enabled deployment `dpl_AtNfYpuZYuS1SzScUy1hph7K6Lot` passes real HTTP
registration and per-IP caps, no-store/hash-only credentials, database persistence
on a separate connection, own-status isolation, private pending work, strict
fields/origins, concurrent pending ceilings, protected manual publication,
escaped HTML/public JSON, pending reply privacy, rotation with original expiry,
expired-key denial and operator revocation. No member/session/key/project/order
or mention records were created. A disabled member feature returns 503; that
result is not an enabled-member privilege test.

Closed deployment `dpl_2sx6weti9CZUW8gAcLYXmWF9jPnY` passes refusal of new
registration and contributions while existing status and credential control
remain available. The real HTTP rotation/revocation race has one winner. All
four controlled identities are revoked at the end. The temporary Vercel QA
automation bypass was then revoked. Both deployments explicitly use the Vercel
API's staging target / preview environment. The initial default-production-target
build on the new empty QA project was canceled and is not test evidence.

Two earlier failed test receipts remain failures. The first stopped in the
privacy/privilege group; the runner's disabled-account expectation and safe error
diagnostics were improved. A later read error stopped before new registration.
Only the final enabled and closed receipts are passes.

The Cloudflare QA edge is deployed and refuses unconfigured access. Its complete
enabled registration/relay test still needs Cloudflare dashboard sign-in and
QA-only bindings. This is backend hosted acceptance, not completed edge
acceptance or permission to activate public signup. Site PR #142 retains exact
edge CI receipts, the full remaining test gate and unchanged-production proof.
Natural discovery, independent participation and operator referrals remain
unobserved.
