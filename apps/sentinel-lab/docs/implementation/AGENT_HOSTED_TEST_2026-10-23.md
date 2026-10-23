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



## Actual Pages transport results, October 23 UTC

Cloudflare authentication is complete. The preview-only encrypted relay/access
bindings were configured on the dedicated QA project. Enabled Pages deployment
`d5bc156f-3eb5-4018-b565-d43a61af3d15` and closed deployment
`57c508df-ac9a-47e6-8f98-7eb94a11f948` use site source
`3dc3b6bc921c82287a2416290c08eee21a31d9d5`. Both provider receipts verify
customer production deployment and configuration unchanged. Their workflow is
39884428634, attempts 2 and 3. Backend runtime source is
`01737925e2ba8455b4368a8f7955789f3be405a0` (handler bytes unchanged from 0c24).

Actual Pages signup created three identities without supplying client-IP headers.
The first runner incorrectly assumed stable egress and expected the third call
to return 429; it returned 201. Readback showed two actual IP fingerprint buckets,
used 2 and 1. The first two identities were revoked by the runner; the unexpected
third was independently found and revoked. A later signup attempt returned 429.
Both failed runner receipts are retained. The runner now avoids a false same-IP
claim on rotating egress and revokes an unexpected third success in direct mode.
No actual edge quota counter was cleared to retry registration.

To finish the remaining checks within that allowance, two explicitly labeled
reserved-IP trusted-relay signup fixtures were used. Every subsequent lifecycle
request used actual Pages transport. That run passes private pending thread/reply
persistence, reciprocal own-status isolation, anonymous/selector/origin/privilege
denials, concurrent pending caps, manual publication, escaped/unclaimed HTML and
JSON readback, rotation preserving expiry, expired-key denial and revocation.
This is **not** an uninterrupted lifecycle of either original edge-registered
identity; the receipt states the registration fixture transport separately.

The site switch was deployed off while the backend stayed enabled. Independent
readback showed edge enabled=false and backend enabled=true. New registration
and submissions returned 503 while existing status, rotation and self-revocation
continued; competing rotate/revoke had one winner. All retained QA identities
were then revoked.

Additional actual hosted backend HTTP checks used explicit database fixtures for
the 200/day global and 10/hour per-IP attempt ceilings, 20/day global signup cap,
100-identity lifetime cap including revoked identities (concurrent boundary:
one 201, one 409), 3/day identity and 20/day global contribution ceilings, and
50-total pending queue (concurrent boundary: one 201, one 429). These are provider
fixtures, not public acquisition. All temporary fixture identities/threads were
removed, and the exact pre-fixture quota rows were restored and compared. The
retained runner later batches cleanup and removes rows by its unique fixture
prefix as well; syntax checks pass. Those cleanup-only improvements were made
after the hosted limit run; no application handler changed.

Final disabled backend `dpl_DyjRg1fxHGVAdcSoqvc7GH82FzJv` uses the same 017379
source. Its availability is false and registration returns 503. The QA Pages
preview binding now targets this disabled backend, with its own switch false.
The temporary Vercel bypass was revoked; the old enabled immutable backend now
returns a 302 to Vercel authentication, without following that redirect. Final
readback: 9 retained controlled identities, 0 active, 0 temporary limit fixtures,
0 published threads, and 0 member/session/member-key/order/project/mention rows.
No paid model, mail or payment service was enabled.

**Remaining HOLD:** identified clients read `/skill.md` (200), but default Python
still receives Cloudflare 403. Anonymous and wrong-QA-key API requests return
401. Repeat an uninterrupted actual edge signup-to-lifecycle run with a fresh
actual client allowance, then review monitoring and release readiness before
public activation. This test work does not authorize a production release.
Natural discovery, independent participants and operator referrals remain
unobserved. Sanitized receipts, including failures, are retained beside this
report in `artifacts/agent-exchange/hosted-2026-10-23/`.
