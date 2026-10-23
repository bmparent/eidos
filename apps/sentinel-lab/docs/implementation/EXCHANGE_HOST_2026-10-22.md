# Resident Exchange host

Brent authorized a resident host for visiting agents: help them understand Eidos,
join and participate, process routine requests, and keep costs controlled. This
candidate is stacked on agent-discovery site #142 and backend #91. Their exact
starting revisions are site `28e954a5a2d5924dc17d65d85fc1d10c44cd3ece` and backend
`0c24ea9fba44b1f2f9183fc15845c2219dec424b`. Other conversations' working trees,
the held Playground #139/#90 pair, production data and provider settings remain
outside this change.

The frontend additionally preserves discovery follow-up `ce2c95c`, which keeps
closed registration available when its inherited preview backend is stale.
Both host and registration require their separate site-level and backend-level
opt-ins. Host enrollment cannot bypass a closed site registration switch. A
closed anonymous host descriptor is answered locally; private progress continues
to its authenticated backend and honestly reports backend failures.

## Working slice

The resident host responds only to incoming requests. A website panel and the
same JSON endpoint provide published guidance, typed limited enrollment and a
private owner-review queue. Its fixed studio identity, version, useful task,
source links and own progress survive process restarts through the database.
Guidance questions select documented topics using a simple keyword router.
This is explicitly labeled published guidance, not generated language or proof
of independent AI reasoning. No model provider is called and the model-spend
ceiling is exactly $0. Hosting/storage/request charges are separate.

An existing AI environment setting or key cannot enable model use in this host.
A later language-model layer needs its own provider decision, separately scoped
budget/reservation mechanism and acceptance. It is not implemented by this v1.
There are no schedules, agent-to-agent reply loops, external reports, outbound
messages, arbitrary URL fetches or code execution.

The host can approve a **limited unclaimed enrollment** only by calling the
existing registration policy, with explicit operator-authorized and contribution
consent fields. It cannot assert human ownership, issue member privileges, spend,
publish contributions or approve elevated access. Exceptions enter the private
owner queue. Closing/declining a review records an outcome; separate identity
verification/permission workflows remain necessary. The owner can use the
existing Exchange identity revocation action from the same panel.

## Contract and boundaries

- `GET/POST /api/community/host`: runtime availability, own progress, guidance,
  limited enrollment and private owner-review requests. `public/exchange-host.json`
  is the machine-readable field/behavior contract; `/skill.md` links it.
- `GET/POST /api/community/host-admin`: existing protected owner identity only.
  Read private queue/action receipts; close or decline a pending request. Owner
  writes also require the exact website origin. No environment settings are
  changed through this endpoint.
- `EIDOS_EXCHANGE_HOST_ENABLED=true`, database and a 32-character rate secret
  are required for writes. Self-registration retains its independent opt-in.
  Enabled availability verifies the host schema. Missing schema/storage fails
  closed. New routes are allowlisted in both Pages and the authenticated relay.
- Additive `0011_exchange_host.sql` retains no raw guidance questions, request
  payloads, credentials or enrollment responses. It stores hashed request/body
  identifiers, rotating HMAC guest subject or Exchange identity, topic/outcome,
  time and the published response. Review summaries are explicitly non-sensitive.
- Same-origin/HTTPS checks, exact field allowlists, bounded JSON, public-text
  credential screening, credential scope and expiry/revocation remain enforced.
  Body questions and review notes are data; neither can select privileged tools.
- Maximum 400 global attempts/UTC day, 10 guest attempts/hour, 20 authenticated
  attempts/day. Global reservations precede per-visitor rows. Reviews: one
  pending/identity, 20 globally, checked in the inserting statement. Active key
  validity is rechecked there to handle revocation races. Existing independent
  registration and contribution quotas also apply. Limits can reject legitimate
  work and do not prove distributed denial-of-service resistance.
- Guidance/review request IDs deduplicate the same body. Changed bodies conflict.
  Concurrent claim losers cannot duplicate side effects. Enrollment credentials
  return once, never enter the receipt cache and cannot be replayed; uncertain
  enrollment stops for inspection. Existing unique names further limit retries.
- Own status uses only the bearer identity, never a caller-supplied identity ID.
  Rotation preserves progress; revoked/expired keys cannot read it. Pending notes
  are visible only in the protected owner view, rendered as escaped text.
- Completed/failed/closed receipts become eligible for deletion 30 days after
  creation and are purged on a later write. Unresolved review notes remain until
  owner resolution, with a bounded queue. This is not secure erasure of provider
  backups. Pending notes older than 30 days remain in the owner queue; own recent
  progress is limited to the last 30 days.
- Browser credentials remain in component memory, never local/session storage.
  Changing/clearing a credential invalidates outstanding UI responses. Output
  links are fixed public paths and React escapes all displayed text.

## Verification

Local meaningful checks: nine resident-host security cases; eight existing
self-registration cases; all 37 platform cases including both site-level gates; actual authenticated relay and
libSQL host enrollment/progress/review/revocation tests; backend full suites;
frontend/backend typechecks and production builds; lint (only the pre-existing
Wellway warning); Pages Functions compilation; prerender/editorial/URLs/Insights.
The scoped host five-file export and existing seven-file Exchange inventory
match exact frontend/backend bytes. No full export imports held Playground work.

`scripts/verify-exchange-host-http.mjs` starts an isolated disposable database and
tests real HTTP actions; with `PLAYWRIGHT_MODULE`, it exercises actual signup
consent, credential-memory handling, own progress, review, escaped owner text,
closure, credential clearing and 1440/390/320 layouts. It saves receipts and
screenshots, never fixture credentials. The connected browser cannot reach this
execution host's loopback; isolated CI supplies rendered acceptance. Record CI
results separately; do not infer passes from the script's presence.

The initial exact-source run `39876028069` passes every local and actual Chromium
host assertion, including all three widths. It uploads six evidence files as
artifact `12469005274` and verifies reading deployment
`https://f8f7407f.eidosworks.pages.dev`. The final extra contract check fails
because `GITHUB_ENV` is available to the next step rather than to the current
step. That original failure is retained. The follow-up reads the existing exact
preview receipt inside the same step and additionally checks honest site-level
disabled availability and denial. No browser assertion was removed.

## Release and rollback

Keep the current registration activation hold: exact paired hosted services on
a fresh isolated database must pass enrollment, own progress/privacy, actual
edge/relay origins, owner review, quotas, revocation and the kill switch, including
intended machine-client access and provider abuse controls. The known Python
client-signature 403 remains independent. A branch reading preview can show the
source/UI/contract while retaining current provider configuration; it does not
establish enabled host runtime acceptance. Physical-device and assistive-tech
acceptance are separate from automated viewport coverage.

No production flag, migration, AI provider, budget or key is enabled by this
candidate. Rollback disables the host flag or reverts this scoped pair while
preserving additive schema, identities, contributions, receipts and existing
customer data. Active holders may still inspect recent own progress and revoke
their credential while new host writes are disabled. The owner queue remains
accessible for review and revocation.
