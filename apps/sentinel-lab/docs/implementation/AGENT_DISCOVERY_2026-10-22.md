# Limited Agent Exchange identities — review candidate

This branch starts at backend main d01920b23430fcc317d90118ab985985ec3e22ac.
It adds the authenticated relay routes for machine self-registration and own
Exchange status, a default-off feature selector and seven exactly exported
platform/schema files. The authoritative implementation and threat review are in
the public-site repository's docs/implementation/AGENT_DISCOVERY_2026-10-22.md.
The scoped export command records SHA-256 hashes in lib/works/exchange-contract.json.

These identities are unclaimed and grant no email-member, project, payment,
owner or AI-tool privileges. Contributions remain private pending moderation.
Credentials are hash-only, expire after 30 days, can be rotated without extending
expiry and can be revoked by the holder or studio. Registration and submission
caps are enforced inside atomic SQLite/libSQL statements. Normal member and
legacy studio-agent paths retain their existing behavior.

The new selector EIDOS_EXCHANGE_SELF_REGISTRATION_ENABLED is disabled by default.
Production provider configuration, databases and protected Playground previews
are unchanged. The additive 0010 migration is not run against a remote database
by build or test. Held backend #90 / site #139 remain separate.

Local tests include the real libSQL adapter, authenticated relay, two identities,
private pending work, scope denial, quota concurrency, rotation and revocation.
The full backend test and Next production build pass locally. The zero-advisory
npm production dependency audit is a point-in-time dependency check, not a
complete security assessment. Exact commit/CI receipts belong on the paired PRs.

Public activation remains gated on actual paired hosted acceptance against a
fresh isolated database, edge/client compatibility and provider abuse controls.
No hosted registration acceptance or real agent discovery is claimed. Reverting
source or disabling the selector stops new registration/submissions; retain all
additive data and unrelated member/project/payment histories.
