# Works public profiles — October 23, 2026

Paired with site `fix/public-profiles-20261023` from public main `1dc8323`.
Backend source starts from production-compatible main `d01920b` and changes
only four vendored community/member handlers plus a libSQL integration test.

The exact username directory read exposes public identity, bounded published
posts and published replies with published parents. It binds immutable member
ownership, excludes disabled accounts, and returns no email, owner IDs,
sessions, keys, preferences, private projects or saved reading. Profile reads
do not depend on signup/mail activation. Existing prefix lookup remains as-is.
Community responses supply a member username only from a valid identity join;
thread authors and replies link back to that public profile.

Local typecheck, baseline tests/build and the new real-libSQL authenticated
relay test pass. Exact CI, deployment identity and production Pages pairing
remain recorded on the release PR; source validation is not a live claim.
There is no migration or provider setting change in this source patch.
Rollback preserves all data and restores the prior compatible backend source.
