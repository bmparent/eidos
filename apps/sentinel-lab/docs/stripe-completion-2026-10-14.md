# Stripe completion — October 14, 2026

This continuation preserves the guarded payment, entitlement, receipt, recovery
and refund implementation from PR72. It repairs a deployment gap: the private
buyer archives are ignored by Git, so a Git-only deployment can report READY
without the archives required for paid downloads.

The new prebuild gate checks every current and retained archive against a frozen
byte count and SHA-256 whenever template commerce or a template source revision
is configured. Missing, altered, duplicate or symbolic-link archives stop the
build. The secret-free bundle tool includes the complete archive inventory and
uses the existing isolated TEST environment branch; no credentials are read.

The paired site restore script recovered all 32 archives (8 current, 24 retained)
from committed source with exact historical hashes. Nightjar developer 1.0.2
was an uncommitted transition; reversing the documented 1.0.3 typography patch
reproduced its previously frozen byte count and SHA-256 exactly.

Local verification: 19 MJS tests and 57 TypeScript tests passed, including the new
missing/altered/duplicate/symlink archive deployment gate. TypeScript passed.
Actual hosted/provider results and unresolved mail, payment and refund gates
are recorded in the dated acceptance ledger after deployment.

Deployment steps from the backend repository root:

```text
node apps/sentinel-lab/scripts/bundle-template-preview.mjs --out=/private/task/path/deployment.json
```

Upload that request through the authenticated Vercel connector as Preview only.
Use its bundle hash and exact backend commit in EIDOS_TEMPLATE_SOURCE_REVISION
on Preview branch codex/paid-templates-20261010. Pin the returned deployment in
the paired site and verify its authenticated catalog before configuring Pages.
Do not use a Git-only deployment for enabled commerce. Production sales stay
closed until actual TEST receipt/recovery delivery and signed webhook, refund
and replay results are accepted on the final pair.
