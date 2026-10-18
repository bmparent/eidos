# Observatory first-milestone operations

This is a private candidate, not an unattended monitoring service. Keep the Site
owner-private. Do not enable billing or an automatic model loop from this runbook.

## Deployment and source identity

Reconciled start: PR79 `c184ae06819e7ad326f141f378b2d83f9b91152c`, stacked on PR77.
Runtime candidate: `4a9bd8b9da652a1d6587412316e18d97d35d76f8`.
Node collector: `dpl_7iTnN6c4qSwsHN5w4U9EtQ7ShyZP`, immutable preview
`https://eidos-sentinel-qam5wbiqu-1brentbm-1876s-projects.vercel.app`.
Sites version 17 source: `0295a79cfea528efa8295a766c64c5d2b3f4524c`.
The precise release and environment revision are in the local `release.json`.
No branch alias or unrelated production Vercel deployment was promoted.
The original collector secret was restricted to PR77's preview branch. The
candidate reuses that credential as an encrypted variable scoped only to its
own preview branch; no key was rotated or broadened to production. The first
new deployment therefore returned unavailable until it was redeployed with the
existing credential. That failed hosted receipt is retained.

## Reproduce and verify

Use [the harness README](fault-harness/README.md) from repository root. Lock
checks must pass before execution. All benchmark outputs belong in a new run
folder; preserve unsuccessful attempts. A successful deployment is separate
from served-source, retained API, browser and model/provider acceptance.

## Private rollback

The freshly reconciled rollback is Sites version **16**:
`appgprj_6ace6f7f4a5481918adb34a5cb366de3~appgver_4671520c5d888191ae59e2dc26d5315d`.
Restore only `OBSERVATORY_TARGET_COLLECTOR_URL` to the previous immutable
collector's `/api/observatory/ip-observe` URL, then deploy that saved private
version so the restored environment revision applies. Preserve all other secret
values, D1 data and audience. The prior collector URL is in `release.json`.
Version 15 remains an older rollback. A rollback drill has not been executed.

## Storage, backup and restore

No migration was added. D1/companion migrations 0001–0003 remain compatible;
old v1 evidence stays readable and new attempt fields are additive. The original
recorded capture SHA-256 remains
`a96bada34c90cc6a13cf93682f1b99be030470020407a63367794016b364bb8f`.
Before any future ownership migration, export authoritative evidence and use
the provider's supported database backup to a restricted destination. Restore
into an isolated database and verify counts, hashes, duplicate exclusion and
checkpoint history before changing bindings. Never overwrite the live database
as a test. This document describes the drill; no provider backup/restore acceptance
is claimed. Mutable preview SQLite files are excluded from the proof mirror;
portable JSON records/exports are retained instead.

## Capacity, health and pilot boundary

One collector and foreground sampling remain. Current rate/concurrency Maps are
per process, not shared admission. The global 2048-record/64-baseline caps and
target-only checkpoint keys are unsuitable for a two-observer seven-day pilot.
Alternating collectors resets a baseline. Implement tenant/monitor/observer and
settings-qualified keys before independent observations can be combined safely.

A conservative planning example is two authorized targets × two genuinely
independent observers × five-minute cadence: 1152 checks/day, 8064 in seven days,
before retries. Actual hosting entitlement, queue/Node/storage and notification
costs are unknown. Do not start a pilot until durable slots/leases, quotas,
coverage accounting, incidents/outbox and a target allowlist are implemented.
`pilot_manifest.json` therefore has status not_started and a null run identity.

Current platform reference points, not verified account entitlements:
[D1 transactions](https://developers.cloudflare.com/d1/worker-api/d1-database/),
[at-least-once Queues](https://developers.cloudflare.com/queues/reference/delivery-guarantees/),
[Cron configuration](https://developers.cloudflare.com/workers/configuration/cron-triggers/),
[Vercel Cron limits](https://vercel.com/docs/cron-jobs/usage-and-pricing).
The Sites cloud-task schedule tool does not establish runtime monitor scheduling.

## Proof Logic + Meaning

The operating gate is partial: source/deployment provenance and reversible
private publication are recorded, while backup/restore and continuous operation
are not proved. The former state mixed a live Site, stacked PRs and an older
collector. Exact immutable identities and additive contracts make that split
auditable. Bounded attempts share D = start + timeout; a failed address never
authorizes an unmeasured host-wide claim. This represents explanation and
reproducibility before automation. It strengthens the north-star's revisitable
evidence, while the pilot, tenant and incident portions remain open. See
release.json, hosted/receipt.json, production-trace.json, the acceptance matrix,
benchmark summary and Drive manifest for the actual receipts and limitations.
