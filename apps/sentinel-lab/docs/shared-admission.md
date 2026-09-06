# Shared Sandbox admission and retry recovery

The Sandbox execution backend uses the existing remote libSQL configuration (`EIDOS_DATABASE_URL`, `EIDOS_DATABASE_AUTH_TOKEN`). It creates only `sentinel_lab_admissions`; Eidos Works tables are untouched. Missing or ephemeral storage blocks preflight and allocation. No new service, plan, capacity, profile or engine behavior is enabled.

Clients must supply a 16–128 character `Idempotency-Key` containing letters, digits, underscores or hyphens. The browser writes its key and locked settings into session storage before dispatch; the operator credential stays in page memory. Retrying after a lost response or reload returns the same job, including across releases. Reusing a key with different settings returns 409. Preparing another experiment explicitly clears the intent. Retry records are retained as tombstones so old requests cannot unexpectedly launch new compute.

## Atomic admission

One primary write transaction expires unused reservations, counts occupied slots, conditionally inserts a reservation, and returns the existing or inserted receipt. A key hash and project scope identify a unique record. Each reservation has a random owner; only that owner can atomically transition an unexpired `RESERVED` row to `ALLOCATING`. No provider call occurs before that transition.

`occupied = count(RESERVED + ALLOCATING + RUNNING)`

`admit iff occupied < configured_capacity`

Because the count and insert share the write transaction, competing Vercel instances cannot independently consume the final slot. Provider listing remains a conservative guard for jobs from older releases, in addition to the shared admission check.

## Recovery and limits

- An unused reservation expires after six minutes, exceeding the dispatch route's five-minute maximum lifetime. A stale allocator is rejected by an atomic phase/owner/deadline check.
- Allocation intent occupies capacity for the configured compute timeout plus a six-minute dispatch grace period. Expiry alone never frees an uncertain allocation. After the deadline, a provider read must confirm the named Sandbox stopped or does not exist. Provider errors preserve the slot and remain retryable. Reconciliation runs during subsequent launches; no paid scheduler is added.
- Monitoring a terminal result stops its session before releasing its reservation. A failed stop is surfaced as a retryable retrieval error; capacity is retained until reconciliation.
- Retries return the stable job ID without re-running source discovery, allocation or the launcher. The unique provider name additionally binds compute to that identity.
- A reservation with no retrievable Sandbox returns a queued or expired durable receipt. Existing completed, crash and expiry receipts stay in the provider snapshot. Snapshot retention remains seven days.
- The invariant covers admitted experiment jobs. Snapshot retrieval sessions are short-lived provider operations; the existing API is not a global quota on all provider VMs, and concurrent retrieval requests can still temporarily overlap. No claim is made that a provider outage guarantees immediate retrieval cleanup.
- This release qualifies the production Sandbox path. The optional external FastAPI runner still requires its own distributed admission design before a multi-instance deployment; forwarding a stable key alone does not qualify that backend.

## Reproduction

From the repository root:

```sh
npm ci --prefix apps/sentinel-lab --no-audit --no-fund
npm test --prefix apps/sentinel-lab
npm run lint --prefix apps/sentinel-lab
npm run build --prefix apps/sentinel-lab
```

With the separately configured validation database in the process environment:

```sh
node apps/sentinel-lab/node_modules/tsx/dist/cli.mjs --tsconfig apps/sentinel-lab/tsconfig.json apps/sentinel-lab/scripts/verify-admission.ts
```

The remote check creates 12 independent clients, competes for capacity one, retries the winning intent 12 times, expires an unused reservation, fences its stale allocator and confirms that an uncertain allocation remains occupied. It allocates no Sandbox and writes a sanitized JSON receipt under `artifacts/sentinel-production-audit-20260906/`.

## Proof Logic + Meaning

The shared-admission gate is an engineering reliability check. Previously, two requests could both observe spare capacity and start duplicate or excess jobs. Atomic counting and insertion plus a stable retry identity remove that race; compare-and-update fencing prevents expired owners from allocating. This is restraint before execution: budget limits become enforceable across instances and operator retries preserve their meaning. The change strengthens reproducible operation of the streaming system without altering thresholds, source ordering, label isolation, predictions, evaluation partitions or proof scoring. The remote database receipt supports the admission claims; it does not establish successful authenticated Vercel engine execution, useful detection quality or held-out generalization. Zero proof gates advance.

The transaction contract follows the [Turso TypeScript reference](https://docs.turso.tech/sdk/ts/reference). Persistent-session behavior follows the [Vercel Sandbox documentation](https://vercel.com/kb/guide/vercel-sandbox-duration-and-persistence).
