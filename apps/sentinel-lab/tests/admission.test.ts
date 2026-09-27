import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createClient } from "@libsql/client";
import { AdmissionStore, admissionConfigured, validateRetryKey } from "../lib/experiments/admission";

const digest = "a".repeat(64), commit = "b".repeat(40);

test("independent clients cannot exceed capacity and retries return the same job", async () => {
  const dir = await mkdtemp(join(tmpdir(), "sentinel-admission-"));
  const clients = Array.from({ length: 12 }, () => createClient({ url: `file:${join(dir, "shared.db")}` }));
  try {
    const stores = clients.map(client => new AdmissionStore(client, "test"));
    await stores[0].initialize();
    const outcomes = await Promise.allSettled(stores.map((store, i) => store.reserve(`competing-request-${i}`, digest, commit, 1)));
    const winners = outcomes.filter(result => result.status === "fulfilled");
    assert.equal(winners.length, 1);
    for (const result of outcomes) if (result.status === "rejected") assert.match(result.reason.message, /RUNNER_CAPACITY_OCCUPIED/);
    const winner = outcomes.findIndex(result => result.status === "fulfilled");
    const original = winners[0].value;
    const retries = await Promise.all(stores.map(store => store.reserve(`competing-request-${winner}`, digest, commit, 1)));
    assert.ok(retries.every(value => !value.acquired && value.admission.jobId === original.admission.jobId));
    await assert.rejects(stores[1].reserve(`competing-request-${winner}`, "c".repeat(64), commit, 1), /IDEMPOTENCY_LOCK_CONFLICT/);
  } finally { clients.forEach(client => client.close()); await rm(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }); }
});

test("abandoned reservations recover, stale allocators are fenced and old retry keys never relaunch", async () => {
  const client = createClient({ url: ":memory:" });
  try {
    const store = new AdmissionStore(client, "test");
    const first = await store.reserve("abandoned-request", digest, commit, 1);
    await client.execute("UPDATE sentinel_lab_admissions SET expires_at=0");
    const next = await store.reserve("replacement-request", digest, commit, 1);
    assert.equal(next.acquired, true);
    await assert.rejects(store.beginAllocation(first.admission, 300_000), /ADMISSION_RESERVATION_EXPIRED/);
    const oldRetry = await store.reserve("abandoned-request", digest, commit, 1);
    assert.equal(oldRetry.acquired, false);
    assert.equal(oldRetry.admission.phase, "ABANDONED");
    assert.equal(oldRetry.admission.jobId, first.admission.jobId);
  } finally { client.close(); }
});

test("allocation uncertainty retains its slot beyond expiry until provider reconciliation", async () => {
  const client = createClient({ url: ":memory:" });
  try {
    const store = new AdmissionStore(client, "test");
    const first = await store.reserve("uncertain-allocation", digest, commit, 1);
    await store.beginAllocation(first.admission, 300_000);
    await assert.rejects(store.beginAllocation(first.admission, 300_000), /ADMISSION_RESERVATION_EXPIRED/);
    await client.execute("UPDATE sentinel_lab_admissions SET expires_at=0");
    await assert.rejects(store.reserve("competing-allocation", digest, commit, 1), /RUNNER_CAPACITY_OCCUPIED/);
    assert.equal((await store.active()).length, 1);
    assert.equal(await store.expired(first.admission.jobId), true);
    await store.release(first.admission.jobId); // provider-confirmed stop is required by caller
    assert.equal((await store.reserve("competing-allocation", digest, commit, 1)).acquired, true);
  } finally { client.close(); }
});

test("admission fails closed without remote storage or a valid retry key", () => {
  assert.equal(admissionConfigured({ EIDOS_DATABASE_URL: "file:/tmp/ephemeral", EIDOS_DATABASE_AUTH_TOKEN: "x" }), false);
  assert.equal(admissionConfigured({ EIDOS_DATABASE_URL: "libsql://example.turso.io", EIDOS_DATABASE_AUTH_TOKEN: "x" }), true);
  for (const key of [null, "", "short", "invalid header key", "x".repeat(129)]) assert.throws(() => validateRetryKey(key), /IDEMPOTENCY_KEY_REQUIRED/);
});
