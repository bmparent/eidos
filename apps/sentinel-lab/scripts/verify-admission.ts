import assert from "node:assert/strict";
import { createClient } from "@libsql/client";
import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { AdmissionStore } from "../lib/experiments/admission";

// Run from repo root. Only the existing separate validation database is eligible.
const url = process.env.EIDOS_DATABASE_URL || "";
assert.ok(url.includes("eidos-works-validation"), "A separate validation database is required");
const scope = `sentinel-audit-${Date.now()}`;
const clients = Array.from({ length: 12 }, () => createClient({ url, authToken: process.env.EIDOS_DATABASE_AUTH_TOKEN }));
const stores = clients.map(client => new AdmissionStore(client, scope));
const output = resolve("artifacts/sentinel-production-audit-20260906");
mkdirSync(output, { recursive: true });
try {
  await stores[0].initialize();
  const results = await Promise.allSettled(stores.map((store, i) => store.reserve(`remote-request-${i.toString().padStart(3, "0")}`, "a".repeat(64), "b".repeat(40), 1)));
  const admitted = results.flatMap(result => result.status === "fulfilled" ? [result.value] : []);
  assert.equal(admitted.length, 1);
  for (const result of results) if (result.status === "rejected") assert.match(result.reason.message, /RUNNER_CAPACITY_OCCUPIED/);
  const winner = results.findIndex(result => result.status === "fulfilled");
  const key = `remote-request-${winner.toString().padStart(3, "0")}`;
  const retries = await Promise.all(stores.map(store => store.reserve(key, "a".repeat(64), "b".repeat(40), 1)));
  assert.ok(retries.every(result => !result.acquired && result.admission.jobId === admitted[0].admission.jobId));
  await clients[0].execute({ sql: "UPDATE sentinel_lab_admissions SET expires_at=0 WHERE scope=?", args: [scope] });
  const recovered = await stores[1].reserve("replacement-request", "a".repeat(64), "b".repeat(40), 1);
  assert.equal(recovered.acquired, true);
  await assert.rejects(stores[0].beginAllocation(admitted[0].admission, 300_000), /ADMISSION_RESERVATION_EXPIRED/);
  await stores[1].beginAllocation(recovered.admission, 300_000);
  await clients[0].execute({ sql: "UPDATE sentinel_lab_admissions SET expires_at=0 WHERE scope=?", args: [scope] });
  await assert.rejects(stores[2].reserve("blocked-uncertainty", "a".repeat(64), "b".repeat(40), 1), /RUNNER_CAPACITY_OCCUPIED/);
  await stores[1].release(recovered.admission.jobId);
  const receipt = { timestamp_utc: new Date().toISOString(), status: "PASS", backend: "remote libSQL validation database", scope,
    independentClients: 12, competingRequests: 12, capacity: 1, admitted: 1, rejected: 11,
    retries: 12, duplicateJobs: 0, abandonedReservationRecovered: true, staleAllocatorFenced: true,
    uncertainAllocationRetainsCapacity: true, realSandboxAllocations: 0, proofGatesAdvanced: 0 };
  writeFileSync(resolve(output, "shared-admission-verification.json"), JSON.stringify(receipt, null, 2) + "\n");
  console.log(JSON.stringify(receipt));
} catch (error) {
  // Provider errors can contain URLs; keep this output sanitized.
  console.error("Admission verification failed: " + (error instanceof Error ? error.name : "unknown error"));
  process.exitCode = 1;
} finally { clients.forEach(client => client.close()); }
