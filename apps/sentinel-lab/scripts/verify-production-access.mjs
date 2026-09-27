import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { cloneDefaultExperiment } from "../lib/experiments/shared.js";

const base = "https://eidos-sentinel-lab.vercel.app";
const spec = cloneDefaultExperiment();
spec.dataContract.maxRows = 1000;
const preflightResponse = await fetch(base + "/api/experiments/preflight", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec), signal: AbortSignal.timeout(30_000) });
const lock = await preflightResponse.json();
const routes = ["/api/experiments/rd-aaaaaaaaaaaa-bbbbbbbb", "/api/experiments/rd-aaaaaaaaaaaa-bbbbbbbb/artifacts/metrics.json"];
const reads = await Promise.all(routes.map(async path => {
  const response = await fetch(base + path, { signal: AbortSignal.timeout(30_000) });
  const body = await response.json();
  return { path, status: response.status, error: body.error, diagnosticId: body.diagnosticId ?? null };
}));
const dispatch = await fetch(base + "/api/experiments", { method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": "unauthenticated-audit-check" }, body: JSON.stringify({ spec: lock.spec, lockDigest: lock.digest }), signal: AbortSignal.timeout(30_000) });
const body = await dispatch.json();
const receipt = { timestamp_utc: new Date().toISOString(), url: base,
  preflight: { httpStatus: preflightResponse.status, readyToDispatch: lock.readyToDispatch, executionBackend: lock.executionBackend, digest: lock.digest, blockers: lock.issues?.filter(issue => issue.severity === "blocker") },
  unauthorizedReads: reads, unauthorizedDispatch: { status: dispatch.status, diagnosticId: body.diagnosticId, error: body.error },
  pinnedSpec: spec, authenticatedLiveJob: null, realComputeAllocated: false,
  verdict: "ACCESS_BLOCKED", reason: "The configured non-revealable production operator secret is not available in this task. Settings and authentication checks are not a successful production experiment." };
const out = resolve("artifacts/sentinel-production-audit-20260906");
mkdirSync(out, { recursive: true });
writeFileSync(resolve(out, "production-access-verification.json"), JSON.stringify(receipt, null, 2) + "\n");
writeFileSync(resolve(out, "pinned-run-lock.json"), JSON.stringify(lock, null, 2) + "\n");
console.log(JSON.stringify(receipt));
if (preflightResponse.status !== 200 || dispatch.status !== 401 || reads.some(read => read.status !== 401)) process.exitCode = 1;
