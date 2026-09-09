import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import { resolve, join } from "node:path";
import { randomBytes } from "node:crypto";
import { Sandbox, FileSystem } from "@vercel/sandbox";
import { AdmissionStore, sharedAdmission } from "../experiments/admission";
import { verifySandboxSource } from "../experiments/sandbox-source";
import { guidedStore, hash, LabError, LIMITS, type Document, type GuidedStore } from "./store";
import { summarizePatterns } from "./patterns";

const terminal = new Set(["completed", "failed", "cancelled", "expired"]);
const local = () => process.env.EIDOS_GUIDED_LOCAL === "1" && !process.env.VERCEL;
const credentials = () => process.env.VERCEL_TOKEN && process.env.VERCEL_TEAM_ID && process.env.VERCEL_PROJECT_ID ? {
  token: process.env.VERCEL_TOKEN, teamId: process.env.VERCEL_TEAM_ID, projectId: process.env.VERCEL_PROJECT_ID,
} : {};
function sourceCommit() {
  const value = process.env.VERCEL_GIT_COMMIT_SHA || process.env.EIDOS_SOURCE_COMMIT || "";
  if (!/^[a-f0-9]{40}$/i.test(value)) throw new LabError(503, "The analysis worker requires an exact source commit.");
  return value;
}
function admission(store: GuidedStore) { return local() ? new AdmissionStore(store.client, "local:sandbox") : sharedAdmission(); }
function localDirectory(id: string) {
  if (!/^gj-[a-f0-9]{32}$/.test(id)) throw new LabError(400, "Invalid job identity.");
  return resolve(process.env.EIDOS_GUIDED_JOB_ROOT || "artifacts/guided-jobs", id);
}

export async function startJob(owner: string, id: string, store = guidedStore()) {
  let job = await store.job(owner, id);
  if (job.status !== "queued") return job;
  const revision = sourceCommit();
  const lease = await store.claim(owner, id);
  if (!lease) return store.job(owner, id);
  const gate = admission(store);
  let reserved;
  try { reserved = await gate.reserve(`guided_${id}`, job.request_hash, revision, 1); }
  catch (error) {
    if (String(error).includes("CAPACITY_OCCUPIED")) {
      await store.updateJob(owner, id, "queued", { error: "Waiting for the shared compute slot. Retry this same job when it is free." }, lease);
      return store.job(owner, id);
    }
    await store.updateJob(owner, id, "failed", { error: "Durable admission failed; no worker was started." }, lease); throw error;
  }
  const reservation = reserved.admission;
  const callbackToken = randomBytes(32).toString("base64url");
  const callbackOrigin = local() ? process.env.EIDOS_GUIDED_CALLBACK_ORIGIN : process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : undefined;
  const provider: Document = { kind: local() ? "local" : "sandbox", admissionId: reservation.jobId,
    name: `eidos-${reservation.jobId}`, directory: local() ? localDirectory(id) : `/vercel/sandbox/guided/${id}`, allocatedAt: Date.now(),
    callbackHash: hash(callbackToken) };
  await store.updateJob(owner, id, "preparing", { provider }, lease);
  if (!reserved.acquired) return store.job(owner, id);
  await gate.beginAllocation(reservation, LIMITS.jobSeconds * 1000);
  const payload = JSON.stringify({ ...job.request, sourceCommit: revision });
  let allocationStage = "create";
  try {
    if (local()) {
      await mkdir(provider.directory, { recursive: true });
      await writeFile(join(provider.directory, "request.json"), payload);
      const stream = createWriteStream(join(provider.directory, "runner.log"), { flags: "a" });
      const child = spawn(/*turbopackIgnore: true*/ process.env.EIDOS_GUIDED_PYTHON || "python", ["-m", "sentinel_runner.guided.cli", "--request", join(provider.directory, "request.json"), "--out", provider.directory], {
        cwd: resolve(/*turbopackIgnore: true*/ process.env.EIDOS_GUIDED_REPO_ROOT || "../.."), windowsHide: true,
        // Child execution has no provider/account/database credentials.
        env: { NODE_ENV: process.env.NODE_ENV, PATH: process.env.PATH, SystemRoot: process.env.SystemRoot, USERPROFILE: process.env.USERPROFILE,
          TEMP: process.env.TEMP, TMP: process.env.TMP, LOCALAPPDATA: process.env.LOCALAPPDATA, APPDATA: process.env.APPDATA,
          EIDOS_CALLBACK_URL: callbackOrigin ? `${callbackOrigin}/api/lab/v1/worker/${id}` : undefined,
          EIDOS_CALLBACK_TOKEN: callbackToken,
          PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8", OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1",
          HF_HUB_DISABLE_TELEMETRY: "1", EIDOS_ARTIFACT_ROOT: provider.directory },
        stdio: ["ignore", "pipe", "pipe"],
      });
      child.stdout.pipe(stream); child.stderr.pipe(stream);
      child.on("error", async () => { await store.updateJob(owner, id, "failed", { error: "Local Python worker could not start." }); await gate.release(reservation.jobId); });
      provider.pid = child.pid;
      // Parent restart does not own the durable result. The worker keeps writing to its directory.
      child.unref();
    } else {
      const allocated = await Sandbox.create({ name: provider.name,
        source: { type: "git", url: process.env.EIDOS_SOURCE_REPOSITORY || "https://github.com/bmparent/eidos.git", revision, depth: 1 },
        image: "vercel/sandbox/python:3.14", resources: { vcpus: 4 }, timeout: LIMITS.jobSeconds * 1000,
        persistent: true, snapshotExpiration: 7 * 86400000,
        tags: { eidos: "sentinel", evidence: "product-engineering", job: reservation.jobId },
        env: { PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8", OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1", HF_HUB_DISABLE_TELEMETRY: "1",
          ...(callbackOrigin ? { EIDOS_CALLBACK_URL: `${callbackOrigin}/api/lab/v1/worker/${id}`, EIDOS_CALLBACK_TOKEN: callbackToken } : {}) }, ...credentials() });
      allocationStage = "bootstrap";
      const session = allocated.currentSession();
      const fs = new FileSystem(session);
      const verified = await verifySandboxSource(session, revision);
      provider.sourceReceipt = verified;
      await fs.mkdir(provider.directory, { recursive: true });
      await session.writeFiles([{ path: `${provider.directory}/request.json`, content: payload }]);
      const command = await session.runCommand({ cmd: verified.interpreter,
        args: [`${verified.repositoryRoot}/services/sentinel-runner/sentinel_runner/guided/bootstrap.py`, "--repo", verified.repositoryRoot,
          "--request", `${provider.directory}/request.json`, "--out", provider.directory], cwd: verified.repositoryRoot,
        detached: true, timeoutMs: LIMITS.jobSeconds * 1000 - 15000 });
      provider.commandId = command.cmdId;
    }
    await gate.running(reservation.jobId);
    await store.updateJob(owner, id, "running", { provider, error: "" }, lease);
  } catch (error) {
    const failure = error as { response?: { status?: number }; status?: number; statusCode?: number; json?: { error?: { code?: string }; code?: string }; name?: string };
    const httpStatus = failure.response?.status ?? failure.statusCode ?? failure.status;
    const code = String(failure.json?.error?.code || failure.json?.code || "unknown").replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 80);
    provider.failureReceipt = { stage: allocationStage, httpStatus: httpStatus ?? null, code, errorType: failure.name || "Error" };
    // An explicit allocation rejection cannot have launched this worker. Do
    // not turn a quota/auth/validation rejection into an occupied mystery job.
    if (!local() && allocationStage === "create" && httpStatus && [400, 401, 402, 403, 404, 413, 422, 429].includes(httpStatus)) {
      await gate.release(reservation.jobId);
      await store.updateJob(owner, id, "failed", { provider, error: `Sandbox allocation was rejected (HTTP ${httpStatus}; ${code}). No worker started. Check provider access or budget, then retry from the saved source.` }, lease);
      return store.job(owner, id);
    }
    // Provider allocation may have succeeded despite a lost response. Keep the
    // admission occupied; polling reconciles the deterministic provider name.
    await store.updateJob(owner, id, "preparing", { provider, error: `Worker allocation or bootstrap was interrupted (${allocationStage}; HTTP ${httpStatus ?? "unknown"}; ${code}). Resume this job to reconcile its stable identity.` }, lease);
  }
  return store.job(owner, id);
}

async function stopWorker(job: Document) {
  if (!job.provider) return;
  if (job.provider.kind === "local") {
    // Worker reads its own cancellation marker. Never kill a persisted PID that
    // the operating system may have reassigned after a parent restart.
    await mkdir(job.provider.directory, { recursive: true });
    await writeFile(join(job.provider.directory, "cancel.requested"), "cancelled by owner");
  } else {
    try {
      const sandbox = await Sandbox.get({ name: job.provider.name, resume: false, ...credentials() });
      if (sandbox.status === "running" || sandbox.status === "pending") await sandbox.currentSession().stop();
    } catch (error) {
      const value = error as { response?: { status?: number }; status?: number; statusCode?: number };
      // Confirmed provider absence releases capacity; network/auth uncertainty does not.
      if ((value.response?.status ?? value.statusCode ?? value.status) !== 404) throw error;
    }
  }
}

async function publish(owner: string, job: Document, result: Document, store: GuidedStore) {
  const latest = await store.job(owner, job.id);
  if (terminal.has(latest.status)) return;
  if (job.request.operation === "parse") {
    const datasetId = job.request.datasetId;
    await store.put(owner, "dataset", datasetId, { ...result, id: datasetId, createdAt: new Date(Number(job.created)).toISOString(),
      confirmed: null, sourceJob: job.id, expiresAt: new Date(Number(job.expires)).toISOString() });
    result = { datasetId, receipt: result.receipt };
  } else if (job.request.operation === "confirm") {
    const dataset = await store.get(owner, "dataset", job.request.datasetId);
    await store.put(owner, "dataset", dataset.id, { ...dataset, confirmed: result });
    result = { datasetId: dataset.id, confirmed: result };
  } else if (job.request.operation === "analyze") {
    await store.put(owner, "run", job.id, { ...result, id: job.id, datasetId: job.request.datasetId,
      ...(job.request.options?.mode === "patterns" && job.request.dataset.kind === "table" ? { patterns: summarizePatterns(job.request.dataset, job.request.mapping) } : {}),
      inputSha256: job.request.dataset.sha256, createdAt: new Date(Number(job.created)).toISOString(), expiresAt: new Date(Number(job.expires)).toISOString() });
    result = { runId: job.id, receipt: result.receipt };
  } else if (job.request.operation === "telemetry") {
    const { commitTelemetry } = await import("./telemetry");
    await commitTelemetry(owner, job, result, store);
    result = { monitorId: job.request.monitorId, processedOffset: result.processedOffset, receipt: result.receipt };
  }
  await store.updateJob(owner, job.id, "completed", { result });
}

export async function pollJob(owner: string, id: string, store = guidedStore()) {
  let job = await store.job(owner, id);
  if (terminal.has(job.status)) {
    if (job.provider) {
      await stopWorker(job).then(() => admission(store).release(job.provider.admissionId)).catch(() => undefined);
    }
    return job;
  }
  if (job.status === "queued") return job;
  if (!job.provider) {
    if (Date.now() - Number(job.updated) > 360000) await store.updateJob(owner, id, "failed", { error: "Dispatch stopped before admission. Retry with a new intent; no worker was allocated." });
    return store.job(owner, id);
  }
  let sandbox: Sandbox | undefined;
  try {
    let read: (name: string) => Promise<string>;
    if (job.provider.kind === "local") read = name => readFile(join(job.provider.directory, name), "utf8");
    else {
      sandbox = await Sandbox.get({ name: job.provider.name, resume: false, ...credentials() });
      if (sandbox.status !== "running") sandbox = await Sandbox.get({ name: job.provider.name, ...credentials() });
      const fs = new FileSystem(sandbox.currentSession());
      read = name => fs.readFile(`${job.provider.directory}/${name}`, "utf8");
    }
    const status = JSON.parse(await read("status.json"));
    if (status.status === "completed") {
      const text = await read("result.json");
      if (hash(text) !== status.resultSha256) throw new LabError(502, "Worker result checksum failed; no success was recorded.");
      await publish(owner, job, JSON.parse(text), store);
      if (sandbox) await sandbox.currentSession().stop();
      await admission(store).release(job.provider.admissionId);
    } else if (status.status === "failed") {
      await store.updateJob(owner, id, "failed", { error: `${status.error}: ${status.detail}`.slice(0, 800) });
      if (sandbox) await sandbox.currentSession().stop();
      await admission(store).release(job.provider.admissionId);
    } else if (Date.now() - job.provider.allocatedAt > LIMITS.jobSeconds * 1000 + 30000) {
      await stopWorker(job); await admission(store).release(job.provider.admissionId);
      await store.updateJob(owner, id, "expired", { error: "The 15-minute worker limit was reached. Original data and prior checkpoints remain saved." });
    } else {
      await store.updateJob(owner, id, status.status === "evaluating" ? "evaluating" : "running");
    }
  } catch (error) {
    if (Date.now() - job.provider.allocatedAt > LIMITS.jobSeconds * 1000 + 360000) {
      // An uncertain provider must not be treated as spare capacity.
      await stopWorker(job);
      await admission(store).release(job.provider.admissionId);
      await store.updateJob(owner, id, "failed", { error: "Worker stopped without a readable result. Retry from the saved dataset; no success was inferred." });
    } else if (error instanceof LabError) throw error;
  }
  return store.job(owner, id);
}

export async function cancelJob(owner: string, id: string, store = guidedStore()) {
  const job = await store.job(owner, id);
  if (terminal.has(job.status)) return job;
  // First fence publication; the same identity can never transition to completed.
  await store.updateJob(owner, id, "cancelled", { error: "Cancelled by the workspace owner." });
  await stopWorker(job);
  if (job.provider) await admission(store).release(job.provider.admissionId);
  return store.job(owner, id);
}

export function publicJob(job: Document) {
  return { id: job.id, status: job.status, createdAt: Number(job.created), updatedAt: Number(job.updated),
    operation: job.request.operation, error: job.error || null, result: job.result,
    bounds: { maximumSeconds: LIMITS.jobSeconds, progressKind: "stage; not a completion estimate" } };
}
