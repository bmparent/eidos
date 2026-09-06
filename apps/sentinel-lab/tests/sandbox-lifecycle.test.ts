import assert from "node:assert/strict";
import test, { mock } from "node:test";
import { FileSystem, Sandbox } from "@vercel/sandbox";
import { dispatchSandboxExperiment, fetchSandboxArtifact, fetchSandboxStatus } from "../lib/experiments/sandbox";
import { cloneDefaultExperiment } from "../lib/experiments/shared.js";
import type { LockedExperiment } from "../lib/experiments/types";
import { GET as download } from "../app/api/experiments/[jobId]/artifacts/[artifactName]/route";
import { GET as getStatus } from "../app/api/experiments/[jobId]/route";
import { createClient } from "@libsql/client";
import { AdmissionStore } from "../lib/experiments/admission";

const jobId = "rd-5eba19350bb0-feedface";
const dir = `/vercel/sandbox/jobs/${jobId}`;
const revision = "a".repeat(40);
const root = "/home/vercel-sandbox/actual-checkout";
const initial = { jobId, status: "RUNNING_FULL_ENGINE", updatedAt: "2026-01-01T00:00:00Z", gatesAdvanced: 0 };

function provider(state = "running", receipt: Record<string, unknown> = initial) {
  const files = new Map<string, string>([
    [`${dir}/status.json`, JSON.stringify(receipt)],
    [`${dir}/launcher_command.json`, JSON.stringify({ schema: "eidos.sentinel-lab.launcher-command.v0.2", jobId, commandId: "cmd-1", startedAt: 1 })],
    [`${dir}/runner.log`, "fixture process output\n"],
  ]);
  const commands: Record<string, unknown>[] = [];
  let stops = 0, resumes = 0;
  const session = {
    status: state,
    async stop() { stops++; session.status = "stopped"; },
    async getCommand() { return { exitCode: null, output: async () => "" }; },
    async writeFiles(values: { path: string; content: string }[]) { for (const value of values) files.set(value.path, value.content); },
    async runCommand(options: Record<string, unknown>) {
      commands.push(options);
      return options.detached
        ? { cmdId: "cmd-1", startedAt: Date.now() }
        : { exitCode: 0, stdout: async () => JSON.stringify({ repositoryRoot: root, workingDirectory: "/home/vercel-sandbox", commit: revision, interpreter: "/usr/bin/python3", launcherPath: `${root}/services/sentinel-runner/sentinel_runner/sandbox_launcher.py`, files: {} }) };
    },
  };
  const sandbox = { get status() { return session.status; }, currentSession: () => session, async resume() { resumes++; session.status = "running"; } };
  mock.method(Sandbox, "get", async (options: { resume?: boolean }) => {
    if (options.resume) await sandbox.resume();
    return sandbox;
  });
  mock.method(FileSystem.prototype, "mkdir", async () => undefined);
  mock.method(FileSystem.prototype, "readFile", async (path: string, encoding?: string) => {
    if (!files.has(path)) throw Object.assign(new Error(`ENOENT: ${path}`), { code: "ENOENT" });
    return encoding ? files.get(path) : Buffer.from(files.get(path)!);
  });
  return { sandbox, session, files, commands, stops: () => stops, resumes: () => resumes };
}

test.afterEach(() => mock.restoreAll());

test("full immutable verification resumes only a completed snapshot and reports provider cleanup", async () => {
  const p = provider("stopped", { ...initial, status: "COMPLETED_ENGINEERING" });
  mock.method(p.session, "runCommand", async () => ({ exitCode: 0, stdout: async () => JSON.stringify({ allMatched: true, declaredCount: 25, matchedCount: 25 }) }));
  const receipt = JSON.parse((await fetchSandboxArtifact(jobId, "artifact_verification.json")).body.toString());
  assert.equal(receipt.allMatched, true);
  assert.equal(receipt.resumedForRetrieval, true);
  assert.equal(receipt.providerStatusAfterRetrieval, "stopped");
  assert.equal(p.stops(), 1);
  assert.equal(p.resumes(), 1);
});

test("immutable verification is unavailable for active jobs and does not stop them", async () => {
  const p = provider();
  await assert.rejects(fetchSandboxArtifact(jobId, "artifact_verification.json"), /Artifact not found/);
  assert.equal(p.stops(), 0);
  assert.equal(p.commands.length, 0);
});

test("dispatch uses the verified checkout and commits its source and command receipts", async () => {
  const p = provider();
  const old = { ...process.env };
  const client = createClient({ url: ":memory:" });
  try {
    process.env.KAGGLE_API_TOKEN = "test-only";
    process.env.EIDOS_SOURCE_COMMIT = revision;
    delete process.env.VERCEL_GIT_COMMIT_SHA;
    mock.method(Sandbox, "list", async () => ({ toArray: async () => [] }));
    mock.method(Sandbox, "create", async () => p.sandbox);
    const admission = new AdmissionStore(client, "test");
    const lock = { digest: "5eba19350bb0" + "0".repeat(52), spec: cloneDefaultExperiment() } as LockedExperiment;
    const result = await dispatchSandboxExperiment(lock, "stable-launch-request", admission);
    const retry = await dispatchSandboxExperiment(lock, "stable-launch-request", admission);
    assert.equal(retry.jobId, result.jobId);
    assert.equal(p.commands.length, 2, "retry cannot start another launcher");
    assert.equal(result.status, "BOOTSTRAPPING_RUNTIME");
    assert.equal(p.commands[0].cwd, undefined);
    assert.equal(p.commands[1].cwd, root);
    assert.equal((p.commands[1].env as Record<string, string>).EIDOS_ENGINE_PATH, `${root}/eidos/EIDOS_BRAIN_UNIFIED_v0_4.7.02.py`);
    assert.ok(p.files.has(`/vercel/sandbox/jobs/${result.jobId}/source_receipt.json`));
    assert.ok(p.files.has(`/vercel/sandbox/jobs/${result.jobId}/launcher_command.json`));
    assert.equal(p.stops(), 0);
  } finally { client.close(); for (const key of Object.keys(process.env)) if (!(key in old)) delete process.env[key]; Object.assign(process.env, old); }
});

test("reading or missing an artifact never stops a running experiment", async () => {
  const p = provider();
  assert.equal((await fetchSandboxArtifact(jobId, "runner.log")).body.toString(), "fixture process output\n");
  await assert.rejects(fetchSandboxArtifact(jobId, "metrics.json"), /Artifact not found/);
  assert.equal(p.stops(), 0);
  assert.equal(p.resumes(), 0);
});

for (const artifact of ["runner.log", "metrics.json"]) test(`stopped snapshot retrieval cleans up after ${artifact}`, async () => {
  const p = provider("stopped");
  if (artifact === "metrics.json") await assert.rejects(fetchSandboxArtifact(jobId, artifact), /Artifact not found/);
  else await fetchSandboxArtifact(jobId, artifact);
  assert.equal(p.resumes(), 1);
  assert.equal(p.stops(), 1);
  assert.equal(p.commands.length, 0, "retrieving a snapshot must not relaunch the engine");
});

test("expired active receipts are persisted once and survive later retrieval", async () => {
  const p = provider("stopped");
  const first = await fetchSandboxStatus(jobId);
  assert.equal(first.status, "EXPIRED");
  const committed = JSON.parse(p.files.get(`${dir}/status.json`)!);
  assert.equal(committed.status, "EXPIRED");
  assert.equal((await fetchSandboxStatus(jobId)).updatedAt, first.updatedAt);
  assert.equal(p.stops(), 2);
  assert.equal(p.commands.length, 0);
});

test("transient monitoring and snapshot read failures do not fabricate engine failure or expiry", async () => {
  const p = provider();
  mock.method(p.session, "getCommand", async () => { throw new Error("503 provider unavailable"); });
  await assert.rejects(fetchSandboxStatus(jobId), /503/);
  assert.equal(JSON.parse(p.files.get(`${dir}/status.json`)!).status, "RUNNING_FULL_ENGINE");
  assert.equal(p.stops(), 0);
  p.session.status = "stopped";
  mock.method(FileSystem.prototype, "readFile", async () => { throw new Error("503 provider unavailable"); });
  await assert.rejects(fetchSandboxStatus(jobId), /503/);
  assert.equal(p.stops(), 1, "temporary retrieval session must close on errors too");
});

test("confirmed launcher exit commits a failure and releases compute", async () => {
  const p = provider();
  mock.method(p.session, "getCommand", async () => ({ exitCode: 1, output: async () => "process crashed" }));
  const result = await fetchSandboxStatus(jobId);
  assert.equal(result.status, "FAILED");
  assert.equal(result.error, "SANDBOX_LAUNCHER_EXITED");
  assert.equal(result.gatesAdvanced, 0);
  assert.ok(p.files.get(`${dir}/launcher_failure.log`)?.includes("process crashed"));
  assert.equal(p.stops(), 1);
});

test("completed receipts retain results and stop compute", async () => {
  const p = provider("running", { ...initial, status: "COMPLETED_ENGINEERING", metrics: { evaluated_rows: 600 } });
  const result = await fetchSandboxStatus(jobId);
  assert.equal(result.status, "COMPLETED_ENGINEERING");
  assert.deepEqual(result.metrics, { evaluated_rows: 600 });
  assert.equal(p.stops(), 1);
});

test("artifact and status routes enforce authentication, byte delivery, allowlists and retryable errors", async () => {
  const oldToken = process.env.EIDOS_OPERATOR_TOKEN, oldBackend = process.env.EIDOS_EXECUTION_BACKEND;
  const p = provider();
  process.env.EIDOS_OPERATOR_TOKEN = "route-test-only";
  process.env.EIDOS_EXECUTION_BACKEND = "sandbox";
  const context = { params: Promise.resolve({ jobId, artifactName: "runner.log" }) };
  const authorized = new Request("http://test", { headers: { Authorization: "Bearer route-test-only" } });
  try {
    assert.equal((await download(new Request("http://test"), context)).status, 401);
    const response = await download(authorized, context);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("content-disposition"), 'attachment; filename="runner.log"');
    assert.equal(response.headers.get("cache-control"), "no-store");
    assert.equal(await response.text(), "fixture process output\n");
    assert.equal((await download(authorized, { params: Promise.resolve({ jobId, artifactName: "../../request.json" }) })).status, 404);
    mock.method(p.session, "getCommand", async () => { throw new Error("503 token=secret-value"); });
    const failed = await getStatus(authorized, context);
    assert.equal(failed.status, 502);
    const diagnostic = await failed.json();
    assert.equal(diagnostic.retryable, true);
    assert.match(diagnostic.diagnosticId, /^[a-f0-9-]{36}$/);
    assert.equal(JSON.stringify(diagnostic).includes("secret-value"), false);
    assert.equal(p.stops(), 0);
  } finally {
    if (oldToken === undefined) delete process.env.EIDOS_OPERATOR_TOKEN; else process.env.EIDOS_OPERATOR_TOKEN = oldToken;
    if (oldBackend === undefined) delete process.env.EIDOS_EXECUTION_BACKEND; else process.env.EIDOS_EXECUTION_BACKEND = oldBackend;
  }
});
