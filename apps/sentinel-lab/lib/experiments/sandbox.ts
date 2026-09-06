import { FileSystem, Sandbox, type Session } from "@vercel/sandbox";
import type { ExperimentStatus, LockedExperiment, RunnerDispatch } from "@/lib/experiments/types";
import { redactDispatchDiagnostic, withDispatchStage } from "@/lib/experiments/dispatch-diagnostics";
import { verifySandboxSource } from "./sandbox-source";
import { AdmissionStore, admissionConfigured, sharedAdmission } from "./admission";
import {
  ACTIVE_SANDBOX_STATUSES,
  LAUNCHER_FAILURE_FILENAME,
  LAUNCHER_RECEIPT_FILENAME,
  LAUNCHER_RECEIPT_GRACE_MS,
  TERMINAL_SANDBOX_STATUSES,
  failedLauncherStatus,
  shouldInspectLauncher,
  statusIsOlderThan,
} from "@/lib/experiments/sandbox-state";

const JOB_ID = /^rd-[a-f0-9]{12}-[a-f0-9]{8}$/;
const COMMIT_SHA = /^[a-f0-9]{40}$/i;
const LAUNCHER_RECEIPT_SCHEMA = "eidos.sentinel-lab.launcher-command.v0.2";
const SANDBOX_ROOT = "/vercel/sandbox";

type JobSession = Session & { fs: FileSystem };

function jobSession(sandbox: Sandbox): JobSession {
  const session = sandbox.currentSession();
  // Pin all operations to this VM. Sandbox-level reads auto-resume stopped
  // sessions, which could otherwise revive an expired job during monitoring.
  return Object.assign(session, { fs: new FileSystem(session) });
}

export const SANDBOX_ARTIFACTS = {
  "source_receipt.json": "application/json; charset=utf-8",
  "run_manifest.json": "application/json; charset=utf-8",
  "dataset_receipt.json": "application/json; charset=utf-8",
  "metrics.json": "application/json; charset=utf-8",
  "evaluation_trace.jsonl": "application/x-ndjson; charset=utf-8",
  "engine_diagnostics.json": "application/json; charset=utf-8",
  "engine_trace.jsonl": "application/x-ndjson; charset=utf-8",
  "runner.log": "text/plain; charset=utf-8",
  "launcher_failure.log": "text/plain; charset=utf-8",
  "bootstrap_failure_traceback.log": "text/plain; charset=utf-8",
  "failure_traceback.log": "text/plain; charset=utf-8",
} as const;

type ArtifactName = keyof typeof SANDBOX_ARTIFACTS;

type LauncherReceipt = {
  schema: typeof LAUNCHER_RECEIPT_SCHEMA;
  jobId: string;
  commandId: string;
  startedAt: number;
  lastCheckedAt?: number;
};

function required(name: string) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`${name}_NOT_CONFIGURED`);
  return value;
}

function integerSetting(name: string, fallback: number, minimum: number, maximum: number) {
  const parsed = Number(process.env[name]);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(maximum, Math.max(minimum, Math.trunc(parsed)));
}

function credentials() {
  const token = process.env.VERCEL_TOKEN?.trim();
  const teamId = process.env.VERCEL_TEAM_ID?.trim();
  const projectId = process.env.VERCEL_PROJECT_ID?.trim();
  return token && teamId && projectId ? { token, teamId, projectId } : {};
}

function source() {
  const owner = process.env.VERCEL_GIT_REPO_OWNER?.trim();
  const slug = process.env.VERCEL_GIT_REPO_SLUG?.trim();
  const url = process.env.EIDOS_SOURCE_REPOSITORY?.trim() || (owner && slug ? `https://github.com/${owner}/${slug}.git` : "https://github.com/bmparent/eidos.git");
  const revision = process.env.VERCEL_GIT_COMMIT_SHA?.trim() || process.env.EIDOS_SOURCE_COMMIT?.trim() || "";
  if (!COMMIT_SHA.test(revision)) throw new Error("EIDOS_SOURCE_COMMIT_NOT_CONFIGURED");
  return { type: "git" as const, url, revision, depth: 1 };
}

function sandboxName(jobId: string) {
  if (!JOB_ID.test(jobId)) throw new Error("Invalid experiment job ID.");
  return `eidos-${jobId}`;
}

function jobDirectory(jobId: string) {
  return `${SANDBOX_ROOT}/jobs/${jobId}`;
}

function isNotFound(error: unknown) {
  if (error && typeof error === "object") {
    const value = error as { response?: { status?: number }; status?: number; statusCode?: number; code?: string };
    const status = value.response?.status ?? value.statusCode ?? value.status;
    if (status !== undefined) return status === 404;
    if (value.code === "ENOENT") return true;
  }
  const message = error instanceof Error ? error.message : String(error);
  return /\bnot[ _-]?found\b|\benoent\b|\b404\b/i.test(message);
}

async function readStatus(sandbox: JobSession, jobId: string) {
  return JSON.parse(await sandbox.fs.readFile(`${jobDirectory(jobId)}/status.json`, "utf8")) as ExperimentStatus;
}

async function readLauncherReceipt(sandbox: JobSession, jobId: string): Promise<LauncherReceipt | null> {
  try {
    const receipt = JSON.parse(
      await sandbox.fs.readFile(`${jobDirectory(jobId)}/${LAUNCHER_RECEIPT_FILENAME}`, "utf8"),
    ) as LauncherReceipt;
    if (receipt.schema !== LAUNCHER_RECEIPT_SCHEMA || receipt.jobId !== jobId || !receipt.commandId) return null;
    return receipt;
  } catch (error) {
    if (isNotFound(error)) return null;
    throw error;
  }
}

async function writeJson(sandbox: JobSession, path: string, value: unknown) {
  await sandbox.writeFiles([{ path, content: JSON.stringify(value, null, 2) + "\n" }]);
}

function diagnosticMessage(error: unknown) {
  const value = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  return redactDispatchDiagnostic(value);
}

async function persistLauncherFailure(
  sandbox: JobSession,
  jobId: string,
  status: ExperimentStatus,
  error: string,
  detail: string,
  diagnostic: string,
  exitCode?: number,
) {
  const failed = failedLauncherStatus(status, error, detail, exitCode) as ExperimentStatus;
  await sandbox.writeFiles([
    {
      path: `${jobDirectory(jobId)}/${LAUNCHER_FAILURE_FILENAME}`,
      content: `${diagnostic.trim()}\n`,
    },
    {
      path: `${jobDirectory(jobId)}/status.json`,
      content: JSON.stringify(failed, null, 2) + "\n",
    },
  ]);
  return failed;
}

async function existingSandbox(jobId: string) {
  try {
    return await Sandbox.get({ name: sandboxName(jobId), resume: false, ...credentials() });
  } catch (error) {
    if (isNotFound(error)) throw new Error("Experiment job not found.");
    throw error;
  }
}

async function enforceCapacity(admission: AdmissionStore) {
  const maximum = integerSetting("EIDOS_MAX_CONCURRENT_JOBS", 1, 1, 10);
  const listing = await Sandbox.list({ tags: { eidos: "sentinel" }, ...credentials() });
  const sandboxes = await listing.toArray();
  // Recover allocations abandoned by a terminated request. Never free an
  // uncertain allocation merely because a lease expired: inspect the provider
  // after the full dispatch + compute deadline. Outages fail closed.
  for (const reservation of await admission.active()) {
    if (!await admission.expired(reservation.jobId)) continue;
    try {
      const existing = await existingSandbox(reservation.jobId);
      if (["stopped", "failed", "aborted"].includes(existing.status)) await admission.release(reservation.jobId);
    } catch (error) {
      if (error instanceof Error && error.message === "Experiment job not found.") await admission.release(reservation.jobId);
      else throw error;
    }
  }
  let active = 0;
  for (const candidate of sandboxes) {
    if (candidate.status !== "pending" && candidate.status !== "running") continue;
    const match = /^eidos-(rd-[a-f0-9]{12}-[a-f0-9]{8})$/.exec(candidate.name);
    if (match) {
      const reconciled = await fetchSandboxStatus(match[1]).catch(() => null);
      if (reconciled && TERMINAL_SANDBOX_STATUSES.has(reconciled.status)) continue;
    }
    active += 1;
  }
  if (active >= maximum) throw new Error("RUNNER_CAPACITY_OCCUPIED");
}

export async function dispatchSandboxExperiment(lock: LockedExperiment, retryKey: string, admission = sharedAdmission()): Promise<RunnerDispatch> {
  const kaggleToken = required("KAGGLE_API_TOKEN");
  const timeout = integerSetting("EIDOS_SANDBOX_TIMEOUT_MS", 2_700_000, 300_000, 86_400_000);
  const vcpus = integerSetting("EIDOS_SANDBOX_VCPUS", 4, 1, 8);
  const repositorySource = source();
  const maximum = integerSetting("EIDOS_MAX_CONCURRENT_JOBS", 1, 1, 10);
  // Return a prior receipt before provider capacity checks, even while its
  // allocation is in flight or the provider is temporarily unavailable.
  const prior = await admission.reserve(retryKey, lock.digest, repositorySource.revision, maximum)
    .catch(async (error) => {
      if (!(error instanceof Error) || error.message !== "RUNNER_CAPACITY_OCCUPIED") throw error;
      await withDispatchStage("sandbox_capacity", () => enforceCapacity(admission));
      return admission.reserve(retryKey, lock.digest, repositorySource.revision, maximum);
    });
  const jobId = prior.admission.jobId;
  const receipt: RunnerDispatch = {
    jobId, status: prior.acquired ? "BOOTSTRAPPING_RUNTIME" : "QUEUED",
    statusUrl: `/api/experiments/${jobId}`, executionBackend: "sandbox",
    evidenceClass: "REAL_DATA_ENGINEERING", proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT",
  };
  if (!prior.acquired) return receipt;
  const name = sandboxName(jobId);
  try {
    await withDispatchStage("sandbox_capacity", () => enforceCapacity(admission));
  } catch (error) {
    await admission.release(jobId);
    throw error;
  }
  await admission.beginAllocation(prior.admission, timeout);
  const directory = jobDirectory(jobId);
  const requestPath = `${directory}/request.json`;
  const initialStatus: ExperimentStatus = {
    schema: "eidos.sentinel-runner.status.v0.2",
    jobId,
    status: "QUEUED",
    updatedAt: new Date().toISOString(),
    evidenceClass: "REAL_DATA_ENGINEERING",
    proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT",
    gatesAdvanced: 0,
    lockDigest: lock.digest,
    executionBackend: "sandbox",
  };

  const allocated = await withDispatchStage("sandbox_allocation", () => Sandbox.create({
    name,
    source: repositorySource,
    image: "vercel/sandbox/python:3.14",
    resources: { vcpus },
    timeout,
    persistent: true,
    snapshotExpiration: 7 * 24 * 60 * 60 * 1000,
    keepLastSnapshots: { count: 1, expiration: 7 * 24 * 60 * 60 * 1000, deleteEvicted: true },
    tags: { eidos: "sentinel", evidence: "engineering", job: jobId },
    env: {
      KAGGLE_API_TOKEN: kaggleToken,
      PYTHONUNBUFFERED: "1",
      EIDOS_JOB_ROOT: `${SANDBOX_ROOT}/jobs`,
      EIDOS_MAX_CONCURRENT_JOBS: "1",
    },
    ...credentials(),
  }), { jobId });
  const sandbox = jobSession(allocated);

  try {
    await withDispatchStage("sandbox_bootstrap", async () => {
      await sandbox.fs.mkdir(directory, { recursive: true });
      await sandbox.writeFiles([
        {
          path: requestPath,
          content: JSON.stringify({ schema: "eidos.sentinel-runner.request.v0.2", lockDigest: lock.digest, spec: lock.spec }, null, 2) + "\n",
        },
        { path: `${directory}/status.json`, content: JSON.stringify(initialStatus, null, 2) + "\n" },
      ]);
      const verified = await verifySandboxSource(sandbox, repositorySource.revision);
      const { repositoryRoot: repoRoot, launcherPath, interpreter } = verified;
      await writeJson(sandbox, `${directory}/source_receipt.json`, {
        schema: "eidos.sentinel-lab.source.v0.1", jobId, verifiedAt: new Date().toISOString(), ...verified,
      });
      initialStatus.artifacts = ["source_receipt.json"];
      await writeJson(sandbox, `${directory}/status.json`, {
        ...initialStatus,
        status: "BOOTSTRAPPING_RUNTIME",
        updatedAt: new Date().toISOString(),
        detail: "Checkout, commit, and launcher verified; installing the CPU runtime.",
      });
      const command = await sandbox.runCommand({
        cmd: interpreter,
        args: [
          launcherPath,
          "--request",
          requestPath,
          "--job-dir",
          directory,
          "--repo-root",
          repoRoot,
        ],
        cwd: repoRoot,
        env: {
          PYTHONPATH: `${repoRoot}/services/sentinel-runner:${repoRoot}/eidos/repo/src`,
          EIDOS_ENGINE_PATH: `${repoRoot}/eidos/EIDOS_BRAIN_UNIFIED_v0_4.7.02.py`,
        },
        detached: true,
        timeoutMs: Math.max(240_000, timeout - 30_000),
      });
      await writeJson(sandbox, `${directory}/${LAUNCHER_RECEIPT_FILENAME}`, {
        schema: LAUNCHER_RECEIPT_SCHEMA,
        jobId,
        commandId: command.cmdId,
        startedAt: command.startedAt,
      } satisfies LauncherReceipt);
    }, { jobId });
  } catch (error) {
    await persistLauncherFailure(
      sandbox,
      jobId,
      initialStatus,
      "SANDBOX_BOOTSTRAP_FAILED",
      `The Sandbox was allocated, but launcher bootstrap failed. Inspect ${LAUNCHER_FAILURE_FILENAME}.`,
      diagnosticMessage(error),
    ).catch(() => undefined);
    await sandbox.stop().then(() => admission.release(jobId)).catch(() => undefined);
    throw error;
  }
  await admission.running(jobId);
  return receipt;
}

async function reconcileSandboxStatus(sandbox: JobSession, jobId: string, wasStopped: boolean): Promise<ExperimentStatus> {
  let status: ExperimentStatus;
  try {
    status = await readStatus(sandbox, jobId);
  } catch (error) {
    if (wasStopped && isNotFound(error)) {
      status = {
        schema: "eidos.sentinel-runner.status.v0.2",
        jobId,
        status: "EXPIRED",
        updatedAt: new Date().toISOString(),
        evidenceClass: "REAL_DATA_ENGINEERING",
        proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT",
        gatesAdvanced: 0,
        error: "SANDBOX_SESSION_EXPIRED",
        detail: "The compute session ended before a readable status artifact was committed.",
        executionBackend: "sandbox",
      };
      await writeJson(sandbox, `${jobDirectory(jobId)}/status.json`, status);
      return status;
    }
    throw error;
  }

  if (wasStopped && ACTIVE_SANDBOX_STATUSES.has(status.status)) {
    status = {
      ...status,
      status: "EXPIRED",
      updatedAt: new Date().toISOString(),
      error: "SANDBOX_SESSION_EXPIRED",
      detail: "The compute session reached its resource deadline before the engine job completed.",
    };
    await writeJson(sandbox, `${jobDirectory(jobId)}/status.json`, status);
  }

  if (!wasStopped && ACTIVE_SANDBOX_STATUSES.has(status.status)) {
    const receipt = await readLauncherReceipt(sandbox, jobId);
    if (!receipt && statusIsOlderThan(status, LAUNCHER_RECEIPT_GRACE_MS)) {
      status = await persistLauncherFailure(
        sandbox,
        jobId,
        status,
        "SANDBOX_LAUNCHER_RECEIPT_TIMEOUT",
        `The Sandbox was allocated, but no detached launcher receipt appeared within ${LAUNCHER_RECEIPT_GRACE_MS / 1000} seconds.`,
        `job=${jobId}\nNo ${LAUNCHER_RECEIPT_FILENAME} was committed before the startup deadline.`,
      );
    } else if (receipt) {
      status = { ...status, launcherCommandId: receipt.commandId, launcherStartedAt: receipt.startedAt };
      if (shouldInspectLauncher(receipt)) {
        const inspection = await sandbox.getCommand(receipt.commandId)
          .then((command) => ({ command }))
          .catch((error: unknown) => ({ error }));
        await writeJson(sandbox, `${jobDirectory(jobId)}/${LAUNCHER_RECEIPT_FILENAME}`, {
          ...receipt,
          lastCheckedAt: Date.now(),
        } satisfies LauncherReceipt);

        if ("error" in inspection) {
          // A provider outage is not evidence that the engine failed. Preserve
          // the running job and let the monitor retry the lookup.
          if (!isNotFound(inspection.error)) throw inspection.error;
          if (statusIsOlderThan(status, LAUNCHER_RECEIPT_GRACE_MS)) {
            status = await persistLauncherFailure(
              sandbox,
              jobId,
              status,
              "SANDBOX_LAUNCHER_COMMAND_UNREADABLE",
              `The detached launcher command could not be inspected after the startup deadline. Download ${LAUNCHER_FAILURE_FILENAME} for the API diagnostic.`,
              `job=${jobId}\ncommand=${receipt.commandId}\n${diagnosticMessage(inspection.error)}`,
            );
          }
        } else if (inspection.command.exitCode !== null) {
          const refreshed = await readStatus(sandbox, jobId);
          if (TERMINAL_SANDBOX_STATUSES.has(refreshed.status)) {
            status = {
              ...refreshed,
              launcherCommandId: receipt.commandId,
              launcherStartedAt: receipt.startedAt,
              launcherExitCode: inspection.command.exitCode,
            };
          } else {
            const output = await inspection.command.output("both")
              .catch((error) => `Unable to read command output: ${diagnosticMessage(error)}`);
            status = await persistLauncherFailure(
              sandbox,
              jobId,
              status,
              "SANDBOX_LAUNCHER_EXITED",
              `The detached launcher exited with code ${inspection.command.exitCode} before committing a terminal status. Download ${LAUNCHER_FAILURE_FILENAME} for its process output.`,
              `job=${jobId}\ncommand=${receipt.commandId}\nexitCode=${inspection.command.exitCode}\n\n${output.slice(-64_000)}`,
              inspection.command.exitCode,
            );
          }
        }
      }
    }
  }
  status.executionBackend = "sandbox";
  if (TERMINAL_SANDBOX_STATUSES.has(status.status) && sandbox.status === "running") {
    await sandbox.stop();
    if (admissionConfigured()) await sharedAdmission().release(jobId);
  }
  return status;
}

export async function fetchSandboxStatus(jobId: string): Promise<ExperimentStatus> {
  try {
    return await withReadableSession(jobId, (session, wasStopped) => reconcileSandboxStatus(session, jobId, wasStopped));
  } catch (error) {
    if (!(error instanceof Error) || error.message !== "Experiment job not found." || !admissionConfigured()) throw error;
    const store = sharedAdmission();
    const receipt = await store.lookup(jobId);
    if (!receipt) throw error;
    const expired = ["ABANDONED", "TERMINAL"].includes(receipt.phase) || await store.expired(jobId);
    if (expired) await store.release(jobId);
    return {
      schema: "eidos.sentinel-runner.status.v0.2", jobId,
      status: expired ? "EXPIRED" : "QUEUED", updatedAt: new Date(expired ? Math.min(Date.now(), receipt.expiresAt) : receipt.createdAt).toISOString(),
      lockDigest: receipt.lockDigest, executionBackend: "sandbox", evidenceClass: "REAL_DATA_ENGINEERING",
      proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT", gatesAdvanced: 0,
      ...(expired ? { error: "ALLOCATION_RESERVATION_EXPIRED", detail: "The launch reservation ended without a retrievable Sandbox. Save this receipt, then prepare a new experiment." }
        : { detail: "A shared launch reservation exists. Allocation is pending or its response was interrupted; keep checking this job rather than starting another." }),
    };
  }
}

async function withReadableSession<T>(jobId: string, read: (session: JobSession, wasStopped: boolean) => Promise<T>): Promise<T> {
  let sandbox = await existingSandbox(jobId);
  if (["pending", "stopping", "snapshotting"].includes(sandbox.status)) throw new Error("SANDBOX_SESSION_TRANSITIONING");
  const wasStopped = ["stopped", "failed", "aborted"].includes(sandbox.status);
  if (wasStopped) sandbox = await Sandbox.get({ name: sandboxName(jobId), resume: true, ...credentials() });
  const session = jobSession(sandbox);
  try {
    return await read(session, wasStopped);
  } finally {
    // A resumed snapshot is for retrieval only; detached processes do not
    // survive a stop. Never stop an active experiment merely to read a file.
    if (wasStopped && session.status === "running") await session.stop();
  }
}

export async function fetchSandboxArtifact(jobId: string, artifactName: string) {
  if (!Object.hasOwn(SANDBOX_ARTIFACTS, artifactName)) throw new Error("Artifact not found.");
  return withReadableSession(jobId, async (sandbox) => {
    try {
      const body = await sandbox.fs.readFile(`${jobDirectory(jobId)}/${artifactName}`);
      return { body, contentType: SANDBOX_ARTIFACTS[artifactName as ArtifactName], filename: artifactName };
    } catch (error) {
      if (isNotFound(error)) throw new Error("Artifact not found.");
      throw error;
    }
  });
}
