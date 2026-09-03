import { Sandbox } from "@vercel/sandbox";
import type { ExperimentStatus, LockedExperiment, RunnerDispatch } from "@/lib/experiments/types";
import { withDispatchStage } from "@/lib/experiments/dispatch-diagnostics";
import { SANDBOX_ROOT, sandboxRepositoryRoot } from "@/lib/experiments/sandbox-paths";
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

export const SANDBOX_ARTIFACTS = {
  "run_manifest.json": "application/json; charset=utf-8",
  "dataset_receipt.json": "application/json; charset=utf-8",
  "metrics.json": "application/json; charset=utf-8",
  "evaluation_trace.jsonl": "application/x-ndjson; charset=utf-8",
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
  const message = error instanceof Error ? error.message : String(error);
  return /not.?found|enoent|404/i.test(message);
}

async function readStatus(sandbox: Sandbox, jobId: string) {
  return JSON.parse(await sandbox.fs.readFile(`${jobDirectory(jobId)}/status.json`, "utf8")) as ExperimentStatus;
}

async function readLauncherReceipt(sandbox: Sandbox, jobId: string): Promise<LauncherReceipt | null> {
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

async function writeJson(sandbox: Sandbox, path: string, value: unknown) {
  await sandbox.writeFiles([{ path, content: JSON.stringify(value, null, 2) + "\n" }]);
}

function diagnosticMessage(error: unknown) {
  const value = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  return value.replace(/[\r\n]+/g, " ").slice(0, 2_000);
}

async function persistLauncherFailure(
  sandbox: Sandbox,
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

async function verifySandboxLauncher(sandbox: Sandbox, launcherPath: string, repoRoot: string) {
  const probe = "from pathlib import Path; import sys; path=Path(sys.argv[1]); assert path.is_file(), f'launcher missing: {path}'; print(sys.executable)";
  const attempts: string[] = [];
  for (const interpreter of ["python", "python3"]) {
    try {
      const result = await sandbox.runCommand({
        cmd: interpreter,
        args: ["-c", probe, launcherPath],
        cwd: repoRoot,
        timeoutMs: 30_000,
      });
      if (result.exitCode === 0) return interpreter;
      const output = await result.output("both").catch(() => "");
      attempts.push(`${interpreter}: exit=${result.exitCode}; ${diagnosticMessage(output || "no output")}`);
    } catch (error) {
      attempts.push(`${interpreter}: ${diagnosticMessage(error)}`);
      // Try the other conventional interpreter name before failing closed.
    }
  }
  throw Object.assign(
    new Error(`Sandbox launcher preflight failed. ${attempts.join(" | ")}`),
    { code: "SANDBOX_LAUNCHER_PREFLIGHT_FAILED" },
  );
}

function newJobId(lockDigest: string) {
  const suffix = crypto.randomUUID().replaceAll("-", "").slice(0, 8);
  return `rd-${lockDigest.slice(0, 12)}-${suffix}`;
}

async function existingSandbox(jobId: string) {
  try {
    return await Sandbox.get({ name: sandboxName(jobId), resume: false, ...credentials() });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/not.?found|404/i.test(message)) throw new Error("Experiment job not found.");
    throw error;
  }
}

async function enforceCapacity() {
  const maximum = integerSetting("EIDOS_MAX_CONCURRENT_JOBS", 1, 1, 10);
  const listing = await Sandbox.list({ tags: { eidos: "sentinel" }, ...credentials() });
  const sandboxes = await listing.toArray();
  let active = 0;
  for (const candidate of sandboxes) {
    if (candidate.status !== "pending" && candidate.status !== "running") continue;
    const match = /^eidos-(rd-[a-f0-9]{12}-[a-f0-9]{8})$/.exec(candidate.name);
    if (match) {
      const reconciled = await existingSandbox(match[1])
        .then((sandbox) => reconcileSandboxStatus(sandbox, match[1]))
        .catch(() => null);
      if (reconciled && TERMINAL_SANDBOX_STATUSES.has(reconciled.status)) continue;
    }
    active += 1;
  }
  if (active >= maximum) throw new Error("RUNNER_CAPACITY_OCCUPIED");
}

export async function dispatchSandboxExperiment(lock: LockedExperiment): Promise<RunnerDispatch> {
  const kaggleToken = required("KAGGLE_API_TOKEN");
  await withDispatchStage("sandbox_capacity", () => enforceCapacity());
  const jobId = newJobId(lock.digest);
  const name = sandboxName(jobId);
  const timeout = integerSetting("EIDOS_SANDBOX_TIMEOUT_MS", 2_700_000, 300_000, 86_400_000);
  const vcpus = integerSetting("EIDOS_SANDBOX_VCPUS", 4, 1, 8);
  const repositorySource = source();
  const repoRoot = sandboxRepositoryRoot(repositorySource.url);
  const directory = jobDirectory(jobId);
  const requestPath = `${directory}/request.json`;
  const launcherPath = `${repoRoot}/services/sentinel-runner/sentinel_runner/sandbox_launcher.py`;
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

  const sandbox = await withDispatchStage("sandbox_allocation", () => Sandbox.create({
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
      PYTHONPATH: `${repoRoot}/services/sentinel-runner:${repoRoot}/eidos/repo/src`,
      EIDOS_ENGINE_PATH: `${repoRoot}/eidos/EIDOS_BRAIN_UNIFIED_v0_4.7.02.py`,
      EIDOS_JOB_ROOT: `${SANDBOX_ROOT}/jobs`,
      EIDOS_MAX_CONCURRENT_JOBS: "1",
    },
    ...credentials(),
  }));

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
      const interpreter = await verifySandboxLauncher(sandbox, launcherPath, repoRoot);
      await writeJson(sandbox, `${directory}/status.json`, {
        ...initialStatus,
        status: "BOOTSTRAPPING_RUNTIME",
        updatedAt: new Date().toISOString(),
        detail: "Sandbox Python and launcher source verified; starting the detached engine process.",
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
    await sandbox.stop().catch(() => undefined);
    throw error;
  }

  return {
    jobId,
    status: "QUEUED",
    statusUrl: `/api/experiments/${jobId}`,
    executionBackend: "sandbox",
    evidenceClass: "REAL_DATA_ENGINEERING",
    proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT",
  };
}

async function reconcileSandboxStatus(sandbox: Sandbox, jobId: string): Promise<ExperimentStatus> {
  const wasStopped = ["stopped", "failed", "aborted"].includes(sandbox.status);
  let status: ExperimentStatus;
  try {
    status = await readStatus(sandbox, jobId);
  } catch (error) {
    if (wasStopped) {
      return {
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
    }
    throw error;
  }

  if (wasStopped && ACTIVE_SANDBOX_STATUSES.has(status.status)) {
    status = {
      ...status,
      status: "EXPIRED",
      error: "SANDBOX_SESSION_EXPIRED",
      detail: "The compute session reached its resource deadline before the engine job completed.",
    };
  }

  if (!wasStopped && ACTIVE_SANDBOX_STATUSES.has(status.status)) {
    const receipt = await readLauncherReceipt(sandbox, jobId);
    if (!receipt && statusIsOlderThan(status, LAUNCHER_RECEIPT_GRACE_MS)) {
      status = await persistLauncherFailure(
        sandbox,
        jobId,
        status,
        "SANDBOX_LAUNCHER_RECEIPT_TIMEOUT",
        "The Sandbox was allocated, but no detached launcher receipt appeared within 60 seconds.",
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
  if (TERMINAL_SANDBOX_STATUSES.has(status.status) && sandbox.status === "running") await sandbox.stop().catch(() => undefined);
  return status;
}

export async function fetchSandboxStatus(jobId: string): Promise<ExperimentStatus> {
  return reconcileSandboxStatus(await existingSandbox(jobId), jobId);
}

export async function fetchSandboxArtifact(jobId: string, artifactName: string) {
  if (!(artifactName in SANDBOX_ARTIFACTS)) throw new Error("Artifact not found.");
  const sandbox = await existingSandbox(jobId);
  try {
    const body = await sandbox.fs.readFile(`${jobDirectory(jobId)}/${artifactName}`);
    return { body, contentType: SANDBOX_ARTIFACTS[artifactName as ArtifactName], filename: artifactName };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/not.?found|enoent|404/i.test(message)) throw new Error("Artifact not found.");
    throw error;
  } finally {
    if (sandbox.status === "running") await sandbox.stop().catch(() => undefined);
  }
}
