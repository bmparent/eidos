import { Sandbox } from "@vercel/sandbox";
import type { ExperimentStatus, LockedExperiment, RunnerDispatch } from "@/lib/experiments/types";

const JOB_ID = /^rd-[a-f0-9]{12}-[a-f0-9]{8}$/;
const COMMIT_SHA = /^[a-f0-9]{40}$/i;
const ACTIVE = new Set(["QUEUED", "BOOTSTRAPPING_RUNTIME", "PREPARING_DATASET", "RUNNING_FULL_ENGINE", "EVALUATING_FROZEN_PREDICTIONS"]);
const TERMINAL = new Set(["COMPLETED_ENGINEERING", "FAILED", "EXPIRED"]);

export const SANDBOX_ARTIFACTS = {
  "run_manifest.json": "application/json; charset=utf-8",
  "dataset_receipt.json": "application/json; charset=utf-8",
  "metrics.json": "application/json; charset=utf-8",
  "evaluation_trace.jsonl": "application/x-ndjson; charset=utf-8",
} as const;

type ArtifactName = keyof typeof SANDBOX_ARTIFACTS;

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
  return `/vercel/sandbox/jobs/${jobId}`;
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
  const active = sandboxes.filter((candidate) => candidate.status === "pending" || candidate.status === "running").length;
  if (active >= maximum) throw new Error("RUNNER_CAPACITY_OCCUPIED");
}

export async function dispatchSandboxExperiment(lock: LockedExperiment): Promise<RunnerDispatch> {
  const kaggleToken = required("KAGGLE_API_TOKEN");
  await enforceCapacity();
  const jobId = newJobId(lock.digest);
  const name = sandboxName(jobId);
  const timeout = integerSetting("EIDOS_SANDBOX_TIMEOUT_MS", 2_700_000, 300_000, 86_400_000);
  const vcpus = integerSetting("EIDOS_SANDBOX_VCPUS", 4, 1, 8);
  const repoRoot = "/vercel/sandbox";
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

  const sandbox = await Sandbox.create({
    name,
    source: source(),
    runtime: "python3.13",
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
      EIDOS_JOB_ROOT: `${repoRoot}/jobs`,
      EIDOS_MAX_CONCURRENT_JOBS: "1",
    },
    ...credentials(),
  });

  try {
    await sandbox.fs.mkdir(directory, { recursive: true });
    await sandbox.writeFiles([
      {
        path: requestPath,
        content: JSON.stringify({ schema: "eidos.sentinel-runner.request.v0.2", lockDigest: lock.digest, spec: lock.spec }, null, 2) + "\n",
      },
      { path: `${directory}/status.json`, content: JSON.stringify(initialStatus, null, 2) + "\n" },
    ]);
    await sandbox.runCommand({
      cmd: "python3",
      args: [
        "-m",
        "sentinel_runner.sandbox_launcher",
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
  } catch (error) {
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

export async function fetchSandboxStatus(jobId: string): Promise<ExperimentStatus> {
  const sandbox = await existingSandbox(jobId);
  const wasStopped = ["stopped", "failed", "aborted"].includes(sandbox.status);
  let status: ExperimentStatus;
  try {
    status = JSON.parse(await sandbox.fs.readFile(`${jobDirectory(jobId)}/status.json`, "utf8")) as ExperimentStatus;
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

  if (wasStopped && ACTIVE.has(status.status)) {
    status = {
      ...status,
      status: "EXPIRED",
      error: "SANDBOX_SESSION_EXPIRED",
      detail: "The compute session reached its resource deadline before the engine job completed.",
    };
  }
  status.executionBackend = "sandbox";
  if (TERMINAL.has(status.status) && sandbox.status === "running") await sandbox.stop().catch(() => undefined);
  return status;
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
