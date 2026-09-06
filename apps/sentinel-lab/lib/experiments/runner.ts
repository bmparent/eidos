import type { ExperimentSpec, ExperimentStatus, LockedExperiment, RunnerDispatch } from "@/lib/experiments/types";
import { admissionConfigured, validateRetryKey } from "./admission";

const JOB_ID = /^[a-z0-9][a-z0-9_-]{7,95}$/i;

type ExecutionBlocker = { code: string; message: string };

export type ExecutionReadiness = {
  backend: "sandbox" | "external" | null;
  configured: boolean;
  blockers: ExecutionBlocker[];
};

function externalConfiguration() {
  const url = process.env.EIDOS_RUNNER_URL?.trim();
  const token = process.env.EIDOS_RUNNER_TOKEN?.trim();
  if (!url || !token) return null;
  const parsed = new URL(url);
  const isLocal = parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1";
  if (parsed.protocol !== "https:" && !(process.env.NODE_ENV !== "production" && isLocal)) {
    throw new Error("EIDOS_RUNNER_URL must use HTTPS outside local development.");
  }
  return { url: parsed.toString().replace(/\/$/, ""), token };
}

function sourceCommitConfigured() {
  const revision = process.env.VERCEL_GIT_COMMIT_SHA?.trim() || process.env.EIDOS_SOURCE_COMMIT?.trim() || "";
  return /^[a-f0-9]{40}$/i.test(revision);
}

function sandboxCredentialsAvailable() {
  return Boolean(
    process.env.VERCEL === "1" ||
    process.env.VERCEL_OIDC_TOKEN?.trim() ||
    (process.env.VERCEL_TOKEN?.trim() && process.env.VERCEL_TEAM_ID?.trim() && process.env.VERCEL_PROJECT_ID?.trim()),
  );
}

function sandboxMaxRows() {
  const parsed = Number(process.env.EIDOS_SANDBOX_MAX_ROWS);
  return Number.isSafeInteger(parsed) && parsed >= 1_000 ? Math.min(parsed, 2_000_000) : 25_000;
}

export function getExecutionReadiness(spec?: ExperimentSpec): ExecutionReadiness {
  const requested = process.env.EIDOS_EXECUTION_BACKEND?.trim().toLowerCase();
  if (requested === "sandbox") {
    const blockers: ExecutionBlocker[] = [];
    if (!admissionConfigured()) blockers.push({ code: "ADMISSION_DATABASE_NOT_CONFIGURED", message: "Shared launch admission requires the existing EIDOS_DATABASE_URL and EIDOS_DATABASE_AUTH_TOKEN settings. No compute will start until durable capacity control is available." });
    if (spec?.engine.executionProfile === "full_capacity") blockers.push({ code: "FULL_CAPACITY_REQUIRES_DEDICATED_RUNNER", message: "Select the standard or mechanism-study profile here. The full-size reservoir requires a dedicated external runner with its larger resource budget enabled." });
    if (!process.env.KAGGLE_API_TOKEN?.trim()) blockers.push({ code: "KAGGLE_CREDENTIAL_NOT_CONFIGURED", message: "Add KAGGLE_API_TOKEN to the Vercel project before dispatching a real dataset job." });
    if (!sourceCommitConfigured()) blockers.push({ code: "SOURCE_COMMIT_NOT_PINNED", message: "The Sandbox must clone an exact Git commit. Deploy through Git integration or set EIDOS_SOURCE_COMMIT to a 40-character commit SHA." });
    if (!sandboxCredentialsAvailable()) blockers.push({ code: "SANDBOX_AUTH_NOT_AVAILABLE", message: "Vercel Sandbox authentication is unavailable. Deploy on Vercel or provide scoped VERCEL_TOKEN, VERCEL_TEAM_ID, and VERCEL_PROJECT_ID values." });
    if (spec && spec.dataContract.maxRows > sandboxMaxRows()) blockers.push({ code: "SANDBOX_ROW_BUDGET_EXCEEDED", message: `This Sandbox deployment caps engineering jobs at ${sandboxMaxRows().toLocaleString()} rows; lower MAX ROWS or explicitly raise EIDOS_SANDBOX_MAX_ROWS after a resource review.` });
    return { backend: "sandbox", configured: blockers.length === 0, blockers };
  }

  const hasExternalSignal = Boolean(process.env.EIDOS_RUNNER_URL?.trim() || process.env.EIDOS_RUNNER_TOKEN?.trim() || requested === "external");
  if (hasExternalSignal) {
    const blockers: ExecutionBlocker[] = [];
    if (spec?.engine.executionProfile === "full_capacity" && process.env.EIDOS_ENABLE_FULL_CAPACITY !== "1") blockers.push({ code: "FULL_CAPACITY_NOT_ENABLED", message: "The full-size profile is locked until EIDOS_ENABLE_FULL_CAPACITY=1 is configured on both the control plane and the dedicated runner." });
    try {
      if (!externalConfiguration()) blockers.push({ code: "EXTERNAL_RUNNER_NOT_CONFIGURED", message: "Set both EIDOS_RUNNER_URL and EIDOS_RUNNER_TOKEN for the external execution backend." });
    } catch (error) {
      blockers.push({ code: "EXTERNAL_RUNNER_URL_INVALID", message: error instanceof Error ? error.message : "The external runner URL is invalid." });
    }
    return { backend: "external", configured: blockers.length === 0, blockers };
  }

  return {
    backend: null,
    configured: false,
    blockers: [{ code: "RUNNER_NOT_CONFIGURED", message: "Choose EIDOS_EXECUTION_BACKEND=sandbox for Vercel-native execution, or configure an authenticated external runner." }],
  };
}

export function isRunnerConfigured(spec?: ExperimentSpec) {
  return getExecutionReadiness(spec).configured;
}

async function runnerFetch(path: string, init?: RequestInit) {
  const config = externalConfiguration();
  if (!config) throw new Error("RUNNER_NOT_CONFIGURED");
  const response = await fetch(`${config.url}${path}`, {
    ...init,
    cache: "no-store",
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${config.token}`,
      ...(init?.headers || {}),
    },
    signal: AbortSignal.timeout(20_000),
  });
  const body = await response.json().catch(() => ({ error: `Runner returned HTTP ${response.status}.` }));
  if (!response.ok) throw new Error(typeof body?.detail === "string" ? body.detail : typeof body?.error === "string" ? body.error : `Runner returned HTTP ${response.status}.`);
  return body;
}

export async function dispatchLockedExperiment(lock: LockedExperiment, retryKey: string): Promise<RunnerDispatch> {
  validateRetryKey(retryKey);
  const readiness = getExecutionReadiness(lock.spec);
  if (!readiness.configured) throw new Error(readiness.blockers[0]?.code || "RUNNER_NOT_CONFIGURED");
  if (readiness.backend === "sandbox") {
    const { dispatchSandboxExperiment } = await import("@/lib/experiments/sandbox");
    return dispatchSandboxExperiment(lock, retryKey);
  }
  return runnerFetch("/v1/experiments", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Idempotency-Key": retryKey },
    body: JSON.stringify({ schema: "eidos.sentinel-runner.request.v0.2", lockDigest: lock.digest, spec: lock.spec satisfies ExperimentSpec }),
  }) as Promise<RunnerDispatch>;
}

export async function fetchExperimentStatus(jobId: string) {
  if (!JOB_ID.test(jobId)) throw new Error("Invalid experiment job ID.");
  const readiness = getExecutionReadiness();
  if (readiness.backend === "sandbox") {
    const { fetchSandboxStatus } = await import("@/lib/experiments/sandbox");
    return fetchSandboxStatus(jobId);
  }
  return runnerFetch(`/v1/experiments/${jobId}`) as Promise<ExperimentStatus>;
}

export async function fetchExperimentArtifact(jobId: string, artifactName: string) {
  if (!JOB_ID.test(jobId)) throw new Error("Invalid experiment job ID.");
  const { SANDBOX_ARTIFACTS } = await import("@/lib/experiments/sandbox");
  if (!Object.hasOwn(SANDBOX_ARTIFACTS, artifactName)) throw new Error("Artifact not found.");
  const readiness = getExecutionReadiness();
  if (readiness.backend === "sandbox") {
    const { fetchSandboxArtifact } = await import("@/lib/experiments/sandbox");
    return fetchSandboxArtifact(jobId, artifactName);
  }
  const config = externalConfiguration();
  if (!config) throw new Error("RUNNER_NOT_CONFIGURED");
  const response = await fetch(`${config.url}/v1/experiments/${jobId}/artifacts/${encodeURIComponent(artifactName)}`, {
    cache: "no-store",
    headers: { Authorization: `Bearer ${config.token}`, Accept: "application/octet-stream" },
    signal: AbortSignal.timeout(20_000),
  });
  if (!response.ok) throw new Error(response.status === 404 ? "Artifact not found." : `Runner returned HTTP ${response.status}.`);
  return {
    body: Buffer.from(await response.arrayBuffer()),
    contentType: response.headers.get("content-type") || "application/octet-stream",
    filename: artifactName,
  };
}
