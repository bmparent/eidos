import type { ExperimentSpec, LockedExperiment, RunnerDispatch } from "@/lib/experiments/types";

const JOB_ID = /^[a-z0-9][a-z0-9_-]{7,95}$/i;

function configuration() {
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

export function isRunnerConfigured() {
  try {
    return configuration() !== null;
  } catch {
    return false;
  }
}

async function runnerFetch(path: string, init?: RequestInit) {
  const config = configuration();
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
  if (!response.ok) throw new Error(typeof body?.error === "string" ? body.error : `Runner returned HTTP ${response.status}.`);
  return body;
}

export async function dispatchLockedExperiment(lock: LockedExperiment): Promise<RunnerDispatch> {
  return runnerFetch("/v1/experiments", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Idempotency-Key": crypto.randomUUID() },
    body: JSON.stringify({ schema: "eidos.sentinel-runner.request.v0.2", lockDigest: lock.digest, spec: lock.spec satisfies ExperimentSpec }),
  }) as Promise<RunnerDispatch>;
}

export async function fetchExperimentStatus(jobId: string) {
  if (!JOB_ID.test(jobId)) throw new Error("Invalid experiment job ID.");
  return runnerFetch(`/v1/experiments/${jobId}`);
}
