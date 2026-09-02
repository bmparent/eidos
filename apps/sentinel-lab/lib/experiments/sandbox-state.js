export const ACTIVE_SANDBOX_STATUSES = new Set([
  "QUEUED",
  "BOOTSTRAPPING_RUNTIME",
  "PREPARING_DATASET",
  "RUNNING_FULL_ENGINE",
  "EVALUATING_FROZEN_PREDICTIONS",
]);

export const TERMINAL_SANDBOX_STATUSES = new Set(["COMPLETED_ENGINEERING", "FAILED", "EXPIRED"]);
export const LAUNCHER_RECEIPT_FILENAME = "launcher_command.json";
export const LAUNCHER_FAILURE_FILENAME = "launcher_failure.log";
export const LAUNCHER_RECEIPT_GRACE_MS = 60_000;
export const LAUNCHER_CHECK_INTERVAL_MS = 30_000;

export function statusIsOlderThan(status, maximumAgeMs, now = Date.now()) {
  const updatedAt = Date.parse(status?.updatedAt || "");
  return Number.isFinite(updatedAt) && now - updatedAt >= maximumAgeMs;
}

export function shouldInspectLauncher(receipt, now = Date.now()) {
  const previousCheck = Number(receipt?.lastCheckedAt || receipt?.startedAt);
  return !Number.isFinite(previousCheck) || now - previousCheck >= LAUNCHER_CHECK_INTERVAL_MS;
}

export function failedLauncherStatus(status, error, detail, exitCode, now = Date.now()) {
  const artifacts = [...new Set([...(Array.isArray(status.artifacts) ? status.artifacts : []), LAUNCHER_FAILURE_FILENAME])];
  return {
    ...status,
    status: "FAILED",
    updatedAt: new Date(now).toISOString(),
    error,
    detail,
    artifacts,
    ...(Number.isInteger(exitCode) ? { launcherExitCode: exitCode } : {}),
  };
}
