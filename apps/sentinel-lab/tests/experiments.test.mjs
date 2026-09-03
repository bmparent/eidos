import assert from "node:assert/strict";
import test from "node:test";
import { cloneDefaultExperiment, preflightIssues, validateExperimentSpec } from "../lib/experiments/shared.js";
import { canonicalJson, sha256Canonical } from "../lib/experiments/lock.js";
import {
  LAUNCHER_CHECK_INTERVAL_MS,
  LAUNCHER_FAILURE_FILENAME,
  failedLauncherStatus,
  shouldInspectLauncher,
  statusIsOlderThan,
} from "../lib/experiments/sandbox-state.js";
import {
  DispatchStageError,
  dispatchDiagnostic,
  normalizeDispatchFailure,
  redactDispatchDiagnostic,
  withDispatchStage,
} from "../lib/experiments/dispatch-diagnostics.js";
import { SANDBOX_ROOT, sandboxRepositoryRoot } from "../lib/experiments/sandbox-paths.js";

test("default real-data spec produces a stable canonical lock", () => {
  const first = validateExperimentSpec(cloneDefaultExperiment());
  const second = validateExperimentSpec(JSON.parse(JSON.stringify(first)));
  assert.equal(canonicalJson(first), canonicalJson(second));
  assert.match(sha256Canonical(first), /^[a-f0-9]{64}$/);
  assert.equal(sha256Canonical(first), sha256Canonical(second));
  assert.equal(sha256Canonical(first), "5eba19350bb0cf1a8761f2bfba3ae730f0e073ec1436c360ba9904b6ef9b044a");
  assert.equal(first.dataset.version, 3);
  assert.equal(first.dataset.expectedSha256, "7db47b2bf97ad58c3556ee25e8e1eb1e697cd391670733833865d0e84d8ed82a");
});

test("sandbox git sources resolve beneath the provider working directory", () => {
  assert.equal(
    sandboxRepositoryRoot("https://github.com/bmparent/eidos.git"),
    `${SANDBOX_ROOT}/eidos`,
  );
  assert.equal(
    sandboxRepositoryRoot("git@github.com:bmparent/eidos.git"),
    `${SANDBOX_ROOT}/eidos`,
  );
  assert.equal(
    sandboxRepositoryRoot("https://github.com/bmparent/eidos.git?ref=main", "/custom/root/"),
    "/custom/root/eidos",
  );
  assert.throws(() => sandboxRepositoryRoot("https://github.com/bmparent/.git"), /EIDOS_SOURCE_REPOSITORY_INVALID/);
});

test("real-data spec requires version and rejects path traversal", () => {
  const missingVersion = cloneDefaultExperiment();
  delete missingVersion.dataset.version;
  assert.throws(() => validateExperimentSpec(missingVersion), /positive integer version/);

  const traversal = cloneDefaultExperiment();
  traversal.dataset.file = "../secret.csv";
  assert.throws(() => validateExperimentSpec(traversal), /without traversal/);
});

test("proof and label-isolation locks cannot be relaxed", () => {
  const value = cloneDefaultExperiment();
  value.protocol.heldoutPolicy = "include_in_engineering";
  assert.throws(() => validateExperimentSpec(value), /heldoutPolicy must remain locked/);

  const featureLeak = cloneDefaultExperiment();
  featureLeak.dataContract.featureColumns = ["Label"];
  assert.throws(() => validateExperimentSpec(featureLeak), /label column cannot be selected/);
});

test("preflight blocks dispatch without changing scientific readiness", () => {
  const spec = validateExperimentSpec(cloneDefaultExperiment());
  const issues = preflightIssues(spec, false);
  assert.ok(issues.some((issue) => issue.code === "RUNNER_NOT_CONFIGURED" && issue.severity === "blocker"));
  assert.ok(issues.some((issue) => issue.code === "AUTO_NUMERIC_FEATURES" && issue.severity === "warning"));
});

test("backend-specific blockers remain explicit and fail closed", () => {
  const spec = validateExperimentSpec(cloneDefaultExperiment());
  const issues = preflightIssues(spec, {
    backend: "sandbox",
    configured: false,
    blockers: [
      { code: "KAGGLE_CREDENTIAL_NOT_CONFIGURED", message: "Kaggle credential is absent." },
      { code: "SANDBOX_AUTH_NOT_AVAILABLE", message: "Sandbox authentication is absent." },
    ],
  }, true);
  assert.ok(issues.some((issue) => issue.code === "KAGGLE_CREDENTIAL_NOT_CONFIGURED" && issue.severity === "blocker"));
  assert.ok(issues.some((issue) => issue.code === "SANDBOX_AUTH_NOT_AVAILABLE" && issue.severity === "blocker"));
  assert.equal(issues.some((issue) => issue.code === "OPERATOR_AUTH_NOT_CONFIGURED"), false);
});

test("stale launcher receipts fail closed without advancing proof gates", () => {
  const now = Date.parse("2026-09-02T20:10:00.000Z");
  const queued = {
    status: "QUEUED",
    updatedAt: "2026-09-02T20:08:00.000Z",
    artifacts: ["runner.log"],
    gatesAdvanced: 0,
    proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT",
  };
  assert.equal(statusIsOlderThan(queued, 60_000, now), true);
  const failed = failedLauncherStatus(queued, "SANDBOX_LAUNCHER_EXITED", "launcher stopped", 127, now);
  assert.equal(failed.status, "FAILED");
  assert.equal(failed.launcherExitCode, 127);
  assert.deepEqual(failed.artifacts, ["runner.log", LAUNCHER_FAILURE_FILENAME]);
  assert.equal(failed.gatesAdvanced, 0);
  assert.equal(failed.proofVerdict, "BLOCKED_RESOURCE_BEFORE_HELDOUT");
});

test("launcher command inspection is rate limited", () => {
  const startedAt = Date.parse("2026-09-02T20:10:00.000Z");
  assert.equal(shouldInspectLauncher({ startedAt }, startedAt + LAUNCHER_CHECK_INTERVAL_MS - 1), false);
  assert.equal(shouldInspectLauncher({ startedAt }, startedAt + LAUNCHER_CHECK_INTERVAL_MS), true);
  assert.equal(
    shouldInspectLauncher(
      { startedAt, lastCheckedAt: startedAt + LAUNCHER_CHECK_INTERVAL_MS },
      startedAt + LAUNCHER_CHECK_INTERVAL_MS * 2 - 1,
    ),
    false,
  );
});

test("sandbox allocation errors return a stable diagnostic response instead of a generic 400", () => {
  const error = new DispatchStageError(
    "sandbox_allocation",
    new Error("Bad Request from provider: token=super-secret-value"),
  );
  const failure = normalizeDispatchFailure(error, { diagnosticId: "diag-123" });

  assert.equal(failure.error, "SANDBOX_ALLOCATION_FAILED");
  assert.equal(failure.status, 502);
  assert.equal(failure.stage, "sandbox_allocation");
  assert.equal(failure.jobId, undefined);
  assert.equal(failure.statusUrl, undefined);
  assert.equal(failure.diagnosticId, "diag-123");
  assert.equal(failure.retryable, true);
  assert.equal(failure.detail.includes("super-secret-value"), false);
});

test("dispatch diagnostics preserve useful provider context while redacting credentials", () => {
  const error = Object.assign(new Error("403 Unauthorized Bearer abc123 token=super-secret KGAT_exampletoken"), {
    code: "forbidden",
    statusCode: 403,
  });
  const wrapped = new DispatchStageError("sandbox_allocation", error);
  const failure = normalizeDispatchFailure(wrapped, { diagnosticId: "diag-auth" });
  const diagnostic = dispatchDiagnostic(wrapped);

  assert.equal(failure.error, "SANDBOX_AUTH_FAILED");
  assert.equal(failure.status, 503);
  assert.equal(failure.retryable, false);
  assert.equal(diagnostic.providerCode, "forbidden");
  assert.equal(diagnostic.providerStatusCode, 403);
  assert.equal(diagnostic.message.includes("abc123"), false);
  assert.equal(diagnostic.message.includes("super-secret"), false);
  assert.equal(diagnostic.message.includes("KGAT_exampletoken"), false);
  assert.match(diagnostic.message, /\[REDACTED/);
});

test("known capacity and configuration failures retain their public contracts", () => {
  const occupied = normalizeDispatchFailure(new Error("RUNNER_CAPACITY_OCCUPIED"), { diagnosticId: "diag-capacity" });
  assert.equal(occupied.error, "RUNNER_CAPACITY_OCCUPIED");
  assert.equal(occupied.status, 429);
  assert.equal(occupied.retryable, true);

  const missing = normalizeDispatchFailure(new Error("EIDOS_SOURCE_COMMIT_NOT_CONFIGURED"), { diagnosticId: "diag-config" });
  assert.equal(missing.error, "EIDOS_SOURCE_COMMIT_NOT_CONFIGURED");
  assert.equal(missing.status, 503);
  assert.equal(missing.retryable, false);
});

test("request parse failures and stage wrappers remain explicit", async () => {
  const invalid = normalizeDispatchFailure(new SyntaxError("Unexpected token"), {
    diagnosticId: "diag-json",
    stage: "request_parse",
  });
  assert.equal(invalid.error, "INVALID_EXPERIMENT_REQUEST");
  assert.equal(invalid.status, 400);
  assert.equal(invalid.retryable, false);

  let bootstrapError;
  await assert.rejects(
    withDispatchStage("sandbox_bootstrap", async () => {
      throw new Error("launcher unavailable");
    }, { jobId: "rd-5eba19350bb0-feedface" }),
    (error) => {
      bootstrapError = error;
      return error instanceof DispatchStageError && error.stage === "sandbox_bootstrap" && error.jobId === "rd-5eba19350bb0-feedface";
    },
  );
  const bootstrap = normalizeDispatchFailure(bootstrapError, { diagnosticId: "diag-bootstrap" });
  assert.equal(bootstrap.error, "SANDBOX_BOOTSTRAP_FAILED");
  assert.equal(bootstrap.jobId, "rd-5eba19350bb0-feedface");
  assert.equal(bootstrap.statusUrl, "/api/experiments/rd-5eba19350bb0-feedface");
  assert.equal(redactDispatchDiagnostic("password=hunter2\nnext"), "password=[REDACTED] next");
});
