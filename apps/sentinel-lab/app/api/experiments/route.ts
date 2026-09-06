import { LOCK_SCHEMA, preflightIssues, validateExperimentSpec } from "@/lib/experiments/shared";
import { sha256Canonical } from "@/lib/experiments/lock";
import { dispatchLockedExperiment, getExecutionReadiness } from "@/lib/experiments/runner";
import { authorizeOperator, isOperatorAuthConfigured } from "@/lib/experiments/operator-auth";
import { dispatchDiagnostic, normalizeDispatchFailure } from "@/lib/experiments/dispatch-diagnostics";
import type { ExperimentSpec, LockedExperiment, PreflightIssue } from "@/lib/experiments/types";
import { readExperimentJson, RequestBodyError } from "@/lib/experiments/request-body";

export const runtime = "nodejs";
export const maxDuration = 300;

export async function POST(request: Request) {
  const diagnosticId = crypto.randomUUID();
  const requestId = request.headers.get("x-vercel-id");
  const startedAt = Date.now();
  let stage = "operator_auth";
  let lockDigest: string | undefined;

  console.log(JSON.stringify({
    level: "info",
    event: "experiment_dispatch_started",
    route: "/api/experiments",
    diagnosticId,
    requestId,
  }));

  try {
    authorizeOperator(request);
    stage = "request_parse";
    const body = await readExperimentJson(request);
    stage = "request_validation";
    const spec = validateExperimentSpec(body?.spec) as ExperimentSpec;
    const digest = sha256Canonical(spec);
    lockDigest = digest;
    if (typeof body?.lockDigest !== "string" || body.lockDigest !== digest) {
      return Response.json({ error: "RUN_LOCK_MISMATCH", detail: "Prepare a new run lock after changing any experiment field." }, { status: 409 });
    }
    const execution = getExecutionReadiness(spec);
    const issues = preflightIssues(spec, execution, isOperatorAuthConfigured()) as PreflightIssue[];
    const lock: LockedExperiment = {
      schema: LOCK_SCHEMA,
      algorithm: "sha256",
      digest,
      spec,
      issues,
      runnerConfigured: execution.configured,
      executionBackend: execution.backend,
      readyToDispatch: execution.configured && !issues.some((issue) => issue.severity === "blocker"),
    };
    if (!lock.readyToDispatch) {
      return Response.json(
        { error: "RUNNER_NOT_CONFIGURED", detail: "The immutable experiment is prepared, but no resource-qualified engine runner is attached.", lock },
        { status: 503, headers: { "Cache-Control": "no-store" } },
      );
    }
    stage = "runner_dispatch";
    const dispatch = await dispatchLockedExperiment(lock);
    console.log(JSON.stringify({
      level: "info",
      event: "experiment_dispatch_accepted",
      route: "/api/experiments",
      diagnosticId,
      requestId,
      lockDigest,
      jobId: dispatch.jobId,
      executionBackend: dispatch.executionBackend,
      durationMs: Date.now() - startedAt,
    }));
    return Response.json(dispatch, { status: 202, headers: { "Cache-Control": "no-store", "X-Eidos-Evidence-Class": "real-data-engineering" } });
  } catch (error) {
    if (error instanceof RequestBodyError) return Response.json({ error: error.message }, { status: error.status, headers: { "Cache-Control": "no-store" } });
    const failure = normalizeDispatchFailure(error, { diagnosticId, stage });
    console.error(JSON.stringify({
      level: "error",
      event: "experiment_dispatch_failed",
      route: "/api/experiments",
      diagnosticId,
      requestId,
      lockDigest,
      stage: failure.stage,
      code: failure.error,
      status: failure.status,
      retryable: failure.retryable,
      jobId: failure.jobId,
      durationMs: Date.now() - startedAt,
      diagnostic: dispatchDiagnostic(error),
    }));
    return Response.json(
      {
        error: failure.error,
        detail: `${failure.detail} Diagnostic ID: ${diagnosticId}`,
        diagnosticId,
        stage: failure.stage,
        retryable: failure.retryable,
        ...(failure.jobId ? { jobId: failure.jobId, statusUrl: failure.statusUrl } : {}),
      },
      {
        status: failure.status,
        headers: {
          "Cache-Control": "no-store",
          "X-Eidos-Diagnostic-ID": diagnosticId,
          "X-Eidos-Evidence-Class": "real-data-engineering",
        },
      },
    );
  }
}
