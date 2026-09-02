import { LOCK_SCHEMA, preflightIssues, validateExperimentSpec } from "@/lib/experiments/shared";
import { sha256Canonical } from "@/lib/experiments/lock";
import { dispatchLockedExperiment, isRunnerConfigured } from "@/lib/experiments/runner";
import { authorizeOperator, isOperatorAuthConfigured } from "@/lib/experiments/operator-auth";
import type { ExperimentSpec, LockedExperiment, PreflightIssue } from "@/lib/experiments/types";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    authorizeOperator(request);
    const body = await request.json();
    const spec = validateExperimentSpec(body?.spec) as ExperimentSpec;
    const digest = sha256Canonical(spec);
    if (typeof body?.lockDigest !== "string" || body.lockDigest !== digest) {
      return Response.json({ error: "RUN_LOCK_MISMATCH", detail: "Prepare a new run lock after changing any experiment field." }, { status: 409 });
    }
    const runnerConfigured = isRunnerConfigured();
    const issues = preflightIssues(spec, runnerConfigured, isOperatorAuthConfigured()) as PreflightIssue[];
    const lock: LockedExperiment = {
      schema: LOCK_SCHEMA,
      algorithm: "sha256",
      digest,
      spec,
      issues,
      runnerConfigured,
      readyToDispatch: runnerConfigured && !issues.some((issue) => issue.severity === "blocker"),
    };
    if (!lock.readyToDispatch) {
      return Response.json(
        { error: "RUNNER_NOT_CONFIGURED", detail: "The immutable experiment is prepared, but no resource-qualified engine runner is attached.", lock },
        { status: 503, headers: { "Cache-Control": "no-store" } },
      );
    }
    const dispatch = await dispatchLockedExperiment(lock);
    return Response.json(dispatch, { status: 202, headers: { "Cache-Control": "no-store", "X-Eidos-Evidence-Class": "real-data-engineering" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Experiment dispatch failed.";
    const status = message === "RUNNER_NOT_CONFIGURED" || message === "OPERATOR_AUTH_NOT_CONFIGURED" ? 503 : message === "OPERATOR_AUTH_REQUIRED" ? 401 : 400;
    return Response.json({ error: message }, { status, headers: { "Cache-Control": "no-store" } });
  }
}
