import { LOCK_SCHEMA, preflightIssues, validateExperimentSpec } from "@/lib/experiments/shared";
import { sha256Canonical } from "@/lib/experiments/lock";
import { getExecutionReadiness } from "@/lib/experiments/runner";
import { isOperatorAuthConfigured } from "@/lib/experiments/operator-auth";
import type { ExperimentSpec } from "@/lib/experiments/types";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const spec = validateExperimentSpec(await request.json()) as ExperimentSpec;
    const execution = getExecutionReadiness(spec);
    const issues = preflightIssues(spec, execution, isOperatorAuthConfigured());
    const lock = {
      schema: LOCK_SCHEMA,
      algorithm: "sha256",
      digest: sha256Canonical(spec),
      spec,
      issues,
      runnerConfigured: execution.configured,
      executionBackend: execution.backend,
      readyToDispatch: !issues.some((issue) => issue.severity === "blocker"),
    };
    return Response.json(lock, { headers: { "Cache-Control": "no-store", "X-Eidos-Evidence-Class": "real-data-engineering" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid real-data experiment request.";
    return Response.json({ error: message }, { status: 400, headers: { "Cache-Control": "no-store" } });
  }
}
