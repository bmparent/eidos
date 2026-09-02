import { LOCK_SCHEMA, preflightIssues, validateExperimentSpec } from "@/lib/experiments/shared";
import { sha256Canonical } from "@/lib/experiments/lock";
import { isRunnerConfigured } from "@/lib/experiments/runner";
import { isOperatorAuthConfigured } from "@/lib/experiments/operator-auth";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const spec = validateExperimentSpec(await request.json());
    const runnerConfigured = isRunnerConfigured();
    const issues = preflightIssues(spec, runnerConfigured, isOperatorAuthConfigured());
    const lock = {
      schema: LOCK_SCHEMA,
      algorithm: "sha256",
      digest: sha256Canonical(spec),
      spec,
      issues,
      runnerConfigured,
      readyToDispatch: !issues.some((issue) => issue.severity === "blocker"),
    };
    return Response.json(lock, { headers: { "Cache-Control": "no-store", "X-Eidos-Evidence-Class": "real-data-engineering" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid real-data experiment request.";
    return Response.json({ error: message }, { status: 400, headers: { "Cache-Control": "no-store" } });
  }
}
