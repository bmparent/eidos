import { fetchExperimentStatus } from "@/lib/experiments/runner";
import { authorizeOperator } from "@/lib/experiments/operator-auth";

export const runtime = "nodejs";

export async function GET(_request: Request, context: { params: Promise<{ jobId: string }> }) {
  try {
    authorizeOperator(_request);
    const { jobId } = await context.params;
    const status = await fetchExperimentStatus(jobId);
    return Response.json(status, { headers: { "Cache-Control": "no-store", "X-Eidos-Evidence-Class": "real-data-engineering" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Experiment status lookup failed.";
    const status = message === "RUNNER_NOT_CONFIGURED" || message === "OPERATOR_AUTH_NOT_CONFIGURED" ? 503 : message === "OPERATOR_AUTH_REQUIRED" ? 401 : 400;
    return Response.json({ error: message }, { status, headers: { "Cache-Control": "no-store" } });
  }
}
