import { fetchExperimentStatus } from "@/lib/experiments/runner";
import { authorizeOperator } from "@/lib/experiments/operator-auth";
import { experimentReadError } from "@/lib/experiments/read-error";

export const runtime = "nodejs";
export const maxDuration = 120;

export async function GET(_request: Request, context: { params: Promise<{ jobId: string }> }) {
  try {
    authorizeOperator(_request);
    const { jobId } = await context.params;
    const status = await fetchExperimentStatus(jobId);
    return Response.json(status, { headers: { "Cache-Control": "no-store", "X-Eidos-Evidence-Class": "real-data-engineering" } });
  } catch (error) {
    return experimentReadError(error, "status");
  }
}
