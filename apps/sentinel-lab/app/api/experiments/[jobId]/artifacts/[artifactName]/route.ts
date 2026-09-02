import { fetchExperimentArtifact } from "@/lib/experiments/runner";
import { authorizeOperator } from "@/lib/experiments/operator-auth";

export const runtime = "nodejs";
export const maxDuration = 120;

export async function GET(request: Request, context: { params: Promise<{ jobId: string; artifactName: string }> }) {
  try {
    authorizeOperator(request);
    const { jobId, artifactName } = await context.params;
    const artifact = await fetchExperimentArtifact(jobId, artifactName);
    return new Response(new Uint8Array(artifact.body), {
      headers: {
        "Cache-Control": "private, no-store",
        "Content-Type": artifact.contentType,
        "Content-Disposition": `attachment; filename="${artifact.filename}"`,
        "X-Eidos-Evidence-Class": "real-data-engineering",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Artifact retrieval failed.";
    const status = message === "Artifact not found." || message === "Experiment job not found." ? 404 : message === "OPERATOR_AUTH_REQUIRED" ? 401 : message === "RUNNER_NOT_CONFIGURED" || message === "OPERATOR_AUTH_NOT_CONFIGURED" ? 503 : 400;
    return Response.json({ error: message }, { status, headers: { "Cache-Control": "private, no-store" } });
  }
}
