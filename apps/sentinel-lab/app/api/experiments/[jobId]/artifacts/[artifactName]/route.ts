import { authorizeOperator } from "@/lib/experiments/operator-auth";
import { fetchExperimentArtifact } from "@/lib/experiments/runner";
import { experimentReadError } from "@/lib/experiments/read-error";

export const runtime = "nodejs";
export const maxDuration = 120;

export async function GET(request: Request, context: { params: Promise<{ jobId: string; artifactName: string }> }) {
  try {
    authorizeOperator(request);
    const { jobId, artifactName } = await context.params;
    const artifact = await fetchExperimentArtifact(jobId, artifactName);
    return new Response(new Uint8Array(artifact.body), { headers: {
      "Content-Type": artifact.contentType,
      "Content-Disposition": `attachment; filename="${artifact.filename}"`,
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      "X-Eidos-Evidence-Class": "real-data-engineering",
    } });
  } catch (error) {
    return experimentReadError(error, "artifact");
  }
}
