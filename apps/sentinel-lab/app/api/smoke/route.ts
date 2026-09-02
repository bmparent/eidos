import { simulateSmoke, validateSmokeRequest } from "@/lib/sentinel/simulate";

export async function POST(request: Request) {
  try {
    const input = validateSmokeRequest(await request.json());
    const result = simulateSmoke(input);

    return Response.json(result, {
      headers: {
        "Cache-Control": "no-store",
        "X-Eidos-Evidence-Class": "engineering-smoke",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid smoke request";
    return Response.json({ error: message }, { status: 400 });
  }
}
