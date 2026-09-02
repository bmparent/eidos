import { CURATED_DATASETS, mapKaggleResult } from "@/lib/experiments/shared";

export const runtime = "nodejs";

function curated(query: string, warning?: string) {
  const normalized = query.toLowerCase();
  const results = CURATED_DATASETS.filter((item) => `${item.ref} ${item.title} ${item.subtitle}`.toLowerCase().includes(normalized));
  return Response.json({ mode: "curated", results: results.length ? results : CURATED_DATASETS, warning }, { headers: { "Cache-Control": "no-store" } });
}

export async function GET(request: Request) {
  const query = new URL(request.url).searchParams.get("q")?.trim() || "";
  if (query.length < 2) return Response.json({ error: "Enter at least two search characters." }, { status: 400 });
  if (query.length > 80) return Response.json({ error: "Dataset search is limited to 80 characters." }, { status: 400 });

  const headers: Record<string, string> = { Accept: "application/json" };
  if (process.env.KAGGLE_API_TOKEN?.trim()) headers.Authorization = `Bearer ${process.env.KAGGLE_API_TOKEN.trim()}`;

  try {
    const endpoint = new URL("https://www.kaggle.com/api/v1/datasets/list");
    endpoint.searchParams.set("search", query);
    endpoint.searchParams.set("sortBy", "hottest");
    endpoint.searchParams.set("group", "public");
    endpoint.searchParams.set("page", "1");
    const response = await fetch(endpoint, { headers, cache: "no-store", signal: AbortSignal.timeout(12_000) });
    if (!response.ok) return curated(query, `Kaggle lookup returned HTTP ${response.status}; showing the reviewed protocol fixture instead.`);
    const body = await response.json();
    const rows = Array.isArray(body) ? body : Array.isArray(body?.datasets) ? body.datasets : [];
    const results = rows.map(mapKaggleResult).filter(Boolean).slice(0, 12);
    if (!results.length) return curated(query, "Kaggle returned no usable dataset records; showing the reviewed protocol fixture instead.");
    return Response.json({ mode: "kaggle", results }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    const message = error instanceof Error && error.name === "TimeoutError" ? "Kaggle lookup timed out" : "Kaggle lookup was unavailable";
    return curated(query, `${message}; showing the reviewed protocol fixture instead.`);
  }
}
