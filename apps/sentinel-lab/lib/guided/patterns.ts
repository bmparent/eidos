import type { Document } from "./store";

const finite = (value: unknown): number | null => {
  if (value === null || value === undefined || typeof value === "boolean" || String(value).trim() === "") return null;
  const result = Number(value); return Number.isFinite(result) ? result : null;
};
/** Retrospective description only. Never feeds preprocessing, forecasts or decisions. */
export function summarizePatterns(dataset: Document, mapping: Document) {
  const rows: Document[] = dataset.records || [], features: string[] = mapping.features || [];
  const pairs: Document[] = [];
  for (let a = 0; a < features.length; a++) for (let b = a + 1; b < features.length; b++) {
    const values = rows.map((row, i) => ({ x: finite(row[features[a]]), y: finite(row[features[b]]), id: dataset.recordIds[i] }))
      .filter((p): p is { x: number; y: number; id: string } => p.x !== null && p.y !== null);
    if (values.length < 3) continue;
    const mx = values.reduce((s, p) => s + p.x, 0) / values.length, my = values.reduce((s, p) => s + p.y, 0) / values.length;
    const covariance = values.reduce((s, p) => s + (p.x - mx) * (p.y - my), 0);
    const varianceX = values.reduce((s, p) => s + (p.x - mx) ** 2, 0), varianceY = values.reduce((s, p) => s + (p.y - my) ** 2, 0);
    pairs.push({ x: features[a], y: features[b], completePairs: values.length,
      correlation: varianceX && varianceY ? covariance / Math.sqrt(varianceX * varianceY) : null,
      sourceRecords: values.slice(0, 12).map(p => p.id) });
  }
  return { method: "whole-dataset Pearson association and category counts", schema: "eidos.patterns.v1",
    pairs: pairs.sort((a, b) => Math.abs(b.correlation || 0) - Math.abs(a.correlation || 0)), categories: dataset.categories || {},
    limitations: "Retrospective associations, not causes, forecasts or independent statistical tests. Missing pairs are excluded; repeated entities and trends can induce correlation. No values feed model training." };
}
