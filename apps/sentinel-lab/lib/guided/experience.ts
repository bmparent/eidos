/** Presentation/preflight policy only. Never substitutes for the worker or changes scores. */
export type LabDocument = Record<string, any>;
export type Experience = "easy" | "advanced";
export type AnalysisMode = "unusual" | "forecast" | "patterns";
export type RunOptions = { mode: AnalysisMode; target: string; horizonSeconds: number; windowSeconds: number };
const list = (value: unknown): LabDocument[] => Array.isArray(value) ? value.filter(x => x && typeof x === "object" && !Array.isArray(x)) : [];
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

export function suggestedMapping(dataset: LabDocument): LabDocument {
  if (dataset.confirmed) return structuredClone(dataset.confirmed);
  const columns = list(dataset.columns);
  const features = columns.filter(c => c.type === "number" && !c.labelCandidate && !c.idCandidate && !c.constant).slice(0, 16).map(c => c.name);
  return { features, labels: columns.filter(c => c.labelCandidate).map(c => c.name), timestamp: "", timezone: "", entity: "", session: "",
    units: Object.fromEntries(features.map(name => [name, "unknown"])), meaning: "", reference: "this entity's historical period", missing: "exclude", ordering: "sort" };
}

/** Client preflight is assistance, not authorization. The existing worker validates the contract again. */
export function prepareRunOptions(dataset: LabDocument, input: LabDocument): RunOptions {
  if (!dataset.confirmed) throw new Error("Confirm what the data means before running an analysis.");
  if (!["unusual", "forecast", "patterns"].includes(input.mode)) throw new Error("Choose a supported analysis goal.");
  const temporal = dataset.confirmed.mode === "temporal" && dataset.kind !== "documents";
  if (input.mode === "forecast" && !temporal) throw new Error("Forecasting needs a confirmed time column and numeric measurements. File order is not time.");
  if (temporal) {
    if (!Array.isArray(dataset.confirmed.features) || !dataset.confirmed.features.includes(input.target)) throw new Error("Select a target from your confirmed measurements.");
    for (const [key, label] of [["horizonSeconds", "Forecast horizon"], ["windowSeconds", "Late-match window"]]) {
      if (!finite(input[key]) || !Number.isInteger(input[key]) || input[key] < 1 || input[key] > 86400)
        throw new Error(`${label} must be a whole number from 1 to 86,400 seconds.`);
    }
  }
  return { mode: input.mode, target: temporal ? input.target : "", horizonSeconds: temporal ? input.horizonSeconds : 60, windowSeconds: temporal ? input.windowSeconds : 60 };
}

export function formatMeasured(value: unknown, unit = ""): string {
  return finite(value) ? `${new Intl.NumberFormat("en", { maximumFractionDigits: 3 }).format(value)}${unit ? ` ${unit}` : ""}` : "Not measured";
}

/** Resolve references against the exact saved input, not positions in a sorted/model matrix. */
export function evidenceForFinding(finding: LabDocument, dataset: LabDocument) {
  const records = Array.isArray(dataset.records) ? dataset.records : [];
  const ids = Array.isArray(dataset.recordIds) ? dataset.recordIds : [];
  const passages = list(dataset.passages);
  const index = new Map<string, number>();
  const ambiguous = new Set<string>();
  ids.forEach((id, i) => { if (typeof id === "string") { if (index.has(id)) ambiguous.add(id); else index.set(id, i); } });
  const passageIndex = new Map<string, LabDocument>();
  for (const passage of passages) { if (typeof passage.id === "string") { if (passageIndex.has(passage.id) || index.has(passage.id)) ambiguous.add(passage.id); else passageIndex.set(passage.id, passage); } }
  const seen = new Set<string>();
  const references: { id: string; when: string; value: unknown; member: LabDocument }[] = [];
  let unresolved = 0;
  for (const member of list(finding.members)) {
    const id = member.recordId || member.passageId;
    if (typeof id !== "string" || ambiguous.has(id)) { unresolved++; continue; }
    if (seen.has(id)) continue;
    seen.add(id);
    const rowIndex = index.get(id);
    const value = rowIndex !== undefined ? records[rowIndex] : passageIndex.get(id);
    if (value === undefined || value === null) { unresolved++; continue; }
    const timeKey = dataset.confirmed?.timestamp;
    const rawTime = timeKey && typeof value === "object" ? value[timeKey] : undefined;
    // Preserve the source value: never guess an epoch unit, date convention or timezone.
    const when = rawTime !== undefined && rawTime !== null && rawTime !== "" ? String(rawTime) : "Time not supplied — inspect the source record";
    references.push({ id, when, value, member });
  }
  return { references, unresolved };
}

export function summarizeRun(run: LabDocument, dataset: LabDocument) {
  if (!run || !Array.isArray(run.findings)) throw new Error("The result is missing its findings array. Do not interpret this as zero anomalies.");
  if (run.findings.some((finding: unknown) => !finding || typeof finding !== "object" || Array.isArray(finding)))
    throw new Error("The findings array contains malformed entries. Do not treat the result as verified.");
  if (run.datasetId !== dataset.id || typeof dataset.sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(dataset.sha256) || run.inputSha256 !== dataset.sha256)
    throw new Error("Result provenance does not match this saved input. Open the original run and dataset before interpreting it.");
  const metrics = run.metrics || {};
  const model = metrics.forecastMAE, baseline = metrics.persistenceMAE;
  let comparison = "No measured forecast comparison is available for this run.";
  if (finite(model) && finite(baseline) && model >= 0 && baseline >= 0) {
    comparison = model < baseline
      ? `Average forecast error was ${formatMeasured(model)}, versus ${formatMeasured(baseline)} for carrying the last value forward. Lower error on this run is not proof of general superiority.`
      : model > baseline
        ? `Carrying the last value forward performed better on this run: error ${formatMeasured(baseline)}, versus ${formatMeasured(model)} for the reported forecasting method.`
        : `The reported forecasting method and carrying the last value forward tied at an average error of ${formatMeasured(model)}.`;
  }
  const checked = list(run.findings).map(finding => ({ finding, ...evidenceForFinding(finding, dataset) }));
  const unverified = checked.filter(f => f.unresolved > 0 || f.references.length === 0).length;
  return { count: run.findings.length, checked, unverified, comparison,
    headline: run.findings.length ? `${run.findings.length} finding${run.findings.length === 1 ? "" : "s"} to investigate` : "No findings crossed this run's threshold",
    limitation: "Unusual does not necessarily mean harmful. No findings does not establish safety. Possible causes require additional evidence.",
    accuracy: finite(metrics.precision) && metrics.precision >= 0 && metrics.precision <= 1
      ? `Reported precision: ${formatMeasured(metrics.precision * 100)}%. Inspect the evaluation labels and matching rules in the full result.`
      : "Detection accuracy is not established by this result. An anomaly score is not a probability that something is wrong." };
}

export function kaggleDatasetUrl(ref: unknown): string | null {
  return typeof ref === "string" && /^[A-Za-z0-9_-]+\/[A-Za-z0-9_-]+$/.test(ref) ? `https://www.kaggle.com/datasets/${ref}` : null;
}

export function describeFailure(message: string): string {
  if (/payment_required|\b402\b/i.test(message)) return "The compute provider rejected worker creation. No analysis result was produced. The operator must check provider access, quota or billing; your saved jobs remain available.";
  if (/\b429\b|capacity|quota|admission/i.test(message)) return "The service could not admit this request within its current limits. Check saved jobs before retrying; no successful analysis is implied.";
  return message;
}

/** One key per immutable request. An uncertain response reuses that key, never silently duplicates work. */
export function requestFingerprint(path: string, input: unknown): string {
  const canonical = (value: any): any => Array.isArray(value) ? value.map(canonical)
    : value && typeof value === "object" ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
  return JSON.stringify([path, canonical(input)]);
}
