import test from "node:test";
import assert from "node:assert/strict";
import { suggestedMapping, prepareRunOptions, formatMeasured, evidenceForFinding, summarizeRun, kaggleDatasetUrl, describeFailure, requestFingerprint } from "../lib/guided/experience";

const sha256 = "a".repeat(64);
const dataset = () => ({ id: "ds-one", kind: "table", sha256, columns: [{ name: "latency", type: "number" }],
  confirmed: { mode: "temporal", features: ["latency"], timestamp: "time", timezone: "America/New_York" },
  records: [{ time: "2026-09-10 09:02", latency: 400 }, { time: "2026-09-10 09:01", latency: 20 }], recordIds: ["csv:2", "csv:3"] });
const result = (metrics = {}) => ({ datasetId: "ds-one", inputSha256: sha256, findings: [], metrics });
const options = { mode: "forecast", target: "latency", horizonSeconds: 60, windowSeconds: 60 };

test("suggestions isolate labels, identifiers and constants; never infer chronology", () => {
  const mapping = suggestedMapping({ columns: [{ name: "target", type: "number", labelCandidate: true }, { name: "id", type: "number", idCandidate: true }, { name: "constant", type: "number", constant: true }, { name: "latency", type: "number" }, { name: "time", type: "timestamp" }] });
  assert.deepEqual(mapping.features, ["latency"]); assert.deepEqual(mapping.labels, ["target"]); assert.equal(mapping.timestamp, "");
});
test("suggested measurement count respects the 16-feature envelope", () => {
  assert.equal(suggestedMapping({ columns: Array.from({ length: 30 }, (_, i) => ({ name: `f${i}`, type: "number" })) }).features.length, 16);
});
test("editing a suggestion cannot mutate the saved mapping", () => {
  const d = dataset(); const mapping = suggestedMapping(d); mapping.features.push("other"); assert.deepEqual(d.confirmed.features, ["latency"]);
});
test("analysis cannot start before confirmation", () => { assert.throws(() => prepareRunOptions({}, options), /Confirm/); });
test("unordered tables and documents cannot request temporal forecasts", () => {
  assert.throws(() => prepareRunOptions({ confirmed: { mode: "unordered" } }, options), /confirmed time/);
  assert.throws(() => prepareRunOptions({ ...dataset(), kind: "documents" }, options), /confirmed time/);
});
test("only confirmed targets are accepted", () => { assert.throws(() => prepareRunOptions(dataset(), { ...options, target: "label" }), /confirmed measurements/); });
test("invalid horizons and matching windows fail rather than default silently", () => {
  for (const key of ["horizonSeconds", "windowSeconds"]) for (const value of [NaN, Infinity, -1, 0, 1.5, 86401, "60", "", null])
    assert.throws(() => prepareRunOptions(dataset(), { ...options, [key]: value }), /whole number/);
});
test("valid run options preserve the explicit contract", () => { assert.deepEqual(prepareRunOptions(dataset(), options), options); });
test("unsupported analysis modes fail closed", () => { assert.throws(() => prepareRunOptions(dataset(), { ...options, mode: "automatic_magic" }), /supported/); });
test("missing, nonnumeric and nonfinite measurements are not rendered as zero", () => {
  for (const value of [undefined, null, NaN, Infinity, "0"]) assert.equal(formatMeasured(value), "Not measured");
  assert.equal(formatMeasured(0), "0");
});
test("evidence uses original record identities, not sorted row positions", () => {
  const evidence = evidenceForFinding({ members: [{ recordId: "csv:3" }] }, dataset());
  assert.equal(evidence.references[0].when, "2026-09-10 09:01"); assert.deepEqual(evidence.references[0].value, dataset().records[1]);
});
test("broken, duplicate source IDs and missing members never get invented references", () => {
  assert.equal(evidenceForFinding({ members: [{ recordId: "missing" }, {}] }, dataset()).unresolved, 2);
  assert.equal(evidenceForFinding({ members: [{ recordId: "csv:2" }] }, { ...dataset(), recordIds: ["csv:2", "csv:2"] }).unresolved, 1);
});
test("one source record is displayed once even with repeated finding members", () => {
  assert.equal(evidenceForFinding({ members: [{ recordId: "csv:2" }, { recordId: "csv:2" }] }, dataset()).references.length, 1);
});
test("documents resolve passage references without manufacturing timestamps", () => {
  const evidence = evidenceForFinding({ members: [{ passageId: "p1" }] }, { passages: [{ id: "p1", page: 2, text: "A source passage" }] });
  assert.equal(evidence.references[0].id, "p1"); assert.match(evidence.references[0].when, /Time not supplied/);
});
test("missing timestamps remain explicitly unknown", () => {
  assert.match(evidenceForFinding({ members: [{ recordId: "csv:2" }] }, { ...dataset(), confirmed: {} }).references[0].when, /Time not supplied/);
});
test("zero findings is not a safety conclusion", () => {
  const summary = summarizeRun(result(), dataset()); assert.equal(summary.count, 0); assert.match(summary.limitation, /does not establish safety/);
});
test("missing findings is an invalid result, not zero anomalies", () => { assert.throws(() => summarizeRun({ ...result(), findings: undefined }, dataset()), /missing/); });
test("a result must match both the input identity and byte hash", () => {
  assert.throws(() => summarizeRun({ ...result(), datasetId: "other" }, dataset()), /provenance/);
  assert.throws(() => summarizeRun({ ...result(), inputSha256: "b".repeat(64) }, dataset()), /provenance/);
  assert.throws(() => summarizeRun(result(), { ...dataset(), sha256: undefined }), /provenance/);
});
test("baseline losses and ties remain visible", () => {
  assert.match(summarizeRun(result({ forecastMAE: 3, persistenceMAE: 2 }), dataset()).comparison, /performed better/);
  assert.match(summarizeRun(result({ forecastMAE: 2, persistenceMAE: 2 }), dataset()).comparison, /tied/);
  assert.match(summarizeRun(result({ forecastMAE: 1, persistenceMAE: 2 }), dataset()).comparison, /not proof/);
});
test("missing metrics do not become accuracy claims", () => {
  const summary = summarizeRun(result({ forecastMAE: null, persistenceMAE: 0, precision: null }), dataset());
  assert.match(summary.comparison, /No measured/); assert.match(summary.accuracy, /not established/);
});
test("unresolved evidence is counted explicitly", () => {
  assert.equal(summarizeRun({ ...result(), findings: [{ members: [{ recordId: "gone" }] }] }, dataset()).unverified, 1);
});
test("Kaggle links are constructed only from validated owner/dataset references", () => {
  assert.equal(kaggleDatasetUrl("owner/data-set"), "https://www.kaggle.com/datasets/owner/data-set");
  for (const ref of ["javascript:alert(1)", "owner/../x", "https://evil.test/x", "owner/x?q=y", null]) assert.equal(kaggleDatasetUrl(ref), null);
});
test("compute rejection is never represented as a completed analysis", () => { assert.match(describeFailure("402 payment_required"), /No analysis result was produced/); });
test("idempotency canonicalizes object keys but distinguishes paths and changed options", () => {
  assert.equal(requestFingerprint("analyze", { a: 1, b: 2 }), requestFingerprint("analyze", { b: 2, a: 1 }));
  assert.notEqual(requestFingerprint("analyze", options), requestFingerprint("analyze", { ...options, horizonSeconds: 120 }));
  assert.notEqual(requestFingerprint("analyze", options), requestFingerprint("confirm", options));
});

test("malformed finding entries are not silently discarded", () => {
  for (const entry of [null, "made-up", 5, []]) assert.throws(() => summarizeRun({ ...result(), findings: [entry] }, dataset()), /malformed/);
});
