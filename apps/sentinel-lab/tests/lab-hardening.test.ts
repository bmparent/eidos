import assert from "node:assert/strict";
import test from "node:test";
import { cloneDefaultExperiment, validateExperimentSpec } from "../lib/experiments/shared.js";
import { engineProfile, reviseDatasetSource, selectDataset } from "../lib/experiments/profiles.js";
import { getExecutionReadiness } from "../lib/experiments/runner";
import { authorizeOperator } from "../lib/experiments/operator-auth";
import { readExperimentJson, RequestBodyError } from "../lib/experiments/request-body";
import { POST as preflight } from "../app/api/experiments/preflight/route";
import { POST as dispatch } from "../app/api/experiments/route";
import { sha256Canonical } from "../lib/experiments/lock.js";
import type { ExperimentSpec } from "../lib/experiments/types";

test("changing a dataset source invalidates inherited hashes, filenames and unknown versions", () => {
  const source = cloneDefaultExperiment();
  for (const patch of [{ ref: "owner/new" }, { version: 9 }, { file: "new.csv" }]) assert.equal(reviseDatasetSource(source, patch).dataset.expectedSha256, undefined);
  assert.equal(reviseDatasetSource(source, { file: source.dataset.file }).dataset.expectedSha256, source.dataset.expectedSha256);
  const selected = selectDataset(source, { ref: "owner/new", version: null });
  assert.equal(selected.dataset.version, 0);
  assert.equal(selected.dataset.file, "");
  assert.equal(selected.dataset.expectedSha256, undefined);
  assert.throws(() => validateExperimentSpec(selected));
});

test("only explicit numeric values and allowlisted engine profiles can be locked", () => {
  for (const seed of [false, true, null, "0", 100]) {
    const value = cloneDefaultExperiment(); value.engine.seed = seed;
    assert.throws(() => validateExperimentSpec(value));
  }
  for (const profile of ["__proto__", "constructor", "gpu_unbounded"]) {
    const value = cloneDefaultExperiment(); value.engine.executionProfile = profile;
    assert.throws(() => validateExperimentSpec(value));
    assert.throws(() => engineProfile(value));
  }
  const standard = validateExperimentSpec(cloneDefaultExperiment());
  const study = validateExperimentSpec({ ...standard, engine: { ...standard.engine, executionProfile: "cpu_mechanisms" } });
  assert.notEqual(sha256Canonical(standard), sha256Canonical(study));
  assert.equal(engineProfile(study).fractal_bands, 4);
});

test("operator and full-capacity gates stay closed before any dispatch", async () => {
  const original = { ...process.env };
  try {
    process.env.EIDOS_OPERATOR_TOKEN = "local-test-only";
    assert.throws(() => authorizeOperator(new Request("http://test")), /OPERATOR_AUTH_REQUIRED/);
    authorizeOperator(new Request("http://test", { headers: { Authorization: "Bearer local-test-only" } }));
    const rejected = await dispatch(new Request("http://test/api/experiments", { method: "POST", body: "{}" }));
    assert.equal(rejected.status, 401);
    process.env.EIDOS_EXECUTION_BACKEND = "sandbox";
    const spec = cloneDefaultExperiment() as ExperimentSpec;
    spec.engine.executionProfile = "full_capacity";
    assert.ok(getExecutionReadiness(spec).blockers.some((item) => item.code === "FULL_CAPACITY_REQUIRES_DEDICATED_RUNNER"));
    process.env.EIDOS_EXECUTION_BACKEND = "external";
    delete process.env.EIDOS_ENABLE_FULL_CAPACITY;
    assert.ok(getExecutionReadiness(spec).blockers.some((item) => item.code === "FULL_CAPACITY_NOT_ENABLED"));
  } finally { for (const key of Object.keys(process.env)) if (!(key in original)) delete process.env[key]; Object.assign(process.env, original); }
});

test("bounded JSON handling rejects malformed and chunked oversized requests", async () => {
  const large = new Request("http://test", { method: "POST", body: "x".repeat(65_537) });
  await assert.rejects(readExperimentJson(large), (error) => error instanceof RequestBodyError && error.status === 413);
  const bad = await preflight(new Request("http://test", { method: "POST", body: "{" }));
  assert.equal(bad.status, 400);
  const oversized = await preflight(new Request("http://test", { method: "POST", body: "x".repeat(65_537) }));
  assert.equal(oversized.status, 413);
  const valid = await preflight(new Request("http://test", { method: "POST", body: JSON.stringify(cloneDefaultExperiment()) }));
  assert.equal(valid.status, 200);
  assert.equal(valid.headers.get("cache-control"), "no-store");
  const lock = await valid.json();
  assert.equal(lock.digest, "5eba19350bb0cf1a8761f2bfba3ae730f0e073ec1436c360ba9904b6ef9b044a");
});
