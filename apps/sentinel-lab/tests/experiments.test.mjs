import assert from "node:assert/strict";
import test from "node:test";
import { cloneDefaultExperiment, preflightIssues, validateExperimentSpec } from "../lib/experiments/shared.js";
import { canonicalJson, sha256Canonical } from "../lib/experiments/lock.js";

test("default real-data spec produces a stable canonical lock", () => {
  const first = validateExperimentSpec(cloneDefaultExperiment());
  const second = validateExperimentSpec(JSON.parse(JSON.stringify(first)));
  assert.equal(canonicalJson(first), canonicalJson(second));
  assert.match(sha256Canonical(first), /^[a-f0-9]{64}$/);
  assert.equal(sha256Canonical(first), sha256Canonical(second));
});

test("real-data spec requires version and rejects path traversal", () => {
  const missingVersion = cloneDefaultExperiment();
  delete missingVersion.dataset.version;
  assert.throws(() => validateExperimentSpec(missingVersion), /positive integer version/);

  const traversal = cloneDefaultExperiment();
  traversal.dataset.file = "../secret.csv";
  assert.throws(() => validateExperimentSpec(traversal), /without traversal/);
});

test("proof and label-isolation locks cannot be relaxed", () => {
  const value = cloneDefaultExperiment();
  value.protocol.heldoutPolicy = "include_in_engineering";
  assert.throws(() => validateExperimentSpec(value), /heldoutPolicy must remain locked/);

  const featureLeak = cloneDefaultExperiment();
  featureLeak.dataContract.featureColumns = ["Label"];
  assert.throws(() => validateExperimentSpec(featureLeak), /label column cannot be selected/);
});

test("preflight blocks dispatch without changing scientific readiness", () => {
  const spec = validateExperimentSpec(cloneDefaultExperiment());
  const issues = preflightIssues(spec, false);
  assert.ok(issues.some((issue) => issue.code === "RUNNER_NOT_CONFIGURED" && issue.severity === "blocker"));
  assert.ok(issues.some((issue) => issue.code === "AUTO_NUMERIC_FEATURES" && issue.severity === "warning"));
});
