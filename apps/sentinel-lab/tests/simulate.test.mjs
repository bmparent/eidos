import assert from "node:assert/strict";
import test from "node:test";
import { simulateSmoke, validateSmokeRequest } from "../lib/sentinel/simulate.js";

const request = {
  scenario: "S1_hidden_backdoor",
  seed: 0,
  frames: 240,
  system: "eidos_ms_v1_observer",
};

test("engineering smoke is deterministic for a locked request", () => {
  assert.deepEqual(simulateSmoke(request), simulateSmoke(request));
});

test("held-out seeds are inaccessible", () => {
  assert.throws(() => validateSmokeRequest({ ...request, seed: 100 }), /engineering seeds 0 and 1/);
});

test("smoke output cannot advance proof gates", () => {
  const result = simulateSmoke(request);
  assert.equal(result.evidenceClass, "ENGINEERING_SMOKE");
  assert.equal(result.protocol.verdict, "BLOCKED_RESOURCE_BEFORE_HELDOUT");
  assert.equal(result.protocol.gatesAdvanced, 0);
  assert.match(result.disclaimer, /does not advance proof gates/i);
});

test("incident language remains calibrated", () => {
  const result = simulateSmoke({ ...request, scenario: "S8_dangerous_repeat" });
  const card = Object.values(result.incident).flat().join(" ").toLowerCase();
  for (const prohibited of ["attack confirmed", "system compromised", "seizure detected", "cause established"]) {
    assert.equal(card.includes(prohibited), false);
  }
  assert.match(result.incident.uncertainty, /synthetic engineering projection only/i);
});

test("all registered engineering scenarios return finite traces", () => {
  const scenarios = [
    "S0_nominal",
    "S1_hidden_backdoor",
    "S2_slow_drift",
    "S3_regime_shift",
    "S6_noise_thrash",
    "S7_harmless_repeat",
    "S8_dangerous_repeat",
    "C1_nuisance_subspace",
  ];

  for (const scenario of scenarios) {
    const result = simulateSmoke({ ...request, scenario });
    assert.equal(result.trace.length, request.frames);
    assert.ok(result.trace.every((point) => Object.values(point).every((value) => typeof value === "boolean" || Number.isFinite(value))));
  }
});
