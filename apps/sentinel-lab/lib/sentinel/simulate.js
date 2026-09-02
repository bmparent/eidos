const SCENARIOS = {
  S0_nominal: {
    label: "S0 · Nominal carrier",
    why: "The observer remained near its causal baseline; no registered consequential event was injected.",
  },
  S1_hidden_backdoor: {
    label: "S1 · Hidden backdoor",
    why: "The quotient view amplified phase-locked structure that stayed modest in the pointwise residual.",
  },
  S2_slow_drift: {
    label: "S2 · Slow drift",
    why: "Persistence accumulated a gradual displacement that no single frame established on its own.",
  },
  S3_regime_shift: {
    label: "S3 · Regime shift",
    why: "Residual structure remained displaced after the registered transition instead of behaving like an isolated spike.",
  },
  S6_noise_thrash: {
    label: "S6 · Noise thrash",
    why: "Raw variance rose, but consequence and persistence evidence remained deliberately limited.",
  },
  S7_harmless_repeat: {
    label: "S7 · Harmless repeat",
    why: "The second excursion matched a familiar pattern whose delayed engineering outcome is marked benign.",
  },
  S8_dangerous_repeat: {
    label: "S8 · Consequential repeat",
    why: "Familiarity did not suppress review because delayed engineering outcome metadata marks the earlier pattern consequential.",
  },
  C1_nuisance_subspace: {
    label: "C1 · Nuisance subspace",
    why: "The registered raw-escape path preserved an excursion that the quotient representation could understate.",
  },
};

const ALLOWED_FRAMES = new Set([240, 480, 720]);
const ALLOWED_SEEDS = new Set([0, 1]);

export function validateSmokeRequest(value) {
  if (!value || typeof value !== "object") throw new Error("Request body must be an object.");
  if (!(value.scenario in SCENARIOS)) throw new Error("Unknown engineering scenario.");
  if (!ALLOWED_SEEDS.has(value.seed)) {
    throw new Error("Only preregistered engineering seeds 0 and 1 are available in the lab.");
  }
  if (!ALLOWED_FRAMES.has(value.frames)) throw new Error("Frames must be 240, 480, or 720.");
  if (value.system !== "eidos_ms_v1_observer") throw new Error("Unknown observer system.");

  return {
    scenario: value.scenario,
    seed: value.seed,
    frames: value.frames,
    system: value.system,
  };
}

function hashText(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(random) {
  const u = Math.max(random(), Number.EPSILON);
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) total += a[i] * b[i];
  return total;
}

function norm(vector) {
  return Math.sqrt(dot(vector, vector));
}

function makeDirections(random, count = 5, dimensions = 64) {
  const directions = [];
  for (let index = 0; index < count; index += 1) {
    const vector = Array.from({ length: dimensions }, () => normal(random));
    for (const prior of directions) {
      const projection = dot(vector, prior);
      for (let dimension = 0; dimension < dimensions; dimension += 1) {
        vector[dimension] -= projection * prior[dimension];
      }
    }
    const length = Math.max(norm(vector), Number.EPSILON);
    directions.push(vector.map((value) => value / length));
  }
  return directions;
}

function round(value, digits = 3) {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function clamp(value, minimum = 0, maximum = 1) {
  return Math.min(maximum, Math.max(minimum, value));
}

function scenarioWindows(scenario, frames) {
  if (scenario === "S0_nominal") return [];
  if (scenario === "S7_harmless_repeat" || scenario === "S8_dangerous_repeat") {
    const width = Math.max(12, Math.round(frames * 0.11));
    return [
      { start: Math.round(frames * 0.27), end: Math.round(frames * 0.27) + width, kind: "first" },
      { start: Math.round(frames * 0.64), end: Math.round(frames * 0.64) + width, kind: "repeat" },
    ];
  }
  const start = Math.round(frames * 0.46);
  return [{ start, end: Math.round(frames * 0.69), kind: "registered" }];
}

function inWindow(frame, windows) {
  return windows.some((window) => frame >= window.start && frame <= window.end);
}

function injectScenario(vector, scenario, frame, windows, directions, random) {
  const windowIndex = windows.findIndex((window) => frame >= window.start && frame <= window.end);
  const active = windowIndex >= 0;
  if (!active) return;

  const window = windows[windowIndex];
  const progress = (frame - window.start) / Math.max(1, window.end - window.start);
  let amplitude = 0;
  let direction = directions[3];

  if (scenario === "S1_hidden_backdoor") amplitude = 0.14 * Math.sin((2 * Math.PI * frame) / 7);
  if (scenario === "S2_slow_drift") amplitude = 0.28 * progress;
  if (scenario === "S3_regime_shift") amplitude = 0.31 * (0.65 + 0.35 * Math.sin((2 * Math.PI * frame) / 31));
  if (scenario === "S6_noise_thrash") {
    for (let i = 0; i < vector.length; i += 1) vector[i] += normal(random) * 0.085;
    return;
  }
  if (scenario === "S7_harmless_repeat" || scenario === "S8_dangerous_repeat") {
    amplitude = 0.29 * Math.sin(Math.PI * progress);
  }
  if (scenario === "C1_nuisance_subspace") {
    amplitude = 0.42 * Math.sin(Math.PI * progress);
    direction = directions[4];
  }

  for (let i = 0; i < vector.length; i += 1) vector[i] += amplitude * direction[i];
}

function summarizeDetector(label, id, scores, threshold, windows, note) {
  let firstDetection = null;
  let falseAlerts = 0;
  let peak = 0;
  let streak = 0;

  scores.forEach((score, frame) => {
    peak = Math.max(peak, score);
    streak = score >= threshold ? streak + 1 : 0;
    if (streak === 3) {
      const detectedAt = frame - 2;
      if (inWindow(detectedAt, windows)) firstDetection ??= detectedAt;
      else falseAlerts += 1;
    }
  });

  return {
    id,
    label,
    peak: round(peak),
    eventDetected: windows.length > 0 ? firstDetection !== null : false,
    firstDetection,
    falseAlerts,
    note,
  };
}

export function simulateSmoke(rawInput) {
  const input = validateSmokeRequest(rawInput);
  const { scenario, seed, frames, system } = input;
  const random = mulberry32(hashText(`${scenario}:${seed}:${frames}`));
  const directions = makeDirections(random);
  const windows = scenarioWindows(scenario, frames);
  const dimensions = 64;
  const prediction = Array(dimensions).fill(0);
  const trace = [];
  const rawScores = [];
  const quotientScores = [];
  const persistenceScores = [];
  const ewmaScores = [];
  const cusumScores = [];
  let residualMean = 0;
  let residualM2 = 0;
  let residualCount = 0;
  let quotientMean = 0;
  let quotientM2 = 0;
  let quotientCount = 0;
  let persistence = 0;
  let ewma = 0;
  let cusum = 0;
  const threshold = scenario === "S6_noise_thrash" ? 0.72 : 0.62;

  for (let frame = 0; frame < frames; frame += 1) {
    const vector = Array.from({ length: dimensions }, (_, dimension) => {
      const carrier =
        0.55 * Math.sin((2 * Math.PI * frame) / 61) * directions[0][dimension] +
        0.38 * Math.cos((2 * Math.PI * frame) / 89) * directions[1][dimension] +
        0.24 * Math.sin((2 * Math.PI * frame) / 137 + 0.4) * directions[2][dimension];
      return carrier + normal(random) * 0.018;
    });

    injectScenario(vector, scenario, frame, windows, directions, random);
    const residual = vector.map((value, index) => value - prediction[index]);
    const residualMagnitude = norm(residual) / Math.sqrt(dimensions);
    const quotientMagnitude = Math.abs(dot(residual, directions[3]));

    const residualVariance = residualCount > 1 ? residualM2 / (residualCount - 1) : 0.0004;
    const quotientVariance = quotientCount > 1 ? quotientM2 / (quotientCount - 1) : 0.0004;
    const rawZ = Math.max(0, (residualMagnitude - residualMean) / Math.max(Math.sqrt(residualVariance), 0.012));
    let quotientZ = Math.max(0, (quotientMagnitude - quotientMean) / Math.max(Math.sqrt(quotientVariance), 0.018));
    if (scenario === "C1_nuisance_subspace") quotientZ *= 0.32;

    const rawScore = clamp(1 - Math.exp(-rawZ / 3.1));
    const quotientScore = clamp(1 - Math.exp(-quotientZ / 3.4));
    const consequenceFloor = scenario === "S8_dangerous_repeat" && windows[1] && frame >= windows[1].start ? 0.24 : 0;
    persistence = Math.max(consequenceFloor, persistence * 0.88 + Math.max(rawScore, quotientScore) * 0.12);
    ewma = ewma * 0.9 + rawScore * 0.1;
    cusum = Math.max(0, cusum * 0.92 + rawScore - 0.21);
    const cusumScore = clamp(cusum / 2.8);
    const combined = clamp(Math.max(rawScore * 0.88, quotientScore * 0.92) + persistence * 0.19);
    const rawEscape = rawScore >= 0.76;
    const score = scenario === "C1_nuisance_subspace" && rawEscape ? Math.max(combined, 0.78) : combined;

    rawScores.push(rawScore);
    quotientScores.push(quotientScore);
    persistenceScores.push(score);
    ewmaScores.push(clamp(ewma * 1.75));
    cusumScores.push(cusumScore);
    trace.push({
      frame,
      raw: round(rawScore),
      quotient: round(quotientScore),
      persistence: round(persistence),
      threshold,
      active: inWindow(frame, windows),
    });

    for (let index = 0; index < dimensions; index += 1) prediction[index] = prediction[index] * 0.82 + vector[index] * 0.18;

    if (frame > 8 && !inWindow(frame, windows)) {
      residualCount += 1;
      const residualDelta = residualMagnitude - residualMean;
      residualMean += residualDelta / residualCount;
      residualM2 += residualDelta * (residualMagnitude - residualMean);

      quotientCount += 1;
      const quotientDelta = quotientMagnitude - quotientMean;
      quotientMean += quotientDelta / quotientCount;
      quotientM2 += quotientDelta * (quotientMagnitude - quotientMean);
    }
  }

  const detectorThreshold = threshold;
  const comparisons = [
    summarizeDetector(
      "EIDOS-MS v1 observer",
      "eidos_ms_v1_observer",
      persistenceScores,
      detectorThreshold,
      windows,
      "Raw escape + quotient + persistence; engineering projection only.",
    ),
    summarizeDetector("Rolling z", "rolling_z", rawScores, detectorThreshold, windows, "Causal residual magnitude."),
    summarizeDetector("EWMA", "ewma", ewmaScores, detectorThreshold, windows, "Causal exponentially weighted residual."),
    summarizeDetector("CUSUM", "cusum", cusumScores, detectorThreshold, windows, "Two-sided engineering proxy."),
  ];

  const primaryWindow = windows.at(-1) ?? { start: 0, end: 0, kind: "none" };
  const firstDetection = comparisons[0].firstDetection;
  const peakRaw = Math.max(...rawScores);
  const peakQuotient = Math.max(...quotientScores);
  const persistenceAuc = persistenceScores.reduce((total, value) => total + value, 0) / frames;
  const rawEscapeTriggered = rawScores.some((value, frame) => inWindow(frame, windows) && value >= 0.76);
  const runId = `eng-${scenario.toLowerCase()}-s${seed}-f${frames}`;
  const noEvent = windows.length === 0;
  const observation = noEvent
    ? "No registered event window was injected. The trace stayed available for false-alert inspection."
    : firstDetection === null
      ? `No sustained observer crossing was recorded inside the registered window ${primaryWindow.start}–${primaryWindow.end}.`
      : `A sustained observer excursion crossed the engineering threshold near frame ${firstDetection} inside window ${primaryWindow.start}–${primaryWindow.end}.`;

  return {
    schema: "eidos.sentinel-lab.smoke.v1",
    runId,
    evidenceClass: "ENGINEERING_SMOKE",
    scenario,
    scenarioLabel: SCENARIOS[scenario].label,
    seed,
    frames,
    system,
    generatedAt: `engineering-seed-${seed}`,
    protocol: {
      id: "EIDOS-GP-v1-2026-09-01",
      verdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT",
      gatesAdvanced: 0,
      gateCount: 7,
    },
    eventWindows: windows,
    trace,
    summary: {
      peakRaw: round(peakRaw),
      peakQuotient: round(peakQuotient),
      persistenceAuc: round(persistenceAuc),
      threshold,
      firstDetection,
      candidateWindows: firstDetection === null ? 0 : 1,
      rawEscapeTriggered,
      measuredFields: 7,
      requiredEvidenceFields: 9,
    },
    incident: {
      observation,
      why: SCENARIOS[scenario].why,
      disconfirm: noEvent
        ? "Repeat with engineering seed 1 and verify that alert pressure remains bounded."
        : "Repeat with the registered mechanism removed and compare the same past-only observer configuration.",
      action:
        scenario === "C1_nuisance_subspace"
          ? "Compare the raw-escape path with A6_no_raw_escape on engineering seeds only."
          : "Run engineering seed 1, then compare the targeted preregistered ablation without opening held-out seeds.",
      uncertainty:
        "Synthetic engineering projection only. It does not run the full Torch reservoir, establish natural-domain behavior, or advance a Grand Proof gate.",
      references: [
        `trace://${scenario}/seed-${seed}/frames-${primaryWindow.start}-${primaryWindow.end}`,
        `protocol://EIDOS-GP-v1-2026-09-01/${scenario}`,
        `run://${runId}`,
      ],
    },
    comparisons,
    disclaimer: "Engineering smoke does not advance proof gates.",
  };
}

export const scenarioOptions = Object.entries(SCENARIOS).map(([value, scenario]) => ({
  value,
  label: scenario.label,
}));
