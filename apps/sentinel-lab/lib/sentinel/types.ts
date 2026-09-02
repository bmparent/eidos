export type ScenarioId =
  | "S0_nominal"
  | "S1_hidden_backdoor"
  | "S2_slow_drift"
  | "S3_regime_shift"
  | "S6_noise_thrash"
  | "S7_harmless_repeat"
  | "S8_dangerous_repeat"
  | "C1_nuisance_subspace";

export type SmokeRequest = {
  scenario: ScenarioId;
  seed: 0 | 1;
  frames: 240 | 480 | 720;
  system: "eidos_ms_v1_observer";
};

export type TracePoint = {
  frame: number;
  raw: number;
  quotient: number;
  persistence: number;
  threshold: number;
  active: boolean;
};

export type ComparisonRow = {
  id: string;
  label: string;
  peak: number;
  eventDetected: boolean;
  firstDetection: number | null;
  falseAlerts: number;
  note: string;
};

export type IncidentEvidence = {
  observation: string;
  why: string;
  disconfirm: string;
  action: string;
  uncertainty: string;
  references: string[];
};

export type SmokeResult = {
  schema: "eidos.sentinel-lab.smoke.v1";
  runId: string;
  evidenceClass: "ENGINEERING_SMOKE";
  scenario: ScenarioId;
  scenarioLabel: string;
  seed: 0 | 1;
  frames: number;
  system: "eidos_ms_v1_observer";
  generatedAt: string;
  protocol: {
    id: "EIDOS-GP-v1-2026-09-01";
    verdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT";
    gatesAdvanced: 0;
    gateCount: 7;
  };
  eventWindows: Array<{ start: number; end: number; kind: string }>;
  trace: TracePoint[];
  summary: {
    peakRaw: number;
    peakQuotient: number;
    persistenceAuc: number;
    threshold: number;
    firstDetection: number | null;
    candidateWindows: number;
    rawEscapeTriggered: boolean;
    measuredFields: number;
    requiredEvidenceFields: number;
  };
  incident: IncidentEvidence;
  comparisons: ComparisonRow[];
  disclaimer: string;
};

export type ImportedArtifact = {
  id: string;
  filename: string;
  kind: "verdict" | "run-lock" | "metrics" | "events" | "unknown";
  digest: string;
  importedAt: string;
  verdict?: string;
  protocolId?: string;
  records?: number;
  summary: string;
};
