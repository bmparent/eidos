export type EvidenceClass = "REAL_DATA_ENGINEERING";
export type ExecutionProfile = "cpu_engineering" | "cpu_mechanisms" | "full_capacity";

export type ExperimentSpec = {
  schema: "eidos.sentinel-lab.experiment.v0.2";
  evidenceClass: EvidenceClass;
  dataset: {
    provider: "kaggle";
    ref: string;
    version: number;
    file: string;
    expectedSha256?: string;
  };
  dataContract: {
    labelColumn: string;
    negativeLabels: string[];
    orderMode: "source" | "column";
    orderColumn?: string;
    excludedColumns: string[];
    featureColumns: string[];
    maxRows: number;
  };
  split: {
    calibration: number;
    evaluation: number;
    sealedHoldout: number;
  };
  engine: {
    version: "0.4.7.02";
    features: 64;
    seed: 0 | 1;
    configProfile: "cicids_webattacks";
    executionProfile?: ExecutionProfile;
  };
  protocol: {
    labelPolicy: "sealed_until_prediction_freeze";
    normalization: "calibration_only_zscore";
    projection: "seeded_gaussian_or_pad";
    heldoutPolicy: "exclude_from_engineering_run";
    proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT";
  };
};

export type PreflightIssue = {
  severity: "blocker" | "warning" | "notice";
  code: string;
  message: string;
};

export type LockedExperiment = {
  schema: "eidos.sentinel-lab.lock.v0.2";
  algorithm: "sha256";
  digest: string;
  spec: ExperimentSpec;
  issues: PreflightIssue[];
  runnerConfigured: boolean;
  executionBackend: "sandbox" | "external" | null;
  readyToDispatch: boolean;
};

export type DatasetSearchResult = {
  ref: string;
  title: string;
  subtitle: string;
  version: number | null;
  totalBytes: number | null;
  usabilityRating: number | null;
  source: "kaggle" | "curated";
  url: string;
};

export type RunnerDispatch = {
  jobId: string;
  status: string;
  statusUrl?: string;
  executionBackend?: "sandbox" | "external";
  evidenceClass: EvidenceClass;
  proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT";
};

export type ExperimentMetrics = {
  evaluation_rows_expected: number;
  evaluation_rows_scored: number;
  confusion: { tp: number; fp: number; fn: number; tn: number };
  recall: number | null;
  precision: number | null;
  false_positive_rate: number | null;
  roc_auc: number | null;
  average_precision: number | null;
  mean_detection_delay_frames: number | null;
  missed_attack_windows: number;
  labels_unsealed_after_prediction_freeze: true;
  heldout_evaluated: false;
  prediction_coverage_complete?: boolean;
  positive_rows?: number;
  negative_rows?: number;
  limitations?: string[];
};

export type EngineDiagnostics = {
  execution_profile: ExecutionProfile;
  code_sha256: string;
  reservoir_units: number;
  memory_dimensions: number;
  leak_bands: number;
  trace_seal_enabled: boolean;
  thermodynamics_enabled: boolean;
  processed_rows: number;
  surprise_rows: number;
  memory_writes: number;
  statistics: Record<string, { samples: number; mean: number; min: number; max: number }>;
  trace: Array<Record<string, number | boolean | null>>;
  trace_sampling?: string;
  geometry?: { method: string; variance_explained: number; states_sha256: string; points: Array<{ step: number; x: number; y: number; z: number }> } | null;
  scope: string;
};

export type ExperimentStatus = {
  schema: string;
  jobId: string;
  status: string;
  updatedAt: string;
  evidenceClass: EvidenceClass;
  proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT";
  gatesAdvanced: 0;
  lockDigest?: string;
  rowsSentToEngine?: number;
  metrics?: ExperimentMetrics;
  engineDiagnostics?: EngineDiagnostics;
  artifacts?: string[];
  error?: string;
  detail?: string;
  launcherCommandId?: string;
  launcherStartedAt?: number;
  launcherExitCode?: number;
  executionBackend?: "sandbox" | "external";
};
