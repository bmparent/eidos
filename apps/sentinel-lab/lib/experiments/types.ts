export type EvidenceClass = "REAL_DATA_ENGINEERING";

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
  recall: number;
  precision: number;
  false_positive_rate: number;
  roc_auc: number | null;
  average_precision: number | null;
  mean_detection_delay_frames: number | null;
  missed_attack_windows: number;
  labels_unsealed_after_prediction_freeze: true;
  heldout_evaluated: false;
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
  artifacts?: string[];
  error?: string;
  detail?: string;
  launcherCommandId?: string;
  launcherStartedAt?: number;
  launcherExitCode?: number;
  executionBackend?: "sandbox" | "external";
};
