export const EXPERIMENT_SCHEMA = "eidos.sentinel-lab.experiment.v0.2";
export const LOCK_SCHEMA = "eidos.sentinel-lab.lock.v0.2";

export const CURATED_DATASETS = [
  {
    ref: "dhoogla/cicids2017",
    title: "CIC-IDS2017",
    subtitle: "Verified fixture · version 3 · labeled network-flow telemetry",
    version: 3,
    totalBytes: null,
    usabilityRating: 10,
    source: "curated",
    url: "https://www.kaggle.com/datasets/dhoogla/cicids2017",
  },
];

export const DEFAULT_EXPERIMENT_SPEC = {
  schema: EXPERIMENT_SCHEMA,
  evidenceClass: "REAL_DATA_ENGINEERING",
  dataset: {
    provider: "kaggle",
    ref: "dhoogla/cicids2017",
    version: 3,
    file: "WebAttacks-Thursday-no-metadata.parquet",
    expectedSha256: "7db47b2bf97ad58c3556ee25e8e1eb1e697cd391670733833865d0e84d8ed82a",
  },
  dataContract: {
    labelColumn: "Label",
    negativeLabels: ["BENIGN"],
    orderMode: "source",
    excludedColumns: [],
    featureColumns: [],
    maxRows: 25000,
  },
  split: {
    calibration: 0.2,
    evaluation: 0.6,
    sealedHoldout: 0.2,
  },
  engine: {
    version: "0.4.7.02",
    features: 64,
    seed: 0,
    configProfile: "cicids_webattacks",
  },
  protocol: {
    labelPolicy: "sealed_until_prediction_freeze",
    normalization: "calibration_only_zscore",
    projection: "seeded_gaussian_or_pad",
    heldoutPolicy: "exclude_from_engineering_run",
    proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT",
  },
};

function requireObject(value, path) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path} must be an object.`);
  }
  return value;
}

function requireString(value, path, max = 240) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${path} is required.`);
  if (value.length > max) throw new Error(`${path} is too long.`);
  return value.trim();
}

function stringList(value, path, maxItems = 128) {
  if (!Array.isArray(value) || value.length > maxItems) throw new Error(`${path} must be a short string list.`);
  const normalized = value.map((item, index) => requireString(item, `${path}[${index}]`, 160));
  return [...new Set(normalized)];
}

function exactKeys(value, allowed, path) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length) throw new Error(`${path} contains unsupported field${unknown.length === 1 ? "" : "s"}: ${unknown.join(", ")}.`);
}

function cleanDatasetPath(value) {
  const path = requireString(value, "dataset.file", 500).replaceAll("\\", "/");
  if (path.startsWith("/") || path.split("/").some((part) => part === ".." || part === "")) {
    throw new Error("dataset.file must be an exact relative Kaggle path without traversal.");
  }
  if (!/\.(csv|tsv|parquet|feather|ftr)$/i.test(path)) {
    throw new Error("dataset.file must be CSV, TSV, Parquet, or Feather data.");
  }
  return path;
}

export function validateExperimentSpec(input) {
  const root = requireObject(input, "experiment");
  exactKeys(root, ["schema", "evidenceClass", "dataset", "dataContract", "split", "engine", "protocol"], "experiment");
  if (root.schema !== EXPERIMENT_SCHEMA) throw new Error(`schema must be ${EXPERIMENT_SCHEMA}.`);
  if (root.evidenceClass !== "REAL_DATA_ENGINEERING") throw new Error("v0.2 only permits REAL_DATA_ENGINEERING runs.");

  const dataset = requireObject(root.dataset, "dataset");
  exactKeys(dataset, ["provider", "ref", "version", "file", "expectedSha256"], "dataset");
  if (dataset.provider !== "kaggle") throw new Error("v0.2 only permits the Kaggle provider.");
  const ref = requireString(dataset.ref, "dataset.ref", 120).toLowerCase();
  if (!/^[a-z0-9][a-z0-9_-]{0,38}\/[a-z0-9][a-z0-9._-]{0,79}$/.test(ref)) {
    throw new Error("dataset.ref must use the Kaggle owner/dataset form.");
  }
  const version = Number(dataset.version);
  if (!Number.isSafeInteger(version) || version < 1) throw new Error("dataset.version must pin a positive integer version.");
  const file = cleanDatasetPath(dataset.file);
  let expectedSha256;
  if (dataset.expectedSha256 !== undefined && dataset.expectedSha256 !== "") {
    expectedSha256 = requireString(dataset.expectedSha256, "dataset.expectedSha256", 64).toLowerCase();
    if (!/^[a-f0-9]{64}$/.test(expectedSha256)) throw new Error("dataset.expectedSha256 must be a 64-character SHA-256 digest.");
  }

  const data = requireObject(root.dataContract, "dataContract");
  exactKeys(data, ["labelColumn", "negativeLabels", "orderMode", "orderColumn", "excludedColumns", "featureColumns", "maxRows"], "dataContract");
  const labelColumn = requireString(data.labelColumn, "dataContract.labelColumn", 160);
  const negativeLabels = stringList(data.negativeLabels, "dataContract.negativeLabels", 32);
  if (!negativeLabels.length) throw new Error("dataContract.negativeLabels must declare at least one benign label.");
  if (data.orderMode !== "source" && data.orderMode !== "column") throw new Error("dataContract.orderMode must be source or column.");
  const orderColumn = data.orderMode === "column" ? requireString(data.orderColumn, "dataContract.orderColumn", 160) : undefined;
  const excludedColumns = stringList(data.excludedColumns, "dataContract.excludedColumns");
  const featureColumns = stringList(data.featureColumns, "dataContract.featureColumns");
  const maxRows = Number(data.maxRows);
  if (!Number.isSafeInteger(maxRows) || maxRows < 1000 || maxRows > 2_000_000) {
    throw new Error("dataContract.maxRows must be an integer from 1,000 to 2,000,000.");
  }
  if (featureColumns.includes(labelColumn)) throw new Error("The label column cannot be selected as an engine feature.");

  const split = requireObject(root.split, "split");
  exactKeys(split, ["calibration", "evaluation", "sealedHoldout"], "split");
  const calibration = Number(split.calibration);
  const evaluation = Number(split.evaluation);
  const sealedHoldout = Number(split.sealedHoldout);
  for (const [name, value] of [["calibration", calibration], ["evaluation", evaluation], ["sealedHoldout", sealedHoldout]]) {
    if (!Number.isFinite(value) || value < 0.1 || value > 0.8) throw new Error(`split.${name} must be between 0.1 and 0.8.`);
  }
  if (Math.abs(calibration + evaluation + sealedHoldout - 1) > 1e-9) throw new Error("Experiment splits must total exactly 1.0.");

  const engine = requireObject(root.engine, "engine");
  exactKeys(engine, ["version", "features", "seed", "configProfile"], "engine");
  if (engine.version !== "0.4.7.02") throw new Error("engine.version must remain locked to 0.4.7.02 in v0.2.");
  if (Number(engine.features) !== 64) throw new Error("engine.features must remain locked to 64 in v0.2.");
  const seed = Number(engine.seed);
  if (seed !== 0 && seed !== 1) throw new Error("Real-data engineering runs are restricted to seeds 0 and 1.");
  if (engine.configProfile !== "cicids_webattacks") throw new Error("engine.configProfile must remain cicids_webattacks in v0.2.");

  const protocol = requireObject(root.protocol, "protocol");
  exactKeys(protocol, ["labelPolicy", "normalization", "projection", "heldoutPolicy", "proofVerdict"], "protocol");
  const locks = {
    labelPolicy: "sealed_until_prediction_freeze",
    normalization: "calibration_only_zscore",
    projection: "seeded_gaussian_or_pad",
    heldoutPolicy: "exclude_from_engineering_run",
    proofVerdict: "BLOCKED_RESOURCE_BEFORE_HELDOUT",
  };
  for (const [key, value] of Object.entries(locks)) {
    if (protocol[key] !== value) throw new Error(`protocol.${key} must remain locked to ${value}.`);
  }

  return {
    schema: EXPERIMENT_SCHEMA,
    evidenceClass: "REAL_DATA_ENGINEERING",
    dataset: { provider: "kaggle", ref, version, file, ...(expectedSha256 ? { expectedSha256 } : {}) },
    dataContract: {
      labelColumn,
      negativeLabels,
      orderMode: data.orderMode,
      ...(orderColumn ? { orderColumn } : {}),
      excludedColumns,
      featureColumns,
      maxRows,
    },
    split: { calibration, evaluation, sealedHoldout },
    engine: { version: "0.4.7.02", features: 64, seed, configProfile: "cicids_webattacks" },
    protocol: locks,
  };
}

export function preflightIssues(spec, runnerState, operatorAuthConfigured = false) {
  const issues = [];
  const execution = typeof runnerState === "boolean"
    ? {
        configured: runnerState,
        blockers: runnerState ? [] : [{
          code: "RUNNER_NOT_CONFIGURED",
          message: "The run can be preregistered now, but dispatch is blocked until a resource-qualified execution backend is configured.",
        }],
      }
    : runnerState;
  if (!spec.dataset.expectedSha256) {
    issues.push({
      severity: "notice",
      code: "DIGEST_SEALED_BY_RUNNER",
      message: "The resource-qualified runner will hash the exact downloaded file and add that digest to the immutable run manifest.",
    });
  }
  if (spec.dataContract.featureColumns.length === 0) {
    issues.push({
      severity: "warning",
      code: "AUTO_NUMERIC_FEATURES",
      message: "The runner will select numeric columns after removing the label, order, excluded, and label-like columns, then record the resolved schema.",
    });
  }
  if (spec.dataContract.orderMode === "source") {
    issues.push({
      severity: "notice",
      code: "SOURCE_ORDER_LOCKED",
      message: "Rows will remain in the exact order delivered by the pinned Kaggle file; no shuffle or balancing is permitted.",
    });
  }
  issues.push({
    severity: "notice",
    code: "RESOURCE_QUALIFIED_PROFILE",
    message: "The full Eidos 0.4.7.02 code path will use the locked small CPU engineering profile: 256 reservoir units and 2,048 hippocampal dimensions. This changes no proof gate.",
  });
  for (const blocker of execution?.blockers || []) {
    issues.push({ severity: "blocker", code: blocker.code, message: blocker.message });
  }
  if (!execution?.configured && !(execution?.blockers || []).length) {
    issues.push({
      severity: "blocker",
      code: "RUNNER_NOT_CONFIGURED",
      message: "The run can be preregistered now, but dispatch is blocked until a resource-qualified execution backend is configured.",
    });
  }
  if (!operatorAuthConfigured) {
    issues.push({
      severity: "blocker",
      code: "OPERATOR_AUTH_NOT_CONFIGURED",
      message: "Dispatch stays closed until EIDOS_OPERATOR_TOKEN is configured on Vercel; this prevents a public page from starting expensive engine jobs.",
    });
  }
  return issues;
}

export function mapKaggleResult(item) {
  const ref = typeof item?.ref === "string" ? item.ref : typeof item?.datasetRef === "string" ? item.datasetRef : "";
  if (!/^[^/]+\/[^/]+$/.test(ref)) return null;
  const versionCandidate = item.currentVersionNumber ?? item.versionNumber ?? item.currentVersion?.versionNumber;
  const version = Number.isSafeInteger(Number(versionCandidate)) && Number(versionCandidate) > 0 ? Number(versionCandidate) : null;
  return {
    ref,
    title: String(item.title || item.name || ref),
    subtitle: String(item.subtitle || item.description || "Public Kaggle dataset").slice(0, 180),
    version,
    totalBytes: Number.isFinite(Number(item.totalBytes)) ? Number(item.totalBytes) : null,
    usabilityRating: Number.isFinite(Number(item.usabilityRating)) ? Number(item.usabilityRating) : null,
    source: "kaggle",
    url: `https://www.kaggle.com/datasets/${ref}`,
  };
}

export function cloneDefaultExperiment() {
  return JSON.parse(JSON.stringify(DEFAULT_EXPERIMENT_SPEC));
}
