from __future__ import annotations

import hashlib
import json
import re
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .profiles import EXECUTION_PROFILES


EXPERIMENT_SCHEMA = "eidos.sentinel-lab.experiment.v0.2"
REQUEST_SCHEMA = "eidos.sentinel-runner.request.v0.2"
PROOF_VERDICT = "BLOCKED_RESOURCE_BEFORE_HELDOUT"
DATASET_REF = re.compile(r"^[a-z0-9][a-z0-9_-]{0,38}/[a-z0-9][a-z0-9._-]{0,79}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _keys(value: Mapping[str, Any], allowed: Tuple[str, ...], path: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{path} contains unsupported fields: {', '.join(unknown)}")


def _text(value: Any, path: str, max_length: int = 240) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} is required")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise ValueError(f"{path} is too long")
    return normalized


def _text_list(value: Any, path: str, max_items: int = 128) -> Tuple[str, ...]:
    if not isinstance(value, list) or len(value) > max_items:
        raise ValueError(f"{path} must be a short string list")
    result = []
    for index, item in enumerate(value):
        normalized = _text(item, f"{path}[{index}]", 160)
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def _dataset_path(value: Any) -> str:
    path = _text(value, "dataset.file", 500).replace("\\", "/")
    parts = path.split("/")
    if path.startswith("/") or not parts or any(part in {"", ".."} for part in parts):
        raise ValueError("dataset.file must be an exact relative path without traversal")
    if not path.lower().endswith((".csv", ".tsv", ".parquet", ".feather", ".ftr")):
        raise ValueError("dataset.file must be CSV, TSV, Parquet, or Feather data")
    return path


@dataclass(frozen=True)
class DatasetSpec:
    ref: str
    version: int
    file: str
    expected_sha256: Optional[str]


@dataclass(frozen=True)
class DataContract:
    label_column: str
    negative_labels: Tuple[str, ...]
    order_mode: str
    order_column: Optional[str]
    excluded_columns: Tuple[str, ...]
    feature_columns: Tuple[str, ...]
    max_rows: int


@dataclass(frozen=True)
class SplitSpec:
    calibration: float
    evaluation: float
    sealed_holdout: float


@dataclass(frozen=True)
class EngineSpec:
    seed: int
    execution_profile: str = "cpu_engineering"


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: DatasetSpec
    data_contract: DataContract
    split: SplitSpec
    engine: EngineSpec

    @classmethod
    def from_dict(cls, value: Any) -> "ExperimentSpec":
        root = _object(value, "experiment")
        _keys(root, ("schema", "evidenceClass", "dataset", "dataContract", "split", "engine", "protocol"), "experiment")
        if root.get("schema") != EXPERIMENT_SCHEMA:
            raise ValueError(f"schema must be {EXPERIMENT_SCHEMA}")
        if root.get("evidenceClass") != "REAL_DATA_ENGINEERING":
            raise ValueError("v0.2 only permits REAL_DATA_ENGINEERING")

        dataset = _object(root.get("dataset"), "dataset")
        _keys(dataset, ("provider", "ref", "version", "file", "expectedSha256"), "dataset")
        if dataset.get("provider") != "kaggle":
            raise ValueError("dataset.provider must be kaggle")
        ref = _text(dataset.get("ref"), "dataset.ref", 120).lower()
        if not DATASET_REF.fullmatch(ref):
            raise ValueError("dataset.ref must use Kaggle owner/dataset form")
        version = dataset.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("dataset.version must pin a positive integer")
        expected_sha = dataset.get("expectedSha256")
        if expected_sha is not None:
            expected_sha = _text(expected_sha, "dataset.expectedSha256", 64).lower()
            if not SHA256.fullmatch(expected_sha):
                raise ValueError("dataset.expectedSha256 must be a SHA-256 digest")
        dataset_spec = DatasetSpec(ref=ref, version=version, file=_dataset_path(dataset.get("file")), expected_sha256=expected_sha)

        contract = _object(root.get("dataContract"), "dataContract")
        _keys(contract, ("labelColumn", "negativeLabels", "orderMode", "orderColumn", "excludedColumns", "featureColumns", "maxRows"), "dataContract")
        order_mode = contract.get("orderMode")
        if order_mode not in {"source", "column"}:
            raise ValueError("dataContract.orderMode must be source or column")
        order_column = _text(contract.get("orderColumn"), "dataContract.orderColumn", 160) if order_mode == "column" else None
        negative_labels = _text_list(contract.get("negativeLabels"), "dataContract.negativeLabels", 32)
        if not negative_labels:
            raise ValueError("dataContract.negativeLabels cannot be empty")
        max_rows = contract.get("maxRows")
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1_000 <= max_rows <= 2_000_000:
            raise ValueError("dataContract.maxRows must be between 1,000 and 2,000,000")
        label_column = _text(contract.get("labelColumn"), "dataContract.labelColumn", 160)
        feature_columns = _text_list(contract.get("featureColumns"), "dataContract.featureColumns")
        if label_column in feature_columns:
            raise ValueError("the label column cannot be an engine feature")
        data_contract = DataContract(
            label_column=label_column,
            negative_labels=negative_labels,
            order_mode=order_mode,
            order_column=order_column,
            excluded_columns=_text_list(contract.get("excludedColumns"), "dataContract.excludedColumns"),
            feature_columns=feature_columns,
            max_rows=max_rows,
        )

        split = _object(root.get("split"), "split")
        _keys(split, ("calibration", "evaluation", "sealedHoldout"), "split")
        raw_values = (split.get("calibration"), split.get("evaluation"), split.get("sealedHoldout"))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_values):
            raise ValueError("split fractions must be finite numbers")
        values = tuple(float(value) for value in raw_values)
        if any(not math.isfinite(value) or value < 0.1 or value > 0.8 for value in values) or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split fractions must each be 0.1..0.8 and total 1.0")
        split_spec = SplitSpec(*values)

        engine = _object(root.get("engine"), "engine")
        _keys(engine, ("version", "features", "seed", "configProfile", "executionProfile"), "engine")
        if engine.get("version") != "0.4.7.02" or engine.get("features") != 64 or engine.get("configProfile") != "cicids_webattacks":
            raise ValueError("engine version, feature dimension, and profile must remain locked")
        seed = engine.get("seed")
        if type(seed) is not int or seed not in {0, 1}:
            raise ValueError("real-data engineering seeds are restricted to 0 and 1")
        execution_profile = engine.get("executionProfile", "cpu_engineering")
        if not isinstance(execution_profile, str) or execution_profile not in EXECUTION_PROFILES:
            raise ValueError("unsupported engine execution profile")

        protocol = _object(root.get("protocol"), "protocol")
        expected_protocol = {
            "labelPolicy": "sealed_until_prediction_freeze",
            "normalization": "calibration_only_zscore",
            "projection": "seeded_gaussian_or_pad",
            "heldoutPolicy": "exclude_from_engineering_run",
            "proofVerdict": PROOF_VERDICT,
        }
        if dict(protocol) != expected_protocol:
            raise ValueError("protocol safety locks do not match Sentinel Lab v0.2")
        return cls(dataset=dataset_spec, data_contract=data_contract, split=split_spec, engine=EngineSpec(seed=seed, execution_profile=execution_profile))

    def to_dict(self) -> Dict[str, Any]:
        dataset: Dict[str, Any] = {
            "provider": "kaggle",
            "ref": self.dataset.ref,
            "version": self.dataset.version,
            "file": self.dataset.file,
        }
        if self.dataset.expected_sha256:
            dataset["expectedSha256"] = self.dataset.expected_sha256
        contract: Dict[str, Any] = {
            "labelColumn": self.data_contract.label_column,
            "negativeLabels": list(self.data_contract.negative_labels),
            "orderMode": self.data_contract.order_mode,
            "excludedColumns": list(self.data_contract.excluded_columns),
            "featureColumns": list(self.data_contract.feature_columns),
            "maxRows": self.data_contract.max_rows,
        }
        if self.data_contract.order_column:
            contract["orderColumn"] = self.data_contract.order_column
        return {
            "schema": EXPERIMENT_SCHEMA,
            "evidenceClass": "REAL_DATA_ENGINEERING",
            "dataset": dataset,
            "dataContract": contract,
            "split": {
                "calibration": self.split.calibration,
                "evaluation": self.split.evaluation,
                "sealedHoldout": self.split.sealed_holdout,
            },
            "engine": {"version": "0.4.7.02", "features": 64, "seed": self.engine.seed, "configProfile": "cicids_webattacks",
                       **({"executionProfile": self.engine.execution_profile} if self.engine.execution_profile != "cpu_engineering" else {})},
            "protocol": {
                "labelPolicy": "sealed_until_prediction_freeze",
                "normalization": "calibration_only_zscore",
                "projection": "seeded_gaussian_or_pad",
                "heldoutPolicy": "exclude_from_engineering_run",
                "proofVerdict": PROOF_VERDICT,
            },
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def lock_digest(spec: ExperimentSpec) -> str:
    return hashlib.sha256(canonical_json(spec.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentEnvelope:
    lock_digest: str
    spec: ExperimentSpec

    @classmethod
    def from_dict(cls, value: Any) -> "ExperimentEnvelope":
        root = _object(value, "request")
        _keys(root, ("schema", "lockDigest", "spec"), "request")
        if root.get("schema") != REQUEST_SCHEMA:
            raise ValueError(f"request.schema must be {REQUEST_SCHEMA}")
        supplied = _text(root.get("lockDigest"), "request.lockDigest", 64).lower()
        if not SHA256.fullmatch(supplied):
            raise ValueError("request.lockDigest must be a SHA-256 digest")
        spec = ExperimentSpec.from_dict(root.get("spec"))
        observed = lock_digest(spec)
        if supplied != observed:
            raise ValueError("RUN_LOCK_MISMATCH")
        return cls(lock_digest=supplied, spec=spec)
