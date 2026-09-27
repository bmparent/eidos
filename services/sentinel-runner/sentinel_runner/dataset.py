from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .spec import ExperimentSpec


LABEL_TOKENS = ("label", "class", "target", "attack", "benign", "malicious", "outcome", "groundtruth")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_label_like(value: str) -> bool:
    normalized = _normalized_name(value).replace("_", "")
    return any(token in normalized for token in LABEL_TOKENS)


def _resolve_downloaded_file(root: Path, requested: str) -> Path:
    expected = PurePosixPath(requested)
    if root.is_file():
        if root.name != expected.name:
            raise FileNotFoundError(f"Kaggle returned {root.name!r}, not the locked file {requested!r}")
        return root.resolve()
    candidate = root.joinpath(*expected.parts)
    if not candidate.is_file():
        raise FileNotFoundError(f"locked Kaggle file was not downloaded: {requested}")
    return candidate.resolve()


def download_pinned_kaggle_file(spec: ExperimentSpec, output_dir: Path) -> Path:
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise RuntimeError("kagglehub is required on the resource-qualified runner") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = f"{spec.dataset.ref}/versions/{spec.dataset.version}"
    downloaded = Path(kagglehub.dataset_download(handle, path=spec.dataset.file, output_dir=str(output_dir)))
    return _resolve_downloaded_file(downloaded, spec.dataset.file)


def read_tabular(path: Path, max_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, nrows=max_rows, low_memory=False)
    elif suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t", nrows=max_rows, low_memory=False)
    elif suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        batches = []
        remaining = max_rows
        for batch in pq.ParquetFile(path).iter_batches(batch_size=min(max_rows, 8192)):
            selected = batch.slice(0, remaining)
            batches.append(selected)
            remaining -= len(selected)
            if remaining <= 0:
                break
        frame = pa.Table.from_batches(batches).to_pandas() if batches else pd.DataFrame()
    elif suffix in {".feather", ".ftr"}:
        frame = pd.read_feather(path).head(max_rows)
    else:
        raise ValueError(f"unsupported locked dataset extension: {suffix}")
    if frame.empty:
        raise ValueError("locked dataset file contains no rows")
    return frame


@dataclass(frozen=True)
class LabelVault:
    evaluation_labels: np.ndarray
    evaluation_source_rows: np.ndarray
    holdout_commitment: str
    holdout_rows: int


@dataclass(frozen=True)
class PreparedDataset:
    frames: np.ndarray
    calibration_rows: int
    evaluation_rows: int
    source_rows: np.ndarray
    feature_columns: Tuple[str, ...]
    label_vault: LabelVault
    receipt: Dict[str, Any]

    def make_gen_factory(self) -> Callable[[], Iterable[Tuple[np.ndarray, Dict[str, Any]]]]:
        frames = self.frames
        rows = self.source_rows
        dataset_receipt = self.receipt

        def _gen() -> Iterable[Tuple[np.ndarray, Dict[str, Any]]]:
            for index in range(frames.shape[0]):
                yield frames[index], {
                    "kind": "sentinel_lab_real_data",
                    "dataset_ref": dataset_receipt["dataset"]["ref"],
                    "dataset_version": dataset_receipt["dataset"]["version"],
                    "dataset_file_sha256": dataset_receipt["dataset"]["file_sha256"],
                    "source_row_index": int(rows[index]),
                }

        return _gen


def _projection(values: np.ndarray, target: int, seed: int) -> Tuple[np.ndarray, str]:
    width = values.shape[1]
    if width == target:
        return values.astype(np.float64, copy=False), "identity"
    if width < target:
        return np.pad(values, ((0, 0), (0, target - width)), mode="constant"), "zero_pad"
    rng = np.random.RandomState(seed + width)
    matrix = rng.randn(width, target).astype(np.float64) / math.sqrt(width)
    return values @ matrix, "seeded_gaussian"


def _split_counts(rows: int, calibration: float, evaluation: float) -> Tuple[int, int, int]:
    calibration_rows = int(math.floor(rows * calibration))
    evaluation_rows = int(math.floor(rows * evaluation))
    holdout_rows = rows - calibration_rows - evaluation_rows
    if min(calibration_rows, evaluation_rows, holdout_rows) < 100:
        raise ValueError("each split must contain at least 100 rows")
    return calibration_rows, evaluation_rows, holdout_rows


def _label_commitment(file_sha256: str, source_rows: Sequence[int], labels: np.ndarray) -> str:
    payload = {
        "file_sha256": file_sha256,
        "source_rows": [int(value) for value in source_rows],
        "labels_sha256": hashlib.sha256(labels.astype(np.uint8).tobytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def prepare_dataframe(frame: pd.DataFrame, spec: ExperimentSpec, *, file_sha256: str, source_path: str = "fixture") -> PreparedDataset:
    contract = spec.data_contract
    if contract.label_column not in frame.columns:
        raise ValueError(f"locked label column not found: {contract.label_column}")
    if contract.order_mode == "column":
        assert contract.order_column is not None
        if contract.order_column not in frame.columns:
            raise ValueError(f"locked order column not found: {contract.order_column}")
        if frame[contract.order_column].isna().any():
            raise ValueError("order column contains missing values")
        frame = frame.sort_values(contract.order_column, kind="mergesort")
    frame = frame.head(contract.max_rows).copy()
    source_rows = frame.index.to_numpy(dtype=np.int64, copy=True)

    raw_labels = frame[contract.label_column]
    if raw_labels.isna().any() or raw_labels.astype(str).str.strip().eq("").any():
        raise ValueError("label vault contains missing labels")
    negatives = {value.casefold().strip() for value in contract.negative_labels}
    labels = np.asarray([0 if str(value).casefold().strip() in negatives else 1 for value in raw_labels], dtype=np.uint8)

    excluded = set(contract.excluded_columns) | {contract.label_column}
    if contract.order_column:
        excluded.add(contract.order_column)
    missing_exclusions = sorted(name for name in contract.excluded_columns if name not in frame.columns)

    if contract.feature_columns:
        missing = [name for name in contract.feature_columns if name not in frame.columns]
        if missing:
            raise ValueError(f"locked feature columns not found: {', '.join(missing)}")
        selected = list(contract.feature_columns)
    else:
        selected = []
        for name in frame.columns:
            if name in excluded or _is_label_like(str(name)):
                continue
            converted = pd.to_numeric(frame[name], errors="coerce")
            if float(converted.notna().mean()) >= 0.90:
                selected.append(str(name))
    if not selected:
        raise ValueError("no safe numeric feature columns remain after label isolation")
    suspicious = [name for name in selected if name in excluded or _is_label_like(name)]
    if suspicious:
        raise ValueError(f"label-like or excluded columns cannot enter the engine: {', '.join(suspicious)}")

    numeric = frame[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    rows = len(numeric)
    calibration_rows, evaluation_rows, holdout_rows = _split_counts(rows, spec.split.calibration, spec.split.evaluation)
    evaluation_end = calibration_rows + evaluation_rows
    calibration_values = numeric.iloc[:calibration_rows].to_numpy(dtype=np.float64)
    medians = np.nanmedian(calibration_values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    calibration_filled = np.where(np.isnan(calibration_values), medians, calibration_values)
    means = np.mean(calibration_filled, axis=0)
    scales = np.std(calibration_filled, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)

    engine_values = numeric.iloc[:evaluation_end].to_numpy(dtype=np.float64)
    engine_values = np.where(np.isnan(engine_values), medians, engine_values)
    engine_values = np.nan_to_num((engine_values - means) / scales, nan=0.0, posinf=0.0, neginf=0.0)
    frames, projection_mode = _projection(engine_values, 64, spec.engine.seed)

    evaluation_labels = labels[calibration_rows:evaluation_end].copy()
    evaluation_source_rows = source_rows[calibration_rows:evaluation_end].copy()
    holdout_labels = labels[evaluation_end:].copy()
    holdout_source_rows = source_rows[evaluation_end:].copy()
    commitment = _label_commitment(file_sha256, holdout_source_rows, holdout_labels)
    receipt = {
        "schema": "eidos.sentinel-runner.dataset-receipt.v0.2",
        "dataset": {
            "provider": "kaggle",
            "ref": spec.dataset.ref,
            "version": spec.dataset.version,
            "file": spec.dataset.file,
            "source_path": source_path,
            "file_sha256": file_sha256,
        },
        "rows": {
            "total": rows,
            "calibration": calibration_rows,
            "evaluation": evaluation_rows,
            "sealed_holdout": holdout_rows,
            "sent_to_engine": int(frames.shape[0]),
        },
        "order": {"mode": contract.order_mode, "column": contract.order_column, "shuffle": False},
        "features": {
            "source_columns": selected,
            "source_width": len(selected),
            "engine_width": 64,
            "projection": projection_mode,
            "normalization_fit": "calibration_only",
            "missing_declared_exclusions": missing_exclusions,
        },
        "label_isolation": {
            "label_column": contract.label_column,
            "engine_metadata_contains_labels": False,
            "engine_metadata_contains_split_membership": False,
            "evaluation_unseal": "after_prediction_freeze",
            "heldout_sent_to_engine": False,
            "heldout_commitment_sha256": commitment,
        },
        "transform_receipts": {
            "median_sha256": hashlib.sha256(medians.astype(np.float64).tobytes()).hexdigest(),
            "mean_sha256": hashlib.sha256(means.astype(np.float64).tobytes()).hexdigest(),
            "scale_sha256": hashlib.sha256(scales.astype(np.float64).tobytes()).hexdigest(),
        },
    }
    return PreparedDataset(
        frames=frames,
        calibration_rows=calibration_rows,
        evaluation_rows=evaluation_rows,
        source_rows=source_rows[:evaluation_end].copy(),
        feature_columns=tuple(selected),
        label_vault=LabelVault(evaluation_labels, evaluation_source_rows, commitment, holdout_rows),
        receipt=receipt,
    )


def prepare_kaggle_dataset(spec: ExperimentSpec, input_dir: Path) -> PreparedDataset:
    path = download_pinned_kaggle_file(spec, input_dir)
    digest = sha256_file(path)
    if spec.dataset.expected_sha256 and digest != spec.dataset.expected_sha256:
        raise ValueError(f"DATASET_DIGEST_MISMATCH: expected {spec.dataset.expected_sha256}, observed {digest}")
    frame = read_tabular(path, spec.data_contract.max_rows)
    return prepare_dataframe(frame, spec, file_sha256=digest, source_path=str(path))
