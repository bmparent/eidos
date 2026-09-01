"""Fair, missing-value-safe metrics for EIDOS-GP-v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import lzma
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    event_precision: float | None
    event_recall: float | None
    f2: float | None
    false_negatives: int | None
    false_positives: int | None
    fp_per_10k: float | None
    first_detection_delay: float | None
    event_coverage: float | None
    apr: float | None
    normal_rmse: float | None
    anomaly_rmse: float | None
    nbc: float | None
    evidence_completeness: float | None
    replay_success: float | None
    memory_utility: float | None
    normalized_delay: float | None
    uncertainty: float | None
    runtime_seconds: float | None = None
    p95_latency_ms: float | None = None
    peak_memory_bytes: int | None = None
    crash_count: int = 0
    nonfinite_count: int = 0
    status: str = "OK"
    skip_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def value_vector(self) -> dict[str, float | None]:
        return {
            "F2": self.f2,
            "APR": self.apr,
            "EC": self.evidence_completeness,
            "REP": self.replay_success,
            "MU": self.memory_utility,
            "-NBC": None if self.nbc is None else -self.nbc,
            "-NDL": None if self.normalized_delay is None else -self.normalized_delay,
            "-FPR": None if self.fp_per_10k is None else -self.fp_per_10k,
            "-U": None if self.uncertainty is None else -self.uncertainty,
        }


def contiguous_events(mask: Sequence[bool] | np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    events: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(values):
        if active and start is None:
            start = index
        elif not active and start is not None:
            events.append((start, index - 1))
            start = None
    if start is not None:
        events.append((start, values.size - 1))
    return events


def overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def detector_metrics(alerts: Sequence[bool], labels: Sequence[int]) -> dict[str, Any]:
    alert_mask = np.asarray(alerts, dtype=bool)
    truth_mask = np.asarray(labels, dtype=bool)
    if alert_mask.shape != truth_mask.shape:
        raise ValueError("alerts and labels must have the same shape")
    predicted = contiguous_events(alert_mask)
    truth = contiguous_events(truth_mask)
    truth_hits = [any(overlap(event, candidate) for candidate in predicted) for event in truth]
    pred_hits = [any(overlap(event, target) for target in truth) for event in predicted]
    tp = int(sum(truth_hits))
    fn = len(truth) - tp
    fp = len(predicted) - int(sum(pred_hits))
    precision = tp / (tp + fp) if tp + fp else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else 1.0
    beta2 = 4.0
    f2 = (1.0 + beta2) * precision * recall / (beta2 * precision + recall) if precision + recall else 0.0
    nominal_frames = int((~truth_mask).sum())
    fp_per_10k = fp * 10000.0 / max(nominal_frames, 1)
    delays: list[int] = []
    coverages: list[float] = []
    for target in truth:
        indices = np.flatnonzero(alert_mask[target[0] : target[1] + 1])
        if indices.size:
            delays.append(int(indices[0]))
        covered = int(alert_mask[target[0] : target[1] + 1].sum())
        coverages.append(covered / max(target[1] - target[0] + 1, 1))
    return {
        "event_precision": float(precision),
        "event_recall": float(recall),
        "f2": float(f2),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "fp_per_10k": float(fp_per_10k),
        "first_detection_delay": None if not delays else float(np.mean(delays)),
        "event_coverage": 1.0 if not truth else float(np.mean(coverages)),
        "normalized_delay": None if not delays else float(np.mean(delays) / max(len(labels), 1)),
    }


def anomaly_preservation(fidelities: Sequence[str], labels: Sequence[int]) -> float:
    preserving = {"structured_residual", "raw_frame_plus_full_context", "anomaly_capsule"}
    truth = np.asarray(labels, dtype=bool)
    if not truth.any():
        return 1.0
    kept = np.asarray([value in preserving for value in fidelities], dtype=bool)
    if kept.shape != truth.shape:
        raise ValueError("fidelity and label lengths differ")
    return float(kept[truth].mean())


def split_rmse(original: np.ndarray, reconstructed: np.ndarray, labels: Sequence[int]) -> tuple[float, float]:
    x = np.asarray(original, dtype=np.float64)
    y = np.asarray(reconstructed, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    if x.shape != y.shape or x.shape[0] != truth.size:
        raise ValueError("reconstruction shapes do not align")
    per_frame = np.sqrt(np.mean((x - y) ** 2, axis=1))
    normal = float(per_frame[~truth].mean()) if (~truth).any() else 0.0
    anomaly = float(per_frame[truth].mean()) if truth.any() else 0.0
    return normal, anomaly


REQUIRED_EVIDENCE_FIELDS = (
    "observation",
    "score",
    "threshold",
    "source_range",
    "selected_representation",
    "uncertainty",
    "next_action",
    "config_hash",
    "code_commit",
    "replay_command",
)


def evidence_completeness(wrapper: Mapping[str, Any]) -> float:
    present = 0
    for field in REQUIRED_EVIDENCE_FIELDS:
        value = wrapper.get(field)
        if value is not None and value != "" and value != [] and value != {}:
            present += 1
    if wrapper.get("raw_retained") and not wrapper.get("raw_references"):
        present -= 1
    return float(max(present, 0) / len(REQUIRED_EVIDENCE_FIELDS))


def compression_reference(frames: np.ndarray, method: str) -> dict[str, Any]:
    raw = np.asarray(frames, dtype=np.float64).tobytes(order="C")
    if method == "raw":
        payload = raw
    elif method == "gzip":
        payload = gzip.compress(raw, mtime=0)
    elif method == "lzma":
        payload = lzma.compress(raw)
    elif method == "zstd":
        try:
            import zstandard as zstd  # type: ignore
        except ImportError:
            return {"system": "zstd", "status": "SKIPPED", "reason": "zstandard dependency unavailable", "bytes": None, "nbc": None}
        payload = zstd.ZstdCompressor(level=3).compress(raw)
    else:
        raise ValueError(f"unknown compression reference: {method}")
    return {
        "system": method,
        "status": "OK",
        "reason": None,
        "bytes": len(payload),
        "raw_bytes": len(raw),
        "nbc": len(payload) / max(len(raw), 1),
    }


def pareto_dominates(left: Mapping[str, float | None], right: Mapping[str, float | None]) -> bool:
    keys = tuple(left)
    if set(keys) != set(right):
        raise ValueError("Pareto vectors must have identical components")
    pairs = [(left[key], right[key]) for key in keys]
    if any(a is None or b is None for a, b in pairs):
        return False
    return all(float(a) >= float(b) for a, b in pairs) and any(float(a) > float(b) for a, b in pairs)


def pareto_front(rows: Sequence[Mapping[str, Any]], *, vector_key: str = "value_vector") -> list[int]:
    front: list[int] = []
    for index, row in enumerate(rows):
        vector = row[vector_key]
        if not any(
            other != index and pareto_dominates(rows[other][vector_key], vector)
            for other in range(len(rows))
        ):
            front.append(index)
    return front


def eidos_value_scalar(metrics: MetricResult, weights: Mapping[str, float], penalties: Mapping[str, float], *, epsilon: float = 1e-6) -> float | None:
    benefits = {
        "F2": metrics.f2,
        "APR": metrics.apr,
        "EC": metrics.evidence_completeness,
        "REP": metrics.replay_success,
        "MU": metrics.memory_utility,
    }
    costs = {
        "NBC": metrics.nbc,
        "NDL": metrics.normalized_delay,
        "FPR": metrics.fp_per_10k,
        "U": metrics.uncertainty,
    }
    if any(value is None for value in benefits.values()) or any(value is None for value in costs.values()):
        return None
    total_weight = sum(float(weights[key]) for key in benefits)
    if total_weight <= 0:
        raise ValueError("benefit weights must sum positive")
    log_benefit = sum(
        float(weights[key]) * math.log(epsilon + float(value))
        for key, value in benefits.items()
    ) / total_weight
    penalty = sum(float(penalties[key]) * float(value) for key, value in costs.items())
    return float(math.exp(log_benefit - penalty))


def registered_sensitivity_grid() -> list[dict[str, dict[str, float]]]:
    benefit_sets = [
        {"F2": 2.0, "APR": 2.0, "EC": 1.0, "REP": 1.0, "MU": 1.0},
        {"F2": 1.0, "APR": 1.0, "EC": 1.0, "REP": 1.0, "MU": 2.0},
        {"F2": 1.0, "APR": 2.0, "EC": 2.0, "REP": 2.0, "MU": 1.0},
    ]
    penalty_sets = [
        {"NBC": 1.0, "NDL": 1.0, "FPR": 0.05, "U": 0.5},
        {"NBC": 2.0, "NDL": 0.5, "FPR": 0.1, "U": 1.0},
        {"NBC": 0.5, "NDL": 2.0, "FPR": 0.2, "U": 0.5},
    ]
    return [{"weights": weights, "penalties": penalties} for weights in benefit_sets for penalties in penalty_sets]


def paired_bootstrap(
    full: Sequence[float],
    comparison: Sequence[float],
    *,
    seed: int = 20260901,
    resamples: int = 10000,
) -> dict[str, Any]:
    left = np.asarray(full, dtype=np.float64)
    right = np.asarray(comparison, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("paired bootstrap requires equal non-empty arrays")
    diff = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, diff.size, size=(int(resamples), diff.size))
    means = diff[indices].mean(axis=1)
    # Two-sided, add-one corrected bootstrap probability.  This is retained as
    # a descriptive paired result rather than presented as an independence
    # assumption over frames; the resampling unit supplied here is a whole
    # seed or source window.
    probability_nonpositive = (float(np.count_nonzero(means <= 0.0)) + 1.0) / (means.size + 1.0)
    probability_nonnegative = (float(np.count_nonzero(means >= 0.0)) + 1.0) / (means.size + 1.0)
    return {
        "n": int(diff.size),
        "numerator": int(diff.size),
        "denominator": int(diff.size),
        "mean_difference": float(diff.mean()),
        "median_difference": float(np.median(diff)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "p_value_two_sided": float(min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))),
        "resamples": int(resamples),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted
