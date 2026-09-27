from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .dataset import PreparedDataset


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(scores)
    rank_sum = float(np.sum(ranks[labels == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    tp = np.cumsum(ordered == 1)
    # Integrate recall increments at distinct thresholds. Equal scores must
    # never gain ranking information from source-row order.
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(order) - 1]
    precision = tp[ends] / (ends + 1)
    recall_increments = np.diff(np.r_[0, tp[ends]]) / positives
    return float(np.sum(precision * recall_increments))


def _delays(labels: np.ndarray, predictions: np.ndarray) -> Tuple[float | None, int]:
    delays: List[int] = []
    missed = 0
    index = 0
    while index < len(labels):
        if labels[index] == 0:
            index += 1
            continue
        end = index + 1
        while end < len(labels) and labels[end] == 1:
            end += 1
        hits = np.flatnonzero(predictions[index:end])
        if hits.size:
            delays.append(int(hits[0]))
        else:
            missed += 1
        index = end
    return (float(np.mean(delays)) if delays else None), missed


def evaluate_frozen_predictions(step_rows: Sequence[Dict[str, Any]], dataset: PreparedDataset) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    start = dataset.calibration_rows
    end = start + dataset.evaluation_rows
    labels_by_step = {
        start + index: int(label)
        for index, label in enumerate(dataset.label_vault.evaluation_labels)
    }
    aligned = {}
    for row in step_rows:
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise RuntimeError("INVALID_PREDICTION_STEP: engine steps must be integers")
        if step < 0 or step >= end:
            raise RuntimeError("PREDICTION_OUTSIDE_ENGINE_PARTITION: unexpected or held-out step")
        if step < start:
            continue
        if step in aligned:
            raise RuntimeError("DUPLICATE_EVALUATION_PREDICTION")
        try:
            score = float(row["z"])
            threshold = float(row["z_thresh_eff"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("INVALID_EVALUATION_PREDICTION") from exc
        if not np.isfinite(score) or not np.isfinite(threshold):
            raise RuntimeError("NONFINITE_EVALUATION_PREDICTION")
        if "is_surprise" in row and (type(row["is_surprise"]) is not bool or row["is_surprise"] != bool(score >= threshold)):
            raise RuntimeError("PREDICTION_DECISION_MISMATCH")
        aligned[int(step)] = (row, score, threshold)
    if len(aligned) != dataset.evaluation_rows:
        raise RuntimeError(f"INCOMPLETE_EVALUATION_PREDICTIONS: expected {dataset.evaluation_rows}, received {len(aligned)}")

    rows = []
    labels = []
    scores = []
    thresholds = []
    for step in range(start, end):
        row, score, threshold = aligned[step]
        labels.append(labels_by_step[step])
        scores.append(score)
        thresholds.append(threshold)
        rows.append({
            "step": step,
            "source_row_index": int(dataset.label_vault.evaluation_source_rows[step - start]),
            "z": score,
            "z_threshold": threshold,
            "is_surprise": bool(score >= threshold),
            "status": str(row.get("status", "")),
        })
    if not rows:
        raise RuntimeError("full engine returned no evaluation-aligned prediction rows")
    y = np.asarray(labels, dtype=np.uint8)
    score_values = np.asarray(scores, dtype=np.float64)
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    predictions = score_values >= threshold_values
    tp = int(np.sum(predictions & (y == 1)))
    fp = int(np.sum(predictions & (y == 0)))
    fn = int(np.sum((~predictions) & (y == 1)))
    tn = int(np.sum((~predictions) & (y == 0)))
    delay, missed_windows = _delays(y, predictions)
    metrics = {
        "schema": "eidos.sentinel-runner.metrics.v0.2",
        "evidence_class": "REAL_DATA_ENGINEERING",
        "proof_verdict": "BLOCKED_RESOURCE_BEFORE_HELDOUT",
        "gates_advanced": 0,
        "evaluation_rows_expected": dataset.evaluation_rows,
        "evaluation_rows_scored": len(rows),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "roc_auc": roc_auc(y, score_values),
        "average_precision": average_precision(y, score_values),
        "mean_detection_delay_frames": delay,
        "missed_attack_windows": missed_windows,
        "labels_unsealed_after_prediction_freeze": True,
        "heldout_evaluated": False,
        "prediction_coverage_complete": True,
        "positive_rows": int(np.sum(y == 1)),
        "negative_rows": int(np.sum(y == 0)),
        "order_basis": dataset.receipt["order"]["mode"],
        "limitations": [
            "Engineering evaluation only; held-out generalization is untested.",
            "Detection delay counts consecutive ordered rows, not elapsed time.",
            *(["Source-file order is not independently verified chronology."] if dataset.receipt["order"]["mode"] == "source" else []),
            *(["Only one class is present; class-separation performance cannot be established."] if len(np.unique(y)) < 2 else []),
        ],
    }
    return metrics, rows
