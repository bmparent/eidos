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
    precision = tp / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision[ordered == 1]) / positives)


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
    rows = []
    labels = []
    scores = []
    thresholds = []
    for row in step_rows:
        try:
            step = int(row.get("step"))
            score = float(row.get("z"))
            threshold = float(row.get("z_thresh_eff"))
        except (TypeError, ValueError):
            continue
        if step < start or step >= end or step not in labels_by_step or not np.isfinite(score) or not np.isfinite(threshold):
            continue
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
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "roc_auc": roc_auc(y, score_values),
        "average_precision": average_precision(y, score_values),
        "mean_detection_delay_frames": delay,
        "missed_attack_windows": missed_windows,
        "labels_unsealed_after_prediction_freeze": True,
        "heldout_evaluated": False,
    }
    return metrics, rows
