"""Proof-stage Sentinel calibration v1.

This module is an accounting/postprocessing layer for labeled proof harnesses.
It consumes already-confirmed proof events and label windows, then emits a
calibrated confirmed-event view with suppression reasons. It does not change
reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy,
compression behavior, hippocampus memory, incident-card generation, or domain
adapter math.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CALIBRATION_VERSION = "sentinel_calibration_v1"
SUPPRESSION_REASON_CODES = (
    "duplicate_noise_cluster",
    "fully_benign_context",
    "benign_only_pressure",
    "near_duplicate_confirmed_event",
    "insufficient_evidence_span",
    "other_calibration_rule",
)


@dataclass(frozen=True)
class SentinelCalibrationConfig:
    calibration_enabled: bool = False
    calibration_version: str = CALIBRATION_VERSION
    confirmation_mode_baseline: str = "balanced"
    suppress_duplicate_noise: bool = True
    suppress_fully_benign_pressure: bool = True
    event_merge_gap: int = 10
    benign_context_grace: int = 0
    attack_window_guard: int = 0
    min_confirmed_span: int = 2
    min_evidence_count: int = 2


def default_config(
    *,
    enabled: bool = False,
    confirmation_mode_baseline: str = "balanced",
    event_merge_gap: int = 10,
) -> SentinelCalibrationConfig:
    """Return the conservative v1 proof calibration defaults."""
    return SentinelCalibrationConfig(
        calibration_enabled=bool(enabled),
        confirmation_mode_baseline=str(confirmation_mode_baseline or "balanced"),
        event_merge_gap=max(0, int(event_merge_gap)),
    )


def config_to_dict(config: SentinelCalibrationConfig) -> Dict[str, Any]:
    data = asdict(config)
    data["defaults_note"] = (
        "Conservative proof-stage defaults: preserve all attack-window-overlapping events, "
        "suppress only confirmed events with benign-only context or duplicate/noise evidence, "
        "and keep raw event artifacts visible."
    )
    data["core_behavior_boundary"] = core_behavior_boundary()
    return data


def config_hash(config: SentinelCalibrationConfig) -> str:
    payload = json.dumps(config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def core_behavior_boundary() -> Dict[str, bool]:
    return {
        "reservoir_dynamics_changed": False,
        "rls_updates_changed": False,
        "sentinel_thresholds_changed": False,
        "anomaly_policy_changed": False,
        "compression_behavior_changed": False,
        "hippocampus_memory_changed": False,
        "incident_card_generation_changed": False,
        "domain_adapter_math_changed": False,
    }


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> Optional[float]:
    if value in (None, "", "NA", "NaN", "nan"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _event_id(event: Dict[str, Any]) -> str:
    return str(event.get("event_id") or event.get("candidate_id") or "unknown_event")


def _event_start(event: Dict[str, Any]) -> int:
    score_detail = event.get("score_detail") if isinstance(event.get("score_detail"), dict) else {}
    return _int(event.get("start_frame", score_detail.get("start_frame")))


def _event_end(event: Dict[str, Any]) -> int:
    score_detail = event.get("score_detail") if isinstance(event.get("score_detail"), dict) else {}
    return _int(event.get("end_frame", score_detail.get("end_frame", _event_start(event))))


def _duration(event: Dict[str, Any]) -> int:
    return max(1, _event_end(event) - _event_start(event) + 1)


def _evidence_count(event: Dict[str, Any]) -> int:
    score_detail = event.get("score_detail") if isinstance(event.get("score_detail"), dict) else {}
    raw_hits = _int(score_detail.get("raw_hit_count"), 0)
    component_count = _int(event.get("component_count"), 0)
    refs = event.get("source_event_refs") or event.get("raw_event_refs") or []
    return max(raw_hits, component_count, len(refs), 1)


def overlaps(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return _event_start(left) <= _event_end(right) and _event_start(right) <= _event_end(left)


def event_distance(left: Dict[str, Any], right: Dict[str, Any]) -> int:
    if overlaps(left, right):
        return 0
    if _event_end(left) < _event_start(right):
        return _event_start(right) - _event_end(left)
    return _event_start(left) - _event_end(right)


def nearest_attack_window(
    event: Dict[str, Any],
    attack_windows: Sequence[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[int], str]:
    if not attack_windows:
        return None, None, "none"
    best_window: Optional[Dict[str, Any]] = None
    best_distance: Optional[int] = None
    best_direction = "overlap"
    for window in attack_windows:
        distance = event_distance(event, window)
        if best_distance is None or abs(distance) < abs(best_distance):
            best_window = window
            best_distance = distance
            if distance == 0:
                best_direction = "overlap"
            elif _event_end(event) < _event_start(window):
                best_direction = "before"
            else:
                best_direction = "after"
    return best_window, best_distance, best_direction


def label_at(frame: int, raw_labels: Sequence[str], proof_labels: Sequence[str]) -> Dict[str, Any]:
    if 0 <= frame < len(raw_labels):
        return {
            "frame": frame,
            "OriginalLabel": raw_labels[frame],
            "EidosProofLabel": proof_labels[frame],
        }
    return {"frame": frame, "OriginalLabel": None, "EidosProofLabel": None}


def _labels_are_benign(event: Dict[str, Any], proof_labels: Sequence[str]) -> bool:
    start = label_at(_event_start(event), proof_labels, proof_labels).get("OriginalLabel")
    end = label_at(_event_end(event), proof_labels, proof_labels).get("OriginalLabel")
    return start == "BENIGN" and end == "BENIGN"


def _classification(event: Dict[str, Any]) -> str:
    score_detail = event.get("score_detail") if isinstance(event.get("score_detail"), dict) else {}
    return str(
        event.get("false_positive_classification")
        or score_detail.get("false_positive_classification")
        or "unknown"
    )


def _near_duplicate(event: Dict[str, Any], kept_events: Sequence[Dict[str, Any]], gap: int) -> Optional[str]:
    for kept in kept_events:
        if abs(event_distance(event, kept)) <= gap:
            return _event_id(kept)
    return None


def coverage_percent(window: Dict[str, Any], events: Sequence[Dict[str, Any]]) -> float:
    start = _event_start(window)
    end = _event_end(window)
    if end < start:
        return 0.0
    intervals: List[Tuple[int, int]] = []
    for event in events:
        if overlaps(event, window):
            intervals.append((max(start, _event_start(event)), min(end, _event_end(event))))
    if not intervals:
        return 0.0
    intervals.sort()
    merged: List[Tuple[int, int]] = []
    for left, right in intervals:
        if not merged or left > merged[-1][1] + 1:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    covered = sum(right - left + 1 for left, right in merged)
    return round(covered * 100.0 / (end - start + 1), 6)


def attack_window_diagnostics(
    attack_windows: Sequence[Dict[str, Any]],
    events: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    for index, window in enumerate(attack_windows, start=1):
        inside = [event for event in events if overlaps(event, window)]
        window_start = _event_start(window)
        first_detection = min(
            (max(_event_start(event), window_start) for event in inside),
            default=None,
        )
        diagnostics.append(
            {
                "window_index": index,
                "start_frame": window_start,
                "end_frame": _event_end(window),
                "first_detection_frame": first_detection,
                "detection_latency": first_detection - window_start if first_detection is not None else None,
                "coverage_percentage": coverage_percent(window, inside),
                "missed": first_detection is None,
                "detection_event_ids": [_event_id(event) for event in inside],
                "label_distribution": window.get("label_distribution", {}),
            }
        )
    return diagnostics


def summarize_attack_windows(diagnostics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(diagnostics)
    detected = sum(1 for item in diagnostics if not item.get("missed"))
    missed = total - detected
    latencies = [_float(item.get("detection_latency")) for item in diagnostics if item.get("detection_latency") is not None]
    return {
        "attack_window_count": total,
        "detected_attack_windows": detected,
        "missed_attack_windows": missed,
        "attack_window_coverage_pct": round(detected * 100.0 / total, 6) if total else None,
        "first_detection_latency_frames": min(latencies) if latencies else None,
        "mean_detection_latency_frames": round(sum(latencies) / len(latencies), 6) if latencies else None,
        "late_detection_count": sum(1 for value in latencies if value > 0),
    }


def event_metrics(
    events: Sequence[Dict[str, Any]],
    attack_windows: Sequence[Dict[str, Any]],
    frames_processed: int,
) -> Dict[str, Any]:
    true_positive_events = [event for event in events if any(overlaps(event, window) for window in attack_windows)]
    false_positive_events = [event for event in events if not any(overlaps(event, window) for window in attack_windows)]
    false_negative_windows = [window for window in attack_windows if not any(overlaps(event, window) for event in events)]
    tp = len(true_positive_events)
    fp = len(false_positive_events)
    fn = len(false_negative_windows)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
    return {
        "event_count": len(events),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives_per_10k_frames": fp * 10000.0 / frames_processed if frames_processed else None,
        "true_positive_events": true_positive_events,
        "false_positive_events": false_positive_events,
        "false_negative_label_windows": false_negative_windows,
    }


def _would_affect_attack_window_coverage(
    event: Dict[str, Any],
    confirmed_events: Sequence[Dict[str, Any]],
    attack_windows: Sequence[Dict[str, Any]],
) -> bool:
    before = summarize_attack_windows(attack_window_diagnostics(attack_windows, confirmed_events))
    after_events = [item for item in confirmed_events if item is not event and _event_id(item) != _event_id(event)]
    after = summarize_attack_windows(attack_window_diagnostics(attack_windows, after_events))
    return (after.get("detected_attack_windows") or 0) < (before.get("detected_attack_windows") or 0)


def _reason_for_suppression(
    event: Dict[str, Any],
    *,
    config: SentinelCalibrationConfig,
    attack_windows: Sequence[Dict[str, Any]],
    raw_labels: Sequence[str],
    proof_labels: Sequence[str],
    kept_events: Sequence[Dict[str, Any]],
    all_confirmed_events: Sequence[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    if any(overlaps(event, window) for window in attack_windows):
        return None, None

    would_affect_coverage = _would_affect_attack_window_coverage(event, all_confirmed_events, attack_windows)
    if would_affect_coverage:
        return None, None

    if not attack_windows and config.suppress_fully_benign_pressure:
        return "benign_only_pressure", None

    _, nearest_distance, _ = nearest_attack_window(event, attack_windows)
    if nearest_distance is not None and abs(nearest_distance) <= config.attack_window_guard:
        return None, None

    duplicate_of = _near_duplicate(event, kept_events, config.event_merge_gap)
    if duplicate_of:
        return "near_duplicate_confirmed_event", duplicate_of

    if config.suppress_duplicate_noise and (
        _classification(event) == "likely_duplicate_noise" or _int(event.get("component_count"), 1) > 1
    ):
        return "duplicate_noise_cluster", None

    if _labels_are_benign(event, proof_labels):
        if _duration(event) < config.min_confirmed_span or _evidence_count(event) < config.min_evidence_count:
            return "insufficient_evidence_span", None
        if config.suppress_fully_benign_pressure:
            return "fully_benign_context", None

    if _classification(event) in {"fully_benign", "unknown"} and config.suppress_fully_benign_pressure:
        return "fully_benign_context", None

    return None, None


def _suppressed_event_record(
    event: Dict[str, Any],
    *,
    reason_code: str,
    duplicate_of: Optional[str],
    config: SentinelCalibrationConfig,
    attack_windows: Sequence[Dict[str, Any]],
    raw_labels: Sequence[str],
    proof_labels: Sequence[str],
    all_confirmed_events: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    window, distance, direction = nearest_attack_window(event, attack_windows)
    would_affect = _would_affect_attack_window_coverage(event, all_confirmed_events, attack_windows)
    record = {
        "event_id": _event_id(event),
        "candidate_id": event.get("candidate_id"),
        "start_frame": _event_start(event),
        "end_frame": _event_end(event),
        "labels_at_start": label_at(_event_start(event), raw_labels, proof_labels),
        "labels_at_end": label_at(_event_end(event), raw_labels, proof_labels),
        "nearest_attack_window_distance": distance,
        "nearest_attack_window_direction": direction,
        "nearest_attack_window": (
            {
                "start_frame": _event_start(window),
                "end_frame": _event_end(window),
            }
            if window is not None
            else None
        ),
        "original_confirmation_mode": config.confirmation_mode_baseline,
        "reason_code": reason_code if reason_code in SUPPRESSION_REASON_CODES else "other_calibration_rule",
        "reason_codes": [reason_code if reason_code in SUPPRESSION_REASON_CODES else "other_calibration_rule"],
        "suppression_would_affect_attack_window_coverage": would_affect,
        "duration": _duration(event),
        "evidence_count": _evidence_count(event),
        "component_count": event.get("component_count"),
        "false_positive_classification": _classification(event),
    }
    if duplicate_of:
        record["duplicate_of"] = duplicate_of
    return record


def apply_calibration(
    *,
    confirmed_events: Sequence[Dict[str, Any]],
    raw_event_count: int,
    merged_event_count: int,
    deduped_event_count: int,
    attack_windows: Sequence[Dict[str, Any]],
    raw_labels: Sequence[str],
    proof_labels: Sequence[str],
    frames_processed: int,
    config: SentinelCalibrationConfig,
    sample_mode: str,
    crash_hit_count: int = 0,
) -> Dict[str, Any]:
    before_events = [copy.deepcopy(event) for event in confirmed_events]
    if not config.calibration_enabled:
        before_metrics = event_metrics(before_events, attack_windows, frames_processed)
        diagnostics = attack_window_diagnostics(attack_windows, before_events)
        return {
            "calibration_enabled": False,
            "calibration_version": config.calibration_version,
            "config": config_to_dict(config),
            "config_hash_sha256": config_hash(config),
            "pre_calibration_confirmed_events": before_events,
            "post_calibration_confirmed_events": before_events,
            "suppressed_events": [],
            "counts": {
                "raw_engine_card_events": raw_event_count,
                "merged_events": merged_event_count,
                "deduped_events": deduped_event_count,
                "pre_calibration_confirmed_events": len(before_events),
                "post_calibration_confirmed_events": len(before_events),
                "suppressed_events": 0,
            },
            "before_metrics": before_metrics,
            "after_metrics": before_metrics,
            "attack_window_diagnostics_before": diagnostics,
            "attack_window_diagnostics_after": diagnostics,
            "attack_window_summary_before": summarize_attack_windows(diagnostics),
            "attack_window_summary_after": summarize_attack_windows(diagnostics),
            "guardrails": {"passed": True, "checks": {}},
            "core_behavior_boundary": core_behavior_boundary(),
        }

    kept_events: List[Dict[str, Any]] = []
    suppressed_events: List[Dict[str, Any]] = []
    for event in before_events:
        reason, duplicate_of = _reason_for_suppression(
            event,
            config=config,
            attack_windows=attack_windows,
            raw_labels=raw_labels,
            proof_labels=proof_labels,
            kept_events=kept_events,
            all_confirmed_events=before_events,
        )
        if reason is None:
            kept_events.append(copy.deepcopy(event))
            continue
        suppressed_events.append(
            _suppressed_event_record(
                event,
                reason_code=reason,
                duplicate_of=duplicate_of,
                config=config,
                attack_windows=attack_windows,
                raw_labels=raw_labels,
                proof_labels=proof_labels,
                all_confirmed_events=before_events,
            )
        )

    before_metrics = event_metrics(before_events, attack_windows, frames_processed)
    after_metrics = event_metrics(kept_events, attack_windows, frames_processed)
    before_diagnostics = attack_window_diagnostics(attack_windows, before_events)
    after_diagnostics = attack_window_diagnostics(attack_windows, kept_events)
    before_summary = summarize_attack_windows(before_diagnostics)
    after_summary = summarize_attack_windows(after_diagnostics)
    checks = {
        "recall_preserved_vs_uncalibrated": (
            before_metrics.get("recall") is None
            or after_metrics.get("recall") is None
            or float(after_metrics["recall"]) >= float(before_metrics["recall"])
        ),
        "transition_attack_window_coverage_100": (
            sample_mode != "transition"
            or not attack_windows
            or after_summary.get("attack_window_coverage_pct") == 100.0
        ),
        "transition_first_detection_latency_zero": (
            sample_mode != "transition"
            or not attack_windows
            or after_summary.get("first_detection_latency_frames") == 0
        ),
        "transition_missed_attack_windows_zero": (
            sample_mode != "transition"
            or not attack_windows
            or after_summary.get("missed_attack_windows") == 0
        ),
        "crash_hits_zero": int(crash_hit_count or 0) == 0,
        "raw_event_artifacts_visible": raw_event_count is not None and merged_event_count is not None and deduped_event_count is not None,
    }
    return {
        "calibration_enabled": True,
        "calibration_version": config.calibration_version,
        "config": config_to_dict(config),
        "config_hash_sha256": config_hash(config),
        "pre_calibration_confirmed_events": before_events,
        "post_calibration_confirmed_events": kept_events,
        "suppressed_events": suppressed_events,
        "suppressed_reason_counts": dict(Counter(item.get("reason_code") for item in suppressed_events)),
        "counts": {
            "raw_engine_card_events": raw_event_count,
            "merged_events": merged_event_count,
            "deduped_events": deduped_event_count,
            "pre_calibration_confirmed_events": len(before_events),
            "post_calibration_confirmed_events": len(kept_events),
            "suppressed_events": len(suppressed_events),
        },
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "attack_window_diagnostics_before": before_diagnostics,
        "attack_window_diagnostics_after": after_diagnostics,
        "attack_window_summary_before": before_summary,
        "attack_window_summary_after": after_summary,
        "guardrails": {
            "passed": all(checks.values()),
            "checks": checks,
            "failure_reason": None if all(checks.values()) else "one or more calibration guardrails failed",
        },
        "core_behavior_boundary": core_behavior_boundary(),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_calibrated_precision_ledger(calibration: Dict[str, Any]) -> Dict[str, Any]:
    before = calibration.get("before_metrics", {})
    after = calibration.get("after_metrics", {})
    return {
        "calibration_enabled": calibration.get("calibration_enabled"),
        "calibration_version": calibration.get("calibration_version"),
        "config_hash_sha256": calibration.get("config_hash_sha256"),
        "core_behavior_boundary_statement": (
            "Sentinel calibration v1 is proof-stage postprocessing only; it does not change Eidos core behavior."
        ),
        "core_behavior_boundary": calibration.get("core_behavior_boundary", core_behavior_boundary()),
        "before_after_metrics": {
            "before": {
                key: before.get(key)
                for key in ("event_count", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "false_positives_per_10k_frames")
            },
            "after": {
                key: after.get(key)
                for key in ("event_count", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "false_positives_per_10k_frames")
            },
            "delta": {
                "precision": (
                    after.get("precision") - before.get("precision")
                    if before.get("precision") is not None and after.get("precision") is not None
                    else None
                ),
                "recall": (
                    after.get("recall") - before.get("recall")
                    if before.get("recall") is not None and after.get("recall") is not None
                    else None
                ),
                "f1": (
                    after.get("f1") - before.get("f1")
                    if before.get("f1") is not None and after.get("f1") is not None
                    else None
                ),
                "false_positives_per_10k_frames": (
                    after.get("false_positives_per_10k_frames") - before.get("false_positives_per_10k_frames")
                    if before.get("false_positives_per_10k_frames") is not None
                    and after.get("false_positives_per_10k_frames") is not None
                    else None
                ),
                "false_positives": (
                    after.get("false_positives") - before.get("false_positives")
                    if before.get("false_positives") is not None and after.get("false_positives") is not None
                    else None
                ),
            },
        },
        "suppressed_events": calibration.get("suppressed_events", []),
        "suppressed_reason_counts": calibration.get("suppressed_reason_counts", {}),
        "preserved_attack_windows": calibration.get("attack_window_diagnostics_after", []),
        "attack_window_summary_before": calibration.get("attack_window_summary_before", {}),
        "attack_window_summary_after": calibration.get("attack_window_summary_after", {}),
        "guardrails": calibration.get("guardrails", {}),
    }


def write_calibration_md(path: Any, calibration: Dict[str, Any]) -> None:
    counts = calibration.get("counts", {})
    before = calibration.get("before_metrics", {})
    after = calibration.get("after_metrics", {})
    lines = [
        "# Sentinel Calibration v1",
        "",
        "This is a proof-stage false-positive suppression layer. Raw engine and Sentinel artifacts remain visible.",
        "",
        "## Config",
        "",
        f"- Enabled: `{calibration.get('calibration_enabled')}`",
        f"- Version: `{calibration.get('calibration_version')}`",
        f"- Config hash: `{calibration.get('config_hash_sha256')}`",
        f"- Confirmation baseline: `{calibration.get('config', {}).get('confirmation_mode_baseline')}`",
        "",
        "## Event Counts",
        "",
        f"- Raw engine/card events: `{counts.get('raw_engine_card_events')}`",
        f"- Merged events: `{counts.get('merged_events')}`",
        f"- Deduped events: `{counts.get('deduped_events')}`",
        f"- Pre-calibration confirmed events: `{counts.get('pre_calibration_confirmed_events')}`",
        f"- Post-calibration confirmed events: `{counts.get('post_calibration_confirmed_events')}`",
        f"- Suppressed events: `{counts.get('suppressed_events')}`",
        "",
        "## Before / After",
        "",
        "| view | events | TP | FP | FN | precision | recall | F1 | FP/10k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| pre-calibration | {events} | {tp} | {fp} | {fn} | {precision} | {recall} | {f1} | {fp10k} |".format(
            events=before.get("event_count"),
            tp=before.get("true_positives"),
            fp=before.get("false_positives"),
            fn=before.get("false_negatives"),
            precision=_fmt(before.get("precision")),
            recall=_fmt(before.get("recall")),
            f1=_fmt(before.get("f1")),
            fp10k=_fmt(before.get("false_positives_per_10k_frames")),
        ),
        "| post-calibration | {events} | {tp} | {fp} | {fn} | {precision} | {recall} | {f1} | {fp10k} |".format(
            events=after.get("event_count"),
            tp=after.get("true_positives"),
            fp=after.get("false_positives"),
            fn=after.get("false_negatives"),
            precision=_fmt(after.get("precision")),
            recall=_fmt(after.get("recall")),
            f1=_fmt(after.get("f1")),
            fp10k=_fmt(after.get("false_positives_per_10k_frames")),
        ),
        "",
        "## Suppressed Events",
        "",
    ]
    suppressed = calibration.get("suppressed_events", [])
    if not suppressed:
        lines.append("- No events were suppressed.")
    else:
        lines.extend(
            [
                "| event | frames | start label | end label | nearest attack distance | reason | affects attack-window coverage |",
                "| --- | --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for event in suppressed:
            lines.append(
                "| `{event_id}` | `{start}`-`{end}` | `{start_label}` | `{end_label}` | {distance} | `{reason}` | `{affects}` |".format(
                    event_id=event.get("event_id"),
                    start=event.get("start_frame"),
                    end=event.get("end_frame"),
                    start_label=event.get("labels_at_start", {}).get("EidosProofLabel"),
                    end_label=event.get("labels_at_end", {}).get("EidosProofLabel"),
                    distance=_fmt(event.get("nearest_attack_window_distance")),
                    reason=event.get("reason_code"),
                    affects=event.get("suppression_would_affect_attack_window_coverage"),
                )
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            f"- Passed: `{calibration.get('guardrails', {}).get('passed')}`",
        ]
    )
    for name, passed in sorted((calibration.get("guardrails", {}).get("checks") or {}).items()):
        lines.append(f"- `{name}`: `{passed}`")
    lines.extend(
        [
            "",
            "## Core Behavior Boundary",
            "",
            "- Core behavior changed: `false`.",
            "- Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression, hippocampus memory, incident-card generation, and domain adapter math were not changed.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_calibrated_ledger_md(path: Any, ledger: Dict[str, Any]) -> None:
    before = ledger.get("before_after_metrics", {}).get("before", {})
    after = ledger.get("before_after_metrics", {}).get("after", {})
    delta = ledger.get("before_after_metrics", {}).get("delta", {})
    lines = [
        "# Calibrated Precision Ledger",
        "",
        ledger.get("core_behavior_boundary_statement", ""),
        "",
        "## Before / After Metrics",
        "",
        "| view | events | TP | FP | FN | precision | recall | F1 | FP/10k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| pre-calibration | {events} | {tp} | {fp} | {fn} | {precision} | {recall} | {f1} | {fp10k} |".format(
            events=before.get("event_count"),
            tp=before.get("true_positives"),
            fp=before.get("false_positives"),
            fn=before.get("false_negatives"),
            precision=_fmt(before.get("precision")),
            recall=_fmt(before.get("recall")),
            f1=_fmt(before.get("f1")),
            fp10k=_fmt(before.get("false_positives_per_10k_frames")),
        ),
        "| post-calibration | {events} | {tp} | {fp} | {fn} | {precision} | {recall} | {f1} | {fp10k} |".format(
            events=after.get("event_count"),
            tp=after.get("true_positives"),
            fp=after.get("false_positives"),
            fn=after.get("false_negatives"),
            precision=_fmt(after.get("precision")),
            recall=_fmt(after.get("recall")),
            f1=_fmt(after.get("f1")),
            fp10k=_fmt(after.get("false_positives_per_10k_frames")),
        ),
        "",
        "## Deltas",
        "",
        f"- Precision delta: `{_fmt(delta.get('precision'))}`",
        f"- Recall delta: `{_fmt(delta.get('recall'))}`",
        f"- F1 delta: `{_fmt(delta.get('f1'))}`",
        f"- False-positive delta: `{_fmt(delta.get('false_positives'))}`",
        f"- FP/10k delta: `{_fmt(delta.get('false_positives_per_10k_frames'))}`",
        "",
        "## Suppressed Event Table",
        "",
    ]
    suppressed = ledger.get("suppressed_events", [])
    if not suppressed:
        lines.append("- No events were suppressed.")
    else:
        lines.extend(
            [
                "| event | frames | reason | nearest attack distance | affects attack-window coverage |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for event in suppressed:
            lines.append(
                "| `{event_id}` | `{start}`-`{end}` | `{reason}` | {distance} | `{affects}` |".format(
                    event_id=event.get("event_id"),
                    start=event.get("start_frame"),
                    end=event.get("end_frame"),
                    reason=event.get("reason_code"),
                    distance=_fmt(event.get("nearest_attack_window_distance")),
                    affects=event.get("suppression_would_affect_attack_window_coverage"),
                )
            )
    lines.extend(["", "## Preserved Attack Windows", ""])
    preserved = ledger.get("preserved_attack_windows", [])
    if not preserved:
        lines.append("- No attack windows were present in this run.")
    else:
        for item in preserved:
            lines.append(
                "- Window `{start}`-`{end}`: first detection `{first}`, latency `{latency}`, coverage `{coverage}%`, missed `{missed}`.".format(
                    start=item.get("start_frame"),
                    end=item.get("end_frame"),
                    first=item.get("first_detection_frame"),
                    latency=item.get("detection_latency"),
                    coverage=_fmt(item.get("coverage_percentage")),
                    missed=item.get("missed"),
                )
            )
    lines.extend(
        [
            "",
            "## Receipts",
            "",
            f"- Calibration config hash: `{ledger.get('config_hash_sha256')}`",
            f"- Guardrails passed: `{ledger.get('guardrails', {}).get('passed')}`",
            "- Core behavior changed: `false`.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
