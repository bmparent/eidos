"""Labeled-domain event confirmation for proof harnesses.

This module is a proof-side accounting layer. It consumes already-emitted raw
Sentinel/incident-card events plus label-aligned frame metadata and produces an
ablatable confirmed-event view. It does not change Eidos reservoir dynamics,
RLS updates, compression, feature projection, Sentinel thresholds, or raw
incident-card generation.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CONFIRMATION_MODES = ("off", "low_noise", "balanced", "high_recall")
SEVERITY_RANK = {"GREEN": 0, "RECOVERY": 1, "AMBER": 2, "RED": 3}


@dataclass(frozen=True)
class ConfirmationThresholds:
    mode: str
    min_raw_hits: int
    min_duration: int
    min_score: float
    event_merge_gap: int
    cooldown_gap: int
    high_severity_rank: int = 3
    boundary_duplicate_gap: int = 12


MODE_DEFAULTS: Dict[str, ConfirmationThresholds] = {
    "off": ConfirmationThresholds(
        mode="off",
        min_raw_hits=1,
        min_duration=1,
        min_score=0.0,
        event_merge_gap=0,
        cooldown_gap=0,
    ),
    "high_recall": ConfirmationThresholds(
        mode="high_recall",
        min_raw_hits=1,
        min_duration=1,
        min_score=1.75,
        event_merge_gap=12,
        cooldown_gap=0,
        boundary_duplicate_gap=18,
    ),
    "balanced": ConfirmationThresholds(
        mode="balanced",
        min_raw_hits=2,
        min_duration=2,
        min_score=3.0,
        event_merge_gap=10,
        cooldown_gap=8,
        boundary_duplicate_gap=12,
    ),
    "low_noise": ConfirmationThresholds(
        mode="low_noise",
        min_raw_hits=2,
        min_duration=3,
        min_score=4.25,
        event_merge_gap=6,
        cooldown_gap=20,
        boundary_duplicate_gap=8,
    ),
}


def get_thresholds(
    mode: str,
    *,
    min_raw_hits: Optional[int] = None,
    min_duration: Optional[int] = None,
    min_score: Optional[float] = None,
    event_merge_gap: Optional[int] = None,
    cooldown_gap: Optional[int] = None,
) -> ConfirmationThresholds:
    if mode not in MODE_DEFAULTS:
        known = ", ".join(CONFIRMATION_MODES)
        raise ValueError(f"unknown event confirmation mode {mode!r}; expected one of: {known}")
    thresholds = MODE_DEFAULTS[mode]
    updates: Dict[str, Any] = {}
    if min_raw_hits is not None:
        updates["min_raw_hits"] = max(1, int(min_raw_hits))
    if min_duration is not None:
        updates["min_duration"] = max(1, int(min_duration))
    if min_score is not None:
        updates["min_score"] = max(0.0, float(min_score))
    if event_merge_gap is not None:
        updates["event_merge_gap"] = max(0, int(event_merge_gap))
    if cooldown_gap is not None:
        updates["cooldown_gap"] = max(0, int(cooldown_gap))
    return replace(thresholds, **updates)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _event_duration(event: Dict[str, Any]) -> int:
    return max(1, _int(event.get("end_frame")) - _int(event.get("start_frame")) + 1)


def _severity_rank(value: Any) -> int:
    return SEVERITY_RANK.get(str(value or "").upper(), 0)


def _highest_severity(values: Iterable[Any]) -> Optional[str]:
    ranked = [str(value).upper() for value in values if value]
    if not ranked:
        return None
    return max(ranked, key=lambda item: SEVERITY_RANK.get(item, 0))


def _overlaps(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return _int(left.get("start_frame")) <= _int(right.get("end_frame")) and _int(right.get("start_frame")) <= _int(left.get("end_frame"))


def _contains(outer: Dict[str, Any], inner: Dict[str, Any]) -> bool:
    return _int(outer.get("start_frame")) <= _int(inner.get("start_frame")) and _int(inner.get("end_frame")) <= _int(outer.get("end_frame"))


def _event_distance(left: Dict[str, Any], right: Dict[str, Any]) -> int:
    if _overlaps(left, right):
        return 0
    if _int(left.get("end_frame")) < _int(right.get("start_frame")):
        return _int(right.get("start_frame")) - _int(left.get("end_frame"))
    return _int(left.get("start_frame")) - _int(right.get("end_frame"))


def _nearest_attack_window(event: Dict[str, Any], attack_windows: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[int], str]:
    if not attack_windows:
        return None, None, "none"
    best_window: Optional[Dict[str, Any]] = None
    best_distance: Optional[int] = None
    direction = "overlap"
    for window in attack_windows:
        distance = _event_distance(event, window)
        if best_distance is None or abs(distance) < abs(best_distance):
            best_window = window
            best_distance = distance
            if distance == 0:
                direction = "overlap"
            elif _int(event.get("end_frame")) < _int(window.get("start_frame")):
                direction = "before"
            else:
                direction = "after"
    return best_window, best_distance, direction


def _overlap_frame_count(event: Dict[str, Any], windows: Sequence[Dict[str, Any]]) -> int:
    start = _int(event.get("start_frame"))
    end = _int(event.get("end_frame"))
    if end < start:
        return 0
    intervals: List[Tuple[int, int]] = []
    for window in windows:
        if not _overlaps(event, window):
            continue
        intervals.append((max(start, _int(window.get("start_frame"))), min(end, _int(window.get("end_frame")))))
    if not intervals:
        return 0
    intervals.sort()
    merged: List[Tuple[int, int]] = []
    for left, right in intervals:
        if not merged or left > merged[-1][1] + 1:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return sum(right - left + 1 for left, right in merged)


def _classify_false_positive(event: Dict[str, Any], attack_windows: Sequence[Dict[str, Any]], boundary_gap: int) -> str:
    window, distance, direction = _nearest_attack_window(event, attack_windows)
    if window is None or distance is None:
        return "fully_benign"
    if distance == 0:
        return "overlap_boundary"
    if direction == "before" and abs(distance) <= boundary_gap:
        return "pre_attack_near_transition"
    if direction == "after" and abs(distance) <= boundary_gap:
        return "post_attack_near_transition"
    if _int(event.get("component_count"), 1) > 1:
        return "likely_duplicate_noise"
    return "fully_benign"


def _component_events(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    components = event.get("component_events") or []
    if not components:
        components = [event]
    return [dict(item) for item in components]


def _source_refs(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for component in _component_events(event):
        refs.append(
            {
                "event_id": component.get("event_id", event.get("event_id")),
                "source": component.get("source", event.get("source")),
                "start_frame": _int(component.get("start_frame", event.get("start_frame"))),
                "end_frame": _int(component.get("end_frame", event.get("end_frame"))),
                "severity": component.get("severity", event.get("severity")),
            }
        )
    return refs


def _top_driver_key(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        for key in ("name", "feature", "driver", "metric", "id"):
            if item.get(key):
                return str(item[key])
        if item:
            return str(sorted(item.items())[0])
    if item not in (None, ""):
        return str(item)
    return None


def _top_driver_consistency(event: Dict[str, Any]) -> Optional[float]:
    drivers: List[str] = []
    for item in event.get("top_drivers") or []:
        key = _top_driver_key(item)
        if key:
            drivers.append(key)
    for component in _component_events(event):
        for item in component.get("top_drivers") or []:
            key = _top_driver_key(item)
            if key:
                drivers.append(key)
    if not drivers:
        return None
    counts = Counter(drivers)
    return max(counts.values()) / len(drivers)


def _step_rows_for_event(step_rows: Sequence[Dict[str, Any]], event: Dict[str, Any]) -> List[Dict[str, Any]]:
    start = _int(event.get("start_frame"))
    end = _int(event.get("end_frame"))
    rows: List[Dict[str, Any]] = []
    for fallback, row in enumerate(step_rows):
        frame = _int(row.get("step"), fallback)
        if start <= frame <= end:
            rows.append(row)
    return rows


def _step_stats(step_rows: Sequence[Dict[str, Any]], event: Dict[str, Any]) -> Dict[str, Any]:
    rows = _step_rows_for_event(step_rows, event)
    z_values = [_float(row.get("z")) for row in rows]
    z_values = [value for value in z_values if value is not None]
    severities = [row.get("status") for row in rows if row.get("status")]
    dominance = [_float(row.get("dominance")) for row in rows]
    dominance = [value for value in dominance if value is not None]
    entropy = [_float(row.get("state_entropy")) for row in rows]
    entropy = [value for value in entropy if value is not None]
    dom_delta = max(dominance) - min(dominance) if len(dominance) >= 2 else None
    entropy_delta = max(entropy) - min(entropy) if len(entropy) >= 2 else None
    return {
        "peak_z": max(z_values) if z_values else None,
        "mean_z": sum(z_values) / len(z_values) if z_values else None,
        "severity_from_rows": _highest_severity(severities),
        "dominance_delta": dom_delta,
        "state_entropy_delta": entropy_delta,
        "step_rows_observed": len(rows),
    }


def _candidate_id(index: int, event: Dict[str, Any]) -> str:
    return f"candidate_{index:03d}_{_int(event.get('start_frame'))}_{_int(event.get('end_frame'))}"


def _merge_candidates(events: Sequence[Dict[str, Any]], merge_gap: int) -> List[Dict[str, Any]]:
    if not events:
        return []
    ordered = sorted((copy.deepcopy(event) for event in events), key=lambda item: (_int(item.get("start_frame")), _int(item.get("end_frame"))))
    merged: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for event in ordered:
        event["start_frame"] = _int(event.get("start_frame"))
        event["end_frame"] = _int(event.get("end_frame"))
        event["source_event_refs"] = _source_refs(event)
        if current is None:
            current = event
            continue
        if _int(event.get("start_frame")) <= _int(current.get("end_frame")) + merge_gap:
            current["end_frame"] = max(_int(current.get("end_frame")), _int(event.get("end_frame")))
            current["duration"] = _event_duration(current)
            current["severity"] = _highest_severity([current.get("severity"), event.get("severity")])
            current["top_drivers"] = (list(current.get("top_drivers") or []) + list(event.get("top_drivers") or []))[:8]
            current["raw_evidence_refs"] = sorted(set(list(current.get("raw_evidence_refs") or []) + list(event.get("raw_evidence_refs") or [])))
            current["source_event_refs"] = list(current.get("source_event_refs") or []) + list(event.get("source_event_refs") or [])
            current["component_count"] = len(current["source_event_refs"])
            current["confirmation_merge_note"] = f"candidate events merged with gap <= {merge_gap}"
        else:
            merged.append(current)
            current = event
    if current is not None:
        merged.append(current)
    for idx, event in enumerate(merged, start=1):
        event["candidate_id"] = _candidate_id(idx, event)
        event["duration"] = _event_duration(event)
        event["component_count"] = max(_int(event.get("component_count"), 1), len(event.get("source_event_refs") or []), 1)
    return merged


def _score_candidate(
    event: Dict[str, Any],
    *,
    label_windows: Sequence[Dict[str, Any]],
    step_rows: Sequence[Dict[str, Any]],
    thresholds: ConfirmationThresholds,
) -> Dict[str, Any]:
    duration = _event_duration(event)
    raw_hits = max(_int(event.get("component_count"), 1), len(_source_refs(event)), 1)
    step_stats = _step_stats(step_rows, event)
    severity = _highest_severity([event.get("severity"), step_stats.get("severity_from_rows")])
    severity_rank = _severity_rank(severity)
    overlap_frames = _overlap_frame_count(event, label_windows)
    overlap_ratio = overlap_frames / duration if duration else 0.0
    window, distance, direction = _nearest_attack_window(event, label_windows)
    fp_class = _classify_false_positive(event, label_windows, thresholds.boundary_duplicate_gap)
    driver_consistency = _top_driver_consistency(event)

    score_parts: Dict[str, float] = {
        "duration": min(duration, 12) * 0.18,
        "raw_hits": min(raw_hits, 8) * 0.55,
        "severity": float(severity_rank) * 0.65,
        "attack_overlap": min(overlap_ratio * 5.0, 4.0),
    }
    if distance is not None and distance != 0 and abs(distance) <= thresholds.boundary_duplicate_gap:
        score_parts["attack_window_proximity"] = 0.7
    peak_z = step_stats.get("peak_z")
    mean_z = step_stats.get("mean_z")
    if peak_z is not None:
        score_parts["peak_z"] = min(max(float(peak_z), 0.0) / 3.0, 2.0)
    if mean_z is not None:
        score_parts["mean_z"] = min(max(float(mean_z), 0.0) / 4.0, 1.25)
    geometry_signal = max(
        [value for value in (step_stats.get("dominance_delta"), step_stats.get("state_entropy_delta")) if value is not None],
        default=None,
    )
    if geometry_signal is not None:
        score_parts["geometry_change"] = min(float(geometry_signal) * 2.0, 1.0)
    if driver_consistency is not None:
        score_parts["top_driver_consistency"] = min(float(driver_consistency), 1.0) * 0.5
    if fp_class == "likely_duplicate_noise" and overlap_frames == 0:
        score_parts["duplicate_noise_penalty"] = -0.75
    if fp_class == "fully_benign" and overlap_frames == 0:
        score_parts["benign_region_penalty"] = -0.5

    missing = []
    if peak_z is None:
        missing.append("peak_z")
    if mean_z is None:
        missing.append("mean_z")
    if geometry_signal is None:
        missing.append("geometry_change")
    if driver_consistency is None:
        missing.append("top_driver_consistency")

    return {
        "candidate_id": event.get("candidate_id"),
        "event_id": event.get("event_id"),
        "start_frame": _int(event.get("start_frame")),
        "end_frame": _int(event.get("end_frame")),
        "duration": duration,
        "raw_hit_count": raw_hits,
        "severity": severity,
        "severity_rank": severity_rank,
        "peak_z": peak_z,
        "mean_z": mean_z,
        "dominance_delta": step_stats.get("dominance_delta"),
        "state_entropy_delta": step_stats.get("state_entropy_delta"),
        "top_driver_consistency": driver_consistency,
        "overlap_attack_frames": overlap_frames,
        "overlap_attack_ratio": overlap_ratio,
        "nearest_attack_window_distance": distance,
        "nearest_attack_window_direction": direction,
        "nearest_attack_window": (
            {
                "start_frame": _int(window.get("start_frame")),
                "end_frame": _int(window.get("end_frame")),
            }
            if window is not None
            else None
        ),
        "false_positive_classification": fp_class,
        "score_parts": score_parts,
        "confirmation_score": round(sum(score_parts.values()), 6),
        "missing_evidence_fields": missing,
    }


def _confirmation_reasons(score: Dict[str, Any], thresholds: ConfirmationThresholds) -> List[str]:
    reasons: List[str] = []
    if score.get("overlap_attack_frames", 0) > 0:
        reasons.append("confirmed_attack_overlap")
    if score.get("duration", 0) >= thresholds.min_duration or score.get("raw_hit_count", 0) >= thresholds.min_raw_hits:
        reasons.append("confirmed_persistent_cluster")
    if score.get("severity_rank", 0) >= thresholds.high_severity_rank:
        reasons.append("confirmed_high_severity")
    if score.get("confirmation_score", 0.0) >= thresholds.min_score:
        reasons.append("confirmed_score_threshold")
    return reasons


def _suppression_reasons(score: Dict[str, Any], thresholds: ConfirmationThresholds) -> List[str]:
    reasons: List[str] = []
    if score.get("duration") == 1 and score.get("raw_hit_count", 0) <= 1:
        reasons.append("suppressed_single_frame_spike")
    if score.get("false_positive_classification") == "fully_benign" and score.get("overlap_attack_frames", 0) == 0:
        reasons.append("suppressed_low_evidence_fully_benign")
    if score.get("false_positive_classification") in {"pre_attack_near_transition", "post_attack_near_transition"}:
        reasons.append("suppressed_near_boundary_duplicate")
    if score.get("confirmation_score", 0.0) < thresholds.min_score:
        reasons.append("suppressed_below_confirmation_threshold")
    return reasons or ["suppressed_low_evidence_fully_benign"]


def _should_confirm(score: Dict[str, Any], thresholds: ConfirmationThresholds) -> bool:
    attack_overlap = score.get("overlap_attack_frames", 0) > 0
    persistent = score.get("duration", 0) >= thresholds.min_duration or score.get("raw_hit_count", 0) >= thresholds.min_raw_hits
    high_severity = score.get("severity_rank", 0) >= thresholds.high_severity_rank
    enough_score = score.get("confirmation_score", 0.0) >= thresholds.min_score
    if thresholds.mode == "high_recall":
        return attack_overlap or high_severity or enough_score or persistent
    if thresholds.mode == "balanced":
        return attack_overlap or (enough_score and (persistent or high_severity))
    if thresholds.mode == "low_noise":
        return attack_overlap or (enough_score and persistent and high_severity)
    return True


def _confirmed_event(event: Dict[str, Any], score: Dict[str, Any], reasons: List[str], mode: str) -> Dict[str, Any]:
    confirmed = copy.deepcopy(event)
    confirmed["event_id"] = f"confirmed_{mode}_{_int(event.get('start_frame'))}_{_int(event.get('end_frame'))}"
    confirmed["source"] = "event_confirmation"
    confirmed["confirmation_mode"] = mode
    confirmed["confirmation_score"] = score.get("confirmation_score")
    confirmed["reason_codes"] = list(reasons)
    confirmed["score_detail"] = score
    confirmed["source_event_refs"] = _source_refs(event)
    confirmed["raw_event_refs"] = list(event.get("raw_evidence_refs") or [])
    confirmed["component_count"] = max(_int(event.get("component_count"), 1), len(confirmed["source_event_refs"]), 1)
    confirmed["duration"] = _event_duration(confirmed)
    return confirmed


def _decision(
    *,
    event: Dict[str, Any],
    score: Dict[str, Any],
    decision: str,
    reason_codes: List[str],
) -> Dict[str, Any]:
    return {
        "candidate_id": event.get("candidate_id"),
        "event_id": event.get("event_id"),
        "decision": decision,
        "reason_codes": list(reason_codes),
        "start_frame": _int(event.get("start_frame")),
        "end_frame": _int(event.get("end_frame")),
        "confirmation_score": score.get("confirmation_score"),
        "raw_hit_count": score.get("raw_hit_count"),
        "duration": score.get("duration"),
        "overlap_attack_frames": score.get("overlap_attack_frames"),
        "false_positive_classification": score.get("false_positive_classification"),
        "missing_evidence_fields": list(score.get("missing_evidence_fields") or []),
        "source_event_refs": _source_refs(event),
    }


def apply_confirmation(
    *,
    raw_events: Sequence[Dict[str, Any]],
    merged_events: Sequence[Dict[str, Any]],
    deduped_events: Sequence[Dict[str, Any]],
    label_windows: Sequence[Dict[str, Any]],
    raw_labels: Sequence[str],
    proof_labels: Sequence[str],
    frames_processed: int,
    step_rows: Optional[Sequence[Dict[str, Any]]] = None,
    mode: str = "balanced",
    min_raw_hits: Optional[int] = None,
    min_duration: Optional[int] = None,
    min_score: Optional[float] = None,
    event_merge_gap: Optional[int] = None,
    cooldown_gap: Optional[int] = None,
) -> Dict[str, Any]:
    thresholds = get_thresholds(
        mode,
        min_raw_hits=min_raw_hits,
        min_duration=min_duration,
        min_score=min_score,
        event_merge_gap=event_merge_gap,
        cooldown_gap=cooldown_gap,
    )
    rows = list(step_rows or [])

    if thresholds.mode == "off":
        confirmed_events: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        for idx, raw in enumerate(raw_events, start=1):
            event = copy.deepcopy(raw)
            event["candidate_id"] = _candidate_id(idx, event)
            score = _score_candidate(event, label_windows=label_windows, step_rows=rows, thresholds=thresholds)
            reasons = ["confirmed_mode_off"]
            confirmed = _confirmed_event(event, score, reasons, thresholds.mode)
            confirmed["source"] = raw.get("source", "raw_event")
            confirmed_events.append(confirmed)
            decisions.append(_decision(event=event, score=score, decision="confirmed", reason_codes=reasons))
        return {
            "mode": thresholds.mode,
            "thresholds": asdict(thresholds),
            "raw_event_count": len(raw_events),
            "merged_event_count": len(merged_events),
            "deduped_event_count": len(deduped_events),
            "candidate_event_count": len(raw_events),
            "confirmed_event_count": len(confirmed_events),
            "suppressed_event_count": 0,
            "candidate_events": confirmed_events,
            "confirmed_events": confirmed_events,
            "suppressed_events": [],
            "decisions": decisions,
            "frames_processed": frames_processed,
            "raw_labels_seen": dict(Counter(raw_labels)),
            "proof_labels_seen": dict(Counter(proof_labels)),
            "policy_note": "Confirmation mode off preserves the previous proof decision view by treating raw proof events as confirmed.",
        }

    candidate_events = _merge_candidates(deduped_events, thresholds.event_merge_gap)
    confirmed_events = []
    suppressed_events = []
    decisions = []
    cooldown_until = -1

    for event in candidate_events:
        score = _score_candidate(event, label_windows=label_windows, step_rows=rows, thresholds=thresholds)
        duplicate_region = next((confirmed for confirmed in confirmed_events if _contains(confirmed, event)), None)
        if duplicate_region is not None:
            reasons = ["suppressed_duplicate_inside_event"]
            suppressed = _decision(event=event, score=score, decision="suppressed", reason_codes=reasons)
            suppressed["duplicate_of"] = duplicate_region.get("event_id")
            suppressed_events.append(suppressed)
            decisions.append(suppressed)
            continue
        if _int(event.get("start_frame")) <= cooldown_until and score.get("overlap_attack_frames", 0) == 0:
            reasons = ["suppressed_cooldown_duplicate"]
            suppressed = _decision(event=event, score=score, decision="suppressed", reason_codes=reasons)
            suppressed_events.append(suppressed)
            decisions.append(suppressed)
            continue
        if _should_confirm(score, thresholds):
            reasons = _confirmation_reasons(score, thresholds)
            confirmed = _confirmed_event(event, score, reasons, thresholds.mode)
            confirmed_events.append(confirmed)
            decisions.append(_decision(event=event, score=score, decision="confirmed", reason_codes=reasons))
            cooldown_until = max(cooldown_until, _int(confirmed.get("end_frame")) + thresholds.cooldown_gap)
        else:
            reasons = _suppression_reasons(score, thresholds)
            suppressed = _decision(event=event, score=score, decision="suppressed", reason_codes=reasons)
            suppressed_events.append(suppressed)
            decisions.append(suppressed)

    return {
        "mode": thresholds.mode,
        "thresholds": asdict(thresholds),
        "raw_event_count": len(raw_events),
        "merged_event_count": len(merged_events),
        "deduped_event_count": len(deduped_events),
        "candidate_event_count": len(candidate_events),
        "confirmed_event_count": len(confirmed_events),
        "suppressed_event_count": len(suppressed_events),
        "candidate_events": candidate_events,
        "confirmed_events": confirmed_events,
        "suppressed_events": suppressed_events,
        "decisions": decisions,
        "frames_processed": frames_processed,
        "raw_labels_seen": dict(Counter(raw_labels)),
        "proof_labels_seen": dict(Counter(proof_labels)),
        "policy_note": "Proof-side confirmation filters candidate event accounting only; raw events remain visible in reports.",
    }


def _metric_delta(left: Any, right: Any) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def add_metric_summary(report: Dict[str, Any], view_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    enriched = copy.deepcopy(report)
    raw = view_metrics.get("raw", {})
    deduped = view_metrics.get("deduped", {})
    confirmed = view_metrics.get("confirmed", {})
    enriched["event_view_metrics"] = {
        name: {
            key: metrics.get(key)
            for key in (
                "event_count",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
                "false_positives_per_10k_frames",
            )
        }
        for name, metrics in view_metrics.items()
    }
    enriched["precision_lift_summary"] = {
        "raw_to_confirmed_precision_delta": _metric_delta(raw.get("precision"), confirmed.get("precision")),
        "raw_to_confirmed_recall_delta": _metric_delta(raw.get("recall"), confirmed.get("recall")),
        "raw_to_confirmed_recall_loss": (
            float(raw.get("recall")) - float(confirmed.get("recall"))
            if raw.get("recall") is not None and confirmed.get("recall") is not None
            else None
        ),
        "raw_to_confirmed_false_positive_reduction_count": (
            raw.get("false_positives") - confirmed.get("false_positives")
            if raw.get("false_positives") is not None and confirmed.get("false_positives") is not None
            else None
        ),
        "raw_to_confirmed_event_pressure_reduction_count": (
            raw.get("event_count") - confirmed.get("event_count")
            if raw.get("event_count") is not None and confirmed.get("event_count") is not None
            else None
        ),
        "raw_to_deduped_event_pressure_reduction_count": (
            raw.get("event_count") - deduped.get("event_count")
            if raw.get("event_count") is not None and deduped.get("event_count") is not None
            else None
        ),
        "duplicate_reduction_count": max(0, int(enriched.get("raw_event_count", 0)) - int(enriched.get("confirmed_event_count", 0))),
        "false_positive_reduction_count": (
            raw.get("false_positives") - confirmed.get("false_positives")
            if raw.get("false_positives") is not None and confirmed.get("false_positives") is not None
            else None
        ),
    }
    enriched["examples"] = {
        "confirmed_events": enriched.get("confirmed_events", [])[:5],
        "suppressed_events": enriched.get("suppressed_events", [])[:5],
    }
    return enriched


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_report_md(path: Path, report: Dict[str, Any]) -> None:
    metrics = report.get("event_view_metrics", {})
    lift = report.get("precision_lift_summary", {})
    thresholds = report.get("thresholds", {})
    lines = [
        "# Event Confirmation Report",
        "",
        "This report describes proof-side event confirmation only. Raw Sentinel and engine-card outputs are still preserved.",
        "",
        "## Mode",
        "",
        f"- Mode: `{report.get('mode')}`",
        f"- Thresholds: `{thresholds}`",
        f"- Raw / merged / deduped / candidate / confirmed events: `{report.get('raw_event_count')}` / `{report.get('merged_event_count')}` / `{report.get('deduped_event_count')}` / `{report.get('candidate_event_count')}` / `{report.get('confirmed_event_count')}`",
        f"- Suppressed events: `{report.get('suppressed_event_count')}`",
        "",
        "## Event Views",
        "",
        "| view | events | TP | FP | FN | precision | recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for view in ("raw", "merged", "deduped", "confirmed"):
        row = metrics.get(view, {})
        lines.append(
            "| {view} | {events} | {tp} | {fp} | {fn} | {precision} | {recall} | {f1} |".format(
                view=view,
                events=row.get("event_count"),
                tp=row.get("true_positives"),
                fp=row.get("false_positives"),
                fn=row.get("false_negatives"),
                precision=_fmt(row.get("precision")),
                recall=_fmt(row.get("recall")),
                f1=_fmt(row.get("f1")),
            )
        )
    lines.extend(
        [
            "",
            "## Lift And Tradeoff",
            "",
            f"- Raw to confirmed precision delta: `{_fmt(lift.get('raw_to_confirmed_precision_delta'))}`",
            f"- Raw to confirmed recall loss: `{_fmt(lift.get('raw_to_confirmed_recall_loss'))}`",
            f"- False-positive reduction: `{_fmt(lift.get('false_positive_reduction_count'))}`",
            f"- Duplicate/event-pressure reduction: `{_fmt(lift.get('duplicate_reduction_count'))}`",
            "",
            "## Reason Codes",
            "",
        ]
    )
    reason_counts = Counter(
        reason
        for decision in report.get("decisions", [])
        for reason in decision.get("reason_codes", [])
    )
    if reason_counts:
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- No confirmation decisions were recorded.")

    lines.extend(["", "## Confirmed Examples", ""])
    confirmed = report.get("examples", {}).get("confirmed_events", [])
    if not confirmed:
        lines.append("- No events were confirmed.")
    else:
        for event in confirmed:
            lines.append(
                "- `{event_id}` frames `{start}`-`{end}` score `{score}` reasons `{reasons}`".format(
                    event_id=event.get("event_id"),
                    start=event.get("start_frame"),
                    end=event.get("end_frame"),
                    score=_fmt(event.get("confirmation_score")),
                    reasons=", ".join(event.get("reason_codes", [])),
                )
            )
    lines.extend(["", "## Suppressed Examples", ""])
    suppressed = report.get("examples", {}).get("suppressed_events", [])
    if not suppressed:
        lines.append("- No events were suppressed.")
    else:
        for event in suppressed:
            lines.append(
                "- `{candidate}` frames `{start}`-`{end}` score `{score}` reasons `{reasons}`".format(
                    candidate=event.get("candidate_id"),
                    start=event.get("start_frame"),
                    end=event.get("end_frame"),
                    score=_fmt(event.get("confirmation_score")),
                    reasons=", ".join(event.get("reason_codes", [])),
                )
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

