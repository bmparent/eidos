"""Confirmed Sentinel event merging and incident-card records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConfirmedEvent:
    event_id: str
    start_frame: int
    end_frame: int
    duration: int
    peak_score: float
    event_count: int
    severity: str
    confidence: float
    why_flagged: List[str]
    event_type: str = "anomaly"
    top_drivers: List[Dict[str, Any]] = field(default_factory=list)
    similar_past_events: List[Dict[str, Any]] = field(default_factory=list)
    raw_evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration": self.duration,
            "peak_score": self.peak_score,
            "event_count": self.event_count,
            "severity": self.severity,
            "confidence": self.confidence,
            "why_flagged": list(self.why_flagged),
            "event_type": self.event_type,
            "top_drivers": list(self.top_drivers),
            "similar_past_events": list(self.similar_past_events),
            "raw_evidence_refs": list(self.raw_evidence_refs),
        }


def event_types_related(left: str, right: str) -> bool:
    if left == right:
        return True
    collapse_types = {"anomaly", "lifecycle_collapse", "lifecycle_extinction"}
    return left in collapse_types and right in collapse_types


def merge_confirmed_events(events: List[ConfirmedEvent], merge_window: int) -> tuple[List[ConfirmedEvent], int]:
    """Merge nearby related event windows and return merge count."""
    if not events:
        return [], 0
    ordered = sorted(events, key=lambda event: (event.start_frame, event.end_frame))
    merged: List[ConfirmedEvent] = [ordered[0]]
    merge_count = 0
    for event in ordered[1:]:
        current = merged[-1]
        nearby = event.start_frame <= current.end_frame + merge_window + 1
        if nearby and event_types_related(current.event_type, event.event_type):
            current.end_frame = max(current.end_frame, event.end_frame)
            current.duration = current.end_frame - current.start_frame + 1
            current.peak_score = max(current.peak_score, event.peak_score)
            current.event_count += event.event_count
            current.confidence = max(current.confidence, event.confidence)
            current.why_flagged = sorted(set(current.why_flagged + event.why_flagged))
            current.top_drivers = (current.top_drivers + event.top_drivers)[:8]
            current.similar_past_events = (current.similar_past_events + event.similar_past_events)[:5]
            current.raw_evidence_refs = sorted(set(current.raw_evidence_refs + event.raw_evidence_refs))
            current.event_id = f"evt_{current.event_type}_{current.start_frame}_{current.end_frame}"
            if event.severity == "RED" or current.severity == "RED":
                current.severity = "RED"
            elif event.severity == "AMBER" or current.severity == "AMBER":
                current.severity = "AMBER"
            merge_count += 1
        else:
            merged.append(event)
    return merged, merge_count


def event_to_incident_card(event: ConfirmedEvent, *, incident_id: str | None = None) -> Dict[str, Any]:
    """Return the incident-card-compatible record for a confirmed event."""
    card_id = incident_id or f"incident_{event.event_id}"
    return {
        "incident_id": card_id,
        "event_id": event.event_id,
        "start_frame": event.start_frame,
        "end_frame": event.end_frame,
        "severity": event.severity,
        "confidence": event.confidence,
        "why_flagged": list(event.why_flagged),
        "top_drivers": list(event.top_drivers),
        "similar_past_events": list(event.similar_past_events),
        "raw_evidence_refs": list(event.raw_evidence_refs),
        "event_type": event.event_type,
    }
