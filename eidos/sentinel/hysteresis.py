"""Hysteresis-based Sentinel event confirmation.

This module turns raw spike-like evidence into candidate clusters first, then
only confirms events when persistence, accumulated evidence, and non-normal
context agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .calibration import SentinelModeConfig, get_mode_config
from .event_merge import ConfirmedEvent, event_to_incident_card, event_types_related, merge_confirmed_events
from .normal_suppression import evidence_weight


@dataclass(frozen=True)
class EvidenceFrame:
    frame: int
    residual_score: float
    geometry_change: float = 0.0
    novelty: float = 0.0
    severity_hint: Optional[str] = None
    lifecycle_phase: Optional[str] = None
    surprise_rate: Optional[float] = None
    eigen_dominance: Optional[float] = None
    spectral_entropy: Optional[float] = None
    spectral_flatness: Optional[float] = None
    plasticity: Optional[float] = None
    top_drivers: List[Dict[str, Any]] = field(default_factory=list)
    similar_past_events: List[Dict[str, Any]] = field(default_factory=list)
    raw_evidence_ref: Optional[str] = None


@dataclass
class _CandidateCluster:
    start_frame: int
    end_frame: int
    last_candidate_frame: int
    peak_score: float
    evidence_score: float
    candidate_frames: int
    max_geometry: float
    max_novelty: float
    event_type: str
    severity_hint: Optional[str] = None
    top_drivers: List[Dict[str, Any]] = field(default_factory=list)
    similar_past_events: List[Dict[str, Any]] = field(default_factory=list)
    raw_evidence_refs: List[str] = field(default_factory=list)
    confirmed: bool = False
    why_flagged: List[str] = field(default_factory=list)

    def add(self, frame: EvidenceFrame, weight: float) -> None:
        self.end_frame = int(frame.frame)
        self.last_candidate_frame = int(frame.frame)
        self.peak_score = max(self.peak_score, float(frame.residual_score))
        self.evidence_score += weight
        self.candidate_frames += 1
        self.max_geometry = max(self.max_geometry, float(frame.geometry_change))
        self.max_novelty = max(self.max_novelty, float(frame.novelty))
        if frame.severity_hint == "RED" or self.severity_hint is None:
            self.severity_hint = frame.severity_hint or self.severity_hint
        self.top_drivers.extend(frame.top_drivers)
        self.similar_past_events.extend(frame.similar_past_events)
        if frame.raw_evidence_ref:
            self.raw_evidence_refs.append(frame.raw_evidence_ref)


@dataclass
class ConfirmationResult:
    mode: str
    confirmed_events: List[ConfirmedEvent]
    candidate_events: int
    suppressed_candidates: int
    cooldown_suppressions: int
    merged_events: int
    incident_cards: List[Dict[str, Any]]

    @property
    def red_count(self) -> int:
        return sum(1 for event in self.confirmed_events if event.severity == "RED")

    @property
    def amber_count(self) -> int:
        return sum(1 for event in self.confirmed_events if event.severity == "AMBER")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "confirmed_events": [event.to_dict() for event in self.confirmed_events],
            "candidate_events": self.candidate_events,
            "suppressed_candidates": self.suppressed_candidates,
            "cooldown_suppressions": self.cooldown_suppressions,
            "merged_events": self.merged_events,
            "red_count": self.red_count,
            "amber_count": self.amber_count,
            "incident_cards": list(self.incident_cards),
        }


class SentinelEventConfirmer:
    """Confirm Sentinel events with hysteresis, merge windows, and cooldown."""

    def __init__(self, mode: str = "balanced"):
        self.config: SentinelModeConfig = get_mode_config(mode)
        self.mode = mode
        self._active: Optional[_CandidateCluster] = None
        self._raw_events: List[ConfirmedEvent] = []
        self._cooldown_until = -1
        self.candidate_events = 0
        self.suppressed_candidates = 0
        self.cooldown_suppressions = 0

    def process(self, frame: EvidenceFrame) -> None:
        frame_type = self._event_type(frame)
        candidate = self._is_candidate(frame)
        if not candidate:
            self._maybe_close_for_gap(int(frame.frame))
            return

        if self._active is None:
            self._start_candidate(frame, frame_type)
        elif self._should_start_new_cluster(frame, frame_type):
            self._finalize_active()
            self._start_candidate(frame, frame_type)
        else:
            self._active.add(frame, self._frame_weight(frame))
        self._maybe_mark_confirmed()

    def finish(self) -> ConfirmationResult:
        self._finalize_active()
        merged, merge_count = merge_confirmed_events(self._raw_events, self.config.merge_window)
        incident_cards = [event_to_incident_card(event) for event in merged]
        return ConfirmationResult(
            mode=self.mode,
            confirmed_events=merged,
            candidate_events=self.candidate_events,
            suppressed_candidates=self.suppressed_candidates,
            cooldown_suppressions=self.cooldown_suppressions,
            merged_events=merge_count,
            incident_cards=incident_cards,
        )

    def _start_candidate(self, frame: EvidenceFrame, event_type: str) -> None:
        self.candidate_events += 1
        refs = [frame.raw_evidence_ref] if frame.raw_evidence_ref else []
        self._active = _CandidateCluster(
            start_frame=int(frame.frame),
            end_frame=int(frame.frame),
            last_candidate_frame=int(frame.frame),
            peak_score=float(frame.residual_score),
            evidence_score=self._frame_weight(frame),
            candidate_frames=1,
            max_geometry=float(frame.geometry_change),
            max_novelty=float(frame.novelty),
            event_type=event_type,
            severity_hint=frame.severity_hint,
            top_drivers=list(frame.top_drivers),
            similar_past_events=list(frame.similar_past_events),
            raw_evidence_refs=refs,
        )

    def _is_candidate(self, frame: EvidenceFrame) -> bool:
        if self._event_type(frame).startswith("lifecycle_"):
            return True
        return float(frame.residual_score) >= self.config.candidate_score

    def _event_type(self, frame: EvidenceFrame) -> str:
        phase = (frame.lifecycle_phase or "").lower()
        if phase in {"collapse", "extinction"}:
            return f"lifecycle_{phase}"
        if phase == "recovery":
            return "lifecycle_recovery"
        return "anomaly"

    def _frame_weight(self, frame: EvidenceFrame) -> float:
        base = evidence_weight(
            residual_score=float(frame.residual_score),
            geometry_change=float(frame.geometry_change),
            novelty=float(frame.novelty),
            surprise_rate=frame.surprise_rate,
            eigen_dominance=frame.eigen_dominance,
            spectral_entropy=frame.spectral_entropy,
            spectral_flatness=frame.spectral_flatness,
            plasticity=frame.plasticity,
            config=self.config,
        )
        phase = (frame.lifecycle_phase or "").lower()
        if phase in {"collapse", "extinction", "recovery"}:
            base += 1.5
        return base

    def _should_start_new_cluster(self, frame: EvidenceFrame, frame_type: str) -> bool:
        assert self._active is not None
        if not event_types_related(self._active.event_type, frame_type):
            return True
        return int(frame.frame) - self._active.last_candidate_frame > self.config.repeat_window

    def _maybe_close_for_gap(self, current_frame: int) -> None:
        if self._active is None:
            return
        if current_frame - self._active.last_candidate_frame > self.config.repeat_window:
            self._finalize_active()

    def _maybe_mark_confirmed(self) -> None:
        if self._active is None or self._active.confirmed:
            return
        duration = self._active.end_frame - self._active.start_frame + 1
        has_shape_change = (
            self._active.max_geometry >= self.config.geometry_change
            and self._active.max_novelty >= self.config.novelty
        )
        is_lifecycle = self._active.event_type.startswith("lifecycle_")
        persistent = (
            duration >= self.config.min_duration
            and self._active.candidate_frames >= self.config.min_candidate_frames
        )
        accumulated = (
            self._active.evidence_score >= self.config.confirmation_score
            and self._active.candidate_frames >= self.config.min_candidate_frames
        )
        if (has_shape_change or is_lifecycle) and (persistent or accumulated):
            self._active.confirmed = True
            self._active.why_flagged = self._why_flagged(has_shape_change, is_lifecycle, persistent, accumulated)

    def _why_flagged(
        self,
        has_shape_change: bool,
        is_lifecycle: bool,
        persistent: bool,
        accumulated: bool,
    ) -> List[str]:
        reasons: List[str] = []
        if persistent:
            reasons.append("candidate persisted across the confirmation window")
        if accumulated:
            reasons.append("candidate accumulated enough evidence")
        if has_shape_change:
            reasons.append("geometry change and novelty supported the residual spike")
        if is_lifecycle:
            reasons.append("lifecycle transition evidence was sustained")
        return reasons

    def _finalize_active(self) -> None:
        if self._active is None:
            return
        active = self._active
        self._active = None
        if not active.confirmed:
            self.suppressed_candidates += 1
            return
        if active.start_frame <= self._cooldown_until:
            self.cooldown_suppressions += 1
            return
        event = self._event_from_candidate(active)
        self._raw_events.append(event)
        self._cooldown_until = event.end_frame + self.config.cooldown

    def _event_from_candidate(self, active: _CandidateCluster) -> ConfirmedEvent:
        severity = self._severity(active)
        confidence = min(
            0.99,
            0.45
            + min(active.evidence_score / max(self.config.confirmation_score * 3.0, 1e-9), 0.35)
            + min(active.max_geometry, 1.0) * 0.1
            + min(active.max_novelty, 1.0) * 0.1,
        )
        event_id = f"evt_{active.event_type}_{active.start_frame}_{active.end_frame}"
        return ConfirmedEvent(
            event_id=event_id,
            start_frame=active.start_frame,
            end_frame=active.end_frame,
            duration=active.end_frame - active.start_frame + 1,
            peak_score=round(active.peak_score, 6),
            event_count=active.candidate_frames,
            severity=severity,
            confidence=round(confidence, 6),
            why_flagged=list(active.why_flagged),
            event_type=active.event_type,
            top_drivers=active.top_drivers[:8],
            similar_past_events=active.similar_past_events[:5],
            raw_evidence_refs=sorted(set(active.raw_evidence_refs)),
        )

    def _severity(self, active: _CandidateCluster) -> str:
        if active.event_type in {"lifecycle_collapse", "lifecycle_extinction"}:
            return "RED"
        if active.event_type == "lifecycle_recovery":
            return "RECOVERY"
        if active.severity_hint in {"RED", "AMBER"}:
            return active.severity_hint
        if active.peak_score >= self.config.red_score:
            return "RED"
        if active.peak_score >= self.config.amber_score:
            return "AMBER"
        return "AMBER"


def process_stream(frames: Iterable[EvidenceFrame], mode: str = "balanced") -> ConfirmationResult:
    confirmer = SentinelEventConfirmer(mode=mode)
    for frame in frames:
        confirmer.process(frame)
    return confirmer.finish()
