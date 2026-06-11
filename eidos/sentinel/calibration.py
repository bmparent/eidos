"""Sentinel event-confirmation mode calibration.

The constants here sit above the existing SentinelMonitor output. They do not
change reservoir dynamics, surprise scoring, or Sentinel thresholds; they only
decide when raw spike evidence becomes a confirmed event.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SentinelModeConfig:
    name: str
    candidate_score: float
    geometry_change: float
    novelty: float
    confirmation_score: float
    min_duration: int
    min_candidate_frames: int
    repeat_window: int
    merge_window: int
    cooldown: int
    normal_residual_ceiling: float
    normal_geometry_ceiling: float
    normal_novelty_ceiling: float
    normal_suppression_multiplier: float
    amber_score: float
    red_score: float


MODE_CONFIGS = {
    "off": SentinelModeConfig(
        name="off",
        candidate_score=1_000_000.0,
        geometry_change=1.0,
        novelty=1.0,
        confirmation_score=1_000_000.0,
        min_duration=1_000_000,
        min_candidate_frames=1_000_000,
        repeat_window=0,
        merge_window=0,
        cooldown=0,
        normal_residual_ceiling=1.0,
        normal_geometry_ceiling=0.05,
        normal_novelty_ceiling=0.05,
        normal_suppression_multiplier=1.0,
        amber_score=1_000_000.0,
        red_score=1_000_000.0,
    ),
    "low_noise": SentinelModeConfig(
        name="low_noise",
        candidate_score=3.0,
        geometry_change=0.35,
        novelty=0.35,
        confirmation_score=8.0,
        min_duration=4,
        min_candidate_frames=3,
        repeat_window=6,
        merge_window=8,
        cooldown=24,
        normal_residual_ceiling=1.75,
        normal_geometry_ceiling=0.12,
        normal_novelty_ceiling=0.12,
        normal_suppression_multiplier=1.75,
        amber_score=3.5,
        red_score=5.0,
    ),
    "balanced": SentinelModeConfig(
        name="balanced",
        candidate_score=2.5,
        geometry_change=0.25,
        novelty=0.25,
        confirmation_score=5.25,
        min_duration=3,
        min_candidate_frames=2,
        repeat_window=8,
        merge_window=10,
        cooldown=16,
        normal_residual_ceiling=1.5,
        normal_geometry_ceiling=0.15,
        normal_novelty_ceiling=0.15,
        normal_suppression_multiplier=1.35,
        amber_score=3.0,
        red_score=4.75,
    ),
    "high_recall": SentinelModeConfig(
        name="high_recall",
        candidate_score=2.0,
        geometry_change=0.18,
        novelty=0.18,
        confirmation_score=3.0,
        min_duration=2,
        min_candidate_frames=1,
        repeat_window=10,
        merge_window=12,
        cooldown=8,
        normal_residual_ceiling=1.35,
        normal_geometry_ceiling=0.18,
        normal_novelty_ceiling=0.18,
        normal_suppression_multiplier=1.1,
        amber_score=2.5,
        red_score=4.5,
    ),
}

MODE_NAMES = tuple(MODE_CONFIGS)


def get_mode_config(mode: str) -> SentinelModeConfig:
    """Return a calibrated confirmation mode or raise a clear error."""
    try:
        return MODE_CONFIGS[mode]
    except KeyError as exc:
        known = ", ".join(MODE_NAMES)
        raise ValueError(f"unknown Sentinel confirmation mode {mode!r}; expected one of: {known}") from exc
