"""Normal-context suppression helpers for Sentinel event confirmation."""

from __future__ import annotations

from .calibration import SentinelModeConfig


def is_stable_normal_context(
    *,
    residual_score: float,
    geometry_change: float,
    novelty: float,
    config: SentinelModeConfig,
) -> bool:
    """Return true when evidence looks like ordinary low-motion behavior."""
    return (
        residual_score <= config.normal_residual_ceiling
        and geometry_change <= config.normal_geometry_ceiling
        and novelty <= config.normal_novelty_ceiling
    )


def confirmation_bar_multiplier(
    *,
    residual_score: float,
    geometry_change: float,
    novelty: float,
    config: SentinelModeConfig,
) -> float:
    """Raise the confirmation bar when residual evidence lacks geometry/novelty support."""
    if is_stable_normal_context(
        residual_score=residual_score,
        geometry_change=geometry_change,
        novelty=novelty,
        config=config,
    ):
        return config.normal_suppression_multiplier
    if geometry_change < config.geometry_change and novelty < config.novelty:
        return max(1.0, config.normal_suppression_multiplier * 0.75)
    return 1.0


def evidence_weight(
    *,
    residual_score: float,
    geometry_change: float,
    novelty: float,
    config: SentinelModeConfig,
) -> float:
    """Convert a raw spike frame into confirmation evidence."""
    if residual_score < config.candidate_score:
        return 0.0
    residual_lift = max(0.0, residual_score - config.candidate_score)
    geometry_lift = max(0.0, geometry_change / max(config.geometry_change, 1e-9))
    novelty_lift = max(0.0, novelty / max(config.novelty, 1e-9))
    multiplier = confirmation_bar_multiplier(
        residual_score=residual_score,
        geometry_change=geometry_change,
        novelty=novelty,
        config=config,
    )
    return (1.0 + residual_lift + 0.75 * geometry_lift + 0.75 * novelty_lift) / multiplier
