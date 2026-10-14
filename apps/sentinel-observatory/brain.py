"""Evidence-only adapter around the repository's unchanged Sentinel V3.

The reservoir residual describes the current frame. It is NOT a forecast.
Latency forecasts are a separate, deliberately simple EWMA experiment. Each
forecast is frozen after one observation and only scored against a later one.
"""
from __future__ import annotations

import math
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "eidos" / "repo" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from eidos_brain.sentinel.input_detectors import InputEvidenceDetector
from eidos_brain.sentinel.sentinel_v3 import SentinelV3, SentinelV3Config


class _CadenceNeutralDetector(InputEvidenceDetector):
    """Do not interpret the research fixture's 47-frame period as evidence."""

    def _period_power(self) -> None:
        return None


def _instant(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO 8601 string with timezone")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be an ISO 8601 string with timezone") from exc
    if dt.tzinfo is None:
        raise ValueError("timestamp timezone is required")
    return dt.astimezone(timezone.utc)


def _number(value: Any) -> float:
    # bool is an int subclass, but is not telemetry.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("latency is missing or not numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 86_400_000:
        raise ValueError("latency must be finite and between 0 and 86400000 milliseconds")
    return result


@dataclass
class _Source:
    detector: SentinelV3
    samples: int = 0
    rejected: int = 0
    skipped_forecasts: int = 0
    last_timestamp: datetime | None = None
    last_evidence_id: str | None = None
    last_brain_state: str = "calibrating"
    ewma: float | None = None
    pending: dict[str, Any] | None = None
    scored_count: int = 0
    ewma_abs_error: float = 0.0
    persistence_abs_error: float = 0.0
    ewma_squared_error: float = 0.0
    persistence_squared_error: float = 0.0
    ewma_wins: int = 0
    persistence_wins: int = 0
    ties: int = 0
    seen: deque[str] = field(default_factory=lambda: deque(maxlen=2048))


class BrainMonitor:
    """One independent, sequential Sentinel instance per monitored endpoint.

    Calibration is provisional: twelve probes do not establish a normal traffic
    distribution. No detector state is called a threat or protective decision.
    The adapter owns no network client, model credentials, or defensive actions.
    """

    def __init__(self, warmup: int = 12, min_scored: int = 20,
                 alpha: float = 0.35, max_forecast_gap_seconds: float = 300.0):
        if warmup < 12 or min_scored < 20 or not 0 < alpha <= 1:
            raise ValueError("warmup >= 12, min_scored >= 20 and 0 < alpha <= 1 required")
        if not math.isfinite(max_forecast_gap_seconds) or max_forecast_gap_seconds <= 0:
            raise ValueError("positive finite forecast validity window required")
        self.warmup = warmup
        self.min_scored = min_scored
        self.alpha = alpha
        self.max_forecast_gap_seconds = max_forecast_gap_seconds
        self.sources: dict[str, _Source] = {}

    def _source(self, source_id: str) -> _Source:
        if not isinstance(source_id, str) or not source_id or len(source_id) > 200:
            raise ValueError("source_id must be a nonempty identifier of at most 200 characters")
        if source_id not in self.sources:
            cfg = SentinelV3Config(features=1, reservoir_size=32, warmup=self.warmup,
                                   rolling_window=8, seed=42)
            detector = SentinelV3(cfg)
            detector.input_detector = _CadenceNeutralDetector(cfg.detector_config(), features=1)
            self.sources[source_id] = _Source(detector=detector)
        return self.sources[source_id]

    def observe(self, source_id: str, evidence_id: str, timestamp: str,
                latency_ms: float | None, status: int | str | None) -> dict[str, Any]:
        state = self._source(source_id)
        base: dict[str, Any] = {
            "source_id": source_id, "evidence_id": evidence_id,
            "engine": "Eidos Brain / Sentinel V3",
            "engine_scope": "Current-frame latency behavior; not threat classification",
            "observed_status": status if isinstance(status, (int, str)) and not isinstance(status, bool) else None,
            "calibration": "provisional per-source baseline; no labeled security validation",
        }
        try:
            if not isinstance(evidence_id, str) or not evidence_id or len(evidence_id) > 256:
                raise ValueError("evidence_id must be a nonempty identifier")
            if evidence_id in state.seen:
                raise ValueError("duplicate evidence_id excluded")
            instant = _instant(timestamp)
            if state.last_timestamp is not None and instant <= state.last_timestamp:
                raise ValueError("non-increasing timestamp excluded")
            latency = _number(latency_ms)
        except ValueError as exc:
            state.rejected += 1
            # Missing samples create coverage gaps. Duplicate/out-of-order delivery
            # does not invalidate an otherwise valid already-issued forecast.
            if "latency" in str(exc) and state.pending is not None:
                state.skipped_forecasts += 1
                state.pending = None
            return {**base, "state": "insufficient_data", "reason": str(exc),
                    "warmup_remaining": max(0, self.warmup - state.samples),
                    "event_score": None, "signals": [],
                    "forecast": {"issued": None, "scored": None,
                                 "comparison": self._comparison(state)},
                    "missing_data_policy": "Excluded; never padded or replaced with zero"}

        scored = None
        if state.pending is not None:
            frozen = state.pending
            gap = (instant - _instant(frozen["issued_at"])).total_seconds()
            if 0 < gap <= self.max_forecast_gap_seconds:
                predicted = frozen["predicted_latency_ms"]
                baseline = frozen["persistence_latency_ms"]
                ae = abs(predicted - latency)
                be = abs(baseline - latency)
                state.scored_count += 1
                state.ewma_abs_error += ae
                state.persistence_abs_error += be
                state.ewma_squared_error += ae * ae
                state.persistence_squared_error += be * be
                if abs(ae - be) < 1e-9:
                    state.ties += 1
                elif ae < be:
                    state.ewma_wins += 1
                else:
                    state.persistence_wins += 1
                scored = {**frozen, "status": "scored", "actual_evidence_id": evidence_id,
                          "observed_at": timestamp, "actual_latency_ms": latency,
                          "absolute_error_ms": ae, "persistence_error_ms": be,
                          "horizon_seconds": gap, "issued_before_observation": True}
            else:
                state.skipped_forecasts += 1
                scored = {"forecast_id": frozen["forecast_id"], "status": "unscored",
                          "reason": "observation gap exceeds forecast validity window",
                          "horizon_seconds": gap}

        # Exactly one finite numeric feature, checked before the canonical engine
        # can silently coerce NaN/Inf to zero or pad/truncate a malformed vector.
        frame = [math.log1p(latency)]
        if len(frame) != 1 or not all(math.isfinite(value) for value in frame):
            raise ValueError("invalid Sentinel feature vector")
        row = state.detector.step(frame)
        state.samples += 1
        state.last_timestamp = instant
        state.last_evidence_id = evidence_id
        state.seen.append(evidence_id)
        state.ewma = latency if state.ewma is None else self.alpha * latency + (1 - self.alpha) * state.ewma
        calibrating = bool(row["warmup"])
        brain_state = ("calibrating" if calibrating else
                       "geometry_change" if row["status"] == "RED" else
                       "change_detected" if row["status"] == "AMBER" else "observing")
        state.last_brain_state = brain_state
        signals = []
        if not calibrating:
            for key, label, meaning in (
                ("residual_evidence", "Reconstruction surprise", "Current-frame residual differs from recent residuals; not a future prediction"),
                ("input_evidence", "Latency distribution change", "Latency energy, change, or variation differs from provisional calibration"),
                ("geometry_evidence", "Reservoir geometry change", "Internal state variation changed; this is not a security severity"),
                ("novelty_evidence", "Input novelty", "Projected latency differs from provisional baseline"),
            ):
                signals.append({"key": key, "label": label, "value": float(row[key]),
                                "meaning": meaning, "evidence_ids": [evidence_id]})

        # Issued after the current observation, targeting a strictly later one.
        # The next observe() scores this frozen object before updating the EWMA.
        pending = None
        if state.samples >= self.warmup:
            pending = {"forecast_id": f"latency-next:{source_id}:{evidence_id}",
                       "method": "EWMA latency", "alpha": self.alpha,
                       "target": "next observed probe latency for this source",
                       "issued_at": timestamp, "basis_evidence_id": evidence_id,
                       "predicted_latency_ms": float(state.ewma),
                       "persistence_latency_ms": latency,
                       "max_horizon_seconds": self.max_forecast_gap_seconds,
                       "status": "pending", "claim": "experimental numerical forecast, not incident prediction"}
        state.pending = pending
        return {**base, "state": brain_state,
                "warmup_remaining": max(0, self.warmup - state.samples),
                "sample_count": state.samples,
                "event_score": None if calibrating else float(row["event_score"]),
                "episode_id": int(row["episode_id"]),
                "event_active": False if calibrating else bool(row["event_active"]),
                "signals": signals,
                "diagnostics": {"reconstruction_error": float(row["error_norm"]),
                                "reconstruction_z": float(row["z"]),
                                "feature": "log1p(latency_ms)", "periodic_probe": "disabled",
                                "security_verdict": "not assessed"},
                "forecast": {"issued": dict(pending) if pending else None,
                             "scored": scored, "comparison": self._comparison(state)}}

    def _comparison(self, state: _Source) -> dict[str, Any]:
        count = state.scored_count
        ae = state.ewma_abs_error / count if count else None
        be = state.persistence_abs_error / count if count else None
        enough = count >= self.min_scored
        return {"scored_count": count, "minimum_provisional_comparison": self.min_scored,
                "status": "provisional_observed_window" if enough else "insufficient_evidence",
                "ewma_mae_ms": ae, "persistence_mae_ms": be,
                "ewma_rmse_ms": math.sqrt(state.ewma_squared_error / count) if count else None,
                "persistence_rmse_ms": math.sqrt(state.persistence_squared_error / count) if count else None,
                "paired_mae_improvement_ms": be - ae if count else None,
                "ewma_wins": state.ewma_wins, "persistence_wins": state.persistence_wins,
                "ties": state.ties, "skipped_forecasts": state.skipped_forecasts,
                "skill_established": False,
                "limitation": "Sequential probe latency comparison only; autocorrelation, narrow coverage and short duration prevent general skill claims"}

    def summary(self) -> dict[str, Any]:
        return {"engine": "Eidos Brain / Sentinel V3", "core_modified": False,
                "forecast_method": "Frozen next-observation EWMA vs persistence",
                "sources": [{"source_id": source_id, "sample_count": state.samples,
                             "rejected_count": state.rejected, "state": state.last_brain_state,
                             "warmup_remaining": max(0, self.warmup - state.samples),
                             "last_evidence_id": state.last_evidence_id,
                             "last_observed_at": state.last_timestamp.isoformat() if state.last_timestamp else None,
                             "forecast": self._comparison(state)}
                            for source_id, state in self.sources.items()],
                "limitations": ["Provisional per-source calibration, not labeled threat detection",
                                "Reconstruction residual is not a forecast",
                                "No traffic inspection, protective action, or incident prediction",
                                "Missing or malformed numeric observations are excluded"]}
