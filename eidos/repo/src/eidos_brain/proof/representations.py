"""Causal Meaningful Surprise representation lifts and past-only calibration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


LIFT_IDS = ("raw", "spectral", "multiscale", "geometry", "memory", "consensus")


def _vector(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def common_phase_coherence(phases: Sequence[float], weights: Sequence[float] | None = None) -> float:
    theta = _vector(phases, name="phases")
    if weights is None:
        w = np.full(theta.size, 1.0 / theta.size, dtype=np.float64)
    else:
        w = _vector(weights, name="weights")
        if w.size != theta.size or np.any(w < 0.0) or float(w.sum()) <= 0.0:
            raise ValueError("weights must be nonnegative and match phases")
        w = w / float(w.sum())
    return float(abs(np.sum(w * np.exp(1j * theta))))


class GaussianProjector:
    """Frozen Gaussian projection with JL-style 1/sqrt(output_dim) scaling.

    Each matrix entry is N(0, 1/output_dim), so the projected squared norm is
    preserved in expectation. This is not a per-vector deterministic guarantee.
    """

    def __init__(self, input_dim: int, output_dim: int, *, seed: int) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("projection dimensions must be positive")
        rng = np.random.default_rng(seed)
        self.matrix = rng.normal(0.0, 1.0 / math.sqrt(output_dim), size=(output_dim, input_dim))

    def transform(self, value: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = _vector(value, name="projection input")
        if arr.size != self.matrix.shape[1]:
            raise ValueError("projection input dimension mismatch")
        return self.matrix @ arr


@dataclass(frozen=True)
class LiftEvidence:
    lift_id: str
    representation: list[float]
    prediction: list[float]
    generalized_residual: list[float]
    residual_score: float
    quotient_residual: float | None
    p_value: float
    structural_evidence: float
    persistence: float
    calibration_count: int
    calibration_status: str
    prediction_source: str
    residual_definition: str
    eligible_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "lift_id": self.lift_id,
            "representation": self.representation,
            "prediction": self.prediction,
            "generalized_residual": self.generalized_residual,
            "residual_score": self.residual_score,
            "quotient_residual": self.quotient_residual,
            "p_value": self.p_value,
            "structural_evidence": self.structural_evidence,
            "persistence": self.persistence,
            "calibration_count": self.calibration_count,
            "calibration_status": self.calibration_status,
            "prediction_source": self.prediction_source,
            "residual_definition": self.residual_definition,
            "eligible_reason": self.eligible_reason,
        }


class PastOnlyLiftCalibrator:
    """Prequential predictor and conformal-style score using earlier rows only."""

    def __init__(
        self,
        *,
        p_min: float = 1e-3,
        threshold: float = 0.65,
        persistence_eta: float = 0.2,
        min_calibration: int = 16,
        max_history: int = 2048,
        nuisance_projector: np.ndarray | None = None,
    ) -> None:
        if not 0.0 < p_min < 1.0:
            raise ValueError("p_min must be in (0, 1)")
        self.p_min = float(p_min)
        self.threshold = float(threshold)
        self.persistence_eta = float(persistence_eta)
        self.min_calibration = int(min_calibration)
        self.scores: deque[float] = deque(maxlen=int(max_history))
        self.residuals: deque[np.ndarray] = deque(maxlen=int(max_history))
        self.prediction: np.ndarray | None = None
        self.persistence = 0.0
        self.nuisance_projector = None if nuisance_projector is None else np.asarray(
            nuisance_projector, dtype=np.float64
        )

    def evaluate(self, lift_id: str, representation: Sequence[float] | np.ndarray) -> LiftEvidence:
        z = _vector(representation, name=f"{lift_id} representation")
        pred = np.zeros_like(z) if self.prediction is None else np.resize(self.prediction, z.size)
        raw_residual = z - pred

        if self.residuals:
            stack = np.vstack([np.resize(item, z.size) for item in self.residuals])
            center = np.median(stack, axis=0)
            mad = np.median(np.abs(stack - center), axis=0)
            scale = np.maximum(1.4826 * mad, 1e-6)
        else:
            center = np.zeros_like(z)
            scale = np.ones_like(z)
        generalized = (raw_residual - center) / scale
        score = float(np.linalg.norm(generalized) / math.sqrt(generalized.size))

        if self.scores:
            ge = sum(1 for previous in self.scores if previous >= score)
            p_value = (1.0 + ge) / (1.0 + len(self.scores))
        else:
            p_value = 1.0
        evidence = float(np.clip(-math.log(max(p_value, self.p_min)) / -math.log(self.p_min), 0.0, 1.0))
        indicator = 1.0 if evidence >= self.threshold else 0.0
        self.persistence = (1.0 - self.persistence_eta) * self.persistence + self.persistence_eta * indicator

        quotient: float | None = None
        if self.nuisance_projector is not None:
            projector = self.nuisance_projector
            if projector.shape != (generalized.size, generalized.size):
                raise ValueError(f"{lift_id} nuisance projector shape mismatch")
            quotient = float(np.linalg.norm((np.eye(generalized.size) - projector) @ generalized))

        count = len(self.scores)
        status = "READY" if count >= self.min_calibration else "WARMING"
        result = LiftEvidence(
            lift_id=lift_id,
            representation=z.astype(float).tolist(),
            prediction=pred.astype(float).tolist(),
            generalized_residual=generalized.astype(float).tolist(),
            residual_score=score,
            quotient_residual=quotient,
            p_value=float(p_value),
            structural_evidence=evidence,
            persistence=float(self.persistence),
            calibration_count=count,
            calibration_status=status,
            prediction_source="prequential_ema_from_prior_frames",
            residual_definition="past_scaled_l2_generalized_residual",
            eligible_reason="available_from_causal_live_observation",
        )

        self.scores.append(score)
        self.residuals.append(raw_residual.copy())
        if self.prediction is None:
            self.prediction = z.copy()
        else:
            self.prediction = 0.95 * np.resize(self.prediction, z.size) + 0.05 * z
        return result


class RepresentationPipeline:
    """Bounded-memory causal lifts over versioned live observations."""

    def __init__(
        self,
        *,
        spectral_window: int = 64,
        multiscale_alphas: Sequence[float] = (0.5, 0.1, 0.02),
        min_calibration: int = 16,
        p_min: float = 1e-3,
        thresholds: Mapping[str, float] | None = None,
        raw_nuisance_projector: np.ndarray | None = None,
    ) -> None:
        self.spectral_window = int(spectral_window)
        if self.spectral_window < 8:
            raise ValueError("spectral_window must be at least 8")
        self.scalar_history: deque[float] = deque(maxlen=self.spectral_window)
        self.alphas = tuple(float(alpha) for alpha in multiscale_alphas)
        if not self.alphas or any(not 0.0 < alpha <= 1.0 for alpha in self.alphas):
            raise ValueError("multiscale alphas must be in (0, 1]")
        self.multiscale_state: list[float] | None = None
        threshold_map = {lift: 0.65 for lift in LIFT_IDS}
        threshold_map.update(dict(thresholds or {}))
        self.calibrators = {
            lift: PastOnlyLiftCalibrator(
                p_min=p_min,
                threshold=threshold_map[lift],
                min_calibration=min_calibration,
                nuisance_projector=raw_nuisance_projector if lift == "raw" else None,
            )
            for lift in LIFT_IDS
        }

    def _spectral(self) -> tuple[np.ndarray, float | None]:
        values = np.asarray(self.scalar_history, dtype=np.float64)
        if values.size < 8:
            padded = np.pad(values, (8 - values.size, 0))
        else:
            padded = values
        windowed = (padded - padded.mean()) * np.hanning(padded.size)
        fft = np.fft.rfft(windowed)
        usable = fft[1:5] if fft.size > 1 else np.zeros(4, dtype=np.complex128)
        if usable.size < 4:
            usable = np.pad(usable, (0, 4 - usable.size))
        magnitude = np.log1p(np.abs(usable))
        phases = np.angle(usable)
        coherence = common_phase_coherence(phases)
        return np.concatenate([magnitude, [coherence]]), coherence

    def _multiscale(self, scalar: float) -> np.ndarray:
        if self.multiscale_state is None:
            self.multiscale_state = [scalar for _ in self.alphas]
        else:
            self.multiscale_state = [
                (1.0 - alpha) * state + alpha * scalar
                for state, alpha in zip(self.multiscale_state, self.alphas)
            ]
        states = np.asarray(self.multiscale_state, dtype=np.float64)
        return np.concatenate([states, np.diff(states)])

    @staticmethod
    def _metric(metrics: Mapping[str, Any], name: str) -> float:
        value = metrics.get(name)
        if value is None:
            return 0.0
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite live metric: {name}")
        return parsed

    def observe(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        frame = _vector(observation["frame"], name="frame")
        residual = _vector(observation["raw_residual"], name="raw_residual")
        scalar = float(frame.mean())
        self.scalar_history.append(scalar)
        spectral, phase_coherence = self._spectral()
        multiscale = self._multiscale(scalar)

        sentinel = dict(observation.get("sentinel_metrics") or {})
        hdc = dict(observation.get("hdc_metrics") or {})
        geometry = np.asarray(
            [
                self._metric(sentinel, "eigen_dominance"),
                self._metric(sentinel, "state_entropy"),
                self._metric(sentinel, "state_flatness"),
                self._metric(sentinel, "spectral_entropy"),
                self._metric(sentinel, "spectral_flatness"),
            ],
            dtype=np.float64,
        )
        memory = np.asarray(
            [
                self._metric(hdc, "similarity"),
                self._metric(hdc, "familiarity"),
                1.0 if bool(hdc.get("write", False)) else 0.0,
            ],
            dtype=np.float64,
        )
        thermo = dict(observation.get("thermodynamic_metrics") or {})
        thermo_evidence = float(
            np.clip(
                0.4 * abs(float(thermo.get("rho", 1.0)) - 1.0)
                + 0.3 * min(abs(float(thermo.get("temperature", 0.0))) / 5.0, 1.0)
                + 0.3 * min(abs(float(thermo.get("energy", 0.0))) / 10.0, 1.0),
                0.0,
                1.0,
            )
        )
        raw = residual
        representations = {
            "raw": raw,
            "spectral": spectral,
            "multiscale": multiscale,
            "geometry": geometry,
            "memory": memory,
        }
        first_pass = {
            lift: self.calibrators[lift].evaluate(lift, value)
            for lift, value in representations.items()
        }
        consensus = np.asarray(
            [
                first_pass["raw"].structural_evidence,
                first_pass["spectral"].structural_evidence,
                first_pass["multiscale"].structural_evidence,
                first_pass["geometry"].structural_evidence,
                first_pass["memory"].structural_evidence,
                thermo_evidence,
            ],
            dtype=np.float64,
        )
        first_pass["consensus"] = self.calibrators["consensus"].evaluate("consensus", consensus)
        evidences = {lift: first_pass[lift].as_dict() for lift in LIFT_IDS}
        values = np.asarray([evidences[lift]["structural_evidence"] for lift in LIFT_IDS], dtype=np.float64)
        disagreement = float(np.clip(np.var(values) / 0.25, 0.0, 1.0))
        return {
            "lifts": evidences,
            "representation_disagreement_definition": "normalized_weighted_evidence_variance",
            "representation_disagreement": disagreement,
            "phase_coherence": phase_coherence,
            "thermodynamic_evidence": thermo_evidence,
        }
