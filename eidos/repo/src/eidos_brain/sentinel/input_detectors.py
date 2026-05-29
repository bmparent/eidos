from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Optional

import numpy as np

from .calibration import WelfordStats, finite_scalar, finite_vector


@dataclass(frozen=True)
class InputDetectorConfig:
    warmup: int = 1000
    rolling_window: int = 32
    period: int = 47
    period_window: int = 188
    z_threshold: float = 3.0
    evidence_scale: float = 4.0
    seed: int = 1729


@dataclass
class InputEvidenceDetector:
    config: InputDetectorConfig
    features: int
    energy_stats: WelfordStats = field(default_factory=WelfordStats)
    delta_stats: WelfordStats = field(default_factory=WelfordStats)
    rolling_var_stats: WelfordStats = field(default_factory=WelfordStats)
    period_stats: WelfordStats = field(default_factory=WelfordStats)
    signature_stats: WelfordStats = field(default_factory=WelfordStats)
    prev_frame: Optional[np.ndarray] = None
    rolling_frames: Deque[np.ndarray] = field(default_factory=deque)
    signatures: Deque[float] = field(default_factory=deque)
    probe: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.config.seed + self.features)
        probe = rng.normal(0.0, 1.0, self.features)
        norm = float(np.linalg.norm(probe))
        self.probe = probe / (norm if norm > 0.0 else 1.0)

    def update(self, frame: object, *, warmup: bool) -> dict[str, float]:
        x = finite_vector(frame, self.features)
        energy = finite_scalar(np.linalg.norm(x))
        if self.prev_frame is None:
            delta = 0.0
        else:
            delta = finite_scalar(np.linalg.norm(x - self.prev_frame))

        self.rolling_frames.append(x.copy())
        while len(self.rolling_frames) > self.config.rolling_window:
            self.rolling_frames.popleft()
        rolling_var = self._rolling_variance()

        signature = finite_scalar(np.dot(x, self.probe))
        self.signatures.append(signature)
        while len(self.signatures) > max(self.config.period_window, self.config.period * 2):
            self.signatures.popleft()
        period_power = self._period_power()

        if warmup:
            self.energy_stats.update(energy)
            self.delta_stats.update(delta)
            self.rolling_var_stats.update(rolling_var)
            self.signature_stats.update(signature)
            if period_power is not None:
                self.period_stats.update(period_power)

        energy_z = self.energy_stats.z(energy) if self.energy_stats.count else 0.0
        delta_z = self.delta_stats.z(delta) if self.delta_stats.count else 0.0
        rolling_var_z = self.rolling_var_stats.z(rolling_var) if self.rolling_var_stats.count else 0.0
        signature_z = self.signature_stats.z(signature) if self.signature_stats.count else 0.0
        period_z = self.period_stats.z(period_power or 0.0) if self.period_stats.count >= 4 else 0.0

        high_energy = self._scaled(abs(energy_z))
        high_delta = self._scaled(abs(delta_z))
        high_variance = self._scaled(max(0.0, rolling_var_z))
        low_variance = self._scaled(max(0.0, -rolling_var_z), threshold=2.25)
        period47_score = self._scaled(max(0.0, period_z), threshold=2.0)
        novelty_score = self._scaled(abs(signature_z), threshold=3.5)
        input_evidence = max(high_energy, high_delta, high_variance, low_variance, period47_score)

        self.prev_frame = x.copy()
        return {
            "energy": energy,
            "delta": delta,
            "rolling_var": rolling_var,
            "period_power": finite_scalar(period_power or 0.0),
            "energy_z": finite_scalar(energy_z),
            "delta_z": finite_scalar(delta_z),
            "rolling_var_z": finite_scalar(rolling_var_z),
            "period47_score": finite_scalar(period47_score),
            "input_evidence_score": finite_scalar(input_evidence),
            "novelty_evidence": finite_scalar(novelty_score),
        }

    def _rolling_variance(self) -> float:
        if len(self.rolling_frames) < 2:
            return 0.0
        arr = np.stack(tuple(self.rolling_frames), axis=0)
        return finite_scalar(float(np.mean(np.var(arr, axis=0))))

    def _period_power(self) -> Optional[float]:
        needed = max(self.config.period * 2, min(self.config.period_window, len(self.signatures)))
        if len(self.signatures) < needed:
            return None
        values = np.asarray(list(self.signatures)[-needed:], dtype=np.float64)
        values = values - float(np.mean(values))
        if not np.any(np.isfinite(values)):
            return 0.0
        idx = np.arange(values.size, dtype=np.float64)
        omega = 2.0 * math.pi / float(self.config.period)
        sin_term = float(np.dot(values, np.sin(omega * idx)))
        cos_term = float(np.dot(values, np.cos(omega * idx)))
        amp = math.sqrt(sin_term * sin_term + cos_term * cos_term) / max(1.0, values.size)
        return finite_scalar(amp)

    def _scaled(self, z: float, *, threshold: Optional[float] = None) -> float:
        threshold = self.config.z_threshold if threshold is None else threshold
        z = finite_scalar(z)
        if z <= threshold:
            return 0.0
        return finite_scalar(min(3.0, (z - threshold) / max(1e-9, self.config.evidence_scale)))
