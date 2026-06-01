from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Iterable, Optional

import numpy as np


def finite_last(values: Iterable[float], default: float = 0.0) -> float:
    """Return the last finite scalar in an iterable, or a finite default."""

    fallback = float(default) if math.isfinite(float(default)) else 0.0
    for value in reversed(list(values)):
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scalar):
            return scalar
    return fallback


def finite_scalar(value: object, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        scalar = float(default)
    if not math.isfinite(scalar):
        scalar = float(default)
    if not math.isfinite(scalar):
        scalar = 0.0
    return scalar


def safe_sigma(var_or_sigma: object, floor: float = 1e-9, *, from_variance: bool = False) -> float:
    """Return a finite positive sigma with an explicit floor."""

    value = finite_scalar(var_or_sigma, default=floor)
    if from_variance:
        value = math.sqrt(max(0.0, value))
    if not math.isfinite(value) or value <= 0.0:
        return float(floor)
    return max(float(floor), float(value))


def safe_z(value: float, mean: float, sigma: float, *, cap: float = 99.0) -> float:
    sigma = safe_sigma(sigma)
    z = (finite_scalar(value) - finite_scalar(mean)) / sigma
    if not math.isfinite(z):
        return 0.0
    return float(max(-cap, min(cap, z)))


@dataclass
class WelfordStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    floor: float = 1e-9

    def update(self, value: float) -> None:
        value = finite_scalar(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if not math.isfinite(self.mean) or not math.isfinite(self.m2):
            self.count = 0
            self.mean = 0.0
            self.m2 = 0.0

    @property
    def variance(self) -> float:
        if self.count < 2:
            return self.floor
        return max(self.floor, finite_scalar(self.m2 / (self.count - 1), self.floor))

    @property
    def sigma(self) -> float:
        return safe_sigma(self.variance, self.floor, from_variance=True)

    def z(self, value: float) -> float:
        return safe_z(value, self.mean, self.sigma)


@dataclass
class RollingScalarWindow:
    size: int
    values: Deque[float] = field(default_factory=deque)

    def append(self, value: float) -> None:
        self.values.append(finite_scalar(value))
        while len(self.values) > self.size:
            self.values.popleft()

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return finite_scalar(np.mean(np.asarray(self.values, dtype=np.float64)))

    def std(self, floor: float = 1e-9) -> float:
        if len(self.values) < 2:
            return floor
        return safe_sigma(float(np.std(np.asarray(self.values, dtype=np.float64))), floor)


@dataclass
class RunningResidualStats:
    alpha: float = 0.03
    history_window: int = 256
    sigma_floor: float = 1e-6
    ema_err: float = 0.0
    count: int = 0
    residuals: RollingScalarWindow = field(init=False)

    def __post_init__(self) -> None:
        self.residuals = RollingScalarWindow(size=max(4, int(self.history_window)))

    def update(self, error: float) -> dict[str, float]:
        err = abs(finite_scalar(error))
        prev_ema = self.ema_err if self.count else err
        sigma = self._sigma()
        z = abs(err - prev_ema) / safe_sigma(sigma, self.sigma_floor)
        if not math.isfinite(z):
            z = 0.0

        if self.count == 0:
            self.ema_err = err
        else:
            self.ema_err = (1.0 - self.alpha) * finite_scalar(self.ema_err) + self.alpha * err
        self.ema_err = finite_scalar(self.ema_err)
        self.residuals.append(err)
        self.count += 1

        sigma_after = self._sigma()
        return {
            "ema_err": finite_scalar(self.ema_err),
            "sigma": safe_sigma(sigma_after, self.sigma_floor),
            "z": finite_scalar(z),
            "error": err,
        }

    def _sigma(self) -> float:
        if len(self.residuals.values) >= 8:
            values = np.asarray(self.residuals.values, dtype=np.float64)
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med)))
            if math.isfinite(mad) and mad > 0.0:
                return safe_sigma(1.4826 * mad, self.sigma_floor)
            return self.residuals.std(self.sigma_floor)
        return self.sigma_floor

    def final(self) -> dict[str, float]:
        return {
            "final_ema_err": finite_scalar(self.ema_err),
            "final_sigma": safe_sigma(self._sigma(), self.sigma_floor),
        }


def finite_vector(values: object, length: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if length is not None:
        if arr.size < length:
            arr = np.pad(arr, (0, length - arr.size))
        elif arr.size > length:
            arr = arr[:length]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
