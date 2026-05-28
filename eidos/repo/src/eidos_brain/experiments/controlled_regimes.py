from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np


REGIME_ORDER = ("NORMAL", "BACKDOOR_PERIODIC", "NOISE_CRASH", "FROZEN_LOW_VARIANCE")


@dataclass(frozen=True)
class ControlledFrame:
    index: int
    regime: str
    frame: np.ndarray


def default_regime_frames(frames_per_regime: Optional[int] = None) -> dict[str, int]:
    if frames_per_regime is not None:
        value = int(frames_per_regime)
        return {name: value for name in REGIME_ORDER}
    return {
        "NORMAL": 2000,
        "BACKDOOR_PERIODIC": 3000,
        "NOISE_CRASH": 3000,
        "FROZEN_LOW_VARIANCE": 3000,
    }


def frame_counts_with_warmup(warmup: int, regime_frames: Mapping[str, int]) -> dict[str, int]:
    counts = {"WARMUP": int(warmup)}
    counts.update({name: int(regime_frames.get(name, 0)) for name in REGIME_ORDER})
    return counts


def generate_controlled_regime_stream(
    *,
    features: int = 32,
    warmup: int = 1000,
    seed: int = 42,
    frames_per_regime: Optional[int] = None,
    regime_frames: Optional[Mapping[str, int]] = None,
) -> Iterable[ControlledFrame]:
    """Yield deterministic frames with labels reserved for evaluation only."""

    features = int(features)
    rng = np.random.default_rng(seed)
    dirs = _basis_vectors(rng, features, count=6)
    backdoor_a = dirs[3]
    backdoor_b = dirs[4]
    frozen_anchor = 0.18 * dirs[0] - 0.07 * dirs[1]
    counts = frame_counts_with_warmup(
        warmup,
        dict(regime_frames) if regime_frames is not None else default_regime_frames(frames_per_regime),
    )

    index = 0
    for _ in range(counts["WARMUP"]):
        yield ControlledFrame(index=index, regime="WARMUP", frame=_normal_frame(index, rng, dirs, features))
        index += 1

    for regime in REGIME_ORDER:
        for _ in range(counts[regime]):
            if regime == "NORMAL":
                frame = _normal_frame(index, rng, dirs, features)
            elif regime == "BACKDOOR_PERIODIC":
                phase = 2.0 * np.pi * (index % 47) / 47.0
                frame = (
                    _normal_frame(index, rng, dirs, features)
                    + 1.00 * np.sin(phase) * backdoor_a
                    + 0.42 * np.cos(phase) * backdoor_b
                )
            elif regime == "NOISE_CRASH":
                crash = rng.normal(0.0, 0.95, features)
                burst = (1.0 if (index % 23) < 11 else -1.0) * 0.35 * dirs[5]
                frame = 0.35 * _normal_frame(index, rng, dirs, features) + crash + burst
            elif regime == "FROZEN_LOW_VARIANCE":
                frame = frozen_anchor + rng.normal(0.0, 4e-4, features)
            else:
                raise ValueError(f"unknown regime: {regime}")
            yield ControlledFrame(index=index, regime=regime, frame=frame.astype(np.float64))
            index += 1


def _basis_vectors(rng: np.random.Generator, features: int, *, count: int) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for _ in range(count):
        v = rng.normal(0.0, 1.0, features)
        for existing in vectors:
            v = v - np.dot(v, existing) * existing
        norm = float(np.linalg.norm(v))
        vectors.append(v / (norm if norm > 0.0 else 1.0))
    return vectors


def _normal_frame(index: int, rng: np.random.Generator, dirs: list[np.ndarray], features: int) -> np.ndarray:
    t = float(index)
    frame = (
        0.55 * np.sin(2.0 * np.pi * t / 61.0) * dirs[0]
        + 0.38 * np.cos(2.0 * np.pi * t / 89.0) * dirs[1]
        + 0.24 * np.sin(2.0 * np.pi * t / 137.0 + 0.4) * dirs[2]
    )
    frame += rng.normal(0.0, 0.018, features)
    return frame.astype(np.float64)
