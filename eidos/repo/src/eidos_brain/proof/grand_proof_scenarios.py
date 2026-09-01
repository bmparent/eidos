"""Deterministic, causal EIDOS-GP-v1 synthetic mechanisms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator

import numpy as np

from .frame_observer import canonical_sha256


SCENARIO_IDS = (
    "S0_nominal",
    "S1_hidden_backdoor",
    "S2_slow_drift",
    "S3_regime_shift",
    "S6_noise_thrash",
    "S7_harmless_repeat",
    "S8_dangerous_repeat",
    "C1_nuisance_subspace",
)


@dataclass(frozen=True)
class ScenarioConfig:
    features: int = 64
    warmup_frames: int = 1000
    scored_frames: int = 5000
    noise_sigma: float = 0.018
    s1_amplitude: float = 0.12
    s2_final_amplitude: float = 0.38
    s3_shift_amplitude: float = 0.42
    s6_noise_sigma: float = 0.55
    repeat_amplitude: float = 0.58
    c1_nuisance_amplitude: float = 0.75
    c1_event_amplitude: float = 0.95
    outcome_horizon: int = 64

    @classmethod
    def smoke(cls) -> "ScenarioConfig":
        return cls(warmup_frames=64, scored_frames=256, outcome_horizon=16)

    @property
    def total_frames(self) -> int:
        return self.warmup_frames + self.scored_frames

    @property
    def hash(self) -> str:
        return "sha256:" + canonical_sha256(asdict(self))


@dataclass(frozen=True)
class EventWindow:
    event_id: str
    start: int
    end: int
    consequential: bool
    outcome: str
    feedback_at: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioStream:
    scenario_id: str
    seed: int
    frames: np.ndarray
    labels: np.ndarray
    events: tuple[EventWindow, ...]
    config: ScenarioConfig
    nuisance_projector: np.ndarray | None = None

    def online_frames(self) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
        """Yield inputs and identity only; labels/outcomes remain sealed."""

        for frame_id, frame in enumerate(self.frames):
            yield frame.copy(), {
                "source_id": f"{self.scenario_id}:seed:{self.seed}",
                "frame_id": frame_id,
                "timestamp": float(frame_id),
                "scenario_id": self.scenario_id,
                "seed": self.seed,
            }

    def score_receipt(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "config_hash": self.config.hash,
            "frame_sha256": canonical_sha256(self.frames.astype(float).tolist()),
            "label_sha256": canonical_sha256(self.labels.astype(int).tolist()),
            "events": [event.as_dict() for event in self.events],
        }


def _orthonormal_directions(rng: np.random.Generator, features: int, count: int = 8) -> list[np.ndarray]:
    directions: list[np.ndarray] = []
    for _ in range(count):
        vector = rng.normal(0.0, 1.0, features)
        for previous in directions:
            vector -= float(np.dot(vector, previous)) * previous
        norm = max(float(np.linalg.norm(vector)), 1e-12)
        directions.append(vector / norm)
    return directions


def _carrier(
    config: ScenarioConfig,
    rng: np.random.Generator,
    directions: list[np.ndarray],
) -> np.ndarray:
    t = np.arange(config.total_frames, dtype=np.float64)
    frames = (
        0.55 * np.sin(2.0 * np.pi * t[:, None] / 61.0) * directions[0][None, :]
        + 0.38 * np.cos(2.0 * np.pi * t[:, None] / 89.0) * directions[1][None, :]
        + 0.24 * np.sin(2.0 * np.pi * t[:, None] / 137.0 + 0.4) * directions[2][None, :]
    )
    frames += rng.normal(0.0, config.noise_sigma, size=(config.total_frames, config.features))
    return frames.astype(np.float64)


def _window(config: ScenarioConfig, start_fraction: float, duration_fraction: float) -> tuple[int, int]:
    start = config.warmup_frames + int(config.scored_frames * start_fraction)
    duration = max(4, int(config.scored_frames * duration_fraction))
    return start, min(config.total_frames - 1, start + duration - 1)


def generate_scenario(
    scenario_id: str,
    *,
    seed: int,
    config: ScenarioConfig | None = None,
) -> ScenarioStream:
    if scenario_id not in SCENARIO_IDS:
        raise ValueError(f"unknown Grand Proof scenario: {scenario_id}")
    cfg = config or ScenarioConfig()
    if cfg.features != 64:
        raise ValueError("EIDOS-GP-v1 scenarios are frozen at 64 dimensions")
    rng = np.random.default_rng(int(seed))
    directions = _orthonormal_directions(rng, cfg.features)
    frames = _carrier(cfg, rng, directions)
    labels = np.zeros(cfg.total_frames, dtype=np.int8)
    events: list[EventWindow] = []
    nuisance_projector = None
    t = np.arange(cfg.total_frames, dtype=np.float64)

    if scenario_id == "S0_nominal":
        pass
    elif scenario_id == "S1_hidden_backdoor":
        start, end = _window(cfg, 0.22, 0.12)
        phase = 2.0 * np.pi * t[start : end + 1] / 47.0
        frames[start : end + 1] += cfg.s1_amplitude * np.sin(phase)[:, None] * directions[3][None, :]
        labels[start : end + 1] = 1
        events.append(EventWindow("S1_event", start, end, True, "harmful"))
    elif scenario_id == "S2_slow_drift":
        start, end = _window(cfg, 0.18, 0.34)
        ramp = np.linspace(0.0, cfg.s2_final_amplitude, end - start + 1)
        frames[start : end + 1] += ramp[:, None] * directions[4][None, :]
        consequence_start = start + (end - start) // 2
        labels[consequence_start : end + 1] = 1
        events.append(EventWindow("S2_event", consequence_start, end, True, "harmful"))
    elif scenario_id == "S3_regime_shift":
        start, end = _window(cfg, 0.30, 0.48)
        phase = 2.0 * np.pi * t[start : end + 1] / 31.0
        frames[start : end + 1] = (
            0.42 * frames[start : end + 1]
            + cfg.s3_shift_amplitude * np.cos(phase)[:, None] * directions[5][None, :]
        )
        labels[start : end + 1] = 1
        events.append(EventWindow("S3_event", start, end, True, "harmful"))
    elif scenario_id == "S6_noise_thrash":
        start = cfg.warmup_frames
        frames[start:] += rng.normal(0.0, cfg.s6_noise_sigma, size=frames[start:].shape)
    elif scenario_id in {"S7_harmless_repeat", "S8_dangerous_repeat"}:
        first_start, first_end = _window(cfg, 0.16, 0.06)
        repeat_start, repeat_end = _window(cfg, 0.62, 0.06)
        shape = np.sin(np.linspace(0.0, np.pi, first_end - first_start + 1))
        frames[first_start : first_end + 1] += cfg.repeat_amplitude * shape[:, None] * directions[6][None, :]
        frames[repeat_start : repeat_end + 1] += cfg.repeat_amplitude * shape[:, None] * directions[6][None, :]
        dangerous = scenario_id == "S8_dangerous_repeat"
        if dangerous:
            labels[first_start : first_end + 1] = 1
            labels[repeat_start : repeat_end + 1] = 1
        feedback_at = first_end + cfg.outcome_horizon
        if feedback_at >= repeat_start:
            raise ValueError("repeat must occur after delayed consequence feedback")
        outcome = "harmful" if dangerous else "benign"
        events.extend(
            [
                EventWindow("repeat_first", first_start, first_end, dangerous, outcome, feedback_at),
                EventWindow("repeat_second", repeat_start, repeat_end, dangerous, outcome, None),
            ]
        )
    elif scenario_id == "C1_nuisance_subspace":
        nuisance = directions[7]
        frames += (
            cfg.c1_nuisance_amplitude * np.sin(2.0 * np.pi * t / 43.0)[:, None] * nuisance[None, :]
        )
        start, end = _window(cfg, 0.46, 0.07)
        pulse = np.sin(np.linspace(0.0, np.pi, end - start + 1))
        frames[start : end + 1] += cfg.c1_event_amplitude * pulse[:, None] * nuisance[None, :]
        labels[start : end + 1] = 1
        events.append(EventWindow("C1_event", start, end, True, "harmful"))
        nuisance_projector = np.outer(nuisance, nuisance)

    return ScenarioStream(
        scenario_id=scenario_id,
        seed=int(seed),
        frames=frames,
        labels=labels,
        events=tuple(events),
        config=cfg,
        nuisance_projector=nuisance_projector,
    )
