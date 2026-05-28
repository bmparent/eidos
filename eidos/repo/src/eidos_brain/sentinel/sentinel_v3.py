from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable, Iterator, Mapping, Optional

import numpy as np

from .calibration import RunningResidualStats, WelfordStats, finite_scalar, finite_vector, safe_sigma
from .event_merge import EventMergeConfig, EventMerger
from .hysteresis import EventHysteresis, HysteresisConfig
from .input_detectors import InputDetectorConfig, InputEvidenceDetector
from .normal_suppression import NormalSuppression, NormalSuppressionConfig


@dataclass(frozen=True)
class SentinelV3Config:
    features: int = 32
    reservoir_size: int = 256
    warmup: int = 1000
    seed: int = 42
    leak: float = 0.35
    recurrent_mix: float = 0.18
    forgetting: float = 0.995
    rls_delta: float = 100.0
    rls_p_min: float = 1e-6
    rls_p_max: float = 1e4
    w_value_clip: float = 6.0
    w_norm_clip: float = 250.0
    residual_z_warn: float = 3.5
    residual_z_scale: float = 4.0
    residual_weight: float = 0.35
    input_weight: float = 1.15
    geometry_weight: float = 1.35
    novelty_weight: float = 0.25
    amber_threshold: float = 1.0
    red_geometry_threshold: float = 0.35
    red_persistence: int = 18
    red_adaptation_cooldown: int = 48
    amber_adaptation_scale: float = 0.25
    rolling_window: int = 32
    period: int = 47
    period_window: int = 188
    merge_cooldown: int = 24

    def detector_config(self) -> InputDetectorConfig:
        return InputDetectorConfig(
            warmup=self.warmup,
            rolling_window=self.rolling_window,
            period=self.period,
            period_window=self.period_window,
            seed=self.seed + 1000,
        )


@dataclass
class SafeRLSReservoir:
    features: int
    reservoir_size: int
    seed: int = 42
    leak: float = 0.35
    recurrent_mix: float = 0.18
    forgetting: float = 0.995
    delta: float = 100.0
    p_min: float = 1e-6
    p_max: float = 1e4
    w_value_clip: float = 6.0
    w_norm_clip: float = 250.0
    state: np.ndarray = field(init=False)
    w_in: np.ndarray = field(init=False)
    recurrent_signs: np.ndarray = field(init=False)
    w_out: np.ndarray = field(init=False)
    p_diag: np.ndarray = field(init=False)
    recoveries: int = 0
    last_clipped_p: bool = False
    last_clipped_w: bool = False

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed + self.reservoir_size * 17 + self.features)
        self.state = np.zeros(self.reservoir_size, dtype=np.float64)
        self.w_in = rng.normal(0.0, 1.0 / math.sqrt(max(1, self.features)), (self.reservoir_size, self.features))
        self.recurrent_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=self.reservoir_size)
        self.w_out = np.zeros((self.features, self.reservoir_size), dtype=np.float64)
        self.p_diag = np.full(self.reservoir_size, finite_scalar(self.delta, 100.0), dtype=np.float64)

    def advance(self, frame: object) -> np.ndarray:
        x = finite_vector(frame, self.features)
        recurrent = np.roll(self.state, 1) * self.recurrent_signs
        drive = self.w_in @ x + self.recurrent_mix * recurrent
        next_state = (1.0 - self.leak) * self.state + self.leak * np.tanh(drive)
        self.state = np.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)
        return self.state

    def predict(self) -> np.ndarray:
        y = self.w_out @ self.state
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    def adapt(self, error: object, *, scale: float = 1.0) -> dict[str, bool | int]:
        err = finite_vector(error, self.features)
        scale = max(0.0, finite_scalar(scale, 0.0))
        if scale <= 0.0:
            return {"rls_recovered": False, "rls_p_clipped": False, "w_out_clipped": False, "recoveries": self.recoveries}

        if not self._finite_state():
            self._recover()
            return {"rls_recovered": True, "rls_p_clipped": True, "w_out_clipped": True, "recoveries": self.recoveries}

        r = self.state
        pr = self.p_diag * r
        denom = self.forgetting + float(np.dot(r, pr))
        if not math.isfinite(denom) or denom <= 1e-12:
            self._recover()
            return {"rls_recovered": True, "rls_p_clipped": True, "w_out_clipped": True, "recoveries": self.recoveries}

        gain = pr / denom
        self.w_out += scale * np.outer(err, gain)
        self.p_diag = (self.p_diag - gain * pr) / max(1e-6, self.forgetting)

        p_before = self.p_diag.copy()
        self.p_diag = np.nan_to_num(self.p_diag, nan=self.delta, posinf=self.p_max, neginf=self.p_min)
        self.p_diag = np.clip(self.p_diag, self.p_min, self.p_max)
        self.last_clipped_p = bool(np.any(np.abs(self.p_diag - p_before) > 0.0))

        w_before = self.w_out.copy()
        self.w_out = np.nan_to_num(self.w_out, nan=0.0, posinf=self.w_value_clip, neginf=-self.w_value_clip)
        self.w_out = np.clip(self.w_out, -self.w_value_clip, self.w_value_clip)
        norm = float(np.linalg.norm(self.w_out))
        if math.isfinite(norm) and norm > self.w_norm_clip:
            self.w_out *= self.w_norm_clip / norm
        elif not math.isfinite(norm):
            self._recover()
            return {"rls_recovered": True, "rls_p_clipped": True, "w_out_clipped": True, "recoveries": self.recoveries}
        self.last_clipped_w = bool(np.any(np.abs(self.w_out - w_before) > 0.0))

        return {
            "rls_recovered": False,
            "rls_p_clipped": self.last_clipped_p,
            "w_out_clipped": self.last_clipped_w,
            "recoveries": self.recoveries,
        }

    def _finite_state(self) -> bool:
        return bool(
            np.all(np.isfinite(self.state))
            and np.all(np.isfinite(self.w_out))
            and np.all(np.isfinite(self.p_diag))
        )

    def _recover(self) -> None:
        self.recoveries += 1
        self.state = np.nan_to_num(self.state, nan=0.0, posinf=0.0, neginf=0.0)
        self.w_out = np.nan_to_num(self.w_out, nan=0.0, posinf=0.0, neginf=0.0)
        self.w_out = np.clip(self.w_out, -self.w_value_clip, self.w_value_clip)
        self.p_diag = np.full(self.reservoir_size, min(self.delta, self.p_max), dtype=np.float64)


@dataclass
class SentinelV3:
    config: SentinelV3Config
    reservoir: SafeRLSReservoir = field(init=False)
    residual_stats: RunningResidualStats = field(init=False)
    input_detector: InputEvidenceDetector = field(init=False)
    hysteresis: EventHysteresis = field(init=False)
    normal_suppression: NormalSuppression = field(init=False)
    merger: EventMerger = field(init=False)
    plasticity_stats: WelfordStats = field(default_factory=WelfordStats)
    state_var_stats: WelfordStats = field(default_factory=WelfordStats)
    dominance_stats: WelfordStats = field(default_factory=WelfordStats)
    step_index: int = 0
    collapse_counter: int = 0
    freeze_remaining: int = 0

    def __post_init__(self) -> None:
        cfg = self.config
        self.reservoir = SafeRLSReservoir(
            features=cfg.features,
            reservoir_size=cfg.reservoir_size,
            seed=cfg.seed,
            leak=cfg.leak,
            recurrent_mix=cfg.recurrent_mix,
            forgetting=cfg.forgetting,
            delta=cfg.rls_delta,
            p_min=cfg.rls_p_min,
            p_max=cfg.rls_p_max,
            w_value_clip=cfg.w_value_clip,
            w_norm_clip=cfg.w_norm_clip,
        )
        self.residual_stats = RunningResidualStats(history_window=max(64, cfg.rolling_window * 4))
        self.input_detector = InputEvidenceDetector(config=cfg.detector_config(), features=cfg.features)
        self.hysteresis = EventHysteresis(HysteresisConfig(amber_threshold=cfg.amber_threshold))
        self.normal_suppression = NormalSuppression(NormalSuppressionConfig())
        self.merger = EventMerger(EventMergeConfig(cooldown=cfg.merge_cooldown))

    def step(self, frame: object) -> dict[str, object]:
        cfg = self.config
        t = self.step_index
        warmup = t < cfg.warmup
        x = finite_vector(frame, cfg.features)
        prev_state = self.reservoir.state.copy()
        self.reservoir.advance(x)
        y = self.reservoir.predict()
        err_vec = x - y
        err_norm = finite_scalar(np.linalg.norm(err_vec) / math.sqrt(max(1, cfg.features)))
        residual = self.residual_stats.update(err_norm)

        input_fields = self.input_detector.update(x, warmup=warmup)
        geometry_fields = self._geometry_fields(prev_state, self.reservoir.state, input_fields, warmup=warmup)

        residual_evidence = self._residual_evidence(residual["z"]) if not warmup else 0.0
        input_evidence = finite_scalar(input_fields["input_evidence_score"]) if not warmup else 0.0
        geometry_evidence = finite_scalar(geometry_fields["geometry_evidence"]) if not warmup else 0.0
        novelty_evidence = finite_scalar(input_fields["novelty_evidence"]) if not warmup else 0.0

        total_evidence = (
            cfg.residual_weight * residual_evidence
            + cfg.input_weight * input_evidence
            + cfg.geometry_weight * geometry_evidence
            + cfg.novelty_weight * novelty_evidence
        )
        threshold_multiplier = self.normal_suppression.update(
            residual_evidence=residual_evidence,
            input_evidence=input_evidence,
            geometry_evidence=geometry_evidence,
        )
        event_score = self.hysteresis.update(total_evidence if not warmup else 0.0)

        collapse_active = geometry_evidence >= cfg.red_geometry_threshold
        self.collapse_counter = self.collapse_counter + 1 if collapse_active and not warmup else 0

        if warmup:
            status = "GREEN"
        elif self.collapse_counter >= cfg.red_persistence:
            status = "RED"
        elif event_score >= cfg.amber_threshold * threshold_multiplier:
            status = "AMBER"
        else:
            status = "GREEN"

        if status == "RED":
            self.freeze_remaining = max(self.freeze_remaining, cfg.red_adaptation_cooldown)

        adaptation_frozen = bool((not warmup) and self.freeze_remaining > 0)
        adapt_scale = 1.0
        if adaptation_frozen:
            adapt_scale = 0.0
        elif status == "AMBER":
            adapt_scale = cfg.amber_adaptation_scale
        safety = self.reservoir.adapt(err_vec, scale=adapt_scale)

        if self.freeze_remaining > 0 and status != "RED":
            self.freeze_remaining -= 1

        merge_fields = self.merger.update(status)
        final_sigma = safe_sigma(residual["sigma"], floor=1e-6)
        result: dict[str, object] = {
            "t": t,
            "status": status,
            "is_surprise": bool(residual_evidence > 0.0),
            "residual_evidence": finite_scalar(residual_evidence),
            "input_evidence": finite_scalar(input_evidence),
            "geometry_evidence": finite_scalar(geometry_evidence),
            "novelty_evidence": finite_scalar(novelty_evidence),
            "event_score": finite_scalar(event_score),
            "adaptation_frozen": adaptation_frozen,
            "ema_err": finite_scalar(residual["ema_err"]),
            "sigma": final_sigma,
            "z": finite_scalar(residual["z"]),
            "error_norm": finite_scalar(err_norm),
            "threshold_multiplier": finite_scalar(threshold_multiplier),
            "collapse_counter": int(self.collapse_counter),
            "freeze_remaining": int(self.freeze_remaining),
            "warmup": bool(warmup),
        }
        result.update(input_fields)
        result.update(geometry_fields)
        result.update(safety)
        result.update(merge_fields)

        self.step_index += 1
        return _finite_result(result)

    def _residual_evidence(self, z: float) -> float:
        z = finite_scalar(z)
        if z <= self.config.residual_z_warn:
            return 0.0
        return finite_scalar(min(3.0, (z - self.config.residual_z_warn) / max(1e-9, self.config.residual_z_scale)))

    def _geometry_fields(
        self,
        prev_state: np.ndarray,
        state: np.ndarray,
        input_fields: Mapping[str, float],
        *,
        warmup: bool,
    ) -> dict[str, float]:
        plasticity = finite_scalar(np.linalg.norm(state - prev_state) / math.sqrt(max(1, state.size)))
        state_variance = finite_scalar(np.var(state))
        state_norm = finite_scalar(np.linalg.norm(state))
        eigen_dominance = finite_scalar(float(np.max(np.abs(state))) / (state_norm / math.sqrt(max(1, state.size)) + 1e-9))

        if warmup:
            self.plasticity_stats.update(plasticity)
            self.state_var_stats.update(state_variance)
            self.dominance_stats.update(eigen_dominance)

        plasticity_z = self.plasticity_stats.z(plasticity) if self.plasticity_stats.count else 0.0
        state_var_z = self.state_var_stats.z(state_variance) if self.state_var_stats.count else 0.0
        dominance_z = self.dominance_stats.z(eigen_dominance) if self.dominance_stats.count else 0.0
        rolling_var_z = finite_scalar(input_fields.get("rolling_var_z", 0.0))

        low_plasticity_score = _scaled(max(0.0, -plasticity_z), threshold=2.0, scale=3.0)
        low_state_var_score = _scaled(max(0.0, -state_var_z), threshold=2.0, scale=3.0)
        high_dominance_score = _scaled(max(0.0, dominance_z), threshold=3.0, scale=4.0)
        low_input_var_score = _scaled(max(0.0, -rolling_var_z), threshold=2.0, scale=3.0)
        rank_collapse_score = max(low_plasticity_score, low_state_var_score, high_dominance_score)
        geometry_evidence = max(rank_collapse_score, min(3.0, 0.7 * low_input_var_score + 0.5 * low_plasticity_score))

        return {
            "plasticity": plasticity,
            "state_variance": state_variance,
            "eigen_dominance": eigen_dominance,
            "plasticity_z": finite_scalar(plasticity_z),
            "state_var_z": finite_scalar(state_var_z),
            "eigen_dominance_z": finite_scalar(dominance_z),
            "low_plasticity_score": finite_scalar(low_plasticity_score),
            "low_state_var_score": finite_scalar(low_state_var_score),
            "rank_collapse_score": finite_scalar(rank_collapse_score),
            "geometry_evidence": finite_scalar(geometry_evidence),
        }


def _scaled(value: float, *, threshold: float, scale: float) -> float:
    value = finite_scalar(value)
    if value <= threshold:
        return 0.0
    return finite_scalar(min(3.0, (value - threshold) / max(1e-9, scale)))


def _finite_result(result: Mapping[str, object]) -> dict[str, object]:
    clean: dict[str, object] = {}
    for key, value in result.items():
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, (int, np.integer)):
            clean[key] = int(value)
        elif isinstance(value, str):
            clean[key] = value
        elif isinstance(value, float) or isinstance(value, np.floating):
            clean[key] = finite_scalar(value)
        else:
            clean[key] = value
    return clean


def run_sentinel_v3_stream(
    frames: Iterable[object],
    *,
    features: int,
    reservoir_size: int,
    warmup: int,
    seed: int = 42,
    config_overrides: Optional[Mapping[str, object]] = None,
) -> Iterator[dict[str, object]]:
    data = asdict(SentinelV3Config(features=features, reservoir_size=reservoir_size, warmup=warmup, seed=seed))
    if config_overrides:
        data.update(dict(config_overrides))
    sentinel = SentinelV3(SentinelV3Config(**data))
    for frame in frames:
        yield sentinel.step(frame)
