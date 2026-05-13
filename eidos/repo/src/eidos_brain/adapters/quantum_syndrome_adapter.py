"""Quantum-system telemetry adapter for defensive Eidos monitoring."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


BASE_FEATURES = [
    "bit_flip_rate",
    "syndrome_burst_rate",
    "parity_violation_rate",
    "measurement_entropy",
    "distribution_shift",
    "readout_bias",
    "crosstalk_proxy",
    "temporal_drift_score",
    "decoder_confidence_anomaly",
    "logical_error_rate",
    "mean_gate_error_rate",
    "readout_fidelity_loss",
    "calibration_age_score",
    "shot_count_log",
    "syndrome_density_delta",
    "rare_outcome_rate",
]


@dataclass
class QuantumSyndromeAdapter:
    """Convert quantum telemetry dictionaries into fixed-dimensional Eidos frames."""

    features: int = 64
    hash_seed: int = 123
    update_reference: bool = True

    def __post_init__(self) -> None:
        if self.features < len(BASE_FEATURES):
            raise ValueError(f"features must be at least {len(BASE_FEATURES)}")
        self.feature_names = [*BASE_FEATURES, *[f"quantum_hash_{i}" for i in range(self.features - len(BASE_FEATURES))]]
        self._reference_distribution: np.ndarray | None = None
        self._previous_syndrome_density: float | None = None
        self._previous_t1: np.ndarray | None = None
        self._previous_t2: np.ndarray | None = None

    def reset(self) -> None:
        self._reference_distribution = None
        self._previous_syndrome_density = None
        self._previous_t1 = None
        self._previous_t2 = None

    def transform(self, event: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        vector = np.zeros(self.features, dtype=np.float64)
        syndrome = _numeric_array(event.get("syndrome_bits", event.get("syndrome", [])))
        distribution = _distribution_vector(
            event.get("shot_counts", event.get("measurement_histogram", event.get("qubit_measurement_histograms", {})))
        )
        readout_fidelity = _numeric_array(event.get("readout_fidelity", []))
        gate_errors = _numeric_array(event.get("gate_error_rates", []))
        t1 = _numeric_array(event.get("t1", event.get("T1", [])))
        t2 = _numeric_array(event.get("t2", event.get("T2", [])))
        measurements = _numeric_matrix(event.get("qubit_measurements", []))

        syndrome_density = float(np.mean(syndrome)) if syndrome.size else 0.0
        vector[0] = syndrome_density
        vector[1] = _burst_rate(syndrome)
        vector[2] = float(np.sum(syndrome) % 2) if syndrome.size else 0.0
        vector[3] = _entropy(distribution)
        vector[4] = self._distribution_shift(distribution, event)
        vector[5] = _readout_bias(distribution)
        vector[6] = _crosstalk_proxy(measurements, syndrome)
        vector[7] = self._temporal_drift(t1, t2, event)
        vector[8] = max(0.0, 1.0 - _safe_float(event.get("decoder_confidence"), 1.0))
        vector[9] = _safe_float(event.get("logical_error_rate"), 0.0)
        vector[10] = float(np.mean(gate_errors)) if gate_errors.size else 0.0
        vector[11] = max(0.0, 1.0 - float(np.mean(readout_fidelity))) if readout_fidelity.size else 0.0
        vector[12] = min(1.0, _safe_float(_metadata(event).get("calibration_age_hours"), 0.0) / 168.0)
        vector[13] = math.log1p(float(np.sum(distribution))) / 12.0 if distribution.size else 0.0
        vector[14] = 0.0 if self._previous_syndrome_density is None else abs(syndrome_density - self._previous_syndrome_density)
        vector[15] = _rare_outcome_rate(distribution)

        self._hash_remainder(event, vector)
        self._previous_syndrome_density = syndrome_density
        if t1.size:
            self._previous_t1 = t1.copy()
        if t2.size:
            self._previous_t2 = t2.copy()
        if self.update_reference and distribution.size:
            if self._reference_distribution is None:
                self._reference_distribution = _probability(distribution)
            else:
                current = _probability(distribution)
                self._reference_distribution = 0.98 * self._reference_distribution + 0.02 * current

        metadata = {
            "source_id": event.get("source_id", "quantum"),
            "timestamp": event.get("timestamp"),
            "feature_names": self.feature_names,
            "is_anomaly": bool(event.get("is_anomaly", False)),
            "scenario": event.get("scenario", "unknown"),
            "top_drivers": _top_vector_drivers(vector, self.feature_names),
        }
        return np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0), metadata

    def transform_many(self, events: list[Mapping[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        frames: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for event in events:
            frame, meta = self.transform(event)
            frames.append(frame)
            metadata.append(meta)
        return np.vstack(frames), metadata

    def _distribution_shift(self, distribution: np.ndarray, event: Mapping[str, Any]) -> float:
        expected = _distribution_vector(event.get("expected_distribution", []))
        if expected.size:
            return _l1_shift(distribution, expected)
        if self._reference_distribution is None:
            return 0.0
        return _l1_shift(distribution, self._reference_distribution)

    def _temporal_drift(self, t1: np.ndarray, t2: np.ndarray, event: Mapping[str, Any]) -> float:
        baseline_t1 = _numeric_array(event.get("baseline_t1", []))
        baseline_t2 = _numeric_array(event.get("baseline_t2", []))
        pieces = []
        if t1.size and baseline_t1.size:
            pieces.append(_relative_delta(t1, baseline_t1))
        elif t1.size and self._previous_t1 is not None:
            pieces.append(_relative_delta(t1, self._previous_t1))
        if t2.size and baseline_t2.size:
            pieces.append(_relative_delta(t2, baseline_t2))
        elif t2.size and self._previous_t2 is not None:
            pieces.append(_relative_delta(t2, self._previous_t2))
        return float(np.mean(pieces)) if pieces else 0.0

    def _hash_remainder(self, event: Mapping[str, Any], vector: np.ndarray) -> None:
        start = len(BASE_FEATURES)
        width = self.features - start
        if width <= 0:
            return
        reserved = {
            "syndrome_bits",
            "syndrome",
            "shot_counts",
            "measurement_histogram",
            "qubit_measurement_histograms",
            "readout_fidelity",
            "gate_error_rates",
            "t1",
            "T1",
            "t2",
            "T2",
            "qubit_measurements",
            "expected_distribution",
            "decoder_confidence",
            "logical_error_rate",
            "baseline_t1",
            "baseline_t2",
        }
        for key, value in event.items():
            if key in reserved:
                continue
            for scalar in _flatten_scalars(value):
                digest = hashlib.blake2b(f"{self.hash_seed}:{key}:{scalar}".encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % width
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[start + bucket] += sign * float(scalar) * 0.01


def generate_quantum_telemetry_stream(
    scenario: str = "normal",
    n_frames: int = 96,
    seed: int = 7,
    n_qubits: int = 16,
) -> list[dict[str, Any]]:
    """Create deterministic quantum telemetry for tests and local benchmarks."""

    rng = np.random.default_rng(seed)
    events: list[dict[str, Any]] = []
    baseline_t1 = rng.normal(85.0, 3.0, size=n_qubits)
    baseline_t2 = rng.normal(62.0, 2.0, size=n_qubits)
    for idx in range(n_frames):
        is_anomaly = scenario != "normal" and idx >= n_frames // 2
        event = _base_quantum_event(rng, idx, n_qubits, baseline_t1, baseline_t2)
        event["scenario"] = scenario
        event["is_anomaly"] = is_anomaly
        if is_anomaly:
            _apply_quantum_scenario(event, scenario, rng, idx - n_frames // 2, n_frames // 2)
        events.append(event)
    return events


def _base_quantum_event(
    rng: np.random.Generator,
    idx: int,
    n_qubits: int,
    baseline_t1: np.ndarray,
    baseline_t2: np.ndarray,
) -> dict[str, Any]:
    syndrome = rng.binomial(1, 0.015, size=n_qubits).astype(int)
    outcomes = rng.multinomial(2048, [0.252, 0.248, 0.25, 0.25])
    return {
        "frame_id": idx,
        "timestamp": float(idx),
        "source_id": "quantum.synthetic",
        "syndrome_bits": syndrome.tolist(),
        "shot_counts": {"00": int(outcomes[0]), "01": int(outcomes[1]), "10": int(outcomes[2]), "11": int(outcomes[3])},
        "gate_error_rates": np.clip(rng.normal(0.0015, 0.0002, size=n_qubits), 0.0, None).tolist(),
        "readout_fidelity": np.clip(rng.normal(0.988, 0.003, size=n_qubits), 0.0, 1.0).tolist(),
        "t1": np.clip(baseline_t1 + rng.normal(0.0, 0.7, size=n_qubits), 1.0, None).tolist(),
        "t2": np.clip(baseline_t2 + rng.normal(0.0, 0.5, size=n_qubits), 1.0, None).tolist(),
        "baseline_t1": baseline_t1.tolist(),
        "baseline_t2": baseline_t2.tolist(),
        "logical_error_rate": float(max(0.0, rng.normal(0.001, 0.0002))),
        "decoder_confidence": float(np.clip(rng.normal(0.965, 0.01), 0.0, 1.0)),
        "calibration_metadata": {"calibration_age_hours": float(rng.uniform(1.0, 8.0))},
    }


def _apply_quantum_scenario(
    event: dict[str, Any],
    scenario: str,
    rng: np.random.Generator,
    anomaly_idx: int,
    anomaly_span: int,
) -> None:
    progress = (anomaly_idx + 1) / max(anomaly_span, 1)
    if scenario == "decoherence_drift":
        event["t1"] = (np.asarray(event["t1"]) * (1.0 - 0.35 * progress)).tolist()
        event["t2"] = (np.asarray(event["t2"]) * (1.0 - 0.45 * progress)).tolist()
        event["logical_error_rate"] = 0.004 + 0.012 * progress
        event["decoder_confidence"] = max(0.45, event["decoder_confidence"] - 0.35 * progress)
    elif scenario == "readout_bias_shift":
        total = 2048
        event["shot_counts"] = {"00": int(total * 0.58), "01": int(total * 0.19), "10": int(total * 0.16), "11": total - int(total * 0.93)}
        event["readout_fidelity"] = np.clip(np.asarray(event["readout_fidelity"]) - 0.08, 0.0, 1.0).tolist()
    elif scenario == "syndrome_burst":
        syndrome = np.zeros(len(event["syndrome_bits"]), dtype=int)
        start = int(rng.integers(0, max(1, len(syndrome) - 5)))
        syndrome[start : start + 5] = 1
        event["syndrome_bits"] = syndrome.tolist()
        event["decoder_confidence"] = 0.55
    elif scenario == "crosstalk_regime_change":
        shots = 64
        base = rng.binomial(1, 0.5, size=(shots, 1))
        noise = rng.binomial(1, 0.08, size=(shots, len(event["syndrome_bits"])))
        event["qubit_measurements"] = np.bitwise_xor(base, noise).astype(int).tolist()
        event["gate_error_rates"] = (np.asarray(event["gate_error_rates"]) * 2.5).tolist()
    elif scenario == "calibration_failure":
        event["gate_error_rates"] = np.clip(rng.normal(0.018, 0.004, len(event["syndrome_bits"])), 0.0, None).tolist()
        event["readout_fidelity"] = np.clip(rng.normal(0.88, 0.02, len(event["syndrome_bits"])), 0.0, 1.0).tolist()
        event["calibration_metadata"] = {"calibration_age_hours": 220.0, "status": "failed"}
        event["decoder_confidence"] = 0.42
    elif scenario == "rare_measurement_distribution_shift":
        total = 2048
        event["shot_counts"] = {"00": 52, "01": 61, "10": 74, "11": total - 187}
        event["logical_error_rate"] = 0.02


def _metadata(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("calibration_metadata", {})
    return value if isinstance(value, Mapping) else {}


def _numeric_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _numeric_matrix(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 0), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    if arr.ndim != 2:
        return np.empty((0, 0), dtype=np.float64)
    return np.nan_to_num(arr)


def _distribution_vector(value: Any) -> np.ndarray:
    if isinstance(value, Mapping):
        values = list(value.values())
    else:
        values = value
    return _numeric_array(values)


def _probability(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    clipped = np.clip(values.astype(np.float64), 0.0, None)
    total = float(np.sum(clipped))
    if total <= 0.0:
        return np.full_like(clipped, 1.0 / clipped.size)
    return clipped / total


def _entropy(values: np.ndarray) -> float:
    probs = _probability(values)
    if probs.size <= 1:
        return 0.0
    entropy = -float(np.sum([p * math.log2(p) for p in probs if p > 0.0]))
    return entropy / math.log2(probs.size)


def _l1_shift(current: np.ndarray, reference: np.ndarray) -> float:
    current_probs = _probability(current)
    reference_probs = _probability(reference)
    width = max(current_probs.size, reference_probs.size)
    if width == 0:
        return 0.0
    current_pad = np.pad(current_probs, (0, width - current_probs.size))
    reference_pad = np.pad(reference_probs, (0, width - reference_probs.size))
    return float(np.sum(np.abs(current_pad - reference_pad)) / 2.0)


def _burst_rate(bits: np.ndarray) -> float:
    if bits.size == 0:
        return 0.0
    max_run = 0
    current = 0
    for bit in bits.astype(int):
        current = current + 1 if bit else 0
        max_run = max(max_run, current)
    return float(max_run / bits.size)


def _readout_bias(distribution: np.ndarray) -> float:
    probs = _probability(distribution)
    if probs.size == 0:
        return 0.0
    return float(np.max(probs) - np.min(probs))


def _crosstalk_proxy(measurements: np.ndarray, syndrome: np.ndarray) -> float:
    if measurements.size and measurements.shape[1] > 1:
        corr = np.corrcoef(measurements, rowvar=False)
        upper = corr[np.triu_indices_from(corr, k=1)]
        finite = upper[np.isfinite(upper)]
        return float(np.mean(np.abs(finite))) if finite.size else 0.0
    if syndrome.size > 1:
        return float(np.mean(syndrome[:-1] == syndrome[1:]))
    return 0.0


def _relative_delta(current: np.ndarray, baseline: np.ndarray) -> float:
    width = min(current.size, baseline.size)
    if width == 0:
        return 0.0
    denom = np.maximum(np.abs(baseline[:width]), 1e-6)
    return float(np.mean(np.abs(current[:width] - baseline[:width]) / denom))


def _rare_outcome_rate(distribution: np.ndarray) -> float:
    probs = _probability(distribution)
    if probs.size == 0:
        return 0.0
    return float(np.sum(probs < 0.01))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flatten_scalars(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        scalars: list[float] = []
        for nested in value.values():
            scalars.extend(_flatten_scalars(nested))
        return scalars
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=object).reshape(-1)
        return [_safe_float(item) for item in arr if _is_number_like(item)]
    if _is_number_like(value):
        return [_safe_float(value)]
    return []


def _is_number_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _top_vector_drivers(vector: np.ndarray, names: list[str], limit: int = 5) -> list[dict[str, Any]]:
    indices = np.argsort(np.abs(vector))[::-1][:limit]
    return [{"feature": names[int(idx)], "index": int(idx), "value": float(vector[int(idx)])} for idx in indices]
