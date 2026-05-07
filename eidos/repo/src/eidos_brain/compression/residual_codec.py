"""Residual-first, anomaly-preserving token codec."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import repeat
from typing import Any, Mapping, Sequence

import numpy as np

from .policy import CompressionPolicy, CompressionPolicyConfig


Token = dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _as_vector(frame: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("frame must contain at least one feature")
    if not np.all(np.isfinite(arr)):
        raise ValueError("frame contains non-finite values")
    return arr


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


class ResidualFirstCodec:
    """Predict, encode residuals, and preserve anomalies as replayable Eidos tokens."""

    def __init__(
        self,
        policy: CompressionPolicy | CompressionPolicyConfig | Mapping[str, Any] | None = None,
        predictor: Any | None = None,
        feature_names: Sequence[str] | None = None,
        source_id: str = "unknown",
        store_prediction_on_anomalies: bool = True,
    ) -> None:
        self.policy = policy if isinstance(policy, CompressionPolicy) else CompressionPolicy(policy)
        self.predictor = predictor
        self.feature_names = list(feature_names or [])
        self.source_id = source_id
        self.store_prediction_on_anomalies = bool(store_prediction_on_anomalies)
        self._encoder_last: np.ndarray | None = None
        self._decoder_last: np.ndarray | None = None
        self._next_frame_id = 0

    def reset(self) -> None:
        self._encoder_last = None
        self._decoder_last = None
        self._next_frame_id = 0

    def encode_stream(
        self,
        frames: Iterable[Sequence[float] | np.ndarray],
        metadata: Iterable[Mapping[str, Any] | None] | None = None,
    ) -> list[Token]:
        metas = metadata if metadata is not None else repeat(None)
        return [self.encode_frame(frame, meta) for frame, meta in zip(frames, metas)]

    def decode_stream(self, tokens: Iterable[Mapping[str, Any]]) -> np.ndarray:
        decoded = [self.decode_token(token) for token in tokens]
        if not decoded:
            return np.empty((0, 0), dtype=np.float64)
        return np.vstack(decoded)

    def encode_frame(self, frame: Sequence[float] | np.ndarray, metadata: Mapping[str, Any] | None = None) -> Token:
        meta = dict(metadata or {})
        x = _as_vector(frame)
        prediction = self._predict_for_encode(x, meta)
        residual = x - prediction
        residual_norm = float(np.linalg.norm(residual) / np.sqrt(residual.size))
        prediction_error = float(np.mean(np.abs(residual)))

        sentinel_status = meta.get("sentinel_status", meta.get("status", "GREEN"))
        surprise_z = _safe_float(meta.get("surprise_z", meta.get("surprise_score", 0.0)))
        decision = self.policy.decide(sentinel_status, residual_norm=residual_norm, surprise_z=surprise_z)
        frame_id = meta.get("frame_id", self._next_frame_id)
        self._next_frame_id = max(self._next_frame_id + 1, int(frame_id) + 1 if isinstance(frame_id, int) else self._next_frame_id)

        feature_names = list(meta.get("feature_names") or self.feature_names or [f"f{i}" for i in range(x.size)])
        top_drivers = list(meta.get("top_drivers") or self._top_drivers(residual, feature_names))
        payload = self._payload_for_decision(x, prediction, residual, decision, meta)
        token = {
            "token_version": 1,
            "frame_id": frame_id,
            "timestamp": meta.get("timestamp"),
            "source_id": meta.get("source_id", self.source_id),
            "prediction_error": prediction_error,
            "residual_norm": residual_norm,
            "surprise_z": surprise_z,
            "sentinel_status": decision.status,
            "sentinel_regime": meta.get("sentinel_regime", decision.status),
            "compression_mode": decision.mode,
            "quantization_scale": decision.quantization_scale,
            "feature_names": feature_names,
            "top_drivers": top_drivers,
            "policy_reason": decision.reason,
            "payload": payload,
        }
        self._encoder_last = self._reconstruct_with_prediction(token, prediction)
        return _jsonable(token)

    def decode_token(self, token: Mapping[str, Any]) -> np.ndarray:
        feature_count = len(token.get("feature_names") or [])
        prediction = self._predict_for_decode(feature_count)
        if token.get("payload", {}).get("prediction") is not None:
            prediction = _as_vector(token["payload"]["prediction"])
        frame = self._reconstruct_with_prediction(token, prediction)
        self._decoder_last = frame.copy()
        return frame

    def _predict_for_encode(self, frame: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
        prediction = self._external_prediction(frame, metadata)
        if prediction is None:
            prediction = self._encoder_last if self._encoder_last is not None else np.zeros_like(frame)
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        if prediction.size != frame.size:
            prediction = np.resize(prediction, frame.size)
        return prediction

    def _predict_for_decode(self, feature_count: int) -> np.ndarray:
        if self._decoder_last is not None:
            return self._decoder_last.copy()
        return np.zeros(max(feature_count, 1), dtype=np.float64)

    def _external_prediction(self, frame: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray | None:
        predictor = self.predictor
        if predictor is None:
            return None

        candidates: list[Any] = []
        if hasattr(predictor, "predict"):
            method = predictor.predict
            call_patterns = (
                lambda: method(frame=frame, metadata=metadata),
                lambda: method(frame, metadata),
                lambda: method(frame),
                lambda: method(),
            )
        elif callable(predictor):
            call_patterns = (
                lambda: predictor(frame=frame, metadata=metadata),
                lambda: predictor(frame, metadata),
                lambda: predictor(frame),
                lambda: predictor(),
            )
        else:
            return None

        for call in call_patterns:
            try:
                candidates.append(call())
                break
            except TypeError:
                continue
        if not candidates or candidates[0] is None:
            return None
        return _as_vector(candidates[0])

    def _payload_for_decision(
        self,
        frame: np.ndarray,
        prediction: np.ndarray,
        residual: np.ndarray,
        decision: Any,
        metadata: Mapping[str, Any],
    ) -> Token:
        if decision.allow_null:
            return {"type": "reference_or_null", "reference": "previous_prediction"}

        if decision.preserve_raw_frame:
            payload = {
                "type": decision.mode,
                "raw_frame": frame.astype(float).tolist(),
                "residual": residual.astype(float).tolist(),
                "replay_context": _jsonable(metadata.get("replay_context", {})),
            }
            if self.store_prediction_on_anomalies:
                payload["prediction"] = prediction.astype(float).tolist()
            if decision.include_model_state:
                payload["model_state_summary"] = _jsonable(metadata.get("model_state_summary", {}))
            if decision.include_sentinel_metrics:
                payload["sentinel_metrics"] = _jsonable(metadata.get("sentinel_metrics", {}))
            return payload

        scale = max(float(decision.quantization_scale), 1e-12)
        quantized = np.rint(residual / scale).astype(np.int64)
        payload = {"type": "quantized_residual", "q_residual": quantized.tolist()}
        if decision.preserve_feature_structure:
            payload["residual"] = residual.astype(float).tolist()
            payload["type"] = decision.mode
        return payload

    def _reconstruct_with_prediction(self, token: Mapping[str, Any], prediction: np.ndarray) -> np.ndarray:
        payload = dict(token.get("payload") or {})
        payload_type = payload.get("type", token.get("compression_mode"))
        if payload.get("raw_frame") is not None:
            return _as_vector(payload["raw_frame"])
        if payload_type == "reference_or_null":
            return prediction.copy()
        if payload.get("residual") is not None and token.get("compression_mode") == "structured_residual":
            return prediction + _as_vector(payload["residual"])
        q_residual = payload.get("q_residual")
        if q_residual is not None:
            scale = _safe_float(token.get("quantization_scale"), 1.0)
            return prediction + _as_vector(q_residual) * scale
        return prediction.copy()

    @staticmethod
    def _top_drivers(residual: np.ndarray, feature_names: Sequence[str], limit: int = 5) -> list[dict[str, Any]]:
        if residual.size == 0:
            return []
        names = list(feature_names)
        if len(names) < residual.size:
            names.extend(f"f{i}" for i in range(len(names), residual.size))
        indices = np.argsort(np.abs(residual))[::-1][: min(limit, residual.size)]
        return [
            {"feature": names[int(idx)], "index": int(idx), "residual": float(residual[int(idx)])}
            for idx in indices
            if abs(float(residual[int(idx)])) > 0.0
        ]


def reconstruction_error(original: Sequence[Sequence[float]] | np.ndarray, reconstructed: Sequence[Sequence[float]] | np.ndarray) -> float:
    """Root-mean-square reconstruction error."""

    x = np.asarray(original, dtype=np.float64)
    y = np.asarray(reconstructed, dtype=np.float64)
    if x.size == 0 and y.size == 0:
        return 0.0
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: original={x.shape}, reconstructed={y.shape}")
    return float(np.sqrt(np.mean((x - y) ** 2)))


def anomaly_preservation_score(tokens: Sequence[Mapping[str, Any]], anomaly_labels: Sequence[bool]) -> float:
    """Fraction of labeled anomalies emitted with structured/anomaly-preserving tokens."""

    if len(anomaly_labels) == 0:
        return 1.0
    anomaly_indices = [i for i, label in enumerate(anomaly_labels) if label]
    if not anomaly_indices:
        return 1.0
    preserving_modes = {"structured_residual", "anomaly_capsule", "raw_frame_plus_full_context"}
    preserved = sum(1 for i in anomaly_indices if i < len(tokens) and tokens[i].get("compression_mode") in preserving_modes)
    return float(preserved / len(anomaly_indices))
