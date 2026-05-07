"""Quantum-era crypto-agility telemetry adapter for defensive monitoring."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


BASE_FEATURES = [
    "legacy_rsa_dependency",
    "legacy_ecc_dependency",
    "unknown_kem_identifier",
    "unknown_signature_identifier",
    "downgrade_attempt_indicator",
    "handshake_failure_spike",
    "certificate_rotation_anomaly",
    "key_reuse_indicator",
    "entropy_rng_anomaly",
    "encrypted_exfiltration_volume_proxy",
    "pqc_negotiation_failure_proxy",
    "harvest_now_decrypt_later_risk_proxy",
    "tls_legacy_version",
    "weak_cipher_indicator",
    "process_crypto_inventory_change",
    "destination_concentration_proxy",
]

PQC_KEMS = {"kyber", "ml-kem", "mlkem", "bike", "hqc", "ntru", "frodo"}
PQC_SIGNATURES = {"dilithium", "ml-dsa", "mldsa", "falcon", "sphincs", "slh-dsa", "slhdsa"}
LEGACY_RSA = {"rsa", "rsa-2048", "rsa-3072", "rsa-4096"}
LEGACY_ECC = {"ecdsa", "ecdhe", "ecdh", "secp256r1", "p-256", "p-384", "ed25519"}


@dataclass
class CryptoAgilityAdapter:
    """Feature adapter for TLS/cert/process/network crypto-risk events."""

    features: int = 64
    hash_seed: int = 321

    def __post_init__(self) -> None:
        if self.features < len(BASE_FEATURES):
            raise ValueError(f"features must be at least {len(BASE_FEATURES)}")
        self.feature_names = [*BASE_FEATURES, *[f"crypto_hash_{i}" for i in range(self.features - len(BASE_FEATURES))]]

    def transform(self, event: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        vector = np.zeros(self.features, dtype=np.float64)
        kem = _norm(event.get("kem_id", event.get("key_exchange", event.get("kem", ""))))
        sig = _norm(event.get("signature_alg", event.get("signature", "")))
        cipher = _norm(event.get("cipher", event.get("cipher_suite", "")))
        tls_version = _norm(event.get("tls_version", event.get("protocol", "")))
        cert = event.get("cert", event.get("certificate", {}))
        cert_map = cert if isinstance(cert, Mapping) else {}

        vector[0] = float(any(name in kem or name in sig or name in cipher for name in LEGACY_RSA))
        vector[1] = float(any(name in kem or name in sig or name in cipher for name in LEGACY_ECC))
        vector[2] = float(bool(kem) and not _known_kem(kem))
        vector[3] = float(bool(sig) and not _known_signature(sig))
        vector[4] = _downgrade_indicator(event, tls_version)
        vector[5] = min(1.0, _safe_float(event.get("handshake_failures"), 0.0) / 25.0)
        vector[6] = _rotation_anomaly(event, cert_map)
        vector[7] = min(1.0, _safe_float(event.get("key_reuse_count", cert_map.get("key_reuse_count")), 0.0) / 10.0)
        vector[8] = _rng_anomaly(event)
        vector[9] = min(1.0, math.log1p(_safe_float(event.get("encrypted_bytes_out"), 0.0)) / math.log1p(500_000_000.0))
        vector[10] = min(1.0, _safe_float(event.get("pqc_failure_count"), 0.0) / 10.0)
        vector[11] = _harvest_now_risk(event, vector)
        vector[12] = float(tls_version in {"ssl3", "tls1.0", "tls1.1", "tlsv1", "tlsv1.0", "tlsv1.1"})
        vector[13] = float(any(marker in cipher for marker in ("rc4", "3des", "des", "null", "export", "md5", "sha1")))
        vector[14] = min(1.0, _safe_float(event.get("crypto_inventory_delta"), 0.0))
        vector[15] = min(1.0, _safe_float(event.get("destination_concentration"), 0.0))
        self._hash_remainder(event, vector)

        metadata = {
            "source_id": event.get("source_id", "crypto-agility"),
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

    def _hash_remainder(self, event: Mapping[str, Any], vector: np.ndarray) -> None:
        start = len(BASE_FEATURES)
        width = self.features - start
        if width <= 0:
            return
        reserved = set(BASE_FEATURES) | {
            "kem_id",
            "key_exchange",
            "kem",
            "signature_alg",
            "signature",
            "cipher",
            "cipher_suite",
            "tls_version",
            "protocol",
            "cert",
            "certificate",
            "handshake_failures",
            "key_reuse_count",
            "encrypted_bytes_out",
            "pqc_failure_count",
            "crypto_inventory_delta",
            "destination_concentration",
        }
        for key, value in event.items():
            if key in reserved:
                continue
            for scalar in _flatten_scalars(value):
                digest = hashlib.blake2b(f"{self.hash_seed}:{key}:{scalar}".encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % width
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[start + bucket] += sign * scalar * 0.01


def generate_crypto_agility_stream(
    scenario: str = "normal",
    n_frames: int = 96,
    seed: int = 11,
) -> list[dict[str, Any]]:
    """Create deterministic defensive crypto-agility events."""

    rng = np.random.default_rng(seed)
    events: list[dict[str, Any]] = []
    for idx in range(n_frames):
        is_anomaly = scenario != "normal" and idx >= n_frames // 2
        event = {
            "frame_id": idx,
            "timestamp": float(idx),
            "source_id": "crypto.synthetic",
            "tls_version": "TLS1.3",
            "cipher": "TLS_AES_256_GCM_SHA384",
            "kem_id": "ML-KEM-768",
            "signature_alg": "ML-DSA-65",
            "handshake_success": True,
            "handshake_failures": int(rng.poisson(0.2)),
            "cert": {
                "age_days": int(rng.integers(5, 70)),
                "rotation_days": int(rng.integers(45, 110)),
                "key_reuse_count": int(rng.integers(0, 2)),
            },
            "rng_entropy_bits": float(rng.normal(7.95, 0.04)),
            "encrypted_bytes_out": float(rng.lognormal(11.0, 0.35)),
            "pqc_negotiated": True,
            "pqc_failure_count": int(rng.poisson(0.1)),
            "destination_concentration": float(rng.uniform(0.02, 0.12)),
            "crypto_inventory_delta": 0.0,
            "scenario": scenario,
            "is_anomaly": is_anomaly,
        }
        if is_anomaly:
            _apply_crypto_scenario(event, scenario, rng, idx - n_frames // 2)
        events.append(event)
    return events


def _apply_crypto_scenario(event: dict[str, Any], scenario: str, rng: np.random.Generator, anomaly_idx: int) -> None:
    if scenario in {"downgrade_exfiltration_risk", "crypto_downgrade_exfiltration_risk"}:
        event.update(
            {
                "tls_version": "TLS1.1",
                "cipher": "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
                "kem_id": "RSA-2048",
                "signature_alg": "ECDSA-SHA1",
                "downgrade_attempt": True,
                "handshake_failures": int(12 + rng.poisson(4)),
                "encrypted_bytes_out": float(180_000_000 + anomaly_idx * 2_000_000),
                "pqc_negotiated": False,
                "pqc_failure_count": int(5 + rng.poisson(3)),
                "destination_concentration": 0.82,
                "key_reuse_count": 9,
                "rng_entropy_bits": 6.4,
            }
        )
        event["cert"] = {"age_days": 720, "rotation_days": 560, "key_reuse_count": 9}
    elif scenario == "unknown_pqc_identifier":
        event.update({"kem_id": "vendor-kem-x999", "signature_alg": "unknown-sig-lab", "pqc_failure_count": 4})


def _known_kem(value: str) -> bool:
    if any(name in value for name in PQC_KEMS | LEGACY_RSA | LEGACY_ECC):
        return True
    return value in {"", "none", "x25519", "dh", "dhe"}


def _known_signature(value: str) -> bool:
    if any(name in value for name in PQC_SIGNATURES | LEGACY_RSA | LEGACY_ECC):
        return True
    return value in {"", "none", "rsa-pss"}


def _downgrade_indicator(event: Mapping[str, Any], tls_version: str) -> float:
    if bool(event.get("downgrade_attempt", False)):
        return 1.0
    previous = _norm(event.get("previous_tls_version", ""))
    if previous in {"tls1.3", "tlsv1.3"} and tls_version in {"tls1.2", "tls1.1", "tls1.0", "tlsv1.2", "tlsv1.1"}:
        return 1.0
    return 0.0


def _rotation_anomaly(event: Mapping[str, Any], cert: Mapping[str, Any]) -> float:
    age = _safe_float(event.get("cert_age_days", cert.get("age_days")), 0.0)
    rotation = _safe_float(event.get("cert_rotation_days", cert.get("rotation_days")), 90.0)
    return min(1.0, max(0.0, age - rotation * 2.0) / 365.0)


def _rng_anomaly(event: Mapping[str, Any]) -> float:
    entropy = _safe_float(event.get("rng_entropy_bits", event.get("entropy_bits")), 8.0)
    repeated = _safe_float(event.get("rng_repeated_values", 0.0), 0.0)
    return min(1.0, max(0.0, 7.5 - entropy) / 2.0 + min(1.0, repeated / 10.0))


def _harvest_now_risk(event: Mapping[str, Any], vector: np.ndarray) -> float:
    long_retention = min(1.0, _safe_float(event.get("retention_days"), 0.0) / 3650.0)
    sensitive = min(1.0, _safe_float(event.get("sensitivity_score"), 0.5))
    return min(1.0, (vector[0] + vector[1] + vector[9] + long_retention + sensitive) / 5.0)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


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
