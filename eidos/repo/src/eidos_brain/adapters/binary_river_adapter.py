"""Byte-window feature adapter for defensive binary-stream monitoring."""

from __future__ import annotations

import hashlib
import math
import zlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


BASE_FEATURES = [
    "byte_entropy",
    "rolling_entropy_delta",
    "printable_ratio",
    "zero_ratio",
    "repeated_block_ratio",
    "compression_difficulty_proxy",
    "mz_header_marker",
    "elf_header_marker",
    "zip_header_marker",
    "png_header_marker",
    "pdf_header_marker",
    "gzip_header_marker",
    "anomalous_high_entropy_blob",
    "mean_byte_value",
    "byte_stddev",
    "window_length_log",
]


@dataclass
class BinaryRiverAdapter:
    """Turn byte windows into compact fixed-dimensional Eidos features."""

    features: int = 64
    hash_seed: int = 777

    def __post_init__(self) -> None:
        if self.features < len(BASE_FEATURES) + 4:
            raise ValueError(f"features must be at least {len(BASE_FEATURES) + 4}")
        self.feature_names = [*BASE_FEATURES, *[f"binary_hash_{i}" for i in range(self.features - len(BASE_FEATURES))]]
        self._previous_entropy: float | None = None

    def reset(self) -> None:
        self._previous_entropy = None

    def transform(self, event_or_window: bytes | bytearray | memoryview | Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        if isinstance(event_or_window, Mapping):
            window = bytes(event_or_window.get("bytes", event_or_window.get("window", b"")))
            event = event_or_window
        else:
            window = bytes(event_or_window)
            event = {}
        if not window:
            raise ValueError("byte window is empty")

        data = np.frombuffer(window, dtype=np.uint8)
        entropy = _byte_entropy(data)
        vector = np.zeros(self.features, dtype=np.float64)
        vector[0] = entropy / 8.0
        vector[1] = 0.0 if self._previous_entropy is None else abs(entropy - self._previous_entropy) / 8.0
        vector[2] = float(np.mean((data >= 32) & (data <= 126)))
        vector[3] = float(np.mean(data == 0))
        vector[4] = _repeated_block_ratio(window)
        vector[5] = min(1.0, len(zlib.compress(window, level=6)) / max(1, len(window)))
        vector[6] = float(window.startswith(b"MZ"))
        vector[7] = float(window.startswith(b"\x7fELF"))
        vector[8] = float(window.startswith(b"PK\x03\x04"))
        vector[9] = float(window.startswith(b"\x89PNG"))
        vector[10] = float(window.startswith(b"%PDF"))
        vector[11] = float(window.startswith(b"\x1f\x8b"))
        vector[12] = float(entropy > 7.0 and vector[5] > 0.9)
        vector[13] = float(np.mean(data)) / 255.0
        vector[14] = float(np.std(data)) / 128.0
        vector[15] = math.log1p(len(window)) / 12.0
        self._hash_byte_ngrams(window, vector)
        self._hash_transitions(data, vector)
        self._previous_entropy = entropy

        metadata = {
            "source_id": event.get("source_id", "binary-river"),
            "timestamp": event.get("timestamp"),
            "feature_names": self.feature_names,
            "is_anomaly": bool(event.get("is_anomaly", False)),
            "scenario": event.get("scenario", "unknown"),
            "top_drivers": _top_vector_drivers(vector, self.feature_names),
        }
        return np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0), metadata

    def transform_many(self, events: list[bytes | Mapping[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        frames: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for event in events:
            frame, meta = self.transform(event)
            frames.append(frame)
            metadata.append(meta)
        return np.vstack(frames), metadata

    def _hash_byte_ngrams(self, window: bytes, vector: np.ndarray) -> None:
        start = len(BASE_FEATURES)
        width = max(1, (self.features - start) // 2)
        for idx in range(max(0, len(window) - 1)):
            gram = window[idx : idx + 2]
            digest = hashlib.blake2b(gram + self.hash_seed.to_bytes(4, "big"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % width
            vector[start + bucket] += 1.0 / max(1, len(window) - 1)

    def _hash_transitions(self, data: np.ndarray, vector: np.ndarray) -> None:
        start = len(BASE_FEATURES)
        width = self.features - start
        offset = width // 2
        transition_width = max(1, width - offset)
        for left, right in zip(data[:-1], data[1:]):
            bucket = ((int(left) >> 4) * 16 + (int(right) >> 4) + self.hash_seed) % transition_width
            vector[start + offset + bucket] += 1.0 / max(1, data.size - 1)


def generate_binary_stream(
    scenario: str = "normal",
    n_windows: int = 96,
    seed: int = 19,
    window_size: int = 256,
) -> list[dict[str, Any]]:
    """Create deterministic byte-window streams for tests and benchmarks."""

    rng = np.random.default_rng(seed)
    events: list[dict[str, Any]] = []
    for idx in range(n_windows):
        is_anomaly = scenario != "normal" and idx >= n_windows // 2
        if is_anomaly and scenario == "high_entropy_blob":
            payload = rng.integers(0, 256, size=window_size, dtype=np.uint8).tobytes()
        else:
            template = f"GET /status/{idx % 7} HTTP/1.1\r\nHost: service.local\r\nUser-Agent: eidos\r\n\r\n"
            repeated = (template * ((window_size // len(template)) + 1)).encode("ascii")[:window_size]
            noise = rng.integers(32, 127, size=window_size, dtype=np.uint8)
            base = np.frombuffer(repeated, dtype=np.uint8).copy()
            mask = rng.random(window_size) < 0.08
            base[mask] = noise[mask]
            payload = base.tobytes()
        events.append(
            {
                "frame_id": idx,
                "timestamp": float(idx),
                "source_id": "binary.synthetic",
                "bytes": payload,
                "scenario": scenario,
                "is_anomaly": is_anomaly,
            }
        )
    return events


def _byte_entropy(data: np.ndarray) -> float:
    counts = np.bincount(data, minlength=256).astype(np.float64)
    probs = counts[counts > 0.0] / data.size
    return -float(np.sum(probs * np.log2(probs)))


def _repeated_block_ratio(window: bytes, block_size: int = 8) -> float:
    if len(window) < block_size * 2:
        return 0.0
    blocks = [window[idx : idx + block_size] for idx in range(0, len(window) - block_size + 1, block_size)]
    if not blocks:
        return 0.0
    return 1.0 - (len(set(blocks)) / len(blocks))


def _top_vector_drivers(vector: np.ndarray, names: list[str], limit: int = 5) -> list[dict[str, Any]]:
    indices = np.argsort(np.abs(vector))[::-1][:limit]
    return [{"feature": names[int(idx)], "index": int(idx), "value": float(vector[int(idx)])} for idx in indices]
