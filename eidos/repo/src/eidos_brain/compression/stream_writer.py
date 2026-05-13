"""Streaming token writers for bounded-memory residual compression artifacts."""

from __future__ import annotations

import gzip
import json
import lzma
import struct
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from .entropy_pack import MAGIC, OptionalCodecUnavailable, tokens_to_jsonl


ANOMALY_CAPSULE_MODES = {"anomaly_capsule", "raw_frame_plus_full_context"}


def _token_line(token: Mapping[str, Any]) -> bytes:
    return (json.dumps(token, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


class TokenWriterBase:
    """Common tracking for token stream writers."""

    def __init__(self, codec: str, flush_every: int = 512, flush_bytes: int = 1_048_576) -> None:
        self.codec = codec
        self.flush_every = max(1, int(flush_every))
        self.flush_bytes = max(1, int(flush_bytes))
        self.tokens_written = 0
        self.bytes_written = 0
        self.chunks_written = 0
        self.first_frame_id: Any | None = None
        self.last_frame_id: Any | None = None
        self.anomaly_capsule_count = 0
        self.compression_mode_counts: Counter[str] = Counter()
        self._closed = False
        self._final_summary: dict[str, Any] | None = None

    def __enter__(self) -> "TokenWriterBase":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.finalize()
        return False

    def __del__(self) -> None:
        if not self._closed:
            try:
                self.finalize()
            except Exception:
                pass

    def open(self) -> None:
        return None

    def write(self, token: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self._closed = True

    def finalize(self) -> dict[str, Any]:
        if self._final_summary is not None:
            return self._final_summary
        self.flush()
        self.close()
        self._final_summary = self.summary()
        return self._final_summary

    def summary(self) -> dict[str, Any]:
        return {
            "path": None,
            "codec": self.codec,
            "tokens_written": self.tokens_written,
            "bytes_written": self.bytes_written,
            "chunks_written": self.chunks_written,
            "first_frame_id": self.first_frame_id,
            "last_frame_id": self.last_frame_id,
            "anomaly_capsule_count": self.anomaly_capsule_count,
            "compression_mode_counts": dict(self.compression_mode_counts),
        }

    def _observe(self, token: Mapping[str, Any]) -> None:
        frame_id = token.get("frame_id")
        if self.first_frame_id is None:
            self.first_frame_id = frame_id
        self.last_frame_id = frame_id
        mode = str(token.get("compression_mode", "unknown"))
        self.compression_mode_counts[mode] += 1
        if mode in ANOMALY_CAPSULE_MODES:
            self.anomaly_capsule_count += 1
        self.tokens_written += 1


class JSONLTokenWriter(TokenWriterBase):
    """Write one token stream as JSON Lines in bounded chunks."""

    def __init__(self, path: str | Path, flush_every: int = 512, flush_bytes: int = 1_048_576) -> None:
        super().__init__("jsonl", flush_every=flush_every, flush_bytes=flush_bytes)
        self.path = Path(path)
        self._handle: BinaryIO | None = None
        self._buffer = bytearray()
        self._tokens_since_flush = 0

    def open(self) -> None:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("wb")

    def write(self, token: Mapping[str, Any]) -> None:
        if self._closed:
            raise ValueError("cannot write to a closed token writer")
        self.open()
        payload = _token_line(token)
        self._buffer.extend(payload)
        self._tokens_since_flush += 1
        self._observe(token)
        if self._tokens_since_flush >= self.flush_every or len(self._buffer) >= self.flush_bytes:
            self.flush()

    def flush(self) -> None:
        if self._handle is None:
            return
        if self._buffer:
            self._handle.write(self._buffer)
            self.bytes_written += len(self._buffer)
            self._buffer.clear()
            self._tokens_since_flush = 0
            self.chunks_written += 1
        self._handle.flush()

    def close(self) -> None:
        if not self._closed:
            self.flush()
            if self._handle is not None:
                self._handle.close()
            self._closed = True

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        summary["path"] = str(self.path)
        if self.path.exists():
            summary["bytes_written"] = self.path.stat().st_size
            self.bytes_written = summary["bytes_written"]
        return summary


class PackedTokenWriter(TokenWriterBase):
    """Write packed token streams incrementally.

    ``gzip`` and ``lzma`` are streaming JSONL compressors. ``binary`` uses the
    existing Eidos token binary format: MAGIC followed by length-prefixed JSON.
    ``zstd`` is used only when the optional ``zstandard`` package is installed.
    """

    def __init__(
        self,
        path: str | Path,
        codec: str = "gzip",
        flush_every: int = 512,
        flush_bytes: int = 1_048_576,
    ) -> None:
        codec_key = codec.lower()
        super().__init__(codec_key, flush_every=flush_every, flush_bytes=flush_bytes)
        self.path = Path(path)
        self._raw: BinaryIO | None = None
        self._stream: Any | None = None
        self._buffer = bytearray()
        self._tokens_since_flush = 0
        self.input_bytes = 0
        self._zstd_module: Any | None = None

    def open(self) -> None:
        if self._raw is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = self.path.open("wb")
        if self.codec == "gzip":
            self._stream = gzip.GzipFile(fileobj=self._raw, mode="wb")
        elif self.codec == "lzma":
            self._stream = lzma.LZMAFile(self._raw, mode="wb")
        elif self.codec in {"binary", "jsonl"}:
            self._stream = self._raw
            if self.codec == "binary":
                self._raw.write(MAGIC)
        elif self.codec == "zstd":
            try:
                import zstandard as zstd  # type: ignore
            except ImportError as exc:
                self._raw.close()
                self._raw = None
                raise OptionalCodecUnavailable(
                    "zstd packing requested but the 'zstandard' package is not installed"
                ) from exc
            self._zstd_module = zstd
            self._stream = zstd.ZstdCompressor(level=3).stream_writer(self._raw)
        else:
            self._raw.close()
            self._raw = None
            raise ValueError(f"unknown token packer: {self.codec}")

    def write(self, token: Mapping[str, Any]) -> None:
        if self._closed:
            raise ValueError("cannot write to a closed token writer")
        self.open()
        if self.codec == "binary":
            payload = json.dumps(token, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            encoded = struct.pack(">I", len(payload)) + payload
        else:
            encoded = _token_line(token)
        self._buffer.extend(encoded)
        self.input_bytes += len(encoded)
        self._tokens_since_flush += 1
        self._observe(token)
        if self._tokens_since_flush >= self.flush_every or len(self._buffer) >= self.flush_bytes:
            self.flush()

    def flush(self) -> None:
        if self._stream is None:
            return
        if self._buffer:
            self._stream.write(self._buffer)
            self._buffer.clear()
            self._tokens_since_flush = 0
            self.chunks_written += 1
        try:
            if self.codec == "zstd" and self._zstd_module is not None:
                self._stream.flush(self._zstd_module.FLUSH_BLOCK)
            else:
                self._stream.flush()
        except (AttributeError, ValueError):
            pass
        if self._raw is not None:
            self._raw.flush()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        if self._stream is not None and self._stream is not self._raw:
            self._stream.close()
        if self._raw is not None and not self._raw.closed:
            self._raw.close()
        self._closed = True

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        summary["path"] = str(self.path)
        summary["input_bytes"] = self.input_bytes
        if self.path.exists():
            summary["bytes_written"] = self.path.stat().st_size
            self.bytes_written = summary["bytes_written"]
        return summary


class InMemoryTokenWriter(TokenWriterBase):
    """Bounded-test/backward-compatible writer that keeps tokens in memory."""

    def __init__(self) -> None:
        super().__init__("memory")
        self.tokens: list[dict[str, Any]] = []

    def write(self, token: Mapping[str, Any]) -> None:
        self.tokens.append(dict(token))
        self._observe(token)

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        summary["bytes_written"] = len(tokens_to_jsonl(self.tokens))
        self.bytes_written = summary["bytes_written"]
        return summary


class NullTokenWriter(TokenWriterBase):
    """Discard tokens while retaining stream statistics."""

    def __init__(self) -> None:
        super().__init__("null")

    def write(self, token: Mapping[str, Any]) -> None:
        self._observe(token)
