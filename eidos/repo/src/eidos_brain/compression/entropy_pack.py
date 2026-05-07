"""Entropy packing for Eidos token streams."""

from __future__ import annotations

import gzip
import json
import lzma
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


MAGIC = b"EIDOSTOK1"


class OptionalCodecUnavailable(RuntimeError):
    """Raised when an optional entropy codec is requested but not installed."""


@dataclass(frozen=True)
class PackedTokenStream:
    codec: str
    data: bytes
    token_count: int
    jsonl_bytes: int


def tokens_to_jsonl(tokens: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [json.dumps(token, sort_keys=True, separators=(",", ":"), ensure_ascii=True) for token in tokens]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def tokens_from_jsonl(data: bytes | str) -> list[dict[str, Any]]:
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def tokens_to_binary(tokens: Iterable[Mapping[str, Any]]) -> bytes:
    chunks = [MAGIC]
    for token in tokens:
        payload = json.dumps(token, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        chunks.append(struct.pack(">I", len(payload)))
        chunks.append(payload)
    return b"".join(chunks)


def tokens_from_binary(data: bytes) -> list[dict[str, Any]]:
    if not data.startswith(MAGIC):
        raise ValueError("not an Eidos token binary stream")
    offset = len(MAGIC)
    tokens: list[dict[str, Any]] = []
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError("truncated token length")
        (size,) = struct.unpack(">I", data[offset : offset + 4])
        offset += 4
        payload = data[offset : offset + size]
        if len(payload) != size:
            raise ValueError("truncated token payload")
        tokens.append(json.loads(payload.decode("utf-8")))
        offset += size
    return tokens


def _zstd_module() -> Any:
    try:
        import zstandard as zstd  # type: ignore
    except ImportError as exc:
        raise OptionalCodecUnavailable("zstd packing requested but the 'zstandard' package is not installed") from exc
    return zstd


def available_packers() -> list[str]:
    packers = ["jsonl", "binary", "gzip", "lzma"]
    try:
        _zstd_module()
    except OptionalCodecUnavailable:
        return packers
    return [*packers, "zstd"]


def pack_tokens(tokens: Iterable[Mapping[str, Any]], codec: str = "gzip") -> PackedTokenStream:
    token_list = list(tokens)
    jsonl = tokens_to_jsonl(token_list)
    codec_key = codec.lower()
    if codec_key == "jsonl":
        data = jsonl
    elif codec_key == "binary":
        data = tokens_to_binary(token_list)
    elif codec_key == "gzip":
        data = gzip.compress(jsonl)
    elif codec_key == "lzma":
        data = lzma.compress(jsonl)
    elif codec_key == "zstd":
        zstd = _zstd_module()
        data = zstd.ZstdCompressor(level=3).compress(jsonl)
    else:
        raise ValueError(f"unknown token packer: {codec}")
    return PackedTokenStream(codec=codec_key, data=data, token_count=len(token_list), jsonl_bytes=len(jsonl))


def unpack_tokens(packed: PackedTokenStream | bytes, codec: str | None = None) -> list[dict[str, Any]]:
    if isinstance(packed, PackedTokenStream):
        codec_key = packed.codec
        data = packed.data
    else:
        if codec is None:
            raise ValueError("codec is required when unpacking raw bytes")
        codec_key = codec.lower()
        data = packed

    if codec_key == "jsonl":
        return tokens_from_jsonl(data)
    if codec_key == "binary":
        return tokens_from_binary(data)
    if codec_key == "gzip":
        return tokens_from_jsonl(gzip.decompress(data))
    if codec_key == "lzma":
        return tokens_from_jsonl(lzma.decompress(data))
    if codec_key == "zstd":
        zstd = _zstd_module()
        return tokens_from_jsonl(zstd.ZstdDecompressor().decompress(data))
    raise ValueError(f"unknown token packer: {codec_key}")


def compression_ratio(raw_bytes: int, compressed_bytes: int) -> float:
    if compressed_bytes <= 0:
        return 0.0
    return float(raw_bytes / compressed_bytes)
