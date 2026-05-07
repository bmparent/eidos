"""Compression primitives for Eidos Brain."""

from .entropy_pack import (
    OptionalCodecUnavailable,
    PackedTokenStream,
    available_packers,
    pack_tokens,
    tokens_from_jsonl,
    tokens_to_jsonl,
    unpack_tokens,
)
from .policy import CompressionDecision, CompressionPolicy, CompressionPolicyConfig, CompressionRule
from .residual_codec import ResidualFirstCodec, anomaly_preservation_score, reconstruction_error
from .stream_writer import InMemoryTokenWriter, JSONLTokenWriter, NullTokenWriter, PackedTokenWriter

__all__ = [
    "CompressionDecision",
    "CompressionPolicy",
    "CompressionPolicyConfig",
    "CompressionRule",
    "OptionalCodecUnavailable",
    "PackedTokenStream",
    "ResidualFirstCodec",
    "InMemoryTokenWriter",
    "JSONLTokenWriter",
    "NullTokenWriter",
    "PackedTokenWriter",
    "anomaly_preservation_score",
    "available_packers",
    "pack_tokens",
    "reconstruction_error",
    "tokens_from_jsonl",
    "tokens_to_jsonl",
    "unpack_tokens",
]
