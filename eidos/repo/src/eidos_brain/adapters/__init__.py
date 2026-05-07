"""Defensive telemetry adapters for Eidos Brain."""

from .binary_river_adapter import BinaryRiverAdapter, generate_binary_stream
from .crypto_agility_adapter import CryptoAgilityAdapter, generate_crypto_agility_stream
from .quantum_syndrome_adapter import QuantumSyndromeAdapter, generate_quantum_telemetry_stream

__all__ = [
    "BinaryRiverAdapter",
    "CryptoAgilityAdapter",
    "QuantumSyndromeAdapter",
    "generate_binary_stream",
    "generate_crypto_agility_stream",
    "generate_quantum_telemetry_stream",
]
