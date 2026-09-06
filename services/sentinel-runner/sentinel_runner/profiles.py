"""Allowlisted, receipt-backed profiles; no arbitrary engine overrides from clients."""
import os

EXECUTION_PROFILES = {
    "cpu_engineering": {"reservoir": 256, "hippocampus_dim": 2048, "fractal_bands": 1, "trace_seal_enabled": False},
    "cpu_mechanisms": {"reservoir": 256, "hippocampus_dim": 2048, "fractal_bands": 4, "trace_seal_enabled": True},
    "full_capacity": {"reservoir": 2000, "hippocampus_dim": 10000, "fractal_bands": 1, "trace_seal_enabled": False},
}


def require_profile_capacity(profile: str) -> None:
    if profile not in EXECUTION_PROFILES:
        raise ValueError("unsupported engine execution profile")
    if profile == "full_capacity" and os.environ.get("EIDOS_ENABLE_FULL_CAPACITY") != "1":
        raise ValueError("FULL_CAPACITY_NOT_ENABLED: use a dedicated runner with EIDOS_ENABLE_FULL_CAPACITY=1")
