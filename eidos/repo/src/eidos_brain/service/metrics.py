"""
metrics.py

Exports Prometheus-style metrics or JSON status.
"""

from typing import Dict, Any

def get_metrics() -> Dict[str, Any]:
    return {
        "eidos_sessions_total": 0,
        "eidos_anomalies_total": 0,
        "eidos_memory_usage_bytes": 0
    }
