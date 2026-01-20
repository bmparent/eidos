"""
adapters.py

Wraps the core engine entrypoint to provide a stable API surface.
This isolates the rest of the application from the specific version of the engine.
"""

import logging
from typing import Dict, Any

# Import the specific engine version
# In a real scenario, this might be dynamic or configured, but here we bind to v0.4.7.02
from .eidos_v0_4_7_02 import run as _engine_run

logger = logging.getLogger(__name__)

def run_session(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single Eidos Brain session with the provided configuration.
    
    Args:
        config: Dictionary containing all configuration parameters.
                Structure expected by eidos_v0_4_7_02.run()
    
    Returns:
        Dict containing session summary, status, and artifacts.
    """
    logger.info("Starting Eidos Brain session via adapter")
    try:
        result = _engine_run(config)
        return result
    except Exception as e:
        logger.exception("Critical engine failure")
        return {
            "status": "CRITICAL_FAILURE",
            "error": str(e)
        }
