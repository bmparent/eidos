"""
config.py

Handles configuration loading from YAML and Environment variables.
Priority: Environment Vars > CLI/Runtime Overrides > YAML Config > Defaults
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and apply environment variable overrides.
    """
    config: Dict[str, Any] = {}

    # 1. Load default config (if we had one bundled, but we rely on passed path or empty)
    # For this setup, we'll try to load 'configs/default.yaml' relative to repo root if nothing passed
    if not config_path:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        default_path = os.path.join(base_dir, "configs", "default.yaml")
        if os.path.exists(default_path):
            config_path = default_path

    if config_path and os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning("No config file found or provided. Using defaults.")

    # 2. Env Var Overrides (EIDOS_*)
    # Map Env Vars to config keys
    
    # Data Source
    if os.environ.get("EIDOS_DATA_SOURCE"):
        config["source_type"] = os.environ["EIDOS_DATA_SOURCE"]
    
    # Artifact Root
    if os.environ.get("EIDOS_ARTIFACT_ROOT"):
        config["artifact_root"] = os.environ["EIDOS_ARTIFACT_ROOT"]
        
    # Hive Backend
    if os.environ.get("HIVE_BACKEND"):
        # We might need to pass this specially if the engine doesn't look at config for this
        # The engine looks at os.environ directly for HIVE_BACKEND usually
        pass 

    # Google API Key is handled by engine via env var directly

    return config
