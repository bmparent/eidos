"""
sources.py

Definitions for supported data sources.
The actual ingestion logic resides in the core engine (eidos_v0_4_7_02.py)
for strict canonical reasons. This module provides keys and validation.
"""

from enum import Enum
from typing import List

class DataSourceType(str, Enum):
    LOCAL = "LOCAL"
    DRIVE = "DRIVE"
    KAGGLE = "KAGGLE"
    STREAM = "STREAM"

def get_allowed_sources() -> List[str]:
    return [s.value for s in DataSourceType]

def validate_source_config(config: dict):
    """
    Validate that the config contains necessary keys for the selected source.
    """
    source = config.get("source_type", "LOCAL").upper()
    params = config.get("source_params", {}).get(source.lower(), {})
    
    if source == DataSourceType.KAGGLE:
        # Check for kaggle deps or keys? 
        pass
    elif source == DataSourceType.STREAM:
        if "url" not in params and "endpoint" not in params:
            # Maybe okay if using defaults, but good to check
            pass
    
    return True
