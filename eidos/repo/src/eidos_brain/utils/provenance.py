"""
provenance.py

Computes engine compatibility hashes and run manifests for reproducibility.
"""

import hashlib
import json
import os
import subprocess
from typing import Dict, Any

def get_engine_hash(engine_path: str = None) -> str:
    """Compute SHA256 of the engine file."""
    if not engine_path:
        # Default to the known location of eidos_v0_4_7_02.py
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        engine_path = os.path.join(base_dir, "engine", "eidos_v0_4_7_02.py")
    
    if not os.path.exists(engine_path):
        return "UNKNOWN"
        
    with open(engine_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_repo_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "UNKNOWN"

def get_config_hash(config: Dict[str, Any]) -> str:
    """Compute hash of normalized config."""
    # JSON dump with sort_keys=True ensures normalization
    s = json.dumps(config, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def write_run_manifest(session_id: str, config: Dict[str, Any], artifact_root: str):
    """Write a manifest for a completed run."""
    manifest = {
        "session_id": session_id,
        "engine_hash": get_engine_hash(),
        "repo_commit": get_repo_commit(),
        "config_hash": get_config_hash(config),
        "config": config
    }
    
    path = os.path.join(artifact_root, f"manifest_{session_id}.json")
    os.makedirs(artifact_root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
