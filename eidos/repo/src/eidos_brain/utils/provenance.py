"""
provenance.py

Computes engine compatibility hashes and run manifests for reproducibility.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

def get_engine_hash(engine_path: Optional[str] = None) -> str:
    """Compute SHA256 of the engine file."""
    if not engine_path:
        # Default to the known location of eidos_v0_4_7_02.py
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        engine_path = os.path.join(base_dir, "engine", "eidos_v0_4_7_02.py")
    
    if not os.path.exists(engine_path):
        return "UNKNOWN"
        
    sha = hashlib.sha256()
    try:
        with open(engine_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
    except Exception:
        return "UNKNOWN"
    return sha.hexdigest()

def get_repo_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "UNKNOWN"

def get_config_hash(config: Dict[str, Any]) -> str:
    """Compute hash of normalized config."""
    # JSON dump with sort_keys=True ensures normalization
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()

def _atomic_write_text(path: str, text: str) -> None:
    """Atomically write text to disk (best-effort)."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".manifest_", suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def write_run_manifest(
    session_id: str,
    config: Dict[str, Any],
    artifact_root: str,
    *,
    filename: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Write a manifest for a completed run."""
    manifest = {
        "session_id": session_id,
        "engine_hash": get_engine_hash(),
        "repo_commit": get_repo_commit(),
        "config_hash": get_config_hash(config),
        "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if extra:
        manifest.update(extra)

    file_name = filename or f"manifest_{session_id}.json"
    path = os.path.join(artifact_root, file_name)
    try:
        _atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True, default=str))
    except Exception:
        return None
    return path
