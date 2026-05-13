import sys
import os
import pytest
import importlib.util
from unittest.mock import MagicMock
import tempfile
import shutil
import copy

# --- 1. PRE-IMPORT MOCKING ---
@pytest.fixture(scope="session", autouse=True)
def mock_external_deps():
    """Mock heavy external dependencies before any import happens."""
    mocks = [
        "google",
        "google.cloud", 
        "google.cloud.pubsub_v1", 
        "kagglehub", 
        "websockets", 
        "google.colab",
    ]
    for m in mocks:
        # Create module if missing
        if m not in sys.modules:
            sys.modules[m] = MagicMock()
        
    # Ensure submodule linkage
    # e.g. sys.modules['google'].cloud should be sys.modules['google.cloud']
    for m in mocks:
        parts = m.split(".")
        if len(parts) > 1:
            parent_name = ".".join(parts[:-1])
            child_name = parts[-1]
            if parent_name in sys.modules:
                setattr(sys.modules[parent_name], child_name, sys.modules[m])
    yield

# --- 2. MODULE IMPORT ---
@pytest.fixture(scope="session")
def brain_module():
    """Dynamically import the EIDOS Brain engine as a module.
    
    Priority: refactored package copy > monolith.
    The refactored copy contains all Codex fixes + hardening patches.
    """
    # Primary: refactored package engine
    test_dir = os.path.dirname(os.path.dirname(__file__))
    refactored_path = os.path.join(
        test_dir, "repo", "src", "eidos_brain", "engine", "eidos_v0_4_7_02.py"
    )
    # Fallback: monolith
    monolith_path = os.path.join(test_dir, "EIDOS_BRAIN_UNIFIED_v0_4.7.02.py")
    
    target_path = refactored_path if os.path.exists(refactored_path) else monolith_path
    
    spec = importlib.util.spec_from_file_location("eidos_brain", target_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eidos_brain"] = module
    spec.loader.exec_module(module)
    return module

# --- 3. CONFIG ISOLATION ---
@pytest.fixture(autouse=True)
def fresh_config(brain_module):
    """Save and restore EIDOS_BRAIN_CONFIG and basic globals."""
    # Snapshot
    original_config = copy.deepcopy(brain_module.EIDOS_BRAIN_CONFIG)
    
    yield
    
    # Restore
    brain_module.EIDOS_BRAIN_CONFIG.clear()
    brain_module.EIDOS_BRAIN_CONFIG.update(original_config)

# --- 4. TEMP ARTIFACT ROOT ---
@pytest.fixture
def temp_artifact_root(brain_module, tmp_path, monkeypatch):
    """Route all artifacts to a temp dir."""
    path = str(tmp_path / "artifacts")
    monkeypatch.setenv("EIDOS_ARTIFACT_ROOT", path)
    # Also patch the global variable if it was already resolved
    brain_module.EIDOS_DATA_ROOT = path
    brain_module.EIDOS_ARCHIVE_ROOT = os.path.join(path, "archive")
    os.makedirs(brain_module.EIDOS_ARCHIVE_ROOT, exist_ok=True)
    return path
