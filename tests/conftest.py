"""Test path setup for repository-root proof harness tests."""
from __future__ import annotations
import sys
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Full-repository collection may have imported eidos/proof before this subtree.
# Retain that package's existing modules while making the root proof harnesses
# discoverable. This affects pytest collection only, not benchmark execution.
proof_package = importlib.import_module("proof")
for proof_dir in (ROOT / "proof", ROOT / "eidos" / "proof"):
    if str(proof_dir) not in proof_package.__path__:
        proof_package.__path__.append(str(proof_dir))
