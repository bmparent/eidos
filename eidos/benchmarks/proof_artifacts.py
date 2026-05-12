"""Proof baseline artifact layout helpers.

This module only defines and creates the artifact layout. It deliberately does
not run benchmarks, tests, model code, or metric aggregation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Union

PathLike = Union[str, Path]

EXPECTED_PROOF_ARTIFACT_FILENAMES = (
    "config.json",
    "benchmark_summary.csv",
    "benchmark_summary.md",
    "pytest_results.xml",
    "environment.txt",
    "git_commit.txt",
    "run_manifest.json",
)

EXPECTED_PROOF_ARTIFACT_SUBDIRS = (
    "scenarios",
    "plots",
)


def create_proof_subdirs(
    out_dir: PathLike,
    subdirs: Iterable[str] = EXPECTED_PROOF_ARTIFACT_SUBDIRS,
) -> Dict[str, Path]:
    """Create proof subdirectories under ``out_dir`` and return their paths."""
    root = Path(out_dir)
    created: Dict[str, Path] = {}
    for name in subdirs:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        created[name] = path
    return created


def create_proof_artifact_dir(out_dir: PathLike) -> Path:
    """Create the proof baseline root plus required scenario and plot folders."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    create_proof_subdirs(root)
    return root


def expected_proof_artifact_paths(out_dir: PathLike) -> Dict[str, Path]:
    """Return expected top-level proof artifact file paths without creating them."""
    root = Path(out_dir)
    return {name: root / name for name in EXPECTED_PROOF_ARTIFACT_FILENAMES}
