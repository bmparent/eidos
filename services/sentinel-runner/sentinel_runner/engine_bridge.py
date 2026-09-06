from __future__ import annotations

import importlib.util
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple

from .dataset import PreparedDataset, sha256_file
from .spec import ExperimentSpec
from .profiles import EXECUTION_PROFILES, require_profile_capacity


ENGINE_FILENAME = "EIDOS_BRAIN_UNIFIED_v0_4.7.02.py"
ENGINEERING_PROFILE = {
    "reservoir": 256,
    "warmup_cap": 500,
    "hippocampus_dim": 2048,
    "hippocampus_log_every": 500,
    "trace_seal_diag_every": 500,
}


def discover_engine_path() -> Path:
    configured = os.environ.get("EIDOS_ENGINE_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.extend((parent / "eidos" / ENGINE_FILENAME, parent / ENGINE_FILENAME))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"full Eidos engine not found; set EIDOS_ENGINE_PATH to {ENGINE_FILENAME}")


def load_engine(path: Path, artifact_dir: Path) -> Any:
    os.environ["EIDOS_ARTIFACT_ROOT"] = str(artifact_dir)
    os.environ["EIDOS_DATA_SOURCE_TYPE"] = "LOCAL"
    engine_directory = str(path.parent)
    inserted_engine_directory = engine_directory not in sys.path
    if inserted_engine_directory:
        sys.path.insert(0, engine_directory)
    try:
        spec = importlib.util.spec_from_file_location("eidos_sentinel_lab_engine", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import engine from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if inserted_engine_directory:
            sys.path.remove(engine_directory)
    if not hasattr(module, "run_stream_once"):
        raise RuntimeError("full engine does not expose run_stream_once; use the current main-branch monolith")
    version = str(getattr(module, "ENGINE_VERSION", "0.4.7.02"))
    if version not in {"0.4.7.02", "v0.4.7.02"}:
        raise RuntimeError(f"unexpected Eidos engine version: {version}")
    return module


def run_full_engine(dataset: PreparedDataset, spec: ExperimentSpec, artifact_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    require_profile_capacity(spec.engine.execution_profile)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    engine_path = discover_engine_path()
    engine = load_engine(engine_path, artifact_dir)
    effective_overrides = {
        **ENGINEERING_PROFILE,
        **EXECUTION_PROFILES[spec.engine.execution_profile],
        "domain": "cicids_webattacks",
        "global_seed": spec.engine.seed,
        "deterministic_cuda": True,
        "demo_enable": False,
    }
    results = engine.run_stream_once(
        dataset.make_gen_factory(),
        est_frames=int(dataset.frames.shape[0]),
        features=64,
        profile_label="sentinel_lab_real_data_engineering",
        session_label=f"sentinel_lab_{spec.dataset.ref.replace('/', '_')}_v{spec.dataset.version}_seed{spec.engine.seed}",
        cfg_overrides=effective_overrides,
        return_step_rows=True,
        return_top_surprises=False,
        sample_geometry=True,
        geom_sample_every=max(1, int(dataset.frames.shape[0]) // 360),
        max_geom_samples=360,
        seed=spec.engine.seed,
    )
    if not isinstance(results, dict):
        raise RuntimeError("full engine returned no structured results")
    # Project bounded, actual reservoir samples for the Lab. This is a visual
    # projection, not a proof of dimension, stability, or detection quality.
    geometry_root = artifact_dir / "reservoir_geometry" / "sentinel_lab_real_data_engineering"
    state_files = list(geometry_root.glob("*_reservoir_states_*.npy"))
    geometry_files = list(geometry_root.glob("*_reservoir_geom_*.json"))
    if len(state_files) != 1 or len(geometry_files) != 1:
        raise RuntimeError("ENGINE_GEOMETRY_MISSING: expected one bounded reservoir sample")
    states = np.load(state_files[0], allow_pickle=False)
    steps = json.loads(geometry_files[0].read_text(encoding="utf-8"))["steps"]
    if states.ndim != 2 or not 3 <= len(states) <= 360 or len(steps) != len(states) or not np.isfinite(states).all():
        raise RuntimeError("INVALID_ENGINE_GEOMETRY")
    centered = states.astype(np.float64) - states.mean(axis=0, keepdims=True)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ axes[:3].T
    variance = singular ** 2
    results["lab_geometry"] = {
        "method": "Centered PCA of at most 360 sampled reservoir states; three coordinates per state",
        "variance_explained": float(variance[:3].sum() / variance.sum()) if variance.sum() else 0.0,
        "states_sha256": sha256_file(state_files[0]),
        "points": [{"step": int(step), "x": float(point[0]), "y": float(point[1]), "z": float(point[2])} for step, point in zip(steps, coordinates)],
    }
    effective_config = results.get("config")
    if not isinstance(effective_config, dict) or any(effective_config.get(key) != value for key, value in effective_overrides.items()):
        raise RuntimeError("ENGINE_PROFILE_MISMATCH: full engine did not attest the requested configuration")
    return results, {
        "module": str(engine_path),
        "version": str(getattr(engine, "ENGINE_VERSION", "0.4.7.02")),
        "code_sha256": sha256_file(engine_path),
        "process_isolation": True,
        "config_profile": "cicids_webattacks",
        "effective_overrides": effective_overrides,
        "execution_profile": spec.engine.execution_profile,
        "effective_config": effective_config,
    }
