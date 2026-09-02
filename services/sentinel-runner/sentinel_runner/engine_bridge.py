from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from .dataset import PreparedDataset, sha256_file
from .spec import ExperimentSpec


ENGINE_FILENAME = "EIDOS_BRAIN_UNIFIED_v0_4.7.02.py"


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
    spec = importlib.util.spec_from_file_location("eidos_sentinel_lab_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import engine from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_stream_once"):
        raise RuntimeError("full engine does not expose run_stream_once; use the current main-branch monolith")
    version = str(getattr(module, "ENGINE_VERSION", "0.4.7.02"))
    if version not in {"0.4.7.02", "v0.4.7.02"}:
        raise RuntimeError(f"unexpected Eidos engine version: {version}")
    return module


def run_full_engine(dataset: PreparedDataset, spec: ExperimentSpec, artifact_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    engine_path = discover_engine_path()
    engine = load_engine(engine_path, artifact_dir)
    results = engine.run_stream_once(
        dataset.make_gen_factory(),
        est_frames=int(dataset.frames.shape[0]),
        features=64,
        profile_label="sentinel_lab_real_data_engineering",
        session_label=f"sentinel_lab_{spec.dataset.ref.replace('/', '_')}_v{spec.dataset.version}_seed{spec.engine.seed}",
        cfg_overrides={
            "domain": "cicids_webattacks",
            "global_seed": spec.engine.seed,
            "deterministic_cuda": True,
            "demo_enable": False,
        },
        return_step_rows=True,
        return_top_surprises=False,
        sample_geometry=False,
        seed=spec.engine.seed,
    )
    if not isinstance(results, dict):
        raise RuntimeError("full engine returned no structured results")
    return results, {
        "module": str(engine_path),
        "version": str(getattr(engine, "ENGINE_VERSION", "0.4.7.02")),
        "code_sha256": sha256_file(engine_path),
        "process_isolation": True,
    }
