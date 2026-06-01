"""Repo-root smoke check for the CUDA-safe tensor hotfix.

The script is intentionally small so it can run both locally and in Colab. It
does not create proof artifacts; it only exercises the helper paths that used
to crash when CUDA tensors reached NumPy similarity code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eidos_forecast import ForecastEngine, TrajectoryRecord
from eidos_incident_cards import EpisodeIndex, EpisodeRecord
from eidos_procedural_memory import ProceduralMemory
from eidos_tensor_utils import to_cpu_numpy_1d


def _git(*parts: str) -> str:
    try:
        result = subprocess.run(
            ["git", *parts],
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return f"unknown ({exc})"
    return result.stdout.strip() or f"unknown ({result.stderr.strip()})"


def run_tensor_checks(signature: Any) -> Dict[str, Any]:
    arr = to_cpu_numpy_1d(signature, dtype=np.float32)
    np.testing.assert_allclose(arr, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    index = EpisodeIndex(maxlen=5)
    index.add(
        EpisodeRecord(
            step=7,
            ts=1.0,
            regime="RED",
            z=3.0,
            err=0.25,
            signature=signature,
            entities={},
            exemplars=[],
            top_drivers=[],
        )
    )
    incident_hits = index.topk(signature, regime="RED", k=1)
    assert incident_hits and incident_hits[0]["sim"] > 0.999

    memory = ProceduralMemory(domain="generic", enabled=True)
    memory.update_prototype("ALERT_HUMAN", signature)
    ranked = memory.rank_actions(signature, regime="AMBER")
    assert ranked and ranked[0]["action"] == "ALERT_HUMAN"
    assert ranked[0]["sim"] > 0.999

    forecast = ForecastEngine(window=3, horizons=[10], temp=1.0, enabled=True)
    forecast.trajectories = [
        TrajectoryRecord(
            domain="generic",
            outcome="RED",
            horizon=10,
            sig_seq=[[1.0, 0.0, 0.0]],
            z_seq=[1.0],
            err_seq=[0.1],
        )
    ]
    forecast.update(signature, z=2.0, err=0.2, regime="AMBER", domain="generic")
    risk = forecast.risk(domain="generic", regime="AMBER")
    assert risk["likely_mode"] == "RED"
    assert risk["evidence"] and risk["evidence"][0]["sim"] > 0.999

    return {
        "converted_shape": list(arr.shape),
        "incident_similarity": float(incident_hits[0]["sim"]),
        "procedural_similarity": float(ranked[0]["sim"]),
        "forecast_similarity": float(risk["evidence"][0]["sim"]),
    }


def main() -> int:
    print(f"repo_root={REPO_ROOT}")
    print(f"git_branch={_git('branch', '--show-current')}")
    print(f"git_commit={_git('rev-parse', 'HEAD')}")

    try:
        import torch
    except ImportError as exc:
        print(f"torch_import_error={exc}")
        print("VALIDATION_FAILED torch is required for this tensor smoke check")
        return 1

    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={cuda_available}")
    if cuda_available:
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    else:
        print("cuda_status=unavailable; exercising CPU tensor fallback")

    signature = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True)
    checks = run_tensor_checks(signature)
    print(f"checks={checks}")
    print(f"VALIDATION_OK device={device} cuda_available={cuda_available}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
