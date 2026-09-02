#!/usr/bin/env python
"""Check optional Eidos Brain runtime dependencies for RNG null-proof runs."""
from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "eidos" / "EIDOS_BRAIN_UNIFIED_v0_4.7.02.py"


class EidosBrainRngPredictor:
    """Minimal initialization probe for the Brain-backed RNG predictor path."""

    def __init__(self, reservoir_cls, *, target_space_size: int = 2, n_reservoir: int = 8):
        self.target_space_size = int(target_space_size)
        self.reservoir = reservoir_cls(
            n_inputs=self.target_space_size,
            n_reservoir=int(n_reservoir),
        )


def line(key: str, value: object) -> None:
    print(f"{key}: {value}")


def main() -> int:
    line("python version", platform.python_version())

    torch = None
    torch_ok = False
    torch_version = "unavailable"
    cuda_available = False
    selected_device = "cpu"
    optional_missing = False

    try:
        import torch as imported_torch  # type: ignore
        torch = imported_torch
        torch_ok = True
        torch_version = getattr(torch, "__version__", "unknown")
        cuda_available = bool(torch.cuda.is_available())
        selected_device = "cuda" if cuda_available else "cpu"
    except Exception as exc:
        optional_missing = True
        torch_version = f"unavailable ({exc.__class__.__name__}: {exc})"

    line("torch import ok", str(torch_ok).lower())
    line("torch version", torch_version)
    line("cuda available", str(cuda_available).lower())
    line("selected device", selected_device)
    line("EIDOS_BRAIN_UNIFIED_v0_4.7.02.py exists", str(ENGINE_PATH.exists()).lower())

    rls_imports = False
    predictor_initializes = False
    engine_error = None

    if not ENGINE_PATH.exists():
        engine_error = f"missing engine file: {ENGINE_PATH}"
    elif torch_ok:
        try:
            spec = importlib.util.spec_from_file_location("eidos_brain_unified_runtime_check", ENGINE_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not build import spec for {ENGINE_PATH}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            rls = getattr(module, "RLS_Reservoir")
            rls_imports = True
            predictor = EidosBrainRngPredictor(rls, target_space_size=2, n_reservoir=8)
            predictor_initializes = predictor.reservoir is not None
        except Exception as exc:
            engine_error = f"{exc.__class__.__name__}: {exc}"

    line("RLS_Reservoir imports", str(rls_imports).lower())
    line("EidosBrainRngPredictor can initialize", str(predictor_initializes).lower())
    if engine_error:
        line("engine error", engine_error)

    if predictor_initializes:
        return 0
    if optional_missing:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
