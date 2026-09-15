"""Read-only environment diagnostics."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from typing import Any

from .agent_definitions import SDKUnavailable, build_sdk_graph
from .config import LabConfig
from .model_registry import ModelRegistry


def run_doctor(config: LabConfig, *, live_model_check: bool = False) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, Any] = {}
    checks["python"] = {"ok": sys.version_info >= (3, 11), "version": sys.version.split()[0]}
    checks["api_key"] = {"ok": bool(os.getenv("OPENAI_API_KEY")), "present": bool(os.getenv("OPENAI_API_KEY"))}
    try:
        importlib.import_module("pydantic")
        checks["pydantic"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001 - doctor must report arbitrary import failures
        checks["pydantic"] = {"ok": False, "error": str(exc)}
    try:
        build_sdk_graph(config)
        checks["agents_sdk"] = {"ok": True, "graph": "Director plus 8 bounded specialist tools"}
    except SDKUnavailable as exc:
        checks["agents_sdk"] = {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - surface SDK construction diagnostics
        checks["agents_sdk"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    checks["repo"] = {"ok": (config.repo_root / ".git").exists(), "path": str(config.repo_root)}
    checks["git"] = {"ok": shutil.which("git") is not None, "path": shutil.which("git")}
    try:
        config.artifact_root.mkdir(parents=True, exist_ok=True)
        probe = config.artifact_root / ".doctor-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks["artifact_root"] = {"ok": True, "path": str(config.artifact_root)}
    except OSError as exc:
        checks["artifact_root"] = {"ok": False, "error": str(exc)}
    checks["models"] = {name: {"model": profile.model, "reasoning": profile.reasoning} for name, profile in config.models.items()}
    if live_model_check:
        try:
            available = ModelRegistry(config).check_openai_availability()
            missing = [profile.model for profile in config.models.values() if profile.model not in available]
            checks["model_availability"] = {"ok": not missing, "missing": missing}
        except Exception as exc:  # noqa: BLE001 - availability may fail by network or policy
            checks["model_availability"] = {"ok": False, "error": str(exc)}
    critical = ("python", "pydantic", "agents_sdk", "repo", "git", "artifact_root")
    return all(bool(checks[name].get("ok")) for name in critical), checks
