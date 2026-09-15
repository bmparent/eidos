"""Centralized model, budget, persistence, and safety configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .schemas import BudgetSpec

DEFAULT_MODELS: dict[str, tuple[str, str]] = {
    "director": ("gpt-5.6-terra", "high"),
    "archivist": ("gpt-5.6-luna", "medium"),
    "curie": ("gpt-5.6-terra", "high"),
    "sentry": ("gpt-5.6-sol", "high"),
    "gauss": ("gpt-5.6-sol", "xhigh"),
    "forge": ("gpt-5.3-codex", "high"),
    "bench": ("gpt-5.4-mini", "medium"),
    "auditor": ("gpt-5.6-sol", "xhigh"),
    "council": ("gpt-6-astra", "max"),
}


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if not value else int(value)


def _float_env(name: str) -> float | None:
    value = os.getenv(name)
    return None if value in (None, "") else float(value)


@dataclass(frozen=True)
class ModelProfile:
    agent: str
    model: str
    reasoning: str
    fallback: str | None = None


@dataclass
class LabConfig:
    repo_root: Path
    artifact_root: Path
    models: dict[str, ModelProfile]
    budget: BudgetSpec
    council_require_approval: bool = True
    max_parallel: int = 3
    dry_run: bool = False
    allow_feature_branch_commits: bool = True
    tracing_enabled: bool = True
    pricing_path: Path | None = None

    @classmethod
    def from_env(cls, repo_root: Path | None = None, *, dry_run: bool = False) -> LabConfig:
        root = (repo_root or Path.cwd()).resolve()
        artifact_setting = os.getenv("EIDOS_AGENT_ARTIFACT_ROOT")
        artifact_root = Path(artifact_setting).resolve() if artifact_setting else root / "artifacts" / "agent_lab"
        models: dict[str, ModelProfile] = {}
        for agent, (default_model, default_reasoning) in DEFAULT_MODELS.items():
            prefix = f"EIDOS_AGENT_{agent.upper()}"
            models[agent] = ModelProfile(
                agent=agent,
                model=os.getenv(f"{prefix}_MODEL", default_model),
                reasoning=os.getenv(f"{prefix}_REASONING", default_reasoning),
                fallback=os.getenv(f"{prefix}_FALLBACK_MODEL") or None,
            )
        budget = BudgetSpec(
            task_budget_usd=_float_env("EIDOS_AGENT_TASK_BUDGET_USD"),
            maximum_specialist_calls=_int_env("EIDOS_AGENT_MAX_SPECIALIST_CALLS", 12),
            maximum_retries=_int_env("EIDOS_AGENT_MAX_RETRIES", 1),
            maximum_turns=_int_env("EIDOS_AGENT_MAX_TURNS", 16),
            council_enabled=_bool_env("EIDOS_AGENT_COUNCIL_ENABLED", False),
            human_approval_threshold_usd=_float_env("EIDOS_AGENT_HUMAN_APPROVAL_THRESHOLD_USD"),
        )
        return cls(
            repo_root=root,
            artifact_root=artifact_root,
            models=models,
            budget=budget,
            council_require_approval=_bool_env("EIDOS_AGENT_COUNCIL_REQUIRE_APPROVAL", True),
            max_parallel=_int_env("EIDOS_AGENT_MAX_PARALLEL", 3),
            dry_run=dry_run,
            allow_feature_branch_commits=_bool_env("EIDOS_AGENT_ALLOW_FEATURE_COMMITS", True),
            tracing_enabled=not _bool_env("OPENAI_AGENTS_DISABLE_TRACING", False),
            pricing_path=root / "config" / "agent_lab" / "model_pricing.yaml",
        )

    @classmethod
    def from_yaml(cls, path: Path, *, repo_root: Path | None = None, dry_run: bool = False) -> LabConfig:
        base = cls.from_env(repo_root, dry_run=dry_run)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for agent, values in data.get("models", {}).items():
            current = base.models[agent]
            base.models[agent] = ModelProfile(
                agent=agent,
                model=values.get("model", current.model),
                reasoning=values.get("reasoning", current.reasoning),
                fallback=values.get("fallback", current.fallback),
            )
        return base

    def ensure_directories(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
