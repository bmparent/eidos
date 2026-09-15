"""Model routing and explicit availability/fallback recording."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import LabConfig, ModelProfile


class ModelUnavailableError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, config: LabConfig) -> None:
        self.config = config
        self.substitutions: list[dict[str, str]] = []

    def profile(self, agent: str) -> ModelProfile:
        try:
            return self.config.models[agent.lower()]
        except KeyError as exc:
            raise KeyError(f"unknown agent: {agent}") from exc

    def resolve(self, agent: str, available_models: set[str] | None = None) -> ModelProfile:
        profile = self.profile(agent)
        if available_models is None or profile.model in available_models:
            return profile
        if profile.fallback and profile.fallback in available_models:
            replacement = ModelProfile(agent=profile.agent, model=profile.fallback, reasoning=profile.reasoning)
            self.substitutions.append(
                {"agent": agent, "configured": profile.model, "used": profile.fallback, "reason": "configured model unavailable"}
            )
            return replacement
        raise ModelUnavailableError(
            f"configured model {profile.model!r} for {agent} is unavailable and no explicit available fallback is configured"
        )

    def check_openai_availability(self) -> set[str]:
        try:
            from openai import OpenAI

            return {item.id for item in OpenAI(timeout=20.0, max_retries=0).models.list().data}
        except Exception as exc:  # credentials, network, or organization policy
            raise ModelUnavailableError(f"model availability check failed: {type(exc).__name__}: {exc}") from exc

    def manifest(self) -> dict[str, Any]:
        return {
            "configured": {name: asdict(profile) for name, profile in self.config.models.items()},
            "substitutions": list(self.substitutions),
        }
