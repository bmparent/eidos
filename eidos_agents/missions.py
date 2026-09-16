"""Mission definition loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .schemas import ResearchRequirements


class MissionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mission_id: str
    title: str
    objective: str
    mode: str
    allowed_scope: list[str]
    forbidden_scope: list[str]
    evidence_targets: list[str]
    evaluation_dimensions: list[str]
    workflow: list[str]
    implementation_authorized: bool = False
    required_artifacts: list[str] = Field(default_factory=list)
    research_requirements: ResearchRequirements = Field(default_factory=ResearchRequirements)

    @classmethod
    def load(cls, path: Path) -> MissionDefinition:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
