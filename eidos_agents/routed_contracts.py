"""Strict routed specialist contracts for mandatory research chains.

These contracts keep host-owned accounting and cost artifacts out of specialist prompts and make
the required output type explicit in each routed tool schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .schemas import AgentResult, SpecialistWorkOrder
from .sdk_contracts import SDKAgentResult


class SDKArchivistWorkOrder(SpecialistWorkOrder):
    required_output: Literal["SDKArchivistResult"]


class SDKSentryWorkOrder(SpecialistWorkOrder):
    required_output: Literal["SDKSentryResult"]


class SDKCurieWorkOrder(SpecialistWorkOrder):
    required_output: Literal["SDKCurieResult"]


class SDKSentryResult(SDKAgentResult):
    """Compact Sentinel-specific structured output carried into the generic persisted result."""

    integrity_gates: list[str] = Field(default_factory=list)
    primary_failure_mode: str
    mandatory_detection_guard: str

    def to_internal(self) -> AgentResult:
        data = self._internal_kwargs()
        data["deliverables"] = {
            "items": [item.model_dump(mode="json") for item in self.deliverables],
            "sentry_integrity": {
                "integrity_gates": self.integrity_gates,
                "primary_failure_mode": self.primary_failure_mode,
                "mandatory_detection_guard": self.mandatory_detection_guard,
            },
        }
        return AgentResult(**data)
