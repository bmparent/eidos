"""Strict routed specialist contracts for mandatory research chains.

These contracts keep host-owned accounting and cost artifacts out of specialist prompts and make
the required output type explicit in each routed tool schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .schemas import AgentResult, SpecialistWorkOrder
from .sdk_contracts import SDKAgentResult


def _canonical_archivist_ref(value: str) -> str:
    """Convert a routed Archivist namespace/path reference to its persisted evidence id."""

    text = value.strip()
    if text.startswith("Archivist:"):
        text = text.removeprefix("Archivist:").split("=", 1)[0].strip()
    return text


class SDKArchivistWorkOrder(SpecialistWorkOrder):
    required_output: Literal["SDKArchivistResult"]


class SDKSentryWorkOrder(SpecialistWorkOrder):
    required_output: Literal["SDKSentryResult"]

    @field_validator("evidence_refs")
    @classmethod
    def canonicalize_forwarded_archivist_refs(cls, refs: list[str]) -> list[str]:
        # The host persists Archivist packet ids as E1/E2/... . Director-facing work orders may
        # namespace those refs as Archivist:E1 or annotate them as Archivist:E1=path. Sentry must
        # receive the canonical ids so its structured evidence_refs can be verified exactly against
        # the persisted packet instead of failing provenance checks on presentation differences.
        return [_canonical_archivist_ref(ref) for ref in refs]


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
