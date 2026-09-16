"""Strict OpenAI Agents SDK wire contracts.

Internal persistence models intentionally remain backward compatible with prior Agent Lab
artifacts. These SDK-facing models avoid open-ended dictionaries/Any so the Agents SDK can
convert them to strict Structured Outputs schemas.
"""

from __future__ import annotations

from pydantic import Field

from .schemas import (
    AgentResult,
    ArchivistResult,
    Claim,
    CurieResult,
    EvidencePacket,
    ExperimentSpec,
    StrictModel,
)


class SDKDeliverable(StrictModel):
    artifact_type: str
    artifact_ref: str | None = None
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)


class SDKExperimentVariable(StrictModel):
    name: str
    role: str
    description: str
    unit: str | None = None


class SDKExperimentSpec(StrictModel):
    experiment_id: str
    task_id: str
    hypothesis_ids: list[str]
    dataset: str
    sample_policy: str
    controls: list[str]
    negative_controls: list[str]
    ablations: list[str]
    seeds: list[int]
    variables: list[SDKExperimentVariable]
    metrics: list[str]
    baseline: str
    commands_or_runner_reference: list[str]
    success_observation: str
    failure_observation: str
    ambiguous_observation: str
    artifacts_required: list[str]

    def to_internal(self) -> ExperimentSpec:
        variables = {
            item.name: {
                "role": item.role,
                "description": item.description,
                "unit": item.unit,
            }
            for item in self.variables
        }
        return ExperimentSpec(
            experiment_id=self.experiment_id,
            task_id=self.task_id,
            hypothesis_ids=self.hypothesis_ids,
            dataset=self.dataset,
            sample_policy=self.sample_policy,
            controls=self.controls,
            negative_controls=self.negative_controls,
            ablations=self.ablations,
            seeds=self.seeds,
            variables=variables,
            metrics=self.metrics,
            baseline=self.baseline,
            commands_or_runner_reference=self.commands_or_runner_reference,
            success_observation=self.success_observation,
            failure_observation=self.failure_observation,
            ambiguous_observation=self.ambiguous_observation,
            artifacts_required=self.artifacts_required,
        )


class SDKAgentResult(StrictModel):
    task_id: str
    agent: str
    status: str
    summary: str
    claims: list[Claim] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    counterevidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    risks: list[str] = Field(default_factory=list)
    next_action: str | None = None
    recommended_agent: str | None = None
    deliverables: list[SDKDeliverable] = Field(default_factory=list)

    def _internal_kwargs(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "agent": self.agent,
            "status": self.status,
            "summary": self.summary,
            "claims": self.claims,
            "evidence_refs": self.evidence_refs,
            "counterevidence_refs": self.counterevidence_refs,
            "confidence": self.confidence,
            "risks": self.risks,
            "next_action": self.next_action,
            "recommended_agent": self.recommended_agent,
            "deliverables": {
                "items": [item.model_dump(mode="json") for item in self.deliverables]
            },
        }

    def to_internal(self) -> AgentResult:
        return AgentResult(**self._internal_kwargs())


class SDKArchivistResult(SDKAgentResult):
    evidence_packet: EvidencePacket

    def to_internal(self) -> ArchivistResult:
        return ArchivistResult(
            **self._internal_kwargs(),
            evidence_packet=self.evidence_packet,
        )


class SDKCurieResult(SDKAgentResult):
    experiment_spec: SDKExperimentSpec

    def to_internal(self) -> CurieResult:
        return CurieResult(
            **self._internal_kwargs(),
            experiment_spec=self.experiment_spec.to_internal(),
        )
