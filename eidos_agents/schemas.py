"""Typed contracts used at every agent and workflow boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    def stable_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


class TaskType(StrEnum):
    RESEARCH = "RESEARCH"
    ENGINEERING = "ENGINEERING"
    MATHEMATICAL = "MATHEMATICAL"
    AUDIT = "AUDIT"


class TaskStatus(StrEnum):
    OPEN = "OPEN"
    PAUSED = "PAUSED"
    BLOCKED = "BLOCKED"
    COMPLETE = "COMPLETE"


class WorkflowState(StrEnum):
    CREATED = "CREATED"
    TRIAGED = "TRIAGED"
    EVIDENCE_GATHERING = "EVIDENCE_GATHERING"
    HYPOTHESIS = "HYPOTHESIS"
    EXPERIMENT_SPECIFIED = "EXPERIMENT_SPECIFIED"
    IMPLEMENTATION_SPECIFIED = "IMPLEMENTATION_SPECIFIED"
    IMPLEMENTING = "IMPLEMENTING"
    BENCHMARKING = "BENCHMARKING"
    AUDITING = "AUDITING"
    ESCALATED = "ESCALATED"
    READY_FOR_HUMAN = "READY_FOR_HUMAN"
    BLOCKED = "BLOCKED"
    CLOSED = "CLOSED"


class ClaimStatus(StrEnum):
    HYPOTHESIS = "HYPOTHESIS"
    SUPPORTED = "SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    REFUTED = "REFUTED"
    KNOWN = "KNOWN"


class EvidenceSourceType(StrEnum):
    PROJECT_EVIDENCE = "PROJECT_EVIDENCE"
    EXTERNAL_EVIDENCE = "EXTERNAL_EVIDENCE"
    INFERENCE = "INFERENCE"
    MISSING = "MISSING"
    CONTRADICTION = "CONTRADICTION"


class BenchmarkStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID"


class AuditVerdict(StrEnum):
    PASS = "PASS"
    PASS_WITH_LIMITATIONS = "PASS_WITH_LIMITATIONS"
    INCONCLUSIVE = "INCONCLUSIVE"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"


class CouncilConclusion(StrEnum):
    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_LIMITATIONS = "SUPPORTED_WITH_LIMITATIONS"
    INCONCLUSIVE = "INCONCLUSIVE"
    REFUTED = "REFUTED"
    REQUIRES_NEW_EXPERIMENT = "REQUIRES_NEW_EXPERIMENT"
    REQUIRES_EXTERNAL_EXPERT_REVIEW = "REQUIRES_EXTERNAL_EXPERT_REVIEW"


class FailureKind(StrEnum):
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"
    SCIENTIFIC_FAILURE = "SCIENTIFIC_FAILURE"
    INVALID_EXPERIMENT = "INVALID_EXPERIMENT"
    PERMISSION_BLOCK = "PERMISSION_BLOCK"
    BUDGET_BLOCK = "BUDGET_BLOCK"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class BudgetSpec(StrictModel):
    task_budget_usd: float | None = Field(default=None, ge=0)
    per_agent_budget_usd: dict[str, float] = Field(default_factory=dict)
    maximum_specialist_calls: int = Field(default=12, ge=1)
    maximum_retries: int = Field(default=1, ge=0)
    maximum_turns: int = Field(default=16, ge=1)
    council_enabled: bool = False
    human_approval_threshold_usd: float | None = Field(default=None, ge=0)


class ResearchRequirements(StrictModel):
    current_evidence_required: bool = False
    required_specialists: list[str] = Field(default_factory=list)
    experiment_spec_required: bool = False
    hypothesis_required: bool = False


class TaskSpec(StrictModel):
    task_id: str
    title: str
    objective: str
    task_type: TaskType
    created_at: datetime = Field(default_factory=utc_now)
    status: TaskStatus = TaskStatus.OPEN
    priority: int = Field(default=3, ge=1, le=5)
    repo: str
    base_commit: str
    allowed_scope: list[str]
    forbidden_scope: list[str] = Field(default_factory=list)
    initial_question: str
    success_criteria: list[str]
    failure_criteria: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    research_requirements: ResearchRequirements = Field(default_factory=ResearchRequirements)
    human_approval_requirements: list[str] = Field(default_factory=list)


class EvidenceItem(StrictModel):
    evidence_id: str
    source_type: EvidenceSourceType
    source: str
    location: str
    claim_supported: str
    excerpt_or_summary: str
    timestamp: datetime = Field(default_factory=utc_now)
    confidence: float = Field(ge=0, le=1)
    stale_possible: bool = False
    hash: str | None = None


class EvidencePacket(StrictModel):
    task_id: str
    question: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    prior_results: list[str] = Field(default_factory=list)
    known_failures: list[str] = Field(default_factory=list)
    recommended_context: list[str] = Field(default_factory=list)


class Hypothesis(StrictModel):
    hypothesis_id: str
    statement: str
    mechanism: str
    alternatives: list[str]
    predictions: list[str]
    falsifiers: list[str]
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.HYPOTHESIS

    @model_validator(mode="after")
    def known_requires_reproducible_evidence(self) -> Hypothesis:
        if self.status == ClaimStatus.KNOWN and (len(self.evidence_for) < 2 or self.evidence_against):
            raise ValueError("KNOWN requires at least two evidence references and no unresolved counterevidence")
        return self


class ExperimentSpec(StrictModel):
    experiment_id: str
    task_id: str
    hypothesis_ids: list[str]
    dataset: str
    sample_policy: str
    controls: list[str]
    negative_controls: list[str]
    ablations: list[str]
    seeds: list[int]
    variables: dict[str, Any]
    metrics: list[str]
    baseline: str
    commands_or_runner_reference: list[str]
    success_observation: str
    failure_observation: str
    ambiguous_observation: str
    artifacts_required: list[str]


class ImplementationSpec(StrictModel):
    implementation_id: str
    objective: str
    allowed_files_modules: list[str]
    forbidden_files_modules: list[str]
    behavior_change: str
    invariants: list[str]
    tests_required: list[str]
    benchmark_required: bool = True
    rollback_condition: str


class ImplementationResult(StrictModel):
    implementation_id: str
    agent: str = "forge"
    files_changed: list[str]
    diff_summary: str
    tests_run: list[str]
    results: dict[str, str]
    commit: str | None = None
    known_risks: list[str] = Field(default_factory=list)
    ready_for_bench: bool = False


class BenchmarkReceipt(StrictModel):
    task_id: str
    experiment_id: str
    run_id: str
    status: BenchmarkStatus
    timestamp: datetime = Field(default_factory=utc_now)
    git_commit: str
    git_dirty_state: bool
    engine_version: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str
    dataset_identity: str
    data_hash: str | None = None
    sample_construction: str
    frame_count: int | None = Field(default=None, ge=0)
    seed: int | None = None
    device: str
    cpu_gpu: str
    precision: float | None = None
    merged_precision: float | None = None
    deduplicated_precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    fp_per_10k: float | None = None
    attack_window_coverage: float | None = None
    detection_latency: float | None = None
    runtime_seconds: float | None = Field(default=None, ge=0)
    memory_bytes: int | None = Field(default=None, ge=0)
    commands: list[str]
    tests: dict[str, str]
    metrics: dict[str, Any]
    crash_scan: list[str]
    artifact_paths: list[str]
    failure_kind: FailureKind | None = None


class AuditResult(StrictModel):
    task_id: str
    auditor: str = "auditor"
    claims_reviewed: list[str]
    evidence_for: list[str]
    evidence_against: list[str]
    methodology_issues: list[str]
    reproducibility: str
    regressions: list[str]
    severity: str
    verdict: AuditVerdict
    corrective_actions: list[str] = Field(default_factory=list)


class CouncilDecision(StrictModel):
    task_id: str
    question: str
    competing_positions: list[str]
    evidence: list[str]
    independent_analysis: str
    conclusion: CouncilConclusion
    confidence: float = Field(ge=0, le=1)
    remaining_uncertainty: list[str]
    decisive_next_experiment: str | None = None


class CostReceipt(StrictModel):
    task_id: str
    price_version: str
    price_effective_date: str | None = None
    total_input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    calls_by_model: dict[str, int] = Field(default_factory=dict)
    calls_by_agent: dict[str, int] = Field(default_factory=dict)
    tool_calls: dict[str, int] = Field(default_factory=dict)
    approximate_cost_usd: float | None = Field(default=None, ge=0)
    budget_remaining: float | None = None
    estimate_notes: list[str] = Field(default_factory=list)


class WorkflowTransition(StrictModel):
    task_id: str
    from_state: WorkflowState
    to_state: WorkflowState
    timestamp: datetime = Field(default_factory=utc_now)
    reason: str
    actor: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)


class RunManifest(StrictModel):
    task_id: str
    task_ref: str
    repo_state: dict[str, Any]
    agents: dict[str, Any]
    models: dict[str, Any]
    experiments: list[str] = Field(default_factory=list)
    implementation: str | None = None
    benchmarks: list[str] = Field(default_factory=list)
    audit: str | None = None
    council: str | None = None
    costs: str | None = None
    final_decision: str | None = None
    tracing_available: bool = False
    tracing_note: str | None = None


class Claim(StrictModel):
    statement: str
    status: ClaimStatus
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def known_has_evidence(self) -> Claim:
        if self.status == ClaimStatus.KNOWN and len(self.evidence_refs) < 2:
            raise ValueError("KNOWN claims require at least two evidence references")
        return self


class AgentResult(StrictModel):
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
    deliverables: dict[str, Any] = Field(default_factory=dict)


class ArchivistResult(AgentResult):
    evidence_packet: EvidencePacket


class CurieResult(AgentResult):
    experiment_spec: ExperimentSpec


class SpecialistAccounting(StrictModel):
    task_id: str
    required: list[str] = Field(default_factory=list)
    attempted: list[str] = Field(default_factory=list)
    completed: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)
    skipped: dict[str, str] = Field(default_factory=dict)


class SpecialistWorkOrder(StrictModel):
    task_id: str
    objective: str
    questions: list[str]
    allowed_scope: list[str]
    forbidden_scope: list[str]
    evidence_refs: list[str]
    required_output: str
    success_criteria: list[str]


class ApprovalRecord(StrictModel):
    task_id: str
    action: str
    approved: bool
    actor: str
    timestamp: datetime = Field(default_factory=utc_now)
    note: str = ""


class FinalDecision(StrictModel):
    task_id: str
    decision: str
    what_changed: list[str]
    what_we_learned: list[str]
    evidence: list[str]
    what_remains_uncertain: list[str]
    auditor_verdict: AuditVerdict | None = None
    regressions: list[str] = Field(default_factory=list)
    cost_receipt_ref: str
    recommended_next_action: str
    human_action_required: list[str] = Field(default_factory=list)
