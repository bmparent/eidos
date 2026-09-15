"""Deterministic workflow transitions and non-bypassable stage rules."""

from __future__ import annotations

from .persistence import ArtifactStore
from .schemas import AuditVerdict, WorkflowState, WorkflowTransition


class InvalidTransition(RuntimeError):
    pass


ALLOWED: dict[WorkflowState, set[WorkflowState]] = {
    WorkflowState.CREATED: {WorkflowState.TRIAGED, WorkflowState.BLOCKED},
    WorkflowState.TRIAGED: {WorkflowState.EVIDENCE_GATHERING, WorkflowState.IMPLEMENTATION_SPECIFIED, WorkflowState.BLOCKED},
    WorkflowState.EVIDENCE_GATHERING: {WorkflowState.HYPOTHESIS, WorkflowState.EXPERIMENT_SPECIFIED, WorkflowState.READY_FOR_HUMAN, WorkflowState.BLOCKED},
    WorkflowState.HYPOTHESIS: {WorkflowState.EXPERIMENT_SPECIFIED, WorkflowState.READY_FOR_HUMAN, WorkflowState.BLOCKED},
    WorkflowState.EXPERIMENT_SPECIFIED: {WorkflowState.IMPLEMENTATION_SPECIFIED, WorkflowState.BENCHMARKING, WorkflowState.READY_FOR_HUMAN, WorkflowState.BLOCKED},
    WorkflowState.IMPLEMENTATION_SPECIFIED: {WorkflowState.IMPLEMENTING, WorkflowState.BLOCKED},
    WorkflowState.IMPLEMENTING: {WorkflowState.BENCHMARKING, WorkflowState.BLOCKED},
    WorkflowState.BENCHMARKING: {WorkflowState.AUDITING, WorkflowState.BLOCKED},
    WorkflowState.AUDITING: {WorkflowState.READY_FOR_HUMAN, WorkflowState.ESCALATED, WorkflowState.BLOCKED},
    WorkflowState.ESCALATED: {WorkflowState.READY_FOR_HUMAN, WorkflowState.BLOCKED},
    WorkflowState.READY_FOR_HUMAN: {WorkflowState.CLOSED, WorkflowState.BLOCKED},
    WorkflowState.BLOCKED: {WorkflowState.EVIDENCE_GATHERING, WorkflowState.IMPLEMENTATION_SPECIFIED, WorkflowState.IMPLEMENTING, WorkflowState.BENCHMARKING, WorkflowState.AUDITING},
    WorkflowState.CLOSED: set(),
}


class WorkflowMachine:
    def __init__(self, task_id: str, store: ArtifactStore) -> None:
        self.task_id = task_id
        self.store = store

    @property
    def state(self) -> WorkflowState:
        return WorkflowState(self.store.latest_state(self.task_id))

    def transition(
        self,
        to_state: WorkflowState,
        *,
        actor: str,
        reason: str,
        input_artifact_ids: list[str] | None = None,
        output_artifact_ids: list[str] | None = None,
        tests_passed: bool | None = None,
        audit_verdict: AuditVerdict | None = None,
    ) -> WorkflowTransition:
        current = self.state
        if to_state not in ALLOWED[current]:
            raise InvalidTransition(f"{current} -> {to_state} is not allowed")
        if current == WorkflowState.IMPLEMENTING and to_state != WorkflowState.BLOCKED and tests_passed is not True:
            raise InvalidTransition("implementation cannot progress without required tests")
        if (
            current == WorkflowState.AUDITING
            and to_state == WorkflowState.READY_FOR_HUMAN
            and audit_verdict not in {AuditVerdict.PASS, AuditVerdict.PASS_WITH_LIMITATIONS}
        ):
            raise InvalidTransition("only PASS or PASS_WITH_LIMITATIONS can progress from audit")
        if audit_verdict == AuditVerdict.BLOCK and to_state != WorkflowState.BLOCKED:
            raise InvalidTransition("Auditor BLOCK must enter BLOCKED")
        transition = WorkflowTransition(
            task_id=self.task_id,
            from_state=current,
            to_state=to_state,
            actor=actor,
            reason=reason,
            input_artifact_ids=input_artifact_ids or [],
            output_artifact_ids=output_artifact_ids or [],
        )
        self.store.add_transition(transition)
        return transition
