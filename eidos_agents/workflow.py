"""Deterministic workflow transitions and non-bypassable stage rules."""

from __future__ import annotations

from .persistence import ArtifactStore
from .schemas import (
    AuditVerdict,
    EvidencePacket,
    ExperimentSpec,
    Hypothesis,
    SpecialistAccounting,
    TaskSpec,
    WorkflowState,
    WorkflowTransition,
)


class InvalidTransition(RuntimeError):
    pass


class StageRequirementUnsatisfied(InvalidTransition):
    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__("stage requirements unsatisfied: " + "; ".join(missing))


ALLOWED: dict[WorkflowState, set[WorkflowState]] = {
    WorkflowState.CREATED: {WorkflowState.TRIAGED, WorkflowState.BLOCKED},
    WorkflowState.TRIAGED: {
        WorkflowState.EVIDENCE_GATHERING,
        WorkflowState.IMPLEMENTATION_SPECIFIED,
        WorkflowState.BLOCKED,
    },
    WorkflowState.EVIDENCE_GATHERING: {
        WorkflowState.HYPOTHESIS,
        WorkflowState.EXPERIMENT_SPECIFIED,
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.BLOCKED,
    },
    WorkflowState.HYPOTHESIS: {
        WorkflowState.EXPERIMENT_SPECIFIED,
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.BLOCKED,
    },
    WorkflowState.EXPERIMENT_SPECIFIED: {
        WorkflowState.IMPLEMENTATION_SPECIFIED,
        WorkflowState.BENCHMARKING,
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.BLOCKED,
    },
    WorkflowState.IMPLEMENTATION_SPECIFIED: {
        WorkflowState.IMPLEMENTING,
        WorkflowState.BLOCKED,
    },
    WorkflowState.IMPLEMENTING: {
        WorkflowState.BENCHMARKING,
        WorkflowState.BLOCKED,
    },
    WorkflowState.BENCHMARKING: {
        WorkflowState.AUDITING,
        WorkflowState.BLOCKED,
    },
    WorkflowState.AUDITING: {
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.ESCALATED,
        WorkflowState.BLOCKED,
    },
    WorkflowState.ESCALATED: {
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.BLOCKED,
    },
    WorkflowState.READY_FOR_HUMAN: {
        WorkflowState.CLOSED,
        WorkflowState.BLOCKED,
    },
    WorkflowState.BLOCKED: {
        WorkflowState.EVIDENCE_GATHERING,
        WorkflowState.IMPLEMENTATION_SPECIFIED,
        WorkflowState.IMPLEMENTING,
        WorkflowState.BENCHMARKING,
        WorkflowState.AUDITING,
    },
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
        if (
            current == WorkflowState.IMPLEMENTING
            and to_state != WorkflowState.BLOCKED
            and tests_passed is not True
        ):
            raise InvalidTransition(
                "implementation cannot progress without required tests"
            )
        if (
            current == WorkflowState.AUDITING
            and to_state == WorkflowState.READY_FOR_HUMAN
            and audit_verdict
            not in {AuditVerdict.PASS, AuditVerdict.PASS_WITH_LIMITATIONS}
        ):
            raise InvalidTransition(
                "only PASS or PASS_WITH_LIMITATIONS can progress from audit"
            )
        if audit_verdict == AuditVerdict.BLOCK and to_state != WorkflowState.BLOCKED:
            raise InvalidTransition("Auditor BLOCK must enter BLOCKED")
        self._validate_reason(reason)
        if to_state == WorkflowState.READY_FOR_HUMAN:
            missing = self.unsatisfied_research_requirements()
            if missing:
                raise StageRequirementUnsatisfied(missing)
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

    def unsatisfied_research_requirements(self) -> list[str]:
        try:
            task = self.store.get_record("task", self.task_id, TaskSpec)
        except KeyError:
            return []
        requirements = task.research_requirements
        missing: list[str] = []
        evidence_packets = self.store.records_for_task(
            "evidence", self.task_id, EvidencePacket
        )
        agent_results = self.store.raw_records_for_task("agent_result", self.task_id)
        experiments = self.store.records_for_task(
            "experiment", self.task_id, ExperimentSpec
        )
        hypotheses = self.store.records_for_task(
            "hypothesis", self.task_id, Hypothesis
        )
        accounting_records = self.store.records_for_task(
            "specialist_accounting", self.task_id, SpecialistAccounting
        )
        accounting = accounting_records[-1] if accounting_records else None

        valid_evidence = [
            packet
            for packet in evidence_packets
            if packet.task_id == self.task_id
            and bool(packet.evidence or packet.missing_evidence or packet.contradictions)
        ]
        archivist_result_exists = any(
            str(result.get("agent", "")).lower() == "archivist"
            for result in agent_results
        )
        if requirements.current_evidence_required:
            if not valid_evidence:
                missing.append("persisted valid EvidencePacket for current task")
            if not archivist_result_exists:
                missing.append("persisted Archivist result")

        if requirements.required_specialists:
            if accounting is None:
                missing.append("persisted specialist completion accounting")
            else:
                for specialist in requirements.required_specialists:
                    if specialist in accounting.completed:
                        continue
                    if specialist in accounting.failed:
                        missing.append(
                            f"required specialist failed: {specialist}: "
                            f"{accounting.failed[specialist]}"
                        )
                    elif specialist in accounting.skipped:
                        missing.append(
                            f"required specialist skipped: {specialist}: "
                            f"{accounting.skipped[specialist]}"
                        )
                    else:
                        missing.append(
                            f"required specialist not completed: {specialist}"
                        )
                if all(
                    specialist in accounting.completed
                    for specialist in requirements.required_specialists
                ):
                    positions = [
                        accounting.completed.index(specialist)
                        for specialist in requirements.required_specialists
                    ]
                    if positions != sorted(positions):
                        missing.append(
                            "required specialist order violated: expected "
                            + " -> ".join(requirements.required_specialists)
                        )

        if requirements.experiment_spec_required and not any(
            experiment.task_id == self.task_id for experiment in experiments
        ):
            missing.append("persisted ExperimentSpec for current task")
        if requirements.hypothesis_required and not hypotheses:
            missing.append("persisted Hypothesis for current task")
        return missing

    def _validate_reason(self, reason: str) -> None:
        """Validate factual artifact claims without interpreting free-form status prose.

        Specialist completion is determined exclusively from SpecialistAccounting. In particular,
        phrases such as "specialist not completed" must never be parsed as positive completion.
        """
        normalized = reason.lower()
        if (
            "evidence" in normalized
            and ("gathered" in normalized or "packet" in normalized)
            and not self.store.records_for_task(
                "evidence", self.task_id, EvidencePacket
            )
        ):
            raise StageRequirementUnsatisfied(
                ["transition reason claims evidence without a persisted EvidencePacket"]
            )
        if "experiment specified" in normalized and not self.store.records_for_task(
            "experiment", self.task_id, ExperimentSpec
        ):
            raise StageRequirementUnsatisfied(
                ["transition reason claims an experiment without a persisted ExperimentSpec"]
            )
