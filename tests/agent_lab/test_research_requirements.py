import pytest

from eidos_agents.persistence import ArtifactStore
from eidos_agents.schemas import (
    AgentResult,
    EvidenceItem,
    EvidencePacket,
    EvidenceSourceType,
    ExperimentSpec,
    ResearchRequirements,
    SpecialistAccounting,
    TaskSpec,
    TaskType,
    WorkflowState,
)
from eidos_agents.workflow import StageRequirementUnsatisfied, WorkflowMachine

REQUIRED = ResearchRequirements(
    current_evidence_required=True,
    required_specialists=["archivist", "sentry", "curie"],
    experiment_spec_required=True,
)


def setup_task(tmp_path, requirements=REQUIRED):
    store = ArtifactStore(tmp_path / "lab")
    task = TaskSpec(
        task_id="TASK-REQ",
        title="research gate",
        objective="retrieve evidence and specify experiment",
        task_type=TaskType.RESEARCH,
        repo="repo",
        base_commit="abc",
        allowed_scope=["**"],
        initial_question="question",
        success_criteria=["requirements persist"],
        research_requirements=requirements,
    )
    store.put_task(task)
    workflow = WorkflowMachine(task.task_id, store)
    workflow.transition(WorkflowState.TRIAGED, actor="director", reason="triaged")
    workflow.transition(
        WorkflowState.EVIDENCE_GATHERING,
        actor="director",
        reason="research requirements evaluated",
    )
    return store, workflow, task


def save_result(store, task_id, agent, index=1, status="PASS"):
    result = AgentResult(
        task_id=task_id,
        agent=agent,
        status=status,
        summary=f"{agent} completed",
        confidence=0.5,
    )
    store.save_record(
        "agent_result",
        f"A-{index}-{agent}",
        task_id,
        result,
        f"tasks/{task_id}/agents/{index:02d}-{agent}.json",
    )


def save_accounting(store, task_id, completed):
    accounting = SpecialistAccounting(
        task_id=task_id,
        required=["archivist", "sentry", "curie"],
        attempted=completed,
        completed=completed,
        skipped={
            agent: "not invoked"
            for agent in ["archivist", "sentry", "curie"]
            if agent not in completed
        },
    )
    store.save_record(
        "specialist_accounting",
        f"S-{task_id}",
        task_id,
        accounting,
        f"tasks/{task_id}/specialist_accounting.json",
    )


def missing_packet(task_id):
    return EvidencePacket(
        task_id=task_id,
        question="current evidence",
        evidence=[
            EvidenceItem(
                evidence_id="E-MISSING",
                source_type=EvidenceSourceType.MISSING,
                source="repository",
                location="artifacts",
                claim_supported="no eligible current receipt located",
                excerpt_or_summary="bounded investigation completed",
                confidence=1.0,
            )
        ],
        missing_evidence=["current comparable calibration receipt"],
    )


def experiment(task_id):
    return ExperimentSpec(
        experiment_id="EXP-REQ",
        task_id=task_id,
        hypothesis_ids=[],
        dataset="frozen chronological labeled dataset",
        sample_policy="calibration and evaluation split by time",
        controls=["current profile"],
        negative_controls=["benign replay"],
        ablations=["confirmation off"],
        seeds=[0],
        variables={"independent": "calibration policy"},
        metrics=["precision", "recall", "fp_per_10k"],
        baseline="current commit",
        commands_or_runner_reference=["approved runner"],
        success_observation="precision improves without recall loss",
        failure_observation="recall or coverage regresses",
        ambiguous_observation="seed-sensitive result",
        artifacts_required=["run_manifest.json"],
    )


def test_fluent_director_decision_cannot_bypass_evidence(tmp_path):
    _store, workflow, _task = setup_task(tmp_path)
    with pytest.raises(StageRequirementUnsatisfied) as exc:
        workflow.transition(
            WorkflowState.READY_FOR_HUMAN,
            actor="director",
            reason="fluent final decision",
        )
    assert "EvidencePacket" in str(exc.value)
    assert workflow.state == WorkflowState.EVIDENCE_GATHERING


def test_archivist_call_without_evidence_packet_is_blocked(tmp_path):
    store, workflow, task = setup_task(tmp_path)
    save_result(store, task.task_id, "archivist")
    save_accounting(store, task.task_id, ["archivist"])
    with pytest.raises(StageRequirementUnsatisfied) as exc:
        workflow.transition(WorkflowState.READY_FOR_HUMAN, actor="director", reason="review")
    assert "EvidencePacket" in str(exc.value)


def test_missing_evidence_packet_counts_as_investigated(tmp_path):
    requirements = ResearchRequirements(current_evidence_required=True)
    store, workflow, task = setup_task(tmp_path, requirements)
    save_result(store, task.task_id, "archivist")
    packet = missing_packet(task.task_id)
    store.save_record(
        "evidence",
        "E-MISSING",
        task.task_id,
        packet,
        f"tasks/{task.task_id}/evidence/evidence_packet.json",
    )
    workflow.transition(WorkflowState.READY_FOR_HUMAN, actor="director", reason="review")
    assert workflow.state == WorkflowState.READY_FOR_HUMAN


def test_required_specialist_gate_names_missing_curie(tmp_path):
    requirements = ResearchRequirements(required_specialists=["archivist", "sentry", "curie"])
    store, workflow, task = setup_task(tmp_path, requirements)
    save_accounting(store, task.task_id, ["archivist", "sentry"])
    with pytest.raises(StageRequirementUnsatisfied) as exc:
        workflow.transition(WorkflowState.READY_FOR_HUMAN, actor="director", reason="review")
    assert "curie" in str(exc.value)


def test_curie_prose_without_experiment_spec_is_rejected(tmp_path):
    requirements = ResearchRequirements(experiment_spec_required=True)
    store, workflow, task = setup_task(tmp_path, requirements)
    save_result(store, task.task_id, "curie")
    with pytest.raises(StageRequirementUnsatisfied) as exc:
        workflow.transition(WorkflowState.READY_FOR_HUMAN, actor="director", reason="review")
    assert "ExperimentSpec" in str(exc.value)


def test_successful_research_path_reaches_human_review(tmp_path):
    store, workflow, task = setup_task(tmp_path)
    for index, agent in enumerate(["archivist", "sentry", "curie"], 1):
        save_result(store, task.task_id, agent, index)
    save_accounting(store, task.task_id, ["archivist", "sentry", "curie"])
    store.save_record(
        "evidence",
        "E-CURRENT",
        task.task_id,
        missing_packet(task.task_id),
        f"tasks/{task.task_id}/evidence/evidence_packet.json",
    )
    spec = experiment(task.task_id)
    store.save_record(
        "experiment",
        spec.experiment_id,
        task.task_id,
        spec,
        f"tasks/{task.task_id}/experiments/{spec.experiment_id}/spec.json",
    )
    workflow.transition(
        WorkflowState.EXPERIMENT_SPECIFIED,
        actor="curie",
        reason="persisted ExperimentSpec validated for current task",
    )
    workflow.transition(
        WorkflowState.READY_FOR_HUMAN,
        actor="director",
        reason="persisted research requirements satisfied",
    )
    assert workflow.state == WorkflowState.READY_FOR_HUMAN


def test_transition_reason_cannot_claim_zero_specialists_completed(tmp_path):
    _store, workflow, _task = setup_task(tmp_path, ResearchRequirements())
    with pytest.raises(StageRequirementUnsatisfied):
        workflow.transition(
            WorkflowState.READY_FOR_HUMAN,
            actor="director",
            reason="specialist research completed",
        )
