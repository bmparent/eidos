import json
from decimal import Decimal

import pytest

from eidos_agents.budget import BudgetExceeded, BudgetManager
from eidos_agents.persistence import ArtifactStore
from eidos_agents.pricing import ModelPrice, PriceRegistry
from eidos_agents.resilient_orchestrator import ResilientEidosOrchestrator
from eidos_agents.schemas import (
    AgentResult,
    ArchivistResult,
    BudgetSpec,
    EvidenceItem,
    EvidencePacket,
    EvidenceSourceType,
    ResearchRequirements,
    SpecialistAccounting,
    TaskSpec,
    TaskType,
    WorkflowState,
)
from eidos_agents.sdk_contracts import (
    SDKAgentResult,
    SDKArchivistResult,
    SDKCurieResult,
    SDKExperimentSpec,
)
from eidos_agents.sdk_runtime import LiveTelemetry, RoutingViolation
from eidos_agents.workflow import StageRequirementUnsatisfied, WorkflowMachine


def _prices():
    return PriceRegistry(
        "test",
        {"cheap": ModelPrice(Decimal(1), Decimal("0.1"), Decimal(2), None)},
    )


def _assert_no_open_ended_object(schema):
    if isinstance(schema, dict):
        if "additionalProperties" in schema:
            assert schema["additionalProperties"] is False
        for value in schema.values():
            _assert_no_open_ended_object(value)
    elif isinstance(schema, list):
        for value in schema:
            _assert_no_open_ended_object(value)


def test_sdk_output_contracts_have_no_open_ended_objects():
    for model in (SDKAgentResult, SDKArchivistResult, SDKCurieResult, SDKExperimentSpec):
        _assert_no_open_ended_object(model.model_json_schema())


def test_required_sequence_blocks_sentry_before_archivist():
    telemetry = LiveTelemetry(
        budget=BudgetManager("TASK", BudgetSpec(), _prices()),
        required_sequence=["archivist", "sentry", "curie"],
        task_id="TASK",
    )
    with pytest.raises(RoutingViolation, match="expected 'archivist'"):
        telemetry.verify_start("sentry")


def test_sentry_must_cite_archivist_packet():
    telemetry = LiveTelemetry(
        budget=BudgetManager("TASK", BudgetSpec(), _prices()),
        required_sequence=["archivist", "sentry", "curie"],
        task_id="TASK",
    )
    packet = EvidencePacket(
        task_id="TASK",
        question="q",
        evidence=[
            EvidenceItem(
                evidence_id="E-1",
                source_type=EvidenceSourceType.PROJECT_EVIDENCE,
                source="repo",
                location="receipt.json",
                claim_supported="candidate current receipt",
                excerpt_or_summary="receipt exists",
                confidence=0.8,
            )
        ],
    )
    archivist = ArchivistResult(
        task_id="TASK",
        agent="archivist",
        status="PASS",
        summary="packet ready",
        confidence=0.8,
        evidence_packet=packet,
    )
    telemetry.agent_outputs.append(("archivist", archivist))
    sentry = AgentResult(
        task_id="TASK",
        agent="sentry",
        status="PASS",
        summary="analysis",
        confidence=0.7,
        evidence_refs=[],
    )
    with pytest.raises(RoutingViolation, match="did not cite"):
        telemetry.verify_output("sentry", sentry)


def test_not_completed_phrase_is_not_a_positive_completion_claim(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    task = TaskSpec(
        task_id="TASK-NOT-COMPLETE",
        title="reason parser",
        objective="prove negative prose is harmless",
        task_type=TaskType.RESEARCH,
        repo="repo",
        base_commit="abc",
        allowed_scope=["**"],
        initial_question="q",
        success_criteria=["transition works"],
    )
    store.put_task(task)
    workflow = WorkflowMachine(task.task_id, store)
    workflow.transition(WorkflowState.TRIAGED, actor="director", reason="triaged")
    workflow.transition(
        WorkflowState.EVIDENCE_GATHERING,
        actor="director",
        reason="research started",
    )
    workflow.transition(
        WorkflowState.READY_FOR_HUMAN,
        actor="director",
        reason="specialist not completed because none was required",
    )
    assert workflow.state == WorkflowState.READY_FOR_HUMAN


def test_failed_required_specialist_does_not_satisfy_gate(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    requirements = ResearchRequirements(required_specialists=["archivist"])
    task = TaskSpec(
        task_id="TASK-FAILED",
        title="failed specialist gate",
        objective="block failed specialist",
        task_type=TaskType.RESEARCH,
        repo="repo",
        base_commit="abc",
        allowed_scope=["**"],
        initial_question="q",
        success_criteria=["failure blocks"],
        research_requirements=requirements,
    )
    store.put_task(task)
    accounting = SpecialistAccounting(
        task_id=task.task_id,
        required=["archivist"],
        attempted=["archivist"],
        completed=[],
        failed={"archivist": "schema error"},
    )
    store.save_record(
        "specialist_accounting",
        "S-FAILED",
        task.task_id,
        accounting,
        f"tasks/{task.task_id}/specialist_accounting.json",
    )
    workflow = WorkflowMachine(task.task_id, store)
    workflow.transition(WorkflowState.TRIAGED, actor="director", reason="triaged")
    workflow.transition(
        WorkflowState.EVIDENCE_GATHERING,
        actor="director",
        reason="research started",
    )
    with pytest.raises(StageRequirementUnsatisfied, match="required specialist failed"):
        workflow.transition(
            WorkflowState.READY_FOR_HUMAN,
            actor="director",
            reason="review",
        )


def test_daily_budget_is_aggregate_across_tasks(tmp_path):
    ledger = tmp_path / "daily.sqlite"
    first = BudgetManager(
        "T1",
        BudgetSpec(task_budget_usd=2),
        _prices(),
        daily_budget_usd=2,
        daily_ledger_path=ledger,
    )
    first.record_call(
        "archivist",
        "cheap",
        {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0},
    )
    second = BudgetManager(
        "T2",
        BudgetSpec(task_budget_usd=2),
        _prices(),
        daily_budget_usd=2,
        daily_ledger_path=ledger,
    )
    second.authorize_call("sentry", "cheap")
    second.record_call(
        "sentry",
        "cheap",
        {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0},
    )
    third = BudgetManager(
        "T3",
        BudgetSpec(task_budget_usd=2),
        _prices(),
        daily_budget_usd=2,
        daily_ledger_path=ledger,
    )
    with pytest.raises(BudgetExceeded, match="daily Agent Lab budget exhausted"):
        third.authorize_call("curie", "cheap")


def test_live_failure_recovery_writes_blocked_manifest_and_cost(
    tmp_path, lab_config, monkeypatch
):
    ledger = tmp_path / "daily.sqlite"
    monkeypatch.setenv("EIDOS_AGENT_DAILY_BUDGET_USD", "2")
    monkeypatch.setenv("EIDOS_AGENT_DAILY_LEDGER_DB", str(ledger))
    orchestrator = ResilientEidosOrchestrator(lab_config)
    task = orchestrator.create_task(
        "recovery fixture",
        research_requirements=ResearchRequirements(
            required_specialists=["archivist", "sentry", "curie"]
        ),
    )
    workflow = WorkflowMachine(task.task_id, orchestrator.store)
    workflow.transition(WorkflowState.TRIAGED, actor="director", reason="fixture")

    budget = BudgetManager(task.task_id, task.budget, orchestrator.prices)
    budget.record_call(
        "director",
        "gpt-5.6-terra",
        {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10},
    )
    task_dir = orchestrator.store.task_dir(task.task_id)
    (task_dir / "live_events.jsonl").write_text(
        json.dumps({"event": "agent_start", "agent": "archivist"}) + "\n",
        encoding="utf-8",
    )

    orchestrator._persist_live_failure(task, RuntimeError("fixture boom"))

    assert WorkflowMachine(task.task_id, orchestrator.store).state == WorkflowState.BLOCKED
    assert (task_dir / "run_failure.json").exists()
    assert (task_dir / "run_manifest.json").exists()
    assert (task_dir / "specialist_accounting.json").exists()
    assert (task_dir / "cost_receipt.json").exists()
