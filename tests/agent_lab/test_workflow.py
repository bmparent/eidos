import pytest

from eidos_agents.persistence import ArtifactStore
from eidos_agents.schemas import AuditVerdict, WorkflowState
from eidos_agents.workflow import InvalidTransition, WorkflowMachine


def machine(tmp_path):
    return WorkflowMachine("TASK-TEST", ArtifactStore(tmp_path / "state"))


def test_implementation_cannot_skip_bench(tmp_path):
    wf = machine(tmp_path)
    wf.transition(WorkflowState.TRIAGED, actor="director", reason="x")
    wf.transition(WorkflowState.IMPLEMENTATION_SPECIFIED, actor="director", reason="x")
    wf.transition(WorkflowState.IMPLEMENTING, actor="forge", reason="x")
    with pytest.raises(InvalidTransition):
        wf.transition(WorkflowState.READY_FOR_HUMAN, actor="forge", reason="self-certified", tests_passed=True)


def test_implementation_requires_tests(tmp_path):
    wf = machine(tmp_path)
    wf.transition(WorkflowState.TRIAGED, actor="director", reason="x")
    wf.transition(WorkflowState.IMPLEMENTATION_SPECIFIED, actor="director", reason="x")
    wf.transition(WorkflowState.IMPLEMENTING, actor="forge", reason="x")
    with pytest.raises(InvalidTransition):
        wf.transition(WorkflowState.BENCHMARKING, actor="bench", reason="x", tests_passed=False)


def test_auditor_block_cannot_promote(tmp_path):
    wf = machine(tmp_path)
    wf.transition(WorkflowState.TRIAGED, actor="director", reason="x")
    wf.transition(WorkflowState.IMPLEMENTATION_SPECIFIED, actor="director", reason="x")
    wf.transition(WorkflowState.IMPLEMENTING, actor="forge", reason="x")
    wf.transition(WorkflowState.BENCHMARKING, actor="bench", reason="x", tests_passed=True)
    wf.transition(WorkflowState.AUDITING, actor="director", reason="x")
    with pytest.raises(InvalidTransition):
        wf.transition(WorkflowState.READY_FOR_HUMAN, actor="director", reason="override", audit_verdict=AuditVerdict.BLOCK)


def test_pass_with_limitations_can_promote(tmp_path):
    wf = machine(tmp_path)
    wf.transition(WorkflowState.TRIAGED, actor="director", reason="x")
    wf.transition(WorkflowState.IMPLEMENTATION_SPECIFIED, actor="director", reason="x")
    wf.transition(WorkflowState.IMPLEMENTING, actor="forge", reason="x")
    wf.transition(WorkflowState.BENCHMARKING, actor="bench", reason="x", tests_passed=True)
    wf.transition(WorkflowState.AUDITING, actor="director", reason="x")
    wf.transition(WorkflowState.READY_FOR_HUMAN, actor="director", reason="x", audit_verdict=AuditVerdict.PASS_WITH_LIMITATIONS)
    assert wf.state == WorkflowState.READY_FOR_HUMAN
