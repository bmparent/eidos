import pytest

from eidos_agents.guardrails import (
    AuditSeparationGuard,
    CouncilGate,
    GuardrailBlock,
    MainBranchGuard,
    MergeGate,
    ScopeGuard,
    SecretGuard,
)
from eidos_agents.permissions import Capability, PermissionDenied, PermissionPolicy
from eidos_agents.schemas import AuditResult, AuditVerdict, ImplementationResult


@pytest.mark.parametrize("role", ["archivist", "auditor", "bench", "council"])
def test_read_only_roles_cannot_write_source(role):
    with pytest.raises(PermissionDenied):
        PermissionPolicy().require(role, Capability.WRITE_SOURCE)


def test_forge_can_write_but_cannot_merge_or_deploy():
    policy = PermissionPolicy()
    policy.require("forge", Capability.WRITE_SOURCE)
    assert not policy.can("forge", Capability.MERGE)
    assert not policy.can("forge", Capability.DEPLOY)


def test_main_branch_write_blocked():
    with pytest.raises(GuardrailBlock):
        MainBranchGuard().check_write("main")


def test_scope_guard_blocks_unrelated_file():
    with pytest.raises(GuardrailBlock):
        ScopeGuard().check(["eidos/engine.py"], ["eidos_agents/**"])


def test_secret_guard_detects_key():
    with pytest.raises(GuardrailBlock):
        SecretGuard().check_text("OPENAI_API_KEY='sk-abcdefghijklmnopqrstuvwxyz012345'")


def test_council_disabled_and_non_director_blocked():
    with pytest.raises(GuardrailBlock):
        CouncilGate().check(enabled=False, approved=True, actor="director")
    with pytest.raises(GuardrailBlock):
        CouncilGate().check(enabled=True, approved=True, actor="gauss")


def test_forge_cannot_audit_itself():
    implementation = ImplementationResult(
        implementation_id="I", agent="forge", files_changed=[], diff_summary="x", tests_run=[], results={}, ready_for_bench=True,
    )
    audit = AuditResult(
        task_id="T", auditor="forge", claims_reviewed=[], evidence_for=[], evidence_against=[], methodology_issues=[],
        reproducibility="x", regressions=[], severity="HIGH", verdict=AuditVerdict.PASS,
    )
    with pytest.raises(GuardrailBlock):
        AuditSeparationGuard().check(implementation, audit)


def test_merge_gate_requires_human_and_audit():
    audit = AuditResult(
        task_id="T", claims_reviewed=[], evidence_for=[], evidence_against=[], methodology_issues=[],
        reproducibility="x", regressions=[], severity="LOW", verdict=AuditVerdict.PASS,
    )
    with pytest.raises(GuardrailBlock):
        MergeGate().check(tests_passed=True, benchmark_receipts_exist=True, audit=audit, git_state_understood=True, human_approval=False)
