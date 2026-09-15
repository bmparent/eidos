from pathlib import Path

from eidos_agents.missions import MissionDefinition
from eidos_agents.orchestrator import EidosOrchestrator


def test_full_mocked_workflow(lab_config):
    orchestrator = EidosOrchestrator(lab_config)
    decision = orchestrator.run_mocked("Implement an approved bounded patch", research_only=False)
    task_dir = orchestrator.store.task_dir(decision.task_id)
    assert decision.decision == "MOCK_WORKFLOW_PASS_WITH_LIMITATIONS"
    assert (task_dir / "run_manifest.json").exists()
    assert (task_dir / "benchmarks").exists()
    assert (task_dir / "audit" / "audit_result.json").exists()
    assert (task_dir / "cost_receipt.json").exists()
    assert orchestrator.store.latest_state(decision.task_id) == "READY_FOR_HUMAN"


def test_sentinel_mission_dry_run_does_not_modify_source(lab_config, repo_root: Path):
    mission = MissionDefinition.load(repo_root / "missions" / "sentinel_precision_v1.yaml")
    before = (repo_root / "eidos" / "EIDOS_BRAIN_UNIFIED_v0_4.7.02.py").read_bytes()
    orchestrator = EidosOrchestrator(lab_config)
    decision = orchestrator.run_mocked(mission.objective, research_only=True, mission=mission)
    after = (repo_root / "eidos" / "EIDOS_BRAIN_UNIFIED_v0_4.7.02.py").read_bytes()
    assert before == after
    assert decision.decision.startswith("RESEARCH_SPEC_READY")
    assert not (orchestrator.store.task_dir(decision.task_id) / "implementation").exists()
