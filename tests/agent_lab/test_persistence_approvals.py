import pytest

from eidos_agents.approvals import ApprovalManager, ApprovalRequired
from eidos_agents.persistence import ArtifactStore
from eidos_agents.schemas import ApprovalRecord


def test_approval_pause_and_resume_survives_store_reopen(tmp_path):
    root = tmp_path / "lab"
    store = ArtifactStore(root)
    approvals = ApprovalManager(store)
    with pytest.raises(ApprovalRequired):
        approvals.require("TASK-X", "council", "rare escalation")
    assert ApprovalManager(ArtifactStore(root)).resume("TASK-X")["status"] == "AWAITING_APPROVAL"
    ArtifactStore(root).add_approval(ApprovalRecord(task_id="TASK-X", action="council", approved=True, actor="Brent"))
    assert ApprovalManager(ArtifactStore(root)).resume("TASK-X")["status"] == "APPROVAL_SATISFIED"


def test_immutable_record_cannot_be_overwritten(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    path = store.root / "x.json"
    path.write_text("{}", encoding="utf-8")
    from eidos_agents.schemas import CostReceipt
    with pytest.raises(FileExistsError):
        store.write_model("x.json", CostReceipt(task_id="T", price_version="v"), immutable=True)
