"""Durable human approval interruptions independent of hidden chat state."""

from __future__ import annotations

import json

from .persistence import ArtifactStore
from .schemas import utc_now


class ApprovalRequired(RuntimeError):
    def __init__(self, task_id: str, action: str) -> None:
        super().__init__(f"human approval required: {task_id} {action}")
        self.task_id = task_id
        self.action = action


class ApprovalManager:
    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def require(self, task_id: str, action: str, reason: str) -> None:
        if self.store.is_approved(task_id, action):
            return
        path = self.store.task_dir(task_id) / "pending_approval.json"
        path.write_text(
            json.dumps({"task_id": task_id, "action": action, "reason": reason, "requested_at": utc_now().isoformat()}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise ApprovalRequired(task_id, action)

    def resume(self, task_id: str) -> dict[str, str]:
        path = self.store.task_dir(task_id) / "pending_approval.json"
        if not path.exists():
            return {"status": "NO_PENDING_APPROVAL", "task_id": task_id}
        pending = json.loads(path.read_text(encoding="utf-8"))
        if not self.store.is_approved(task_id, pending["action"]):
            return {"status": "AWAITING_APPROVAL", "task_id": task_id, "action": pending["action"]}
        return {"status": "APPROVAL_SATISFIED", "task_id": task_id, "action": pending["action"]}
