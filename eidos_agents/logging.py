"""Secret-minimizing JSONL event logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .guardrails import SecretGuard
from .schemas import utc_now


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.secrets = SecretGuard()

    def emit(self, *, task_id: str, workflow_state: str, agent: str, model: str, event_type: str, **fields: Any) -> None:
        event = {
            "task_id": task_id,
            "workflow_state": workflow_state,
            "agent": agent,
            "model": model,
            "event_type": event_type,
            "timestamp": utc_now().isoformat(),
            **fields,
        }
        payload = json.dumps(event, sort_keys=True, default=str)
        self.secrets.check_text(payload)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
