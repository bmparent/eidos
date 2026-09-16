"""SQLite metadata index plus reviewable JSON/Markdown task artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .schemas import ApprovalRecord, Hypothesis, TaskSpec, WorkflowTransition, utc_now

T = TypeVar("T", bound=BaseModel)


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS records (
  kind TEXT NOT NULL, record_id TEXT NOT NULL, task_id TEXT, created_at TEXT NOT NULL,
  payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL,
  PRIMARY KEY(kind, record_id)
);
CREATE TABLE IF NOT EXISTS transitions (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, created_at TEXT NOT NULL,
  from_state TEXT NOT NULL, to_state TEXT NOT NULL, actor TEXT NOT NULL, reason TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
  task_id TEXT NOT NULL, action TEXT NOT NULL, created_at TEXT NOT NULL, actor TEXT NOT NULL,
  approved INTEGER NOT NULL, payload_json TEXT NOT NULL,
  PRIMARY KEY(task_id, action, created_at)
);
CREATE TABLE IF NOT EXISTS commands (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, created_at TEXT NOT NULL,
  actor TEXT NOT NULL, command TEXT NOT NULL, exit_code INTEGER, output_digest TEXT
);
"""


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "agent_lab.sqlite"
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def task_dir(self, task_id: str) -> Path:
        if not task_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in task_id):
            raise ValueError("unsafe task id")
        path = self.root / "tasks" / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_model(self, relative_path: str, model: BaseModel, *, immutable: bool = False) -> Path:
        destination = (self.root / relative_path).resolve()
        if self.root not in destination.parents:
            raise ValueError("artifact path escapes root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if immutable and destination.exists():
            raise FileExistsError(destination)
        data = model.model_dump_json(indent=2)
        destination.write_text(data + "\n", encoding="utf-8")
        return destination

    def index(self, kind: str, record_id: str, model: BaseModel, task_id: str | None = None) -> None:
        payload = model.model_dump_json()
        payload_hash = model.stable_hash() if hasattr(model, "stable_hash") else "unavailable"
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO records(kind,record_id,task_id,created_at,payload_json,payload_hash) VALUES(?,?,?,?,?,?)",
                (kind, record_id, task_id, utc_now().isoformat(), payload, payload_hash),
            )

    def put_task(self, task: TaskSpec) -> Path:
        path = self.write_model(f"tasks/{task.task_id}/task.json", task, immutable=not (self.task_dir(task.task_id) / "task.json").exists())
        try:
            self.index("task", task.task_id, task, task.task_id)
        except sqlite3.IntegrityError:
            pass
        return path

    def get_record(self, kind: str, record_id: str, model_type: type[T]) -> T:
        with self.connect() as conn:
            row = conn.execute("SELECT payload_json FROM records WHERE kind=? AND record_id=?", (kind, record_id)).fetchone()
        if row is None:
            raise KeyError(f"{kind}/{record_id}")
        return model_type.model_validate_json(row["payload_json"])

    def add_transition(self, transition: WorkflowTransition) -> None:
        payload = transition.model_dump_json()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO transitions(task_id,created_at,from_state,to_state,actor,reason,payload_json) VALUES(?,?,?,?,?,?,?)",
                (transition.task_id, transition.timestamp.isoformat(), transition.from_state, transition.to_state, transition.actor, transition.reason, payload),
            )
        journal = self.task_dir(transition.task_id) / "transitions.jsonl"
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    def transitions(self, task_id: str) -> list[WorkflowTransition]:
        with self.connect() as conn:
            rows = conn.execute("SELECT payload_json FROM transitions WHERE task_id=? ORDER BY seq", (task_id,)).fetchall()
        return [WorkflowTransition.model_validate_json(row["payload_json"]) for row in rows]

    def latest_state(self, task_id: str) -> str:
        transitions = self.transitions(task_id)
        return str(transitions[-1].to_state) if transitions else "CREATED"

    def add_approval(self, approval: ApprovalRecord) -> None:
        payload = approval.model_dump_json()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO approvals(task_id,action,created_at,actor,approved,payload_json) VALUES(?,?,?,?,?,?)",
                (approval.task_id, approval.action, approval.timestamp.isoformat(), approval.actor, int(approval.approved), payload),
            )

    def is_approved(self, task_id: str, action: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT approved FROM approvals WHERE task_id=? AND action=? ORDER BY created_at DESC LIMIT 1", (task_id, action)
            ).fetchone()
        return bool(row and row["approved"])

    def record_command(self, task_id: str | None, actor: str, command: str, exit_code: int | None, output_digest: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO commands(task_id,created_at,actor,command,exit_code,output_digest) VALUES(?,?,?,?,?,?)",
                (task_id, utc_now().isoformat(), actor, command, exit_code, output_digest),
            )

    def list_records(self, kind: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT record_id,task_id,created_at,payload_hash FROM records WHERE kind=? ORDER BY created_at", (kind,)).fetchall()
        return [dict(row) for row in rows]

    def records_for_task(self, kind: str, task_id: str, model_type: type[T]) -> list[T]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM records WHERE kind=? AND task_id=? ORDER BY created_at",
                (kind, task_id),
            ).fetchall()
        return [model_type.model_validate_json(row["payload_json"]) for row in rows]

    def raw_records_for_task(self, kind: str, task_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM records WHERE kind=? AND task_id=? ORDER BY created_at",
                (kind, task_id),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def add_hypothesis(self, hypothesis: Hypothesis, task_id: str) -> Path:
        path = self.write_model(f"hypotheses/{hypothesis.hypothesis_id}.json", hypothesis, immutable=True)
        self.index("hypothesis", hypothesis.hypothesis_id, hypothesis, task_id)
        return path

    def save_record(self, kind: str, record_id: str, task_id: str, model: BaseModel, relative_path: str) -> Path:
        path = self.write_model(relative_path, model, immutable=True)
        self.index(kind, record_id, model, task_id)
        return path
