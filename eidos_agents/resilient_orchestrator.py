"""Failure-safe wrapper for live Agent Lab execution.

The core orchestrator remains responsible for the successful workflow. This wrapper guarantees that
an exception during live SDK execution still leaves a reviewable BLOCKED run rather than a partial
artifact directory with no manifest.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from .orchestrator import EidosOrchestrator
from .schemas import (
    AgentResult,
    ArchivistResult,
    CostReceipt,
    CurieResult,
    RunManifest,
    SpecialistAccounting,
    TaskSpec,
    WorkflowState,
)
from .workflow import WorkflowMachine


_FAILURE_STATUSES = {"FAIL", "FAILED", "BLOCK", "BLOCKED", "ERROR", "UNSTRUCTURED"}


class ResilientEidosOrchestrator(EidosOrchestrator):
    """EidosOrchestrator with deterministic forensic finalization on live failure."""

    def __init__(self, config):
        super().__init__(config)
        self._active_task: TaskSpec | None = None

    def create_task(self, *args, **kwargs) -> TaskSpec:  # type: ignore[override]
        task = super().create_task(*args, **kwargs)
        self._active_task = task
        return task

    async def run_live(self, *args, **kwargs):  # type: ignore[override]
        self._active_task = None
        try:
            return await super().run_live(*args, **kwargs)
        except Exception as exc:
            if self._active_task is not None:
                self._persist_live_failure(self._active_task, exc)
            raise

    def _persist_live_failure(self, task: TaskSpec, exc: Exception) -> None:
        task_dir = self.store.task_dir(task.task_id)
        events = self._read_jsonl(task_dir / "live_events.jsonl")
        self._recover_sdk_outputs(task, task_dir)
        accounting = self._recover_accounting(task, events)
        accounting_path = task_dir / "specialist_accounting.json"
        if not accounting_path.exists():
            try:
                self.store.save_record(
                    "specialist_accounting",
                    f"SPECIALISTS-RECOVERY-{task.task_id}",
                    task.task_id,
                    accounting,
                    f"tasks/{task.task_id}/specialist_accounting.json",
                )
            except (FileExistsError, sqlite3.IntegrityError):
                pass

        cost_ref = self._recover_cost_receipt(task, task_dir)
        failure_path = task_dir / "run_failure.json"
        failure_payload = {
            "task_id": task.task_id,
            "status": "BLOCKED",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "events_recorded": len(events),
            "specialist_accounting": accounting.model_dump(mode="json"),
        }
        failure_path.write_text(
            json.dumps(failure_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        workflow = WorkflowMachine(task.task_id, self.store)
        if workflow.state not in {WorkflowState.BLOCKED, WorkflowState.CLOSED}:
            try:
                workflow.transition(
                    WorkflowState.BLOCKED,
                    actor="workflow",
                    reason=f"live orchestration exception: {type(exc).__name__}",
                )
            except Exception:
                # Failure finalization must never conceal the original exception.
                pass

        manifest_path = task_dir / "run_manifest.json"
        if not manifest_path.exists():
            experiments = [
                path.relative_to(self.config.artifact_root).as_posix()
                for path in sorted((task_dir / "experiments").glob("*/spec.json"))
            ] if (task_dir / "experiments").exists() else []
            final_path = task_dir / "final_decision.json"
            trace_id = next(
                (
                    event.get("trace_id")
                    for event in events
                    if event.get("event") == "trace" and event.get("trace_id")
                ),
                None,
            )
            manifest = RunManifest(
                task_id=task.task_id,
                task_ref=f"tasks/{task.task_id}/task.json",
                repo_state={
                    "commit": task.base_commit,
                    "branch": self.repo.repo_current_branch(),
                    "dry_run": False,
                    "status": "BLOCKED",
                },
                agents={
                    "events": events,
                    "recovered_after_failure": True,
                    "exception_type": type(exc).__name__,
                    "failure_ref": failure_path.relative_to(
                        self.config.artifact_root
                    ).as_posix(),
                },
                models=self.models.manifest(),
                experiments=experiments,
                costs=cost_ref,
                final_decision=(
                    final_path.relative_to(self.config.artifact_root).as_posix()
                    if final_path.exists()
                    else None
                ),
                tracing_available=bool(trace_id),
                tracing_note=(
                    f"trace_id={trace_id}; live run blocked and recovered"
                    if trace_id
                    else "live run blocked; trace identifier unavailable"
                ),
            )
            self.store.write_model(
                f"tasks/{task.task_id}/run_manifest.json", manifest, immutable=True
            )

    def _recover_sdk_outputs(self, task: TaskSpec, task_dir: Path) -> None:
        entries = self._read_jsonl(task_dir / "sdk_outputs.jsonl")
        for index, entry in enumerate(entries, 1):
            agent = str(entry.get("agent", "")).lower()
            payload = entry.get("output")
            if not isinstance(payload, dict):
                continue
            model_type: type[AgentResult]
            if agent == "archivist":
                model_type = ArchivistResult
            elif agent == "curie":
                model_type = CurieResult
            else:
                model_type = AgentResult
            try:
                output = model_type.model_validate(payload)
            except Exception:
                continue
            agent_path = task_dir / "agents" / f"{index:02d}-{agent}.json"
            if not agent_path.exists():
                try:
                    self.store.save_record(
                        "agent_result",
                        f"AGENT-RECOVERY-{task.task_id}-{index:02d}-{agent}",
                        task.task_id,
                        output,
                        f"tasks/{task.task_id}/agents/{index:02d}-{agent}.json",
                    )
                except (FileExistsError, sqlite3.IntegrityError):
                    pass
            if isinstance(output, ArchivistResult):
                evidence_path = task_dir / "evidence" / "evidence_packet.json"
                if not evidence_path.exists():
                    try:
                        self.store.save_record(
                            "evidence",
                            f"EVID-RECOVERY-{task.task_id}",
                            task.task_id,
                            output.evidence_packet,
                            f"tasks/{task.task_id}/evidence/evidence_packet.json",
                        )
                    except (FileExistsError, sqlite3.IntegrityError):
                        pass
            if isinstance(output, CurieResult):
                spec = output.experiment_spec
                spec_path = task_dir / "experiments" / spec.experiment_id / "spec.json"
                if not spec_path.exists():
                    try:
                        self.store.save_record(
                            "experiment",
                            spec.experiment_id,
                            task.task_id,
                            spec,
                            f"tasks/{task.task_id}/experiments/{spec.experiment_id}/spec.json",
                        )
                    except (FileExistsError, sqlite3.IntegrityError):
                        pass

    def _recover_accounting(
        self, task: TaskSpec, events: list[dict[str, Any]]
    ) -> SpecialistAccounting:
        attempted: list[str] = []
        completed: list[str] = []
        failed: dict[str, str] = {}
        for event in events:
            agent = str(event.get("agent", "")).lower()
            if not agent or agent == "director":
                continue
            if event.get("event") == "agent_start" and agent not in attempted:
                attempted.append(agent)
            if event.get("event") == "agent_end":
                status = str(event.get("status", "UNSTRUCTURED")).upper()
                if status in _FAILURE_STATUSES:
                    failed[agent] = f"specialist ended with status {status}"
                elif agent not in completed:
                    completed.append(agent)
        for agent in attempted:
            if agent not in completed and agent not in failed:
                failed[agent] = "specialist started but no structured completion was persisted"
        skipped = {
            agent: "required specialist was not invoked"
            for agent in task.research_requirements.required_specialists
            if agent not in attempted
        }
        return SpecialistAccounting(
            task_id=task.task_id,
            required=task.research_requirements.required_specialists,
            attempted=attempted,
            completed=completed,
            failed=failed,
            skipped=skipped,
        )

    def _recover_cost_receipt(self, task: TaskSpec, task_dir: Path) -> str | None:
        existing = task_dir / "cost_receipt.json"
        if existing.exists():
            return existing.relative_to(self.config.artifact_root).as_posix()
        ledger_env = os.getenv("EIDOS_AGENT_DAILY_LEDGER_DB")
        if not ledger_env:
            return None
        ledger_path = Path(ledger_env)
        if not ledger_path.exists():
            return None
        try:
            conn = sqlite3.connect(ledger_path)
            rows = conn.execute(
                "SELECT agent,model,cost_usd,usage_json FROM charges WHERE task_id=? ORDER BY id",
                (task.task_id,),
            ).fetchall()
            conn.close()
        except sqlite3.Error:
            return None
        if not rows:
            return None
        tokens: Counter[str] = Counter()
        calls_by_agent: Counter[str] = Counter()
        calls_by_model: Counter[str] = Counter()
        total = Decimal(0)
        for agent, model, cost, usage_json in rows:
            total += Decimal(str(cost))
            calls_by_agent[str(agent)] += 1
            calls_by_model[str(model)] += 1
            usage = json.loads(usage_json)
            for key in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
            ):
                tokens[key] += int(usage.get(key) or 0)
        remaining = (
            None
            if task.budget.task_budget_usd is None
            else float(Decimal(str(task.budget.task_budget_usd)) - total)
        )
        receipt = CostReceipt(
            task_id=task.task_id,
            price_version=self.prices.version,
            price_effective_date=self.prices.effective_date,
            total_input_tokens=tokens["input_tokens"],
            cached_input_tokens=tokens["cached_input_tokens"],
            output_tokens=tokens["output_tokens"],
            reasoning_tokens=tokens["reasoning_tokens"],
            calls_by_model=dict(calls_by_model),
            calls_by_agent=dict(calls_by_agent),
            tool_calls={},
            approximate_cost_usd=float(total),
            budget_remaining=remaining,
            estimate_notes=[
                "Recovered from persistent daily budget ledger after live exception"
            ],
        )
        self.store.write_model(
            f"tasks/{task.task_id}/cost_receipt.json", receipt, immutable=True
        )
        return existing.relative_to(self.config.artifact_root).as_posix()

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows
