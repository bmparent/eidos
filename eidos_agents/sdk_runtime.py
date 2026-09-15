"""Thin live runtime adapter; deterministic workflow remains outside the SDK loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent_definitions import SDKGraph, SDKUnavailable, build_sdk_graph
from .config import LabConfig
from .model_registry import ModelRegistry
from .schemas import FinalDecision, TaskSpec


class AgentsSDKRuntime:
    def __init__(self, config: LabConfig) -> None:
        self.config = config
        self.registry = ModelRegistry(config)
        self.graph: SDKGraph | None = None

    def initialize(self) -> SDKGraph:
        self.graph = build_sdk_graph(self.config, self.registry)
        return self.graph

    async def run_director(
        self, task: TaskSpec, session_db: Path, state_path: Path | None = None
    ) -> tuple[FinalDecision | None, dict[str, int], list[Any]]:
        try:
            from agents import Runner, SQLiteSession, trace
        except Exception as exc:
            raise SDKUnavailable(f"OpenAI Agents SDK runtime unavailable: {type(exc).__name__}: {exc}") from exc
        graph = self.graph or self.initialize()
        session = SQLiteSession(task.task_id, str(session_db))
        prompt = task.model_dump_json(indent=2)
        with trace(
            "Eidos Agent Lab task",
            trace_id=None,
            group_id=task.task_id,
            metadata={"task_id": task.task_id, "repo_commit": task.base_commit, "workflow_state": "TRIAGED"},
            disabled=not self.config.tracing_enabled,
        ):
            result = await Runner.run(
                graph.director,
                prompt,
                session=session,
                max_turns=self.config.budget.maximum_turns,
            )
        usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        raw_usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        if raw_usage:
            for key in usage:
                usage[key] = int(getattr(raw_usage, key, 0) or 0)
        interruptions = list(getattr(result, "interruptions", []))
        if interruptions and state_path is not None:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(result.to_state().to_json(), indent=2) + "\n", encoding="utf-8")
        output = result.final_output if isinstance(result.final_output, FinalDecision) else None
        return output, usage, interruptions

    async def resume_director(
        self, task: TaskSpec, session_db: Path, state_path: Path
    ) -> tuple[FinalDecision | None, dict[str, int], list[Any]]:
        try:
            from agents import Runner, RunState, SQLiteSession
        except Exception as exc:
            raise SDKUnavailable(f"OpenAI Agents SDK runtime unavailable: {type(exc).__name__}: {exc}") from exc
        graph = self.graph or self.initialize()
        state_json = json.loads(state_path.read_text(encoding="utf-8"))
        state = await RunState.from_json(graph.director, state_json)
        for interruption in state.get_interruptions():
            state.approve(interruption)
        session = SQLiteSession(task.task_id, str(session_db))
        result = await Runner.run(graph.director, state, session=session)
        interruptions = list(getattr(result, "interruptions", []))
        if interruptions:
            state_path.write_text(json.dumps(result.to_state().to_json(), indent=2) + "\n", encoding="utf-8")
        else:
            state_path.unlink(missing_ok=True)
        usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        raw_usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        if raw_usage:
            for key in usage:
                usage[key] = int(getattr(raw_usage, key, 0) or 0)
        output = result.final_output if isinstance(result.final_output, FinalDecision) else None
        return output, usage, interruptions
