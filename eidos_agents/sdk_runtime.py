"""Thin live runtime adapter; deterministic workflow remains outside the SDK loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_definitions import SDKGraph, SDKUnavailable, build_sdk_graph
from .budget import BudgetManager
from .config import LabConfig
from .model_registry import ModelRegistry
from .schemas import AgentResult, ArchivistResult, FinalDecision, TaskSpec
from .sdk_contracts import SDKAgentResult


class RoutingViolation(RuntimeError):
    """Raised when the live specialist chain violates a deterministic host contract."""


_FAILURE_STATUSES = {"FAIL", "FAILED", "BLOCK", "BLOCKED", "ERROR"}


@dataclass
class LiveTelemetry:
    budget: BudgetManager
    allowed_specialists: set[str] | None = None
    required_sequence: list[str] = field(default_factory=list)
    task_id: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    agent_outputs: list[tuple[str, AgentResult]] = field(default_factory=list)
    trace_id: str | None = None

    def successful_specialists(self) -> list[str]:
        return [
            agent
            for agent, output in self.agent_outputs
            if output.status.upper() not in _FAILURE_STATUSES
        ]

    def next_required_specialist(self) -> str | None:
        successful = self.successful_specialists()
        for specialist in self.required_sequence:
            if specialist not in successful:
                return specialist
        return None

    def verify_start(self, specialist: str) -> None:
        if specialist not in self.required_sequence:
            return
        expected = self.next_required_specialist()
        if expected != specialist:
            raise RoutingViolation(
                f"required specialist routing violation: expected {expected!r}, got {specialist!r}"
            )

    def verify_output(self, specialist: str, output: AgentResult) -> None:
        if self.task_id is not None and output.task_id != self.task_id:
            raise RoutingViolation(
                f"{specialist} returned task_id {output.task_id!r}; expected {self.task_id!r}"
            )
        if output.agent.strip().lower() != specialist:
            raise RoutingViolation(
                f"{specialist} output declared agent={output.agent!r}"
            )
        if output.status.upper() in _FAILURE_STATUSES:
            return
        if specialist == "sentry" and "archivist" in self.required_sequence:
            archivist = next(
                (
                    item
                    for agent, item in reversed(self.agent_outputs)
                    if agent == "archivist" and isinstance(item, ArchivistResult)
                ),
                None,
            )
            if archivist is None:
                raise RoutingViolation("Sentry completed without a prior Archivist result")
            evidence_ids = {
                item.evidence_id for item in archivist.evidence_packet.evidence
            }
            if evidence_ids and not (evidence_ids & set(output.evidence_refs)):
                raise RoutingViolation(
                    "Sentry did not cite any evidence reference from the Archivist packet"
                )
        if specialist == "curie" and "sentry" in self.required_sequence:
            prior = [
                item
                for agent, item in self.agent_outputs
                if agent == "sentry" and item.status.upper() not in _FAILURE_STATUSES
            ]
            if not prior:
                raise RoutingViolation("Curie completed without a successful Sentry result")
            if not output.evidence_refs:
                raise RoutingViolation(
                    "Curie completed without carrying forward any evidence references"
                )


def _agent_key(name: str) -> str:
    normalized = name.strip().lower()
    aliases = {"eidos director": "director", "eidos council": "council"}
    return aliases.get(normalized, normalized)


def _usage_dict(usage: Any) -> dict[str, int]:
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
    }


def _internal_output(output: Any) -> AgentResult | None:
    if isinstance(output, SDKAgentResult):
        return output.to_internal()
    if isinstance(output, AgentResult):
        return output
    return None


def _telemetry_hooks(telemetry: LiveTelemetry, config: LabConfig) -> Any:
    from agents.lifecycle import RunHooksBase

    class Hooks(RunHooksBase):
        async def on_agent_start(self, context: Any, agent: Any) -> None:
            key = _agent_key(agent.name)
            if key != "director":
                telemetry.verify_start(key)
                profile = config.models[key]
                telemetry.budget.authorize_specialist(
                    key, profile.model, council=key == "council"
                )
            telemetry.events.append({"event": "agent_start", "agent": key})

        async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
            key = _agent_key(agent.name)
            telemetry.events.append({"event": "agent_end", "agent": key})
            internal = _internal_output(output)
            if internal is not None:
                telemetry.verify_output(key, internal)
                telemetry.agent_outputs.append((key, internal))

        async def on_llm_start(
            self, context: Any, agent: Any, system_prompt: Any, input_items: Any
        ) -> None:
            key = _agent_key(agent.name)
            profile = config.models[key]
            telemetry.budget.authorize_call(
                key, profile.model, council=key == "council"
            )
            telemetry.events.append(
                {"event": "llm_start", "agent": key, "model": profile.model}
            )

        async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
            key = _agent_key(agent.name)
            profile = config.models[key]
            usage = _usage_dict(response.usage)
            telemetry.budget.record_call(key, profile.model, usage)
            telemetry.events.append(
                {
                    "event": "llm_end",
                    "agent": key,
                    "model": profile.model,
                    "usage": usage,
                }
            )

        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            name = str(getattr(tool, "name", type(tool).__name__))
            telemetry.budget.record_tool(name)
            telemetry.events.append(
                {
                    "event": "tool_start",
                    "agent": _agent_key(agent.name),
                    "tool": name,
                    "arguments": getattr(context, "tool_arguments", None),
                }
            )

        async def on_tool_end(
            self, context: Any, agent: Any, tool: Any, result: Any
        ) -> None:
            telemetry.events.append(
                {
                    "event": "tool_end",
                    "agent": _agent_key(agent.name),
                    "tool": str(getattr(tool, "name", type(tool).__name__)),
                }
            )

    return Hooks()


class AgentsSDKRuntime:
    def __init__(self, config: LabConfig) -> None:
        self.config = config
        self.registry = ModelRegistry(config)
        self.graph: SDKGraph | None = None

    def initialize(self, hooks: Any | None = None) -> SDKGraph:
        self.graph = build_sdk_graph(self.config, self.registry, run_hooks=hooks)
        return self.graph

    async def run_director(
        self,
        task: TaskSpec,
        session_db: Path,
        state_path: Path | None = None,
        *,
        telemetry: LiveTelemetry | None = None,
    ) -> tuple[FinalDecision | None, dict[str, int], list[Any]]:
        try:
            from agents import Runner, SQLiteSession, trace
        except Exception as exc:
            raise SDKUnavailable(
                f"OpenAI Agents SDK runtime unavailable: {type(exc).__name__}: {exc}"
            ) from exc

        if telemetry is not None:
            telemetry.task_id = task.task_id
            telemetry.required_sequence = list(
                task.research_requirements.required_specialists
            )
        hooks = _telemetry_hooks(telemetry, self.config) if telemetry else None
        graph = self.graph or self.initialize(hooks)
        session = SQLiteSession(task.task_id, str(session_db))
        prompt = task.model_dump_json(indent=2)
        with trace(
            "Eidos Agent Lab task",
            trace_id=None,
            group_id=task.task_id,
            metadata={
                "task_id": task.task_id,
                "repo_commit": task.base_commit,
                "workflow_state": "TRIAGED",
            },
            disabled=not self.config.tracing_enabled,
        ) as current_trace:
            if telemetry is not None:
                telemetry.trace_id = getattr(current_trace, "trace_id", None)
            result = await Runner.run(
                graph.director,
                prompt,
                context=telemetry,
                session=session,
                hooks=hooks,
                max_turns=self.config.budget.maximum_turns,
            )

        usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }
        raw_usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        if raw_usage:
            usage = _usage_dict(raw_usage)
        interruptions = list(getattr(result, "interruptions", []))
        if interruptions and state_path is not None:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(result.to_state().to_json(), indent=2) + "\n",
                encoding="utf-8",
            )
        output = (
            result.final_output
            if isinstance(result.final_output, FinalDecision)
            else None
        )
        return output, usage, interruptions

    async def resume_director(
        self, task: TaskSpec, session_db: Path, state_path: Path
    ) -> tuple[FinalDecision | None, dict[str, int], list[Any]]:
        try:
            from agents import Runner, RunState, SQLiteSession
        except Exception as exc:
            raise SDKUnavailable(
                f"OpenAI Agents SDK runtime unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        graph = self.graph or self.initialize()
        state_json = json.loads(state_path.read_text(encoding="utf-8"))
        state = await RunState.from_json(graph.director, state_json)
        for interruption in state.get_interruptions():
            state.approve(interruption)
        session = SQLiteSession(task.task_id, str(session_db))
        result = await Runner.run(graph.director, state, session=session)
        interruptions = list(getattr(result, "interruptions", []))
        if interruptions:
            state_path.write_text(
                json.dumps(result.to_state().to_json(), indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            state_path.unlink(missing_ok=True)
        usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }
        raw_usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        if raw_usage:
            usage = _usage_dict(raw_usage)
        output = (
            result.final_output
            if isinstance(result.final_output, FinalDecision)
            else None
        )
        return output, usage, interruptions
