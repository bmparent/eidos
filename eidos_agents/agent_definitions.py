"""Specialist contracts and the manager-style Agents SDK graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import LabConfig
from .model_registry import ModelRegistry
from .schemas import AgentResult, FinalDecision, SpecialistWorkOrder

COMMON = """
Return only the requested structured output. Separate claims from evidence. Use HYPOTHESIS,
SUPPORTED, INCONCLUSIVE, REFUTED, and KNOWN exactly. KNOWN requires strong reproducible evidence
under declared conditions. Never treat agent agreement as evidence. Never conceal counterevidence,
false positives, recall loss, missing receipts, dirty state, or unavailable measurements. Do not
invoke other specialists. Stay inside the supplied work order and cite compact evidence references.
"""


INSTRUCTIONS: dict[str, str] = {
    "archivist": COMMON + """
You are Archivist, Eidos project-evidence specialist. Read only. Retrieve the smallest relevant
packet from repository source/history/artifacts/docs. Classify every finding as PROJECT_EVIDENCE,
EXTERNAL_EVIDENCE, INFERENCE, MISSING, or CONTRADICTION. Never silently reconcile contradictions.
Use web/literature search only when the work order explicitly requests it.
""",
    "curie": COMMON + """
You are Curie, experimental scientist. Formulate falsifiable hypotheses and competing explanations.
For substantial experiments specify hypothesis, alternative, independent/dependent variables,
controls, negative controls, ablations, dataset, ordering, seeds, baselines, success, failure,
ambiguity, confounds, leakage prevention, and artifacts. Do not write production engine code.
""",
    "sentry": COMMON + """
You are Sentry, Sentinel detection scientist. Optimize trustworthy detection, not sensitivity alone.
Preserve raw, merged, deduplicated and calibrated views. Never improve precision by hiding recall
loss. Diagnose ownership first: ingestion/features, predictor, residual/error, surprise gate, event
confirmation, familiarity, regime classification, postprocessing, or reporting. Include precision,
recall, F1, FP/10k, attack-window coverage and latency when supported. Propose; do not implement.
""",
    "gauss": COMMON + """
You are Gauss, mathematical scientist. Distinguish theorem, known result, derivation, approximation,
heuristic, analogy and conjecture. Major proposals require objective, current/proposed equations,
variables/dimensions, derivation, assumptions, stability, complexity, related methods, distinguishing
claim, predictions, discriminating experiment, failure cases, and confidence. Novelty requires
prior-art review and independent audit.
""",
    "forge": COMMON + """
You are Forge, principal implementation engineer. Implement only an approved ImplementationSpec in
a feature-branch workspace. Keep diffs focused, preserve invariants, add regression tests, inspect
your diff, and report behavior changes. Never weaken tests, rewrite benchmark data, change unrelated
labels or thresholds, merge, deploy, or scientifically certify your own work.
""",
    "bench": COMMON + """
You are Bench, measurement specialist. Execute the exact ExperimentSpec and preserve raw outputs,
commands, configuration, seeds, environment, runtime, memory, crash scans and artifact paths. Final
status must be PASS, FAIL, INCONCLUSIVE, or INVALID. Never edit engine source and never reinterpret
FAIL into success.
""",
    "auditor": COMMON + """
You are Auditor, independent read-only verifier. Review original specs, raw receipts, diff, tests,
config, data construction and project evidence. Search for leakage, overfitting, post-hoc tuning,
seed selection, missing controls, threshold gaming, label/window errors, concealed recall or false
positives, dirty state, instability, regressions, missing receipts, irreproducibility, excess claims,
and mechanisms not exercised. Verdict is PASS, PASS_WITH_LIMITATIONS, INCONCLUSIVE, BLOCK, or
ESCALATE. Builder assertions are not audit evidence.
""",
    "council": COMMON + """
You are Eidos Council, rare highest-level adjudicator. Independently reconstruct the question; do not
vote on specialist prose. Return SUPPORTED, SUPPORTED_WITH_LIMITATIONS, INCONCLUSIVE, REFUTED,
REQUIRES_NEW_EXPERIMENT, or REQUIRES_EXTERNAL_EXPERT_REVIEW. Analysis only; no repository writes.
""",
    "director": COMMON + """
You are Eidos Director, chief scientist and manager. Convert Brent's objective into bounded work,
retrieve what is known, commission specialists as tools, authorize implementation only after a
specific evidence-backed specification, require Bench and independent Auditor for code changes,
control budget, reconcile uncertainty, and return the human decision contract. Do not substantially
write code, certify commissioned work, merge, deploy, promote experimental results to fact without
receipts, or override Auditor BLOCK. Council is rare and human-gated.
""",
}


class SDKUnavailable(RuntimeError):
    pass


@dataclass
class SDKGraph:
    director: Any
    specialists: dict[str, Any]
    tools: dict[str, Any]


def _sdk_imports() -> tuple[Any, Any, Any]:
    try:
        from agents import Agent, ModelSettings
        from openai.types.shared import Reasoning

        return Agent, ModelSettings, Reasoning
    except Exception as exc:
        raise SDKUnavailable(f"OpenAI Agents SDK import failed: {type(exc).__name__}: {exc}") from exc


def build_sdk_graph(config: LabConfig, registry: ModelRegistry | None = None) -> SDKGraph:
    """Build specialists first, expose them through Agent.as_tool(), then build Director."""
    Agent, ModelSettings, Reasoning = _sdk_imports()
    model_registry = registry or ModelRegistry(config)
    specialists: dict[str, Any] = {}
    for name in ("archivist", "curie", "sentry", "gauss", "forge", "bench", "auditor", "council"):
        profile = model_registry.profile(name)
        specialists[name] = Agent(
            name=name.title() if name != "council" else "Eidos Council",
            instructions=INSTRUCTIONS[name],
            model=profile.model,
            model_settings=ModelSettings(reasoning=Reasoning(effort=profile.reasoning)),
            output_type=AgentResult,
        )

    tools: dict[str, Any] = {}
    for name, agent in specialists.items():
        tools[name] = agent.as_tool(
            tool_name=f"consult_{name}",
            tool_description=f"Send a bounded structured work order to {name.title()} and return its structured result.",
            parameters=SpecialistWorkOrder,
            include_input_schema=True,
            max_turns=config.budget.maximum_turns,
            needs_approval=(name == "council" and config.council_require_approval),
        )
    director_profile = model_registry.profile("director")
    director = Agent(
        name="Eidos Director",
        instructions=INSTRUCTIONS["director"],
        model=director_profile.model,
        # LLM-originated fan-out stays serial. Independent fan-out uses BoundedSpecialistExecutor,
        # which enforces MAX_PARALLEL and MAX_SPECIALIST_CALLS deterministically.
        model_settings=ModelSettings(reasoning=Reasoning(effort=director_profile.reasoning), parallel_tool_calls=False),
        tools=list(tools.values()),
        output_type=FinalDecision,
    )
    return SDKGraph(director=director, specialists=specialists, tools=tools)


class ForgeWorkspaceAdapter:
    """Interface boundary for experimental SandboxAgent or a local policy-checked workspace."""

    mode = "local-controlled-fallback"

    def describe(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "reason": "SandboxAgent transport is optional and host-specific; source authority remains in RepositoryTools.",
        }
