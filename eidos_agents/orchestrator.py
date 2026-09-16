"""Manager workflow: deterministic gates around optional LLM judgment."""

from __future__ import annotations

import json
import uuid

from .approvals import ApprovalManager, ApprovalRequired
from .budget import BudgetManager
from .config import LabConfig
from .logging import EventLogger
from .missions import MissionDefinition
from .model_registry import ModelRegistry
from .persistence import ArtifactStore
from .pricing import PriceRegistry
from .rendering import final_decision_markdown
from .repo_tools import RepositoryTools
from .schemas import (
    ApprovalRecord,
    ArchivistResult,
    AuditResult,
    AuditVerdict,
    BenchmarkReceipt,
    BenchmarkStatus,
    ClaimStatus,
    CurieResult,
    EvidenceItem,
    EvidencePacket,
    EvidenceSourceType,
    ExperimentSpec,
    FinalDecision,
    Hypothesis,
    ImplementationResult,
    ImplementationSpec,
    ResearchRequirements,
    RunManifest,
    SpecialistAccounting,
    TaskSpec,
    TaskType,
    WorkflowState,
    utc_now,
)
from .sdk_runtime import AgentsSDKRuntime, LiveTelemetry
from .workflow import StageRequirementUnsatisfied, WorkflowMachine


class EidosOrchestrator:
    def __init__(self, config: LabConfig) -> None:
        self.config = config
        config.ensure_directories()
        self.store = ArtifactStore(config.artifact_root)
        self.repo = RepositoryTools(config.repo_root, self.store)
        self.models = ModelRegistry(config)
        if config.pricing_path is None:
            raise ValueError("pricing_path is required")
        self.prices = PriceRegistry.load(config.pricing_path)

    def create_task(
        self,
        objective: str,
        *,
        task_type: TaskType = TaskType.RESEARCH,
        allowed_scope: list[str] | None = None,
        forbidden_scope: list[str] | None = None,
        required_artifacts: list[str] | None = None,
        research_requirements: ResearchRequirements | None = None,
    ) -> TaskSpec:
        task_id = "TASK-" + utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6].upper()
        task = TaskSpec(
            task_id=task_id,
            title=objective[:100],
            objective=objective,
            task_type=task_type,
            repo=str(self.config.repo_root),
            base_commit=self.repo.repo_current_commit(),
            allowed_scope=allowed_scope or ["eidos_agents/**", "tests/agent_lab/**", "docs/agent_lab/**", "missions/**", "config/agent_lab/**"],
            forbidden_scope=forbidden_scope or ["eidos/EIDOS_BRAIN_UNIFIED_v0_4.7.02.py", "apps/**"],
            initial_question=objective,
            success_criteria=["structured evidence exists", "workflow invariants hold", "final decision cites receipts"],
            failure_criteria=["unsupported claim promoted to KNOWN", "builder self-audits", "Auditor BLOCK bypassed"],
            required_artifacts=required_artifacts or ["run_manifest.json", "cost_receipt.json", "final_decision.json"],
            budget=self.config.budget,
            research_requirements=research_requirements or ResearchRequirements(),
            human_approval_requirements=["merge", "deployment", "Council invocation", "Auditor BLOCK override"],
        )
        self.store.put_task(task)
        logger = EventLogger(self.store.task_dir(task_id) / "events.jsonl")
        logger.emit(task_id=task_id, workflow_state="CREATED", agent="director", model=self.models.profile("director").model, event_type="task_created")
        return task

    def approve(self, task_id: str, action: str, *, actor: str = "Brent") -> None:
        self.store.add_approval(ApprovalRecord(task_id=task_id, action=action, approved=True, actor=actor))

    def run_mocked(self, objective: str, *, research_only: bool = False, mission: MissionDefinition | None = None) -> FinalDecision:
        allowed = mission.allowed_scope if mission else None
        forbidden = mission.forbidden_scope if mission else None
        task = self.create_task(
            objective,
            task_type=TaskType.RESEARCH if research_only else TaskType.ENGINEERING,
            allowed_scope=allowed,
            forbidden_scope=forbidden,
            required_artifacts=mission.required_artifacts if mission else None,
            research_requirements=mission.research_requirements if mission else None,
        )
        workflow = WorkflowMachine(task.task_id, self.store)
        workflow.transition(WorkflowState.TRIAGED, actor="director", reason="mocked deterministic triage")
        workflow.transition(WorkflowState.EVIDENCE_GATHERING, actor="director", reason="Archivist work order issued")

        evidence = self._mock_evidence(task, mission)
        evidence_path = self.store.save_record(
            "evidence", f"EVID-{task.task_id}", task.task_id, evidence,
            f"tasks/{task.task_id}/evidence/evidence_packet.json",
        )
        workflow.transition(
            WorkflowState.HYPOTHESIS, actor="director", reason="bounded evidence packet received",
            output_artifact_ids=[evidence_path.relative_to(self.config.artifact_root).as_posix()],
        )
        hypothesis = Hypothesis(
            hypothesis_id="HYP-" + task.task_id.removeprefix("TASK-"),
            statement=("Sentinel alert pressure is owned by one or more confirmation/calibration layers; current receipts are required to localize it" if research_only else "The manager workflow enforces independent benchmark and audit stages"),
            mechanism="Deterministic state transitions and role-separated capabilities",
            alternatives=["Reporting-only distortion", "Incomplete or stale evidence"],
            predictions=["A code-changing task cannot reach human review before Bench and Auditor"],
            falsifiers=["A direct IMPLEMENTING to READY_FOR_HUMAN transition succeeds"],
            evidence_for=[evidence.evidence[0].evidence_id] if evidence.evidence else [],
            status=ClaimStatus.HYPOTHESIS,
        )
        hypothesis_path = self.store.add_hypothesis(hypothesis, task.task_id)
        experiment = self._mock_experiment(task, hypothesis, research_only)
        experiment_path = self.store.save_record(
            "experiment", experiment.experiment_id, task.task_id, experiment,
            f"tasks/{task.task_id}/experiments/{experiment.experiment_id}/spec.json",
        )
        workflow.transition(
            WorkflowState.EXPERIMENT_SPECIFIED, actor="curie", reason="discriminating experiment specified",
            input_artifact_ids=[hypothesis_path.relative_to(self.config.artifact_root).as_posix()],
            output_artifact_ids=[experiment_path.relative_to(self.config.artifact_root).as_posix()],
        )

        audit: AuditResult | None = None
        changes: list[str] = []
        benchmarks: list[str] = []
        implementation_ref: str | None = None
        audit_ref: str | None = None
        if not research_only:
            implementation = ImplementationSpec(
                implementation_id="IMP-" + task.task_id.removeprefix("TASK-"), objective="Exercise approved mocked workflow",
                allowed_files_modules=task.allowed_scope, forbidden_files_modules=task.forbidden_scope,
                behavior_change="No Eidos engine behavior change", invariants=["main is not modified", "Forge does not audit"],
                tests_required=["mock workflow test"], benchmark_required=True, rollback_condition="any workflow invariant failure",
            )
            implementation_path = self.store.save_record(
                "implementation_spec", implementation.implementation_id, task.task_id, implementation,
                f"tasks/{task.task_id}/implementation/spec.json",
            )
            implementation_ref = implementation_path.relative_to(self.config.artifact_root).as_posix()
            workflow.transition(WorkflowState.IMPLEMENTATION_SPECIFIED, actor="director", reason="implementation spec approved")
            workflow.transition(WorkflowState.IMPLEMENTING, actor="forge", reason="mock adapter started")
            result = ImplementationResult(
                implementation_id=implementation.implementation_id, files_changed=[], diff_summary="Mocked: no source writes",
                tests_run=["mock workflow"], results={"mock workflow": "PASS"}, commit=None,
                known_risks=["No live model call performed"], ready_for_bench=True,
            )
            self.store.save_record("implementation_result", implementation.implementation_id + "-result", task.task_id, result, f"tasks/{task.task_id}/implementation/result.json")
            workflow.transition(WorkflowState.BENCHMARKING, actor="bench", reason="required tests passed", tests_passed=True)
            receipt = self._mock_benchmark(task, experiment)
            receipt_path = self.store.save_record(
                "benchmark", receipt.run_id, task.task_id, receipt,
                f"tasks/{task.task_id}/benchmarks/{receipt.run_id}.json",
            )
            benchmarks.append(receipt_path.relative_to(self.config.artifact_root).as_posix())
            workflow.transition(WorkflowState.AUDITING, actor="director", reason="benchmark receipt ready")
            audit = AuditResult(
                task_id=task.task_id, claims_reviewed=[hypothesis.statement], evidence_for=benchmarks,
                evidence_against=[], methodology_issues=["mocked orchestration does not establish live model availability"],
                reproducibility="Deterministic mocked path", regressions=[], severity="LOW",
                verdict=AuditVerdict.PASS_WITH_LIMITATIONS, corrective_actions=["Run opt-in live smoke on a compatible Python runtime"],
            )
            audit_path = self.store.save_record("audit", "AUD-" + task.task_id, task.task_id, audit, f"tasks/{task.task_id}/audit/audit_result.json")
            audit_ref = audit_path.relative_to(self.config.artifact_root).as_posix()
            workflow.transition(
                WorkflowState.READY_FOR_HUMAN, actor="director", reason="independent audit allows review",
                audit_verdict=audit.verdict, output_artifact_ids=[audit_ref],
            )
            changes = ["No source changes; mocked adapters and gates were exercised"]
        else:
            workflow.transition(
                WorkflowState.READY_FOR_HUMAN, actor="director", reason="research-only mission produced evidence and experiment spec",
                input_artifact_ids=[experiment_path.relative_to(self.config.artifact_root).as_posix()],
            )

        budget = BudgetManager(task.task_id, task.budget, self.prices)
        for tool in ("archivist", "curie", "sentry"):
            budget.record_tool(tool)
        cost = budget.receipt()
        cost_path = self.store.write_model(f"tasks/{task.task_id}/cost_receipt.json", cost, immutable=True)
        decision = FinalDecision(
            task_id=task.task_id,
            decision=("RESEARCH_SPEC_READY_FOR_HUMAN; no implementation authorized" if research_only else "MOCK_WORKFLOW_PASS_WITH_LIMITATIONS"),
            what_changed=changes,
            what_we_learned=["Structured workflow and separation gates complete the mocked path", "Current scientific Sentinel state remains unclaimed"],
            evidence=[evidence_path.relative_to(self.config.artifact_root).as_posix(), experiment_path.relative_to(self.config.artifact_root).as_posix(), *benchmarks, *([audit_ref] if audit_ref else [])],
            what_remains_uncertain=["Live model availability and paid orchestration were not tested", "The Sentinel hypothesis remains untested"],
            auditor_verdict=audit.verdict if audit else None,
            regressions=[], cost_receipt_ref=cost_path.relative_to(self.config.artifact_root).as_posix(),
            recommended_next_action="Review the experiment specification before authorizing implementation or paid execution",
            human_action_required=["Approve any live paid smoke", "Approve merge separately"],
        )
        decision_path = self.store.write_model(f"tasks/{task.task_id}/final_decision.json", decision, immutable=True)
        (self.store.task_dir(task.task_id) / "final_decision.md").write_text(final_decision_markdown(decision), encoding="utf-8")
        manifest = RunManifest(
            task_id=task.task_id, task_ref=f"tasks/{task.task_id}/task.json",
            repo_state={"commit": task.base_commit, "branch": self.repo.repo_current_branch(), "dry_run": True},
            agents={name: {"mode": "mocked"} for name in self.config.models}, models=self.models.manifest(),
            experiments=[experiment_path.relative_to(self.config.artifact_root).as_posix()], implementation=implementation_ref,
            benchmarks=benchmarks, audit=audit_ref, costs=cost_path.relative_to(self.config.artifact_root).as_posix(),
            final_decision=decision_path.relative_to(self.config.artifact_root).as_posix(), tracing_available=False,
            tracing_note="Mock path does not emit remote traces",
        )
        self.store.write_model(f"tasks/{task.task_id}/run_manifest.json", manifest, immutable=True)
        return decision

    async def run_live(
        self,
        objective: str,
        *,
        research_only: bool = False,
        mission: MissionDefinition | None = None,
    ) -> FinalDecision:
        task = self.create_task(
            objective,
            task_type=TaskType.RESEARCH,
            allowed_scope=mission.allowed_scope if mission else None,
            forbidden_scope=mission.forbidden_scope if mission else None,
            required_artifacts=mission.required_artifacts if mission else None,
            research_requirements=mission.research_requirements if mission else None,
        )
        workflow = WorkflowMachine(task.task_id, self.store)
        workflow.transition(WorkflowState.TRIAGED, actor="director", reason="live Director run requested")
        runtime = AgentsSDKRuntime(self.config)
        budget = BudgetManager(task.task_id, task.budget, self.prices)
        required = task.research_requirements.required_specialists
        allowed = (
            set(required)
            if required
            else ({"archivist", "curie", "sentry", "gauss"} if research_only else None)
        )
        telemetry = LiveTelemetry(budget=budget, allowed_specialists=allowed)
        state_path = self.store.task_dir(task.task_id) / "sdk_run_state.json"
        decision, _usage, interruptions = await runtime.run_director(
            task,
            self.config.artifact_root / "director_sessions.sqlite",
            state_path,
            telemetry=telemetry,
        )
        if interruptions:
            workflow.transition(WorkflowState.BLOCKED, actor="director", reason="human approval required")
            ApprovalManager(self.store).require(task.task_id, "council", f"{len(interruptions)} SDK approval interruption(s)")
        if decision is None:
            workflow.transition(WorkflowState.BLOCKED, actor="director", reason="Director returned malformed structured output")
            raise RuntimeError("Director did not return FinalDecision")
        experiment_refs: list[str] = []
        for index, (agent, output) in enumerate(telemetry.agent_outputs, 1):
            self.store.save_record(
                "agent_result",
                f"AGENT-{task.task_id}-{index:02d}-{agent}",
                task.task_id,
                output,
                f"tasks/{task.task_id}/agents/{index:02d}-{agent}.json",
            )
            if isinstance(output, ArchivistResult):
                packet = output.evidence_packet
                if packet.task_id != task.task_id:
                    raise StageRequirementUnsatisfied(["Archivist EvidencePacket task_id mismatch"])
                self.store.save_record(
                    "evidence",
                    f"EVID-{task.task_id}-{index:02d}",
                    task.task_id,
                    packet,
                    f"tasks/{task.task_id}/evidence/evidence_packet.json",
                )
            if isinstance(output, CurieResult):
                experiment = output.experiment_spec
                if experiment.task_id != task.task_id:
                    raise StageRequirementUnsatisfied(["Curie ExperimentSpec task_id mismatch"])
                experiment_path = self.store.save_record(
                    "experiment",
                    experiment.experiment_id,
                    task.task_id,
                    experiment,
                    f"tasks/{task.task_id}/experiments/{experiment.experiment_id}/spec.json",
                )
                experiment_refs.append(
                    experiment_path.relative_to(self.config.artifact_root).as_posix()
                )
        accounting = self._specialist_accounting(task, telemetry)
        self.store.save_record(
            "specialist_accounting",
            f"SPECIALISTS-{task.task_id}",
            task.task_id,
            accounting,
            f"tasks/{task.task_id}/specialist_accounting.json",
        )
        events_path = self.store.task_dir(task.task_id) / "live_events.json"
        events_path.write_text(
            json.dumps(telemetry.events, indent=2, default=str) + "\n", encoding="utf-8"
        )
        cost = budget.receipt()
        cost_path = self.store.write_model(
            f"tasks/{task.task_id}/cost_receipt.json", cost, immutable=True
        )
        decision = decision.model_copy(
            update={"cost_receipt_ref": cost_path.relative_to(self.config.artifact_root).as_posix()}
        )
        decision_path = self.store.write_model(
            f"tasks/{task.task_id}/final_decision.json", decision, immutable=True
        )
        (self.store.task_dir(task.task_id) / "final_decision.md").write_text(
            final_decision_markdown(decision), encoding="utf-8"
        )
        workflow.transition(
            WorkflowState.EVIDENCE_GATHERING,
            actor="director",
            reason="live research requirements evaluated",
        )
        if experiment_refs:
            workflow.transition(
                WorkflowState.EXPERIMENT_SPECIFIED,
                actor="curie",
                reason="persisted ExperimentSpec validated for current task",
                output_artifact_ids=experiment_refs,
            )
        gate_error: StageRequirementUnsatisfied | None = None
        try:
            workflow.transition(
                WorkflowState.READY_FOR_HUMAN,
                actor="director",
                reason="persisted research requirements satisfied",
            )
        except StageRequirementUnsatisfied as exc:
            gate_error = exc
            workflow.transition(WorkflowState.BLOCKED, actor="workflow", reason=str(exc))
        manifest = RunManifest(
            task_id=task.task_id,
            task_ref=f"tasks/{task.task_id}/task.json",
            repo_state={
                "commit": task.base_commit,
                "branch": self.repo.repo_current_branch(),
                "dry_run": False,
            },
            agents={"events": telemetry.events},
            models=self.models.manifest(),
            experiments=experiment_refs,
            costs=cost_path.relative_to(self.config.artifact_root).as_posix(),
            final_decision=decision_path.relative_to(self.config.artifact_root).as_posix(),
            tracing_available=bool(telemetry.trace_id),
            tracing_note=(
                f"trace_id={telemetry.trace_id}"
                if telemetry.trace_id
                else "SDK trace identifier unavailable"
            ),
        )
        self.store.write_model(
            f"tasks/{task.task_id}/run_manifest.json", manifest, immutable=True
        )
        if gate_error is not None:
            raise gate_error
        return decision

    def _specialist_accounting(
        self, task: TaskSpec, telemetry: LiveTelemetry
    ) -> SpecialistAccounting:
        attempted: list[str] = []
        for event in telemetry.events:
            agent = str(event.get("agent", "")).lower()
            if event.get("event") == "agent_start" and agent != "director" and agent not in attempted:
                attempted.append(agent)
        completed: list[str] = []
        failed: dict[str, str] = {}
        for agent, output in telemetry.agent_outputs:
            normalized = agent.lower()
            if output.status.upper() in {"FAIL", "FAILED", "BLOCK", "BLOCKED", "ERROR"}:
                failed[normalized] = output.summary
            elif normalized not in completed:
                completed.append(normalized)
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

    async def resume_live(self, task_id: str) -> FinalDecision:
        status = ApprovalManager(self.store).resume(task_id)
        if status["status"] != "APPROVAL_SATISFIED":
            raise ApprovalRequired(task_id, status.get("action", "unknown"))
        task = self.store.get_record("task", task_id, TaskSpec)
        state_path = self.store.task_dir(task_id) / "sdk_run_state.json"
        if not state_path.exists():
            raise RuntimeError("no serialized SDK run state exists")
        runtime = AgentsSDKRuntime(self.config)
        decision, _usage, interruptions = await runtime.resume_director(
            task, self.config.artifact_root / "director_sessions.sqlite", state_path
        )
        if interruptions:
            ApprovalManager(self.store).require(task_id, "council", f"{len(interruptions)} SDK approval interruption(s)")
        if decision is None:
            raise RuntimeError("resumed Director did not return FinalDecision")
        (self.store.task_dir(task_id) / "pending_approval.json").unlink(missing_ok=True)
        workflow = WorkflowMachine(task_id, self.store)
        workflow.transition(WorkflowState.EVIDENCE_GATHERING, actor="director", reason="approved research run resumed")
        workflow.transition(WorkflowState.READY_FOR_HUMAN, actor="director", reason="resumed research decision ready")
        return decision

    def _mock_evidence(self, task: TaskSpec, mission: MissionDefinition | None) -> EvidencePacket:
        target = "eidos/docs/proof_runs"
        latest = sorted((self.config.repo_root / target).rglob("*"), key=lambda p: p.as_posix(), reverse=True) if (self.config.repo_root / target).exists() else []
        relevant = [p for p in latest if p.is_file() and any(word in p.name.lower() for word in ("sentinel", "calibration", "precision", "proof", "analysis"))][:8]
        items = [
            EvidenceItem(
                evidence_id=f"EVID-{index+1:04d}", source_type=EvidenceSourceType.PROJECT_EVIDENCE,
                source="repository", location=path.relative_to(self.config.repo_root).as_posix(),
                claim_supported="This is a candidate current project receipt; metrics require content-level audit",
                excerpt_or_summary="Located by deterministic bounded repository scan; not interpreted as a current result",
                confidence=0.7, stale_possible=True,
            ) for index, path in enumerate(relevant)
        ]
        if not items:
            items.append(EvidenceItem(
                evidence_id="EVID-0001", source_type=EvidenceSourceType.MISSING, source="repository",
                location=target, claim_supported="No candidate Sentinel receipt located",
                excerpt_or_summary="Evidence must be gathered before scientific claims", confidence=1.0, stale_possible=False,
            ))
        return EvidencePacket(
            task_id=task.task_id, question=task.initial_question, evidence=items,
            contradictions=[], missing_evidence=["content-level freshness and comparability audit", "new execution under a frozen ExperimentSpec"],
            prior_results=[item.location for item in items if item.source_type == EvidenceSourceType.PROJECT_EVIDENCE],
            known_failures=["Historical metrics must not be assumed current"],
            recommended_context=(mission.evidence_targets if mission else ["task spec", "repository state"]),
        )

    def _mock_experiment(self, task: TaskSpec, hypothesis: Hypothesis, research_only: bool) -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id="EXP-" + task.task_id.removeprefix("TASK-"), task_id=task.task_id,
            hypothesis_ids=[hypothesis.hypothesis_id], dataset="newest eligible frozen CICIDS/WebAttacks receipt or explicit MISSING",
            sample_policy="chronological calibration/evaluation/holdout; labels isolated until prediction freeze",
            controls=["current unmodified Sentinel profile", "raw event view"],
            negative_controls=["fully benign chronological replay"],
            ablations=["confirmation", "familiarity context", "merge/dedup reporting"], seeds=[0, 1, 2],
            variables={"independent": "single bounded layer intervention", "dependent": ["raw precision", "recall", "FP/10k", "coverage", "latency"]},
            metrics=["raw_precision", "merged_precision", "deduplicated_precision", "recall", "f1", "fp_per_10k", "attack_window_coverage", "detection_latency", "runtime", "memory", "crashes"],
            baseline="current commit and frozen config before any intervention",
            commands_or_runner_reference=["eidos/tools/run_labeled_domain_proof.py (exact invocation to be frozen after eligible dataset discovery)"],
            success_observation="predeclared precision/FP improvement without unacceptable recall, coverage, latency, crash, or core-touch regression",
            failure_observation="precision improves by concealing raw events or materially reducing recall/coverage",
            ambiguous_observation="seed-sensitive or incomparable receipts; classify INCONCLUSIVE",
            artifacts_required=["run_manifest.json", "precision_ledger.json", "benchmark_summary.csv", "crash_scan.json", "audit_result.json"],
        )

    def _mock_benchmark(self, task: TaskSpec, experiment: ExperimentSpec) -> BenchmarkReceipt:
        config = {"mode": "mock", "source_modification": False}
        return BenchmarkReceipt(
            task_id=task.task_id, experiment_id=experiment.experiment_id, run_id="RUN-" + uuid.uuid4().hex[:8].upper(),
            status=BenchmarkStatus.PASS, git_commit=task.base_commit, git_dirty_state=False, engine_version=None,
            config=config, config_hash=self.repo.calculate_config_hash(config), dataset_identity="mock-contract-fixture",
            sample_construction="deterministic schema fixture", frame_count=0, seed=0, device="local", cpu_gpu="CPU",
            commands=["python -m eidos_agents run --mock ..."], tests={"mock workflow": "PASS"},
            metrics={"workflow_contract": "PASS", "scientific_metrics": None}, crash_scan=[],
            artifact_paths=[f"tasks/{task.task_id}/transitions.jsonl"],
        )
