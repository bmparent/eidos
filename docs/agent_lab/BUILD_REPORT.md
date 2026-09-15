# Eidos Research Council / Multi-Agent Lab v1 — Build Report

## Status

`READY_FOR_BRENT_REVIEW`

This status means the local architecture, mocked orchestration, current SDK graph, CLI, persistence,
guards, documentation and relevant regression suites passed. It does not claim a paid live research
mission, improved Sentinel metrics, production deployment or scientific validation.

## Git identity

- Repository: `bmparent/eidos`
- Branch: `codex/eidos-agent-lab-v1-2026-09-15`
- Starting commit: `a2da5fd9af21c02573e538fd30aaf7d5c690e457`
- Ending implementation commit: `ad30c8b9bcf46406dcdb7ccf7f9303fc1c8ec0a6`
- Closeout: this report is added in a subsequent documentation-only commit so it can cite the exact
  immutable implementation commit. The final branch HEAD is reported in the Codex handoff.
- Merge/deployment: not performed.

## Architecture implemented

`eidos_agents/` is an independent root package. Director is the manager and specialists are bounded
`Agent.as_tool()` tools with `SpecialistWorkOrder` inputs and Pydantic structured output. LLM-originated
tool fan-out is serial; explicit independent fan-out uses a semaphore with configured parallel/call
limits. Python owns stage transitions, capabilities, approvals, scope, branch safety, budget, evidence
policy and merge/Council gates.

Persistence uses `artifacts/agent_lab/agent_lab.sqlite` as an index and human-readable JSON/Markdown
task bundles. Task transitions, command digests, approvals, hypotheses and immutable experiment/result
records survive restart. Director uses per-task SQLite sessions. SDK `RunState` is serialized for HITL
resume without tracing API keys.

## Files added and modified

- Added 22 modules under `eidos_agents/` for schemas, runtime, agents, orchestration, workflow,
  persistence, budgets, permissions, guardrails, tools, approvals, logging, routing and rendering.
- Added 12 agent-lab test files (47 collected tests including one opt-in live smoke).
- Added `config/agent_lab/models.yaml` and versioned `model_pricing.yaml`.
- Added `missions/sentinel_precision_v1.yaml`.
- Added four operator/science/cost/role docs plus this report.
- Added proof journal and plain-language analysis for 2026-09-15.
- Added root `pyproject.toml`, `.env.example`, ignore rules and README link.
- Modified no Eidos engine, Sentinel algorithm, threshold, label, compression or familiarity source.

## Agent definitions and actual model routing

| Agent | Model | Reasoning | Authority |
|---|---|---|---|
| Director | `gpt-5.6-terra` | high | manager; no source writes |
| Archivist | `gpt-5.6-luna` | medium | read-only evidence |
| Curie | `gpt-5.6-terra` | high | experiment design; read-only |
| Sentry | `gpt-5.6-sol` | high | Sentinel science; read-only |
| Gauss | `gpt-5.6-sol` | xhigh | mathematics; read-only |
| Forge | `gpt-5.3-codex` | high | scoped feature-branch writes/tests |
| Bench | `gpt-5.4-mini` | medium | tests/benchmarks; generated artifacts only |
| Auditor | `gpt-5.6-sol` | xhigh | independent read/test; no writes |
| Council | `gpt-6-astra` | max | rare analysis; approval-gated |

`doctor --live-model-check` verified these identifiers are available to the configured API project on
2026-09-15. No silent fallback occurred. Explicit fallback environment variables exist; substitutions
are written to manifests.

## Tools

The typed repository surface includes status/branch/commit/diff, bounded search/read/tree, artifact
read/list, pytest and safe command execution, feature-branch validation/commit, file/config hashing,
environment collection and crash scans. Destructive git, force push, automatic branch switching,
deployment, path escape and non-allowlisted programs are rejected.

## Permissions and guardrails

- `ScopeGuard` / `ForbiddenScopeGuard`
- `SecretGuard`
- `MainBranchGuard`
- `AuditSeparationGuard`
- `BudgetGuard` through `BudgetManager`
- `EvidenceGuard` plus schema-level KNOWN validation
- `MergeGate`
- `CouncilGate`
- durable human approval and resume
- source-snapshot protection around Bench test execution

Forge cannot write on `main`, merge or deploy. Bench and Auditor cannot write source. Council cannot
write and only Director can request it. Auditor BLOCK cannot progress automatically.

## CLI

Implemented: `doctor`, `investigate`, `run`, `mission`, `task`, `evidence`, `costs`, `approve`, `resume`,
`hypotheses`, and `experiments`, with `--dry-run` / `--mock`. Runtime receipts are ignored and do not
dirty source.

## Tests and results

| Command | Result |
|---|---|
| `.\.venv312\Scripts\python.exe -m pytest tests/agent_lab -q` | 46 passed, 1 opt-in live smoke skipped |
| `.\.venv\Scripts\uvx.exe ruff check eidos_agents tests/agent_lab` | passed |
| from `eidos/`: `python -m pytest -q` with plugin autoload disabled | 140 passed, 1 skipped |
| `python -m pytest eidos/repo/tests -q` | 78 passed, 4 skipped |
| from `services/sentinel-runner/`: `python -m pytest -q` | passed (26 tests) |
| focused root RNG-null suite | 9 passed, 5 pre-existing import-path failures |

The five unrelated failures all stop in existing `eidos/sentinel/event_merge.py` at
`from operator_explanation import enrich_incident_card`; the module exists at
`eidos/operator_explanation.py`. No agent-lab frame appears in those traces. This patch does not mix an
unrelated proof-runner import fix into the infrastructure change.

## Mock workflow result

Task `TASK-20260915T230550Z-084EC5` completed the mocked Director → evidence → hypothesis → experiment
→ Forge → Bench → Auditor → human-review path. Auditor returned `PASS_WITH_LIMITATIONS` because no
live model call occurred. The cost receipt preserves tool use and labels dollar cost unknown.

## Sentinel dry-run result

Task `TASK-20260915T230557Z-AC01B6` completed the research-only mission. It generated an EvidencePacket,
Hypothesis and ExperimentSpec, performed no implementation, and left
`eidos/EIDOS_BRAIN_UNIFIED_v0_4.7.02.py` byte-identical. It does not claim old CICIDS metrics are current.

## Tracing and experimental SDK surfaces

Agents SDK tracing is enabled by default and documented with the OpenAI traces dashboard. It was not
exercised because no paid model run was authorized. The graph uses stable `Agent`, `Runner`,
`Agent.as_tool()`, structured outputs, SQLite sessions and `RunState` HITL serialization.

`SandboxAgent` is not a hard dependency. Forge uses a documented `ForgeWorkspaceAdapter` with the
local policy-checked fallback because sandbox transport is host-specific/experimental. The boundary can
adopt SandboxAgent later without changing workflow authority.

## Persistence and recovery

SQLite metadata plus JSON/Markdown receipts passed restart tests. Approval state persists, and an
approved SDK interruption can reconstruct `RunState` and resume the outer Director run. Experiment
specifications/results are immutable; repeated runs receive new IDs.

## Cost implications

No model inference call was made, so this build incurred no agent-inference token cost. The read-only
model availability call succeeded. Checked-in pricing is deliberately `null`: current reviewed API
prices/tool charges were not established, so CostReceipt returns an unknown estimate rather than a
false exact bill. Brent must add versioned reviewed prices and a task budget before live execution.

## Known limitations

- Paid live agent response quality and remote traces were not tested.
- Python 3.12+ is required; current SDK 0.22.2 failed to import on this host's Python 3.11.0.
- Forge uses the local controlled workspace adapter, not SandboxAgent transport.
- Sentinel precision/calibration remains a hypothesis/experiment plan, not a new measured result.
- Pricing estimates remain unknown until a reviewed registry is supplied.
- The unrelated root RNG-null import defect remains.

## Exact commands Brent should run next

```powershell
git switch codex/eidos-agent-lab-v1-2026-09-15

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m eidos_agents doctor --live-model-check
.\.venv\Scripts\python.exe -m pytest tests/agent_lab -q

# No-cost receipt path
.\.venv\Scripts\python.exe -m eidos_agents run "Review the agent-lab architecture" --mock
.\.venv\Scripts\python.exe -m eidos_agents mission missions/sentinel_precision_v1.yaml --dry-run

# Only after reviewing prices and setting a budget:
$env:EIDOS_AGENT_TASK_BUDGET_USD = "<approved amount>"
$env:EIDOS_AGENT_LIVE_SMOKE = "1"
.\.venv\Scripts\python.exe -m pytest tests/agent_lab/test_live_smoke.py -m live -q
```

Do not merge until Brent reviews the implementation diff, known limitation list, ignored receipt bundle
and an explicitly budgeted live smoke if desired.

## Proof Logic + Meaning

### Goal reached

The architecture gate is `passed` locally; scientific Sentinel improvement is `missing` because no new
measurement was run.

### Previous state

Eidos had proof practices and artifacts but no typed, persistent manager organization that enforced
role separation, budget, approvals, claim vocabulary and audit progression.

### Technical logic utilized

Strict schemas, capability allowlists, path/branch guards, immutable records and a deterministic state
graph turn governance into executable constraints. Director remains manager; specialists are tools;
Forge, Bench and Auditor remain separate.

### Math / scoring logic

The Sentinel mission requires precision, recall, F1, FP/10k, attack-window coverage and latency together.
Cost uses visible regular/cached/output/reasoning token classes against versioned prices. Unknown inputs
remain unknown.

### Philosophical meaning

Independent audit is restraint before belief. Reproducibility is truth that can be revisited.

### Why this is better

Agents cannot make prohibited workflow transitions by writing convincing prose. Failures,
counterevidence, missing pricing and recall regressions remain visible.

### How this moves Eidos closer to the north-star goal

It strengthens reproducibility, self-monitoring and explanation around future streaming intelligence
experiments. It does not itself prove learning, compression or anomaly performance.

### Evidence

The cited test results, mocked task bundles, model availability check, dry-run mission, source diff and
Drive-mirrored build receipt support the architecture claim.

### Remaining uncertainty

Live orchestration, remote tracing, SandboxAgent transport, scientific Sentinel performance, GPU and
production behavior remain unproven.
