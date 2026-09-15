# Eidos Research Council / Multi-Agent Lab v1

This subsystem is a proof-first manager organization for Eidos Brain. It does not change the core
engine. A controlling Eidos Director sends bounded `SpecialistWorkOrder` objects to specialists
exposed through OpenAI Agents SDK `Agent.as_tool()`. Specialist results return to Director as typed
objects. Python, not model prose, enforces repository authority, budgets, approvals, workflow stages,
and independent audit.

## Architecture

```text
Brent -> Director
           |-- Archivist (read-only evidence)
           |-- Curie (experiments)
           |-- Sentry (detection science)
           |-- Gauss (mathematics)
           |-- Forge (feature-branch implementation)
           |-- Bench (measurement and generated receipts)
           |-- Auditor (independent read-only verification)
           `-- Council (rare, disabled/approval-gated)

question -> evidence -> hypothesis -> discriminating experiment
         -> implementation only if authorized -> benchmark -> audit -> human decision
```

Important modules:

- `schemas.py`: strict Pydantic contracts and scientific vocabulary.
- `workflow.py`: persisted deterministic state machine.
- `permissions.py`, `guardrails.py`, `repo_tools.py`: code-enforced authority and safety.
- `agent_definitions.py`: specialist prompts, model attachment, structured outputs, agents-as-tools.
- `sdk_runtime.py`: thin live Agents SDK/session/tracing adapter.
- `orchestrator.py`: task lifecycle and mocked acceptance workflow.
- `persistence.py`: SQLite metadata plus reviewable JSON/Markdown receipts.
- `budget.py`, `pricing.py`: visible tokens/calls and versioned estimated costs.
- `concurrency.py`: semaphore-bounded fan-out and technical-failure-only retries.

Forge workspace execution is behind `ForgeWorkspaceAdapter`. v1 uses a local policy-checked fallback;
the experimental `SandboxAgent` transport is not a hard dependency. The fallback never grants other
roles source-write authority, never switches branches automatically, and rejects protected-branch or
out-of-scope changes.

## Quickstart

Python 3.12+ is required. The current Agents SDK failed during import on this host's Python 3.11.0,
so the package fails honestly through `doctor` rather than masking that runtime incompatibility.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m eidos_agents doctor --live-model-check
.\.venv\Scripts\python.exe -m eidos_agents run "Your bounded objective" --mock
.\.venv\Scripts\python.exe -m eidos_agents mission missions/sentinel_precision_v1.yaml --dry-run
```

Ordinary tests never require an API key:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/agent_lab -q
```

Paid/live execution is opt-in: omit `--mock` and `--dry-run`. Configure a task dollar budget first.
The model-list check is read-only; it does not run an agent turn.

## CLI

```text
python -m eidos_agents doctor [--live-model-check]
python -m eidos_agents investigate "question" [--dry-run|--mock]
python -m eidos_agents run "objective" [--dry-run|--mock]
python -m eidos_agents mission missions/sentinel_precision_v1.yaml --dry-run
python -m eidos_agents task TASK_ID
python -m eidos_agents evidence TASK_ID
python -m eidos_agents costs TASK_ID
python -m eidos_agents approve TASK_ID ACTION
python -m eidos_agents resume TASK_ID
python -m eidos_agents hypotheses
python -m eidos_agents experiments
```

`--dry-run` and `--mock` do not modify source or perform consequential external actions. Mission v1
is research-only and refuses live execution until its experiment is separately authorized.

## Permission matrix

| Agent | Repo read | Source write | Artifact write | Tests | Web | Council | Merge/deploy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Director | yes | no | orchestrator only | no | no | gate only | no |
| Archivist | yes | no | no | no | explicit request | no | no |
| Curie / Sentry / Gauss | yes | no | no | no | Gauss explicit | no | no |
| Forge | yes | feature branch only | no | yes | no | no | no |
| Bench | yes | no | generated only | yes | no | no | no |
| Auditor | yes | no | no | reproduction only | no | no | no |
| Council | yes | no | no | no | no | n/a | no |

## Human approvals

Merge, deployment, destructive migration, external side effects, Council, and Auditor-BLOCK override
are human actions. Approval requests are persisted as `pending_approval.json`; `approve` records a
durable decision, and `resume` reconstructs state from artifacts rather than hidden conversation.
An Auditor BLOCK cannot be promoted automatically even if Director argues otherwise.

## Persistence and receipts

Runtime output defaults to ignored `artifacts/agent_lab/`:

```text
agent_lab.sqlite
tasks/TASK_ID/
  task.json
  transitions.jsonl
  events.jsonl
  evidence/
  hypotheses/
  experiments/
  implementation/
  benchmarks/
  audit/
  council/
  cost_receipt.json
  run_manifest.json
  final_decision.json
  final_decision.md
```

Experiment specifications and results are immutable records. Conclusions store evidence references,
not uncontrolled transcripts. The SQLite database is an index; JSON/Markdown remains reviewable.

## Tracing and sessions

Agents SDK tracing is enabled unless `OPENAI_AGENTS_DISABLE_TRACING=1`. Task metadata includes task ID,
repo commit and workflow state; SDK agent/function/guardrail spans are emitted automatically. Open the
[OpenAI traces dashboard](https://platform.openai.com/traces) for the API project owning the key. If
organization policy disables tracing, execution continues and the manifest records that limitation.
Director uses a per-task SQLite session. Specialists receive bounded work orders; there is no global
shared multi-agent chat.

## Troubleshooting

- `doctor` reports SDK import failure: use Python 3.12+, rebuild the venv, and rerun.
- configured model missing: set an explicit `EIDOS_AGENT_<ROLE>_FALLBACK_MODEL`; substitutions are
  recorded. There is no silent downgrade.
- budget block: inspect `cost_receipt.json`, increase the configured limit deliberately, then resume.
- dirty/protected branch: move work to a feature worktree; the tool will not reset, clean, or switch it.
- tracing absent: check organization data policy and `OPENAI_AGENTS_DISABLE_TRACING`.
