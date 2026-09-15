# Codex Journal — 2026-09-15

## What happened today

Built the Eidos Research Council / Multi-Agent Lab v1 as a separate root Python package. The active
`main` checkout was dirty and 70 commits behind, so work moved to an isolated worktree created from
current `origin/main` at `a2da5fd`. No core Eidos model or Sentinel behavior changed.

## What was accomplished

- Added strict task, evidence, hypothesis, experiment, implementation, benchmark, audit, Council,
  cost, transition, manifest and final-decision schemas.
- Added Director plus eight structured Agents SDK specialists using manager-style `Agent.as_tool()`.
- Added model and pricing registries, budgets, permissions, guardrails, bounded concurrency, safe repo
  tools, SQLite/JSON persistence, approvals/resume, CLI, documentation and a Sentinel precision mission.
- Exercised no-cost mocked full and research-only workflows.

## Tests and commands run

- `.\.venv312\Scripts\python.exe -m pytest tests/agent_lab -q` — 46 passed, 1 opt-in live test skipped.
- `.\.venv\Scripts\uvx.exe ruff check eidos_agents tests/agent_lab` — passed.
- From `eidos/`: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest -q` — 140 passed, 1 skipped.
- `python -m pytest eidos/repo/tests -q` — 78 passed, 4 skipped.
- From `services/sentinel-runner/`: `python -m pytest -q` — passed (26 collected tests).
- Root RNG-null focused suite — 9 passed, 5 failed from the pre-existing top-level
  `operator_explanation` import path; no agent-lab code is in those traces.
- `.\.venv312\Scripts\python.exe -m eidos_agents doctor --live-model-check` — passed; SDK graph and
  configured model availability verified without an agent inference call.
- Mocked workflow and `missions/sentinel_precision_v1.yaml --dry-run` — passed.

## Problems encountered

OpenAI Agents SDK 0.22.2 failed to import under the host's Python 3.11.0 due to a runtime typing error
inside `agents.tool`. It imported and built the complete graph under isolated Python 3.12.14, so the
agent-lab package now honestly requires Python 3.12+. The existing root RNG-null command exposes an
unrelated repo-root import defect: `eidos/sentinel/event_merge.py` imports `operator_explanation` as a
top-level module although the file is under `eidos/`.

## What changed

Only the agent-lab package, its tests/config/mission/docs, root packaging metadata, `.env.example`,
`.gitignore`, and a README link changed.

## What did not change

Reservoir dynamics, RLS, surprise scoring, Sentinel labels/thresholds, anomaly policy, compression,
familiarity, incident logic, forecasting and hosted services were untouched.

## Proof Logic + Meaning

### Goal reached

The v1 architecture acceptance gate is passed locally: typed specialists, deterministic gates,
independent audit, persistent approvals, mocked end-to-end execution, doctor and repository regression
suites are evidenced. Live paid orchestration was not required and was not run.

### Previous state

Proof tooling existed, but there was no unified multi-agent task state machine, hypothesis registry,
role capability enforcement, cost receipt, bounded specialist graph or durable HITL resume layer.

### Technical logic utilized

Pydantic rejects malformed contracts; SQLite and JSON preserve transitions; capability sets and path
guards constrain authority; a directed state graph rejects stage skipping; SDK specialists are tools
owned by Director; Bench and Auditor remain distinct from Forge; Council and consequential actions
fail closed behind approval.

### Math / scoring logic

Sentinel mission specs preserve `precision = TP/(TP+FP)`, `recall = detected windows/total windows`,
`F1 = 2PR/(P+R)`, and `FP_per_10k = FP/benign_frames*10000`. Cost is a token-weighted estimate using
regular, cached, output and reported reasoning tokens. With unreviewed prices, cost remains unknown.

### Philosophical meaning

Independent audit is restraint before belief. Reproducibility is truth that can be revisited.

### Why this is better

Workflow, authority, uncertainty and cost are machine-testable and reviewable rather than prompt-only.
Agent consensus cannot automatically become evidence or bypass a BLOCK.

### How this moves Eidos closer to the north-star goal

It strengthens reproducibility, self-monitoring and human-readable receipts around future Eidos
learning, compression and anomaly experiments. It does not itself establish improved detection or
compression.

### Evidence

`tests/agent_lab/`, ignored `artifacts/agent_lab/` task bundles, `docs/agent_lab/BUILD_REPORT.md`, and
the passing existing Eidos/package/Sentinel-runner suites.

### Remaining uncertainty

No paid live agent mission, remote trace, SandboxAgent transport, scientific Sentinel execution or
current pricing estimate was tested. The root RNG-null import defect remains outside this patch.

## Artifacts generated

Repo-local runtime receipts are under `artifacts/agent_lab/`; build-validation receipts are under
`artifacts/agent_lab/builds/eidos_agent_lab_v1_20260915/`.

## Google Drive archive status

The build-validation receipt folder is mirrored to
`G:\My Drive\Eidos_Brain_Proof_Phase\2026-09-15\eidos_agent_lab_v1_20260915\`. The final manifest
records copied files and hashes.

## Thoughts on improvement

After Brent review, the next safe step is an explicitly budgeted, low-cost live Archivist smoke and
trace inspection, followed by review of the Sentinel experiment specification before any core change.

## Where to improve next

Fix the unrelated root RNG-null import path in a separate PR, then perform a human-approved live agent
smoke on this branch.

## Anything that stands out

The model availability endpoint confirmed all requested identifiers for the configured API project.
No model inference was purchased. Python runtime compatibility was a larger risk than SDK surface
compatibility; the adapter kept that failure explicit.

## End-of-task summary

1. Files changed: agent lab source/tests/config/mission/docs and narrow root metadata.
2. Whether core behavior changed: no.
3. Tests added or skipped: 47 collected; 46 passed; opt-in paid smoke skipped.
4. Repo-root commands run: recorded above.
5. Artifacts generated: task, evidence, hypothesis, experiment, benchmark, audit, cost and decision receipts.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: mirrored; final manifest contains hashes.
9. Known limitations: no paid live agent turn; local Forge fallback; unreviewed price registry.
10. Follow-up tasks not implemented: unrelated RNG import fix, live smoke, Sandbox transport.
11. Proof Logic + Meaning written: yes.
12. Math/logic explanation included: yes.
13. Philosophical meaning included: yes.
14. Why this is better than previous state: explicit above.
15. How this moves Eidos closer to the ultimate goal: reproducibility/self-monitoring receipts.
16. Evidence files cited: tests, build report, task bundles and regression suites.
17. Remaining uncertainty / unproven claims: Sentinel performance and scientific value remain unproven.
