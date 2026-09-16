# Eidos Agent Lab — First Live Multi-Agent Smoke

## Status

`FAIL`

The paid API call and receipt pipeline completed, but the intended multi-agent research path did not.
Director invoked no specialist, retrieved no repository evidence, and answered the question directly as
`INCONCLUSIVE`. This proves live Director execution, tracing, token/cost accounting, budget control and
persistence, but it does not satisfy the required evidence-retrieval, specialist-routing or research-synthesis
acceptance gates.

## Pricing patch

The registry now contains reviewed standard API text-token prices effective 2026-09-15 for every configured
production model. It records the version, effective date, official OpenAI source URLs and exclusions for tool,
regional/data-residency, Fast-mode, long-context and cache-write charges. Estimates are explicitly non-billing.

Focused tests cover model completeness, ordinary/cached/output arithmetic, unknown-model behavior, price
version/effective date and budget enforcement. The full Agent Lab suite and Ruff passed before the live run.

Pricing commit: `da80587` (`agent-lab: populate reviewed model pricing`).

Official sources reviewed:

- <https://developers.openai.com/api/docs/models/compare>
- <https://developers.openai.com/api/docs/models/gpt-5.4-mini>
- <https://developers.openai.com/api/docs/models/gpt-5.3-codex>

## Live task

- Task ID: `TASK-20260916T002609Z-C3E108`
- Local start: 2026-09-15 20:26:06 EDT
- Local end: 2026-09-15 20:26:47 EDT
- Git commit: `b1efa6acef57630f475aac369d33c47a99e73fec`
- Exact prompt: “Determine the current evidence-backed state of Sentinel precision/calibration in this
  repository and identify the single highest-value missing experiment needed before changing Sentinel
  behavior. Retrieve the newest relevant evidence rather than assuming historical CICIDS metrics are current.
  This is research-only. Do not modify source code.”

One local pre-model launch at commit `197e874` failed in the telemetry hook because `Eidos Director` was not
normalized to the `director` configuration key. The exception occurred in `on_llm_start` before an API request,
so it spent zero tokens. Commit `b1efa6a` fixed that identity mapping; the one allowed technical retry became
the paid attempt reported here.

## Actual routing

| Order | Agent | Model | Reasoning | Why invoked |
|---:|---|---|---|---|
| 1 | Director | `gpt-5.6-terra` | high | Required live manager entry point |

Expected Archivist followed by Sentry and/or Curie did not occur. There were no unexpected expensive agent
calls, but the absence of the expected bounded specialists is the principal failure.

## Evidence behavior

Archivist was not invoked. No repository tools were called, no current evidence packet was produced, and no
newest Sentinel/CICIDS/Precision Ledger/calibration receipt was retrieved. Director accurately labeled the
result `INCONCLUSIVE`, but its statement that repository evidence was unavailable was caused by its routing
choice rather than an established absence of evidence.

## Research result

Director recommended a commit-pinned calibration/precision evaluation using a leakage-resistant temporal
holdout, frozen threshold selection, confusion metrics, PR-AUC, Brier score, expected calibration error,
reliability data and confidence intervals. That is a plausible hypothesis for a useful next experiment, but it
is not an `ExperimentSpec`, was not grounded in retrieved project evidence, and is not scientifically validated.

No hypothesis ledger record, evidence packet or specialist-generated ExperimentSpec was created.

## Cost

| Field | Value |
|---|---:|
| Input tokens | 1,513 |
| Cached input tokens | 0 |
| Output tokens | 1,209 |
| Reasoning tokens reported | 184 |
| Estimated `gpt-5.6-terra` cost | $0.017534 |
| Estimated total cost | $0.017534 |
| Authorized budget | $1.00 |
| Estimated budget remaining | $0.982466 |

The estimate uses `openai-standard-api-2026-09-15`. Reasoning tokens are reported separately by the SDK but
are already represented within output-token billing; the registry has no separate reasoning-token price.
Tool-specific charges were zero because no tools ran. The estimate is not billing authority.

## Guardrails

- Council calls: 0 — pass.
- Forge implementation calls: 0 — pass.
- Source modifications by agents: 0 — pass.
- Estimated cost at or below $1.00: pass.
- Secrets printed: none observed — pass.
- Main modified: no — pass.
- Merge: not performed — pass.
- Deployment: not performed — pass.
- Specialist calls: 0 of maximum 4 — below limit, but functionally insufficient.

## Tracing

Remote tracing was enabled and the SDK returned trace ID
`trace_82fdc17964be430eaae51a0b29a8eb35`. Inspect it in the OpenAI traces dashboard at
<https://platform.openai.com/traces>. No raw sensitive trace payload is committed.

## Repository integrity

- Before tree: `3e74d31356cf2f301d5422ffbb750c7aaa301ab9`
- After tree: `3e74d31356cf2f301d5422ffbb750c7aaa301ab9`
- Core file SHA-256 before/after:
  `d1d8db7b5f8e70c0ae6abc96575295620a27ea2943fa50209317d264da50ef50`
- Tracked status before: clean.
- Tracked status after: clean.
- Runtime task artifacts remained ignored under `artifacts/agent_lab/`.
- Task and cost receipts reloaded successfully through the existing CLI after process exit.
- Eleven closeout/task files were mirrored to
  `G:\My Drive\Eidos_Brain_Proof_Phase\2026-09-15\eidos_agent_lab_live_smoke_20260915\` with zero
  SHA-256 mismatches; the ignored bundle contains `drive_manifest.json`.

Persisted transitions were `CREATED → TRIAGED → EVIDENCE_GATHERING → READY_FOR_HUMAN`. The transition reason
“live bounded specialist research completed” is inaccurate because no specialist ran; this is a telemetry/state
semantics defect to correct before the next live mission.

## Problems found

1. Director can bypass evidence retrieval in a research task even when the objective explicitly requires it.
2. The research-only state machine marked evidence gathering complete without an EvidencePacket or Archivist call.
3. Director claimed execution access was unavailable although bounded Archivist repository tools were present.
4. No specialist output, evidence packet, hypothesis ledger item or ExperimentSpec was produced.
5. The initial pre-model telemetry identity bug required the single allowed technical retry; it is fixed and tested.

## Recommended next step

Not ready for the live `sentinel_precision_v1` research mission. First add a deterministic research-stage gate:
when current project evidence is required, Director must obtain a persisted Archivist result/EvidencePacket
before `EVIDENCE_GATHERING` can complete or `READY_FOR_HUMAN` can be reached. Add a mocked regression proving
that a direct evidence-free Director answer is rejected. Then run a separately authorized small live retry before
the full Sentinel mission.

## Proof Logic + Meaning

### Goal reached

Pricing, paid Director execution, trace creation, token/cost receipts, the $1 budget and persistence passed.
The multi-agent evidence-research gate failed.

### Previous state

Pricing was deliberately unknown and live orchestration was untested. The live persistence path also lacked
deterministic cost and per-agent telemetry wiring.

### Technical logic utilized

Versioned prices convert observed SDK token classes into an estimate. Live hooks authorize each model call,
attribute usage by agent/model, journal tool calls, and persist the manifest. Research-only tool visibility blocks
Forge, Bench, Auditor and Council. Before/after Git trees and core hashes detect source mutation.

### Math / scoring logic

```text
estimated_cost = regular_input_tokens * input_rate / 1e6
               + cached_input_tokens * cached_rate / 1e6
               + output_tokens * output_rate / 1e6

estimated_cost = 1513 * 2.00 / 1e6 + 1209 * 12.00 / 1e6
               = 0.017534 USD
```

No Sentinel precision, recall, F1, FP/10k, coverage or latency was measured.

### Philosophical meaning

Cost receipts are honesty about resources. Calling this smoke a failure despite a valid model response is honesty
about evidence: fluent synthesis is not a substitute for retrieval.

### Why this is better

The system now exposes the exact cost, trace, route and missing specialist calls instead of describing the run as
successful merely because a model returned structured JSON.

### How this moves Eidos closer to the north-star goal

It strengthens reproducibility and self-monitoring around future experiments, while identifying the missing
evidence gate that must be fixed before agent conclusions can credibly support Sentinel behavior changes.

### Evidence

- `artifacts/agent_lab/tasks/TASK-20260916T002609Z-C3E108/run_manifest.json`
- `artifacts/agent_lab/tasks/TASK-20260916T002609Z-C3E108/cost_receipt.json`
- `artifacts/agent_lab/tasks/TASK-20260916T002609Z-C3E108/live_events.json`
- `artifacts/agent_lab/tasks/TASK-20260916T002609Z-C3E108/transitions.jsonl`
- `tests/agent_lab/`

### Remaining uncertainty

Live specialist routing, evidence retrieval, structured ExperimentSpec creation and a scientifically meaningful
Sentinel result remain unproven. Remote trace contents were not independently inspected in this closeout.
