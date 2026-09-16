# Eidos Agent Lab — Live Routing Smoke v2

## STATUS

`FAIL`

The deterministic research-stage patch worked: Director could not promote an evidence-free answer to
`READY_FOR_HUMAN`. The paid smoke did not satisfy the research-organization contract because every specialist
tool call failed strict output-schema validation before the specialist agent started.

## PATCH COMMIT

`3391d4b964aa4bfa5f455011b4bb50e8bc37b3d5` — `agent-lab: enforce research evidence stage`

Patch validation: 61 Agent Lab tests passed, one opt-in paid test skipped, and scoped Ruff passed. Regression
coverage includes evidence bypass, missing EvidencePacket, valid MISSING evidence, missing required specialist,
prose-only Curie output, successful structured research path, and truthful transition semantics.

## TASK ID

`TASK-20260916T004858Z-7DE619`

- Start: 2026-09-15 20:48:56 EDT
- End: 2026-09-15 20:49:55 EDT
- Base commit: `3391d4b964aa4bfa5f455011b4bb50e8bc37b3d5`
- Budget: `$0.75`

## ACTUAL ROUTING

1. Director initial turn (`gpt-5.6-terra`, high).
2. `consult_archivist` attempted; rejected before Archivist started.
3. Director retry turn.
4. `consult_archivist` retried once; rejected before Archivist started.
5. Director continuation.
6. `consult_sentry` attempted; rejected before Sentry started.
7. Director continuation.
8. `consult_curie` attempted; rejected before Curie started.
9. Director final escalation response.
10. Deterministic stage gate rejected promotion.

No specialist model inference occurred. Council, Forge, Bench, Auditor and Gauss were not called.

## REQUIRED VS ACTUAL SPECIALISTS

| Specialist | Required | Tool attempted | Agent started | Completed |
|---|---:|---:|---:|---:|
| Archivist | yes | yes, twice | no | no |
| Sentry | yes | yes | no | no |
| Curie | yes | yes | no | no |

Persisted `SpecialistAccounting` correctly records all three as skipped/not invoked. Tool attempts do not count as
successful specialist invocation.

## REPOSITORY TOOLS USED

None. Archivist never started, so `repo_status`, `repo_search`, `repo_read_file`, and `repo_list_tree` were not
called. Their exposure remains covered by the SDK graph unit test, but live use remains unproven.

## EVIDENCE PACKET

No. No EvidencePacket was returned or persisted. The workflow gate explicitly named both the missing current-task
EvidencePacket and missing Archivist result.

## SENTRY RESULT

No. `consult_sentry` failed before model inference and returned no AgentResult.

## CURIE EXPERIMENT SPEC

No. `consult_curie` failed before model inference and returned no ExperimentSpec. Director's prose hypothesis about
a frozen labeled holdout is not treated as an ExperimentSpec.

## DIRECTOR FINAL DECISION

Director returned an escalation stating that no Sentinel behavior change was authorized and that the required
artifacts were absent. The deterministic gate then rejected `READY_FOR_HUMAN` with:

```text
persisted valid EvidencePacket for current task
persisted Archivist result
required specialist not completed: archivist
required specialist not completed: sentry
required specialist not completed: curie
persisted ExperimentSpec for current task
```

This is a valid safety outcome, not successful completion of the research task. No Sentinel performance claim was
established.

## WORKFLOW TRANSITIONS

Persisted:

```text
CREATED -> TRIAGED
TRIAGED -> EVIDENCE_GATHERING
```

Not persisted:

```text
EVIDENCE_GATHERING -> READY_FOR_HUMAN
```

The new stage gate therefore repaired the original unsafe promotion. A secondary defect prevented the intended
`EVIDENCE_GATHERING -> BLOCKED` transition: the truthful error text “required specialist not completed” was
misread by the transition-reason validator as a positive claim that specialist work had completed.

## COST BY AGENT/MODEL

All billed model usage was Director `gpt-5.6-terra`; failed specialist tools made no specialist API calls.

| Turn | Role | Input | Cached input | Output | Reasoning | Estimated cost |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Director initial | 1,568 | 0 | 465 | 70 | `$0.0087160` |
| 2 | Director after Archivist failure | 2,099 | 1,565 | 315 | 35 | `$0.0051610` |
| 3 | Director after retry failure | 2,480 | 2,096 | 338 | 27 | `$0.0052432` |
| 4 | Director after Sentry failure | 2,883 | 2,477 | 323 | 0 | `$0.0051834` |
| 5 | Director finalization | 3,271 | 2,880 | 1,162 | 259 | `$0.0153020` |
| — | Archivist | 0 | 0 | 0 | 0 | `$0.0000000` |
| — | Sentry | 0 | 0 | 0 | 0 | `$0.0000000` |
| — | Curie | 0 | 0 | 0 | 0 | `$0.0000000` |

Reasoning tokens are reported separately by the SDK but represented within output-token billing. Estimates use
the versioned `openai-standard-api-2026-09-15` registry and are not billing authority.

## TOTAL COST

- Input tokens: 12,301
- Cached input tokens: 9,018
- Output tokens: 2,603
- Reasoning tokens reported: 391
- Estimated total: `$0.0396056`
- Budget remaining: `$0.7103944`
- Budget check: pass (`$0.0396056 <= $0.75`)

## TRACE ID

Remote tracing was enabled, but the trace identifier was not persisted because the secondary BLOCKED-transition
exception occurred before `run_manifest.json` was written. Trace display is therefore `unavailable from local
receipts`; this is a telemetry limitation in addition to the smoke failure.

## SOURCE INTEGRITY

- Before tree: `de80186bb8d44baa21b9cb2726a37e448b6cf485`
- After tree: `de80186bb8d44baa21b9cb2726a37e448b6cf485`
- Core SHA-256 before/after:
  `d1d8db7b5f8e70c0ae6abc96575295620a27ea2943fa50209317d264da50ef50`
- Source modifications by agents: 0
- Council calls: 0
- Forge calls: 0
- Merge: not performed
- Deployment: not performed
- Eleven report/task receipts plus `drive_manifest.json` were mirrored to
  `G:\My Drive\Eidos_Brain_Proof_Phase\2026-09-15\eidos_agent_lab_live_smoke_v2_20260915\` with zero
  SHA-256 mismatches.

## PROBLEMS FOUND

1. All specialist output schemas fail Agents SDK strict-schema conversion with:
   `additionalProperties should not be set for object types`.
2. Local schema inspection identifies `AgentResult.deliverables: dict[str, Any]` as the open
   `additionalProperties: true` node. It invalidates AgentResult, ArchivistResult and CurieResult as strict model
   outputs.
3. Tool-start/tool-end telemetry alone is insufficient to prove specialist invocation; the new accounting
   correctly requires actual nested agent start/completion.
4. A negative phrase containing “not completed” triggers the transition-reason completion-claim detector.
5. The secondary exception prevented `run_manifest.json`, trace ID and an explicit BLOCKED transition from being
   persisted, though task, cost, events, accounting and final-decision receipts survived.

## READY FOR SENTINEL_PRECISION_V1: NO

Before another separately authorized live smoke, replace the open-ended `deliverables` output field with closed
typed specialist schemas (or another SDK-supported strict structure), add a graph test that runs strict-schema
conversion for every specialist output type, and make transition-reason truth checking distinguish negative
“not completed” text. Do not run the full Sentinel mission until Archivist, Sentry and Curie complete live and
the persisted evidence/experiment gates pass.

## Proof Logic + Meaning

### Goal reached

The unsafe evidence bypass is repaired and tested. The live multi-agent routing goal failed.

### Previous state

An evidence-free Director answer could advance to human review and falsely claim specialist research completed.

### Technical logic utilized

Task-persisted requirements are checked against immutable EvidencePacket, AgentResult, SpecialistAccounting,
ExperimentSpec and Hypothesis records. Model prose cannot satisfy these lookups.

### Math / scoring logic

```text
estimated_cost = regular_input * input_rate / 1e6
               + cached_input * cached_rate / 1e6
               + output * output_rate / 1e6
               = 0.0396056 USD
```

No Sentinel precision, recall, F1, FP/10k, coverage or calibration metric was measured.

### Philosophical meaning

Evidence before consensus means a fluent Director answer must fail when the research organization did not
actually produce evidence.

### Why this is better

The system now blocks rather than promotes an unsupported conclusion and names every missing receipt.

### How this moves Eidos closer to the north-star goal

It strengthens reproducibility, self-monitoring and human control around future Sentinel research without
changing the detector.

### Evidence

- `artifacts/agent_lab/tasks/TASK-20260916T004858Z-7DE619/task.json`
- `artifacts/agent_lab/tasks/TASK-20260916T004858Z-7DE619/specialist_accounting.json`
- `artifacts/agent_lab/tasks/TASK-20260916T004858Z-7DE619/live_events.json`
- `artifacts/agent_lab/tasks/TASK-20260916T004858Z-7DE619/cost_receipt.json`
- `artifacts/agent_lab/tasks/TASK-20260916T004858Z-7DE619/final_decision.json`
- `artifacts/agent_lab/tasks/TASK-20260916T004858Z-7DE619/transitions.jsonl`

### Remaining uncertainty

Live specialist execution, repository evidence retrieval, a structured ExperimentSpec, remote trace retrieval and
the present scientific state of Sentinel precision/calibration remain unproven.
