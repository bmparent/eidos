# Scientific Method and Claim Discipline

## Claim vocabulary

- `HYPOTHESIS`: proposed explanation, not adequately tested.
- `SUPPORTED`: evidence favors it under stated conditions; alternatives may remain.
- `INCONCLUSIVE`: present evidence does not decide it.
- `REFUTED`: evidence materially contradicts it.
- `KNOWN`: directly established under declared conditions by strong reproducible evidence.

“Proved” and “confirmed” are not casual substitutes. A better metric does not establish the mechanism
that allegedly caused it. The schema rejects `KNOWN` without multiple evidence references and rejects
unresolved counterevidence for a KNOWN hypothesis.

## Experiment discipline

Curie specifies before execution: hypothesis, competing explanation, independent/dependent variables,
controls, negative controls, ablations, dataset and sample policy, order assumptions, seeds, baseline,
success/failure/ambiguous observations, and required receipts. Evaluation labels remain isolated until
prediction freeze when the protocol calls for it. Failed science is not retried unchanged.

Bench measures the specification and retains raw output. Auditor receives original specs, raw receipts,
diff, tests, config, data construction and evidence. Forge output is never sufficient audit evidence.

For Sentinel, raw, merged, deduplicated and calibrated views remain visible together. Always report
recall/coverage regressions beside precision. Unknown metrics are `null`/`unknown`, never zero.

## Proof Logic + Meaning

### Goal reached

The v1 lab establishes a reproducible organizational gate: a builder cannot certify itself and a code
change cannot skip measurement and independent audit.

### Previous state

Eidos had strong proof artifacts but no unified typed agent lifecycle, persistent approval state,
hypothesis ledger, model/cost registry, or deterministic separation-of-duties workflow.

### Technical logic utilized

Pydantic contracts constrain boundaries; SQLite and JSON persist state; explicit capability sets limit
authority; a transition graph rejects skipped stages; cost and Council gates fail closed; receipts tie
decisions to source evidence.

### Math / scoring logic

Detection work retains:

```text
precision = TP / (TP + FP)
recall = detected_attack_windows / total_attack_windows
F1 = 2 * precision * recall / (precision + recall)
FP_per_10k = false_positive_events / benign_frames * 10000
```

Cost estimates retain cached tokens rather than hiding them:

```text
estimated_cost = regular_input * input_price
               + cached_input * cached_input_price
               + output * output_price
               + reported_reasoning * reasoning_price
```

No dollar value is produced when reviewed pricing is absent.

### Philosophical meaning

Independent audit is restraint before belief. Reproducibility is truth that can be revisited.
Institutional memory is memory before ambition.

### Why this is better

Authority, evidence status, cost and workflow state are now explicit and testable instead of embedded
only in prompts or conversation. Contradiction, failure and missing evidence remain visible.

### How this moves Eidos closer to the north-star goal

It strengthens the reproducible and self-monitoring parts of the claim by making research decisions,
experiments, implementation, measurement and audit replayable. It does not itself prove better
compression, anomaly preservation or detection.

### Evidence

See the agent-lab tests, mocked task bundles, `missions/sentinel_precision_v1.yaml`, and BUILD_REPORT.

### Remaining uncertainty

No paid live multi-agent research mission was run. The Sentinel mission produced only a research plan;
its hypothesis and current performance remain untested. SandboxAgent transport is not enabled in v1.
