# Month 2 Incident Card Quality Starter - 2026-07-06

This is the first Month 2 proof receipt created after the Month 1 final package. It is intentionally a structural and interpretability audit only. It does not change incident-card generation, Sentinel thresholds, reservoir dynamics, RLS behavior, hippocampus behavior, or compression behavior.

## Scope

Audited incident-card JSON files from the two July 6 next-harder CPU guardrail runs:

- `artifacts/proof_runs/2026-07-06/next_harder_guardrails/natural_attack_windows_3_cpu/off/incident_cards`
- `artifacts/proof_runs/2026-07-06/next_harder_guardrails/normal_only_2k_cpu/off/incident_cards`

## Structural Counts

| Run | Total cards | Engine cards | Confirmed-event cards | Engine cards missing required fields | Confirmed cards missing required fields |
| --- | ---: | ---: | ---: | ---: | ---: |
| `natural_attack_windows_3_cpu` | 34 | 21 | 13 | 0 | 0 |
| `normal_only_2k_cpu` | 78 | 65 | 13 | 0 | 0 |

Required engine-card fields checked:

```text
incident_id, severity, confidence, domain, evidence, actions, hypotheses
```

Required confirmed-event fields checked:

```text
incident_id, event_id, start_frame, end_frame, severity, confidence, why_flagged, raw_evidence_refs
```

## Quality Signals

| Run | Severity distribution | Engine-card attack flags | Why-flagged coverage |
| --- | --- | --- | --- |
| `natural_attack_windows_3_cpu` | `RED=30`, `AMBER=4` | `True=1`, `False=20`, `unknown=13` | `13` confirmed cards explain accumulated evidence and geometry/novelty; `8` also cite persistence |
| `normal_only_2k_cpu` | `RED=50`, `AMBER=28` | `False=65`, `unknown=13` | `13` confirmed cards explain accumulated evidence and geometry/novelty; `9` also cite persistence |

## Interpretation

The incident-card files are structurally complete for the required fields in this starter audit. The more important Month 2 finding is qualitative: the normal-only run still produces many engine cards on benign data, even though calibrated proof views suppress the operator-facing false-positive count. That means the next proof layer should not simply count cards; it should score whether each card is useful, concise, label-aware, and safe to show an operator.

## Proof Logic + Meaning

Goal reached: a Month 2 incident-card quality track has started. Gate status is `partial` because this proves structural completeness, not human-quality explanation.

Previous state: incident-card artifacts existed, but they were only cited as evidence, not audited for structure or explanatory quality.

Technical logic utilized: JSON field validation, card-type counts, severity distribution, attack-flag distribution, and `why_flagged` explanation coverage.

Math / scoring logic:

```text
required_field_pass_rate = cards_with_required_fields / total_cards
natural_required_field_pass_rate = 34 / 34
normal_required_field_pass_rate = 78 / 78
```

Philosophical meaning: explanation before automation. Eidos should not only raise or suppress events; it should produce receipts that help a human understand what happened and why.

Why this is better: Month 2 now has a concrete starting point for incident-card quality instead of treating card existence as enough.

How this moves Eidos closer to the north-star goal: it strengthens the `emits human-readable incident receipts` part of the claim while preserving the distinction between structural existence and human usefulness.

Evidence:

- `artifacts/proof_runs/2026-07-06/next_harder_guardrails/natural_attack_windows_3_cpu/off/incident_cards`
- `artifacts/proof_runs/2026-07-06/next_harder_guardrails/normal_only_2k_cpu/off/incident_cards`
- `docs/proof_runs/2026-07-06/month1_final_proof_package.md`

Remaining uncertainty:

- This does not prove the cards are high quality for human operators.
- This does not tune card generation or reduce benign engine-card volume.
- This does not replace a human review rubric.
- This does not prove production readiness.
