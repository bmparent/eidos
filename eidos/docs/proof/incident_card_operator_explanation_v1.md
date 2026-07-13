# Incident-card operator explanation v1

## Purpose

Every emitted Eidos incident card now carries one operator-facing explanation
contract with five sections: **what happened**, **why it matters**, **evidence**,
**uncertainty**, and **next action**. The contract makes anomaly observations
usable for triage without turning them into unsupported attack diagnoses.

The original incident-card fields remain unchanged for compatibility. The new
authoritative object is `operator_explanation`; `operator_narrative` is a
readable rendering of that object.

## Authority boundary

`operator_explanation` is composed deterministically from fields already in the
card. It may summarize at most three drivers, three raw references, and one
similar event, while retaining counts of omitted references. It separates
detection, hypothesis, and forecast confidence and states what each quantity
does and does not mean.

Severity controls triage urgency only. A card does not by itself establish
maliciousness, compromise, root cause, business impact, or authority to contain
an asset. Machine-learned action rankings are preserved as
`recommendation_only`; the deterministic next action is contextual
investigation and corroboration.

## Feature meaning

Registered names are humanized without changing their meaning. Anonymous model
indices and projected components are described as such and carry an explicit
limitation. The composer never invents a source-feature interpretation for a
latent or projected component.

## Optional AI renderer

An optional callable can reword the five section summaries. It receives a deep
copy of the structured facts, their SHA-256 digest, and explicit constraints.
It must return exactly five non-empty strings. Unsupported certainty or response
claims, schema drift, excess length, exceptions, and malformed output trigger a
deterministic fallback. Renderer output never modifies the structured facts and
is explicitly marked non-authoritative with the digest of the facts it renders.
The renderer is instructed to reword rather than add claims; because wording
validation cannot prove factual grounding, downstream decisions must use the
structured explanation, not rendered prose, as their authority.

## Integration points

- Confirmed Sentinel events are enriched when converted to incident cards.
- Engine cards are enriched before JSONL emission.
- The labeled proof harness enriches both confirmed and legacy engine cards
  before writing proof artifacts.

This is an explanation-layer change. It does not alter detector thresholds,
event confirmation, severity, merging, calibration, metrics, or raw evidence.
