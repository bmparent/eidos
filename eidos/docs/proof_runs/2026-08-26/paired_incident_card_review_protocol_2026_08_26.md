# Eidos Paired Incident-Card Review Protocol — 2026-08-26

## Registration status

Locked before fetching the 153 frozen raw card payloads or generating their post-change operator views. Protocol ID: `PICQR-2026-08-26-v1`.

This is a paired, exhaustive, single-reviewer operator-style proxy evaluation. The reviewer is ChatGPT acting from a security-operator perspective. It is not independent human validation and cannot establish inter-rater reliability.

## Goal

Test whether the deterministic operator-explanation layer merged at `f676fe2342b98886bc04cd8f4b0e943fce77ec9a` materially improves the operator-facing quality of the same 153 frozen cards that failed HICQR on 2026-07-13, while holding detector outputs, identities, ordering inputs, and raw evidence fixed.

## Frozen population

- Legacy repository baseline: `2da0e78a460ade628ecdaf7159ecdad346645cd7`.
- Explanation implementation: `f676fe2342b98886bc04cd8f4b0e943fce77ec9a`.
- Natural receipt: `natural_attack_windows_3_cpu_spread/off`, 75 cards.
- Normal-control receipt: `normal_only_2k_cpu/off`, 78 cards.
- Population: all 153 cards; no sampling or early stopping.
- Legacy scores and blind mapping: HICQR protocol `HICQR-2026-07-13-v1`.

Every raw payload must match its locked SHA-256 value before enrichment. Any missing or mismatched payload forces a `PARTIAL_INVALID_INPUT` verdict.

## Intervention boundary

For each card, apply only `enrich_incident_card` from the explanation implementation. Do not alter detector output, severity, confidence, hypotheses, evidence, raw references, recommended actions, event boundaries, or truth labels.

The enriched operator surface is the intended five-field presentation:

1. what happened
2. why it matters
3. evidence
4. uncertainty
5. next action

The structured `operator_explanation` is authoritative. The deterministic `operator_narrative` is included as its readable rendering. Compatibility fields remain available for audit but are excluded from the primary operator surface so duplicated back-end JSON does not masquerade as interface content.

## Pairing and blinding

- Preserve the original card identity internally but never expose receipt, filename, truth label, or legacy/new label on the scoring surface.
- Create two variants per card: legacy (`L`) and enriched (`E`).
- Assign 306 presentation IDs using a deterministic shuffle keyed by:

```text
SHA256("eidos-paired-review-2026-08-26|2da0e78a460ade628ecdaf7159ecdad346645cd7|f676fe2342b98886bc04cd8f4b0e943fce77ec9a")
```

- Score presentation IDs in shuffled order. Restore variant and receipt identities only after scores are locked.
- Exact-content duplicates remain in the operational population. Identical operator surfaces may reuse the same anchored score, but every card remains a separate paired row and duplicate counts are reported.

## Locked rubric

Use the same six 1–5 anchored dimensions as HICQR:

- Usefulness, 20%: 1 unusable; 3 partially useful; 5 quickly clarifies what happened and why it matters.
- Evidence traceability, 20%: 1 unsupported; 3 partial linkage; 5 concrete evidence can be followed to the conclusion.
- Explanation specificity, 15%: 1 generic; 3 partly incident-specific; 5 specific without overreach.
- Actionability, 15%: 1 no usable or unsafe next step; 3 plausible but underspecified; 5 bounded, relevant, executable next step.
- Operator safety/calibration, 20%: 1 dangerous or materially overconfident; 3 caveated but imperfect; 5 clearly separates observation, inference, and uncertainty.
- Concision/scannability, 10%: 1 too sparse or bloated; 3 usable with friction; 5 compact and easy to scan.

```text
composite = 0.20U + 0.20E + 0.15S + 0.15A + 0.20C + 0.10B
paired_delta = enriched_composite - legacy_composite
```

Preserve a one-sentence rationale for every dimension scored 2 or lower and for every enriched dimension that is lower than its paired legacy score.

## Acceptance gates

An `OPERATOR_PROXY_IMPROVEMENT_SUPPORTED` verdict requires all of the following:

1. All 153 raw payloads verify and all 153 pairs are scoreable.
2. Enriched median composite is at least 4.0 overall and within both receipts.
3. At least 80% of enriched cards have composite at least 4.0 overall and within both receipts.
4. No enriched card has safety/calibration 1; at most 5% have safety/calibration 2 or lower overall and within each receipt.
5. Enriched median actionability is at least 3 and evidence traceability at least 4 overall and within each receipt.
6. Paired median composite improvement is at least +0.75 overall and within both receipts.
7. At least 80% of pairs improve and no more than 5% regress overall and within both receipts.
8. At most 5% of enriched normal-control surfaces state attack or compromise as fact without direct card-level evidence.

If paired improvement gates pass but the enriched cards miss one or more absolute HICQR quality gates, the verdict is `IMPROVED_BUT_GATE_NOT_MET`. A safety gate failure yields `FAILED_SAFETY`. No material paired improvement yields `FAILED_NO_MATERIAL_IMPROVEMENT`.

## Interpretation boundary

This experiment can show whether the merged deterministic explanation changes this single reviewer’s quality judgment on the fixed population. It cannot establish detector correctness, deployment performance, independent human usefulness, inter-rater agreement, comprehension time, downstream decision quality, or generalization to new cards.

## Required artifacts

- this locked protocol;
- raw identity and SHA verification receipt;
- deterministic presentation map;
- all 306 dimension scores and rationales;
- 153 paired deltas;
- aggregate results by receipt and card type;
- negative examples and any regressions;
- a Proof Logic + Meaning report with limits and one next experiment.
