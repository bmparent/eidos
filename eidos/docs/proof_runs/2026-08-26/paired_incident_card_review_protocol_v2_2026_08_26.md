# Eidos Paired Incident-Card Review Protocol v2 — 2026-08-26

## Registration status

Locked after the engineering replay and before any independent operator receives a scoring surface. Protocol ID: `PICQR-2026-08-26-v2`.

The implementation team and the ChatGPT reviewer that produced the engineering projection are ineligible to provide the primary score. The primary reviewer must be an actual security operator who has not seen the legacy scores, projected scores, source receipts, truth labels, or variant mapping. A second independent operator is strongly preferred and must be reported separately if available.

## Goal

Test whether the five-field explanation plus strict-window evidence linkage at repository commit `4a639cd693701fb764fe30ba672d4811bdbf5a75` improves operator-facing quality on the same 153 frozen cards, without changing detector outputs, event boundaries, severity, confidence, calibration, or source observations.

## Frozen population and input identity

- Legacy detector/card commit: `2da0e78a460ade628ecdaf7159ecdad346645cd7`.
- Explanation baseline: `f676fe2342b98886bc04cd8f4b0e943fce77ec9a`.
- Evidence-link implementation: `4a639cd693701fb764fe30ba672d4811bdbf5a75`.
- Natural receipt: 75 cards.
- Normal-control receipt: 78 cards.
- Total: 153 cards, comprising 119 engine cards and 34 confirmed-event cards.
- Input manifest: `frozen_identity_manifest.csv`.
- Locked manifest SHA-256: `a943ea3c3a9f03abb122d7a90b83d0864da990b660c38b2773fba7404d5fbcb5`.

Before rendering, all 153 downloaded bytes must match `downloaded_byte_sha256` in the manifest. The unreproduced 2026-07-13 raw/sanitized hash columns are retained as audit metadata but are not used as v2 input validators. Any current-byte mismatch, duplicate Drive ID, missing file, or title mismatch forces `PARTIAL_INVALID_INPUT`.

## Intervention boundary

Create two surfaces per frozen card:

1. `LEGACY`: the original card, rendered with the v2 sanitizer and layout only.
2. `PATCHED`: the same card after the f676 five-field explanation and, for confirmed events, strict-window engine-evidence linkage from commit `4a639cd`.

The v2 sanitizer and layout must be identical across variants. They may hide receipt, filename, source run, dataset truth, attack/benign labels, and variant identity. They may not add, remove, summarize, reinterpret, or reorder substantive card evidence differently between variants. Compatibility fields remain available in the sealed audit artifact but are excluded from the primary operator display.

Strict-window linkage may use only engine cards whose step is inside the confirmed event's inclusive frame window. Every linked source card ID and step must resolve to the frozen manifest. No nearest-window or semantic matching is allowed in this experiment.

## Blinding and presentation order

- Render 306 surfaces and assign presentation IDs `V2P001`–`V2P306`.
- Shuffle with SHA-256 seed text:

```text
eidos-paired-review-v2-2026-08-26|a943ea3c3a9f03abb122d7a90b83d0864da990b660c38b2773fba7404d5fbcb5|4a639cd693701fb764fe30ba672d4811bdbf5a75
```

- Do not place paired variants adjacent deliberately.
- Hide receipt, card type, truth label, legacy/patched identity, source filename, code commit, and prior score.
- Lock all 306 scores before unblinding.
- Exact-content duplicates remain separate operational rows; duplicate hashes are reported.

## Locked rubric

Score each dimension 1–5 using the original HICQR anchors:

- Usefulness, 20%: 1 unusable; 3 partially useful; 5 quickly clarifies what happened and why it matters.
- Evidence traceability, 20%: 1 unsupported; 3 partial linkage; 5 concrete evidence can be followed to the conclusion.
- Explanation specificity, 15%: 1 generic; 3 partly incident-specific; 5 specific without overreach.
- Actionability, 15%: 1 no usable or unsafe next step; 3 plausible but underspecified; 5 bounded, relevant, executable next step.
- Operator safety/calibration, 20%: 1 dangerous or materially overconfident; 3 caveated but imperfect; 5 separates observation, inference, and uncertainty.
- Concision/scannability, 10%: 1 too sparse or bloated; 3 usable with friction; 5 compact and easy to scan.

```text
composite = 0.20U + 0.20E + 0.15S + 0.15A + 0.20C + 0.10B
paired_delta = patched_composite - legacy_composite
```

Preserve one sentence for every dimension score of 2 or lower and every patched dimension that is lower than legacy. No early stopping and no scorebook changes after the first surface is shown.

## Acceptance gates

A bounded `SINGLE_OPERATOR_IMPROVEMENT_SUPPORTED` verdict requires every condition below for the primary operator:

1. All 153 inputs verify and all 153 pairs are scoreable.
2. Patched median composite is at least 4.0 overall and within both receipts.
3. At least 80% of patched cards score at least 4.0 overall and within both receipts.
4. No patched safety score is 1; at most 5% are 2 or lower overall and within each receipt.
5. Patched median actionability is at least 3 and median evidence traceability at least 4 overall and within each receipt.
6. Paired median composite improvement is at least +0.75 overall and within both receipts.
7. At least 80% of pairs improve and no more than 5% regress overall and within both receipts.
8. At most 5% of normal-control patched surfaces state attack or compromise as fact without direct card-level evidence.
9. Every displayed linked driver resolves to a strict-window source card, and no unlinked confirmed card implies ranked drivers are present.

If paired improvement passes but an absolute quality gate fails, return `IMPROVED_BUT_GATE_NOT_MET`. A safety or evidence-integrity failure returns `FAILED_SAFETY_OR_INTEGRITY`. No material paired improvement returns `FAILED_NO_MATERIAL_IMPROVEMENT`.

With two independent operators, report each verdict separately plus exact agreement, weighted Cohen's kappa per dimension, and paired-delta agreement. Do not collapse disagreement into a consensus score after unblinding.

## Interpretation boundary

This study can support only a bounded human-factor claim for this frozen CICIDS/WebAttacks card population and reviewer. It cannot establish detector correctness, attack truth, production readiness, decision latency, real-world outcomes, multi-operator consensus, compression advantage, or cross-domain generalization.

## Required artifacts

- this locked protocol and its SHA-256;
- the locked input manifest and byte verification receipt;
- sanitizer/renderer source and exact commit;
- 306 blinded surface hashes and the sealed variant map;
- all raw dimension scores and rationales;
- 153 paired deltas and subgroup results for receipt, card type, linked confirmed events, and unlinked confirmed events;
- acceptance checks, failures, negative examples, and reviewer limitations;
- a Proof Logic + Meaning report with exactly one next experiment.
