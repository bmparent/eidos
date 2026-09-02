# Eidos Revival Proof Report — 2026-08-26

## Executive verdict

Eidos is not at square one. Current `main` is an AMBER, proof-first research system with a 72.5/100 Month 1 package, 129 previously reported passing tests, a larger 170,366-row labeled guardrail dataset, and a deterministic five-field operator explanation at `f676fe2342b98886bc04cd8f4b0e943fce77ec9a`. The engine is still not validated for production, broad generalization, or human-operator usefulness.

Today's paired protocol does **not** produce a promotable proof verdict: `PARTIAL_INVALID_INPUT`. All 153 Drive files reconcile by ID, title, size, and population, but 0/153 locked `raw_sha256` values match the downloaded bytes or standard canonical JSON, and the old raw/sanitized hash-generation procedure is absent. Changing that gate after seeing the mismatch would manufacture certainty.

The engineering replay still isolates a high-leverage result. The merged f676 explanation is projected to lift composite pass rate from the legacy 0/153 to 119/153 (77.8%), narrowly missing the locked 80% gate because confirmed-event cards preserve no ranked drivers. A strict-window evidence-link patch at `4a639cd693701fb764fe30ba672d4811bdbf5a75` connects 17/34 confirmed events to already-existing engine evidence without changing event boundaries, severity, confidence, detector output, or calibration. Under the same transparent scorebook, the projected pass rate becomes 136/153 (88.9%) and every numerical gate passes. This is an engineering projection, not human validation.

## Where the project stands

| Layer | Current evidence | Bounded status |
|---|---|---|
| Core Eidos/Sentinel | Streaming prediction/residual/anomaly machinery with preserved receipts | Demonstrated research prototype |
| Detector calibration | Larger labeled CICIDS/WebAttacks guardrails; strict calibrated views can suppress false positives while retaining narrow-event recall | Promising but narrow |
| Operator explanation | Deterministic what/why/evidence/uncertainty/action schema on main | Implemented; f676 human effect not formally validated |
| Human factors | Legacy 153-card review failed: mean 2.635/5, 0/153 at 4.0, safety <=2 on 153/153 | Failed legacy proxy; actual operators not tested |
| Reproducibility | Repo/Drive receipts exist, but no exact-SHA CI and legacy card hash semantics are missing | AMBER |
| Generality / production | Cross-domain proof, long-run compression quality, operational latency, and independent human decision quality remain incomplete | Not established |

## Paired replay math

| Surface | Mean composite | Median | Cards >=4 | Natural >=4 | Normal >=4 | Bounded result |
|---|---:|---:|---:|---:|---:|---|
| Legacy frozen cards | 2.635 | 2.750 | 0/153 | 0/75 | 0/78 | Failed locked HICQR proxy |
| f676 explanation projection | 3.972 | 4.050 | 119/153 | 54/75 | 65/78 | Improved, but gate not met |
| Evidence-link patch projection | 4.028 | 4.050 | 136/153 | 62/75 | 74/78 | All scorebook gates pass; not validated |

Scorebook anchors were fixed by card structure: f676 engine 4.05; f676 confirmed without drivers 3.70; patched linked confirmed 4.05; patched unlinked confirmed 3.85. Every legacy-to-new proxy pair improves, no proxy pair regresses, and all enriched safety scores are 5. These are inspectable reviewer judgments rather than learned metrics.

## Code leap completed

The patch:

- reads legacy engine drivers from `evidence.drivers` and references from `evidence.exemplars`;
- links only engine cards whose step falls inside a confirmed event window;
- ranks/deduplicates linked drivers and records exact source card IDs/steps;
- leaves detector and calibration outputs untouched;
- stops telling operators to inspect ranked drivers when none exist;
- recognizes `cicids_webattacks` as a security domain for next-step guidance;
- fixes the “a anomaly” rendering defect.

Strict overlap links 8/21 natural and 9/13 normal confirmed events. The remaining 17 confirmed events explicitly disclose that ranked contributors are absent.

Validation: 39/39 focused explanation, confirmation, and labeled-proof tests pass. A broader CPU-compatible profile passes 122 tests with 6 skips. The unfiltered suite is not clean in this environment because PyTorch is absent; two torch-specific modules fail collection and four engine runtime tests require torch.

## Proof Logic + Meaning

**Goal.** Move Eidos toward a self-monitoring streaming intelligence codec whose anomaly receipts are useful without converting anomaly into attack certainty.

**Logic and math.** The legacy operator proxy failed despite structural completeness. Holding 153 card identities fixed shows the f676 explanation likely produces a large paired improvement, but its projected 119/153 pass rate is below the locked 80% threshold. The missing mass is localized to 34 confirmed-event cards. Linking 17 strict-overlap windows raises the projected pass numerator to 136/153, above 80% overall and within both receipts.

**Philosophy.** The important move is not prettier prose. It is a tighter observation-to-interpretation chain: a confirmed window points to the engine evidence already inside that window, uncertainty stays explicit, and the operator is never told evidence exists when it does not.

**Better than before?** Yes as an engineering result: 39 focused tests pass and evidence linkage is implemented. Not yet as a human-factor claim: the locked input hash gate failed and no independent operator reviewed the patched surfaces.

**Movement toward the north star.** Material but bounded. The receipt layer now carries more of the causal trail from residual observation to confirmed window to human action. Detection correctness, compression advantage, and cross-domain generality did not move today.

**Uncertainty and next action.** Execute locked protocol `PICQR-2026-08-26-v2` (`24fcc766c3b3cadf53988d304919240f968336463187562a97c7a80d3fb72bb3`): render the patched surfaces and run one genuinely blinded review by at least one independent security operator. Do not tune the code or scorebook during that review. That is the shortest path from a gate-passing projection to evidence.
