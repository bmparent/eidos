# Eidos Brain / Eidos Sentinel — Grand Proof Protocol v1

**Protocol ID:** `EIDOS-GP-v1-2026-09-01`
**Status:** design locked; execution lock required before held-out runs
**Scope:** Eidos Brain / Eidos Sentinel only
**Companion contract:** `meaningful_surprise_v1_spec.md`
**Repository baseline inspected:** `00489e358865994ec4b40a5c5bfdfa034560773a`
**Main baseline:** `f676fe2342b98886bc04cd8f4b0e943fce77ec9a`

## 1. Decision this protocol must make

This protocol decides whether the full live Eidos engine, observed through Meaningful Surprise v1, demonstrates a bounded joint advantage that cannot be explained by a proxy detector, unequal alert budget, unequal byte budget, built-in card/replay credit, label leakage, or one favorable synthetic scenario.

It must return an affirmative construction or a useful negative result. “More tests passed” is not a proof verdict.

### 1.1 Primary empirical conjecture

At matched alert and byte budgets on held-out streams, `eidos_ms_full` is on the detection–preservation–memory–compression Pareto frontier and contributes value beyond the strongest tested baseline and its own preregistered ablations, while satisfying raw-escape, dangerous-repeat, label-isolation, replay, and false-positive safety gates.

### 1.2 Why the current evidence is insufficient

- `repo/src/eidos_brain/proof/run_proof.py::eidos_minimal` is a lag-1 residual plus windowed FFT scout, not the live reservoir/HDC/thermodynamic/codec engine.
- `repo/src/eidos_brain/benchmarks/quantum_compression_benchmark.py` uses a Sentinel proxy for its Eidos path.
- The current scalar proof metric grants unequal evidence-completeness and replay credit and sets memory utility to zero.
- Existing live integration proves real prediction/Sentinel/HDC metadata reaches the codec, but not that the combined system adds held-out value.
- Existing CICIDS/WebAttacks results support a bounded labeled guardrail claim, not cross-domain generalization or unique technical advantage.

## 2. Two locks

### 2.1 Design lock

This document freezes hypotheses, comparisons, gates, scenario meanings, analysis rules, and permissible verdicts. It must not be edited after a held-out result is observed. Corrections require a new version and a reasoned diff.

### 2.2 Execution lock

Before any held-out run, create `run_lock.json` containing:

- protocol and Meaningful Surprise document SHA-256 values;
- exact code commit and branch;
- dirty/untracked state;
- exact dataset paths, byte hashes, sizes, row counts, licenses, and splits;
- exact engine and scout configs;
- domain contracts and hashes;
- calibration and test seed lists;
- resource-profile selection receipt;
- baseline availability and explicit skip rules;
- thresholds, byte/alert budgets, metric normalizers, scalar sensitivity grid;
- exact commands and artifact root.

If the lock is missing, changed after test access, or inconsistent with artifacts, the only verdict is `INVALID_RUN`.

## 3. Intervention boundary

The first Grand Proof is shadow-only:

- the current live engine produces prediction, residual, Sentinel, geometry, HDC, thermodynamic, codec, and replay evidence;
- the Meaningful Surprise layer consumes a proof observer stream and writes separate decisions/tokens/cards;
- default engine outputs do not change;
- detector thresholds, event boundaries, confirmation, calibration, HDC write/freeze behavior, codec policy, incident-card facts, and domain math remain unchanged when the feature is off;
- no production action is authorized.

The treatment may simulate alternate shadow alert and fidelity decisions. It may not silently replace the authoritative live outputs.

## 4. Systems under comparison

All systems receive the same chronological input frames, calibration/test split, outcome feedback timing, alert budget, and byte-accounting definition.

| ID | Detector / representation | Memory | Codec / retention | Evidence wrapper |
| --- | --- | --- | --- | --- |
| `eidos_ms_full` | full live Eidos signals plus v1 scout | live HDC plus causal consequence annotations | v1 shadow policy and residual codec | common wrapper plus native references |
| `eidos_live_current` | current live surprise/Sentinel path | current HDC metrics | current residual codec | common wrapper |
| `eidos_minimal` | existing raw/spectral minimal scout | none | common budget wrapper | common wrapper |
| `rolling_z` | causal rolling z-score | none | common budget wrapper | common wrapper |
| `ewma` | causal EWMA residual | none | common budget wrapper | common wrapper |
| `cusum` | two-sided causal CUSUM | none | common budget wrapper | common wrapper |
| `isolation_forest` | frozen calibration-only fit | none | common budget wrapper | common wrapper |
| `knn_episode` | causal raw-feature kNN score | same delayed feedback available to Eidos | common budget wrapper | common wrapper |

`isolation_forest` may be skipped only if its dependency is unavailable before execution lock. `zstd` may be skipped under the same rule. A missing optional baseline is visible and lowers the strength of the claim; it is never scored as zero.

### 4.1 Common evidence/replay wrapper

Every detector baseline receives a deterministic card and replay wrapper built from evidence it actually produced. The wrapper may report score, threshold, time range, top input contributors when defined, uncertainty, and a bounded next action. It may not invent reservoir, HDC, geometry, or causal evidence.

Evidence completeness and replay are then scored from the same schema. Eidos does not receive automatic `EC=1` or `REP=1` merely because the repository contains a card writer.

### 4.2 Compression baselines

At minimum compare raw frames, gzip, lzma, and available zstd. Compression-only methods are not assigned fabricated alerts. They define byte/reconstruction reference points. Joint systems are compared through matched-budget operating points, not by setting missing detection metrics to zero.

## 5. Ablations

All ablations consume the same captured live stream so stochastic engine variation cannot masquerade as a component effect.

| ID | Removed contribution | Targeted question |
| --- | --- | --- |
| `A0_full` | none | full v1 shadow policy |
| `A1_no_hdc` | familiarity and memory consequence | does episodic memory help S8 without harming S7? |
| `A2_no_geometry` | geometry lift and evidence | does state geometry add held-out value? |
| `A3_no_multiscale` | multi-scale/phase lift | does long-horizon structure add value in S1–S3? |
| `A4_no_thermo` | thermodynamic metrics from the scout | do these metrics add value when available? |
| `A5_no_scout` | representation selection; raw only | does lift selection beat raw residual alone? |
| `A6_no_raw_escape` | raw safety OR constraint | counterfactual only; must fail the nuisance counterexample if the counterexample is valid |
| `A7_no_voi` | consequence/VOI term; structural evidence only | is “meaning” adding value beyond surprise? |

An ablation that produces no measurable loss is evidence that the removed component is not part of the supported v1 uniqueness claim. It is not hidden.

## 6. Synthetic mechanism suite

All generators are causal, deterministic by seed, 64-dimensional, and emit truth/outcome fields only to a sealed scorer. Outcome feedback made available online is delayed until its registered horizon closes and is provided identically to memory-capable baselines.

Use Gram–Schmidt orthonormal directions from the scenario RNG. The nominal carrier should reuse the existing controlled-regime form:

\[
x_t=0.55\sin(2\pi t/61)d_0+0.38\cos(2\pi t/89)d_1
+0.24\sin(2\pi t/137+0.4)d_2+\epsilon_t,
\quad \epsilon_t\sim\mathcal N(0,0.018^2I).
\]

Exact amplitudes, frame windows, outcome horizons, and costs are implemented in a scenario config and hashed in the execution lock. The semantic role below may not change.

| ID | Construction and meaning | Required behavior |
| --- | --- | --- |
| `S0_nominal` | carrier only; no consequential event | low false-alert and byte pressure |
| `S1_hidden_backdoor` | weak, phase-locked component below obvious pointwise SNR in a bounded window | structural/spectral or multi-scale lift finds it without excess FPR |
| `S2_slow_drift` | gradual sub-threshold change whose accumulated state crosses a consequential boundary | persistence or multi-scale evidence detects before the outcome horizon |
| `S3_regime_shift` | durable change in the carrier dynamics, not a single spike | identify transition and avoid endless duplicate alerts |
| `S6_noise_thrash` | large high-entropy variance with no consequential outcome | do not confuse unpredictability with meaning; retain only budgeted evidence |
| `S7_harmless_repeat` | a repeated, familiar moderate excursion with verified benign consequence | recurrence may reduce novelty/review cost; no attack/danger statement |
| `S8_dangerous_repeat` | an equally familiar pattern whose first occurrence later receives harmful consequence feedback, then recurs | repeat remains review/escalation-worthy even when novelty falls |
| `C1_nuisance_subspace` | meaningful event placed in the registered nuisance component so quotient/scout evidence can miss it, with raw evidence above escape threshold | full system preserves it via raw escape; `A6_no_raw_escape` exposes the miss |

S7 and S8 must be matched in amplitude, duration, recurrence gap, and nominal context; only delayed consequence differs. Otherwise the test does not isolate familiarity from danger.

### 6.1 Seeds and stages

- engineering smoke: seeds `0,1`, reduced frames; never used for a claim;
- calibration/development: seeds `10..19`;
- held-out test: seeds `100..119`;
- paired bootstrap unit: seed;
- no threshold, coefficient, generator, split, or policy tuning after any held-out seed is opened.

The execution lock records the exact frame count. The default intended shape is at least 1,000 warmup frames and 5,000 scored frames per scenario.

## 7. Real and transfer domains

### 7.1 Domain A — CICIDS/WebAttacks

Required source unless its byte identity fails:

- path: `artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`;
- rows: `170366`;
- registered SHA-256: `d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d`;
- label column: ` Label`;
- benign rows: `168186`;
- attack rows: `2180`.

The current runner’s raw, merged, deduped, confirmed, pre-calibration, calibrated, attack-window, latency, and false-positive-taxonomy views must remain side by side. Labels are evaluation-only.

The execution lock freezes a source-order, non-overlapping calibration/test split or the existing canonical replay windows. Rows from a single attack window may not be split across calibration and test.

### 7.2 Domain B — real non-cyber stream

A real non-cyber dataset is mandatory for the cross-domain or unique-value verdict. The preferred candidate is the previously used `epileptic-seizure-recognition` stream, provided its raw source—not only old reports, geometry, or surprise artifacts—is available and can be hashed.

Before use, lock:

- raw byte hash and license/provenance;
- causal stream construction;
- label-to-outcome contract;
- subject/session/group-aware split when identifiers exist;
- feature normalization fit on calibration data only;
- a statement that the experiment is a signal benchmark, not medical validation or clinical advice.

If no eligible raw non-cyber source exists, continue engineering and synthetic/cyber runs but cap the verdict at `SYNTHETIC_AND_CYBER_ONLY`. The quantum/crypto/binary generators cannot substitute for this gate.

### 7.3 Transfer stress suite

Run the existing quantum, crypto-agility, and binary-river generators as synthetic transfer checks after replacing or clearly separating their Sentinel proxy result from the full-engine result. These tests can reveal brittleness; they cannot establish real-world cross-domain generalization.

## 8. Resource-profile selection

Full-engine proof means all relevant live mechanisms execute; it does not require an impractical reservoir size on every machine.

Before outcome-bearing calibration runs:

1. test reservoir sizes in ascending order from `{128, 256, 512, 1024}` on `S0_nominal`, seed `0`, 2,000 frames;
2. select the largest size whose projected full suite stays within the declared time budget and whose peak memory stays below 70% of available RAM/VRAM;
3. freeze the selected size and all other engine parameters in `run_lock.json`;
4. do not use detection or compression quality to select the size.

A result at a reduced profile supports that profile only. Report it plainly.

## 9. Budget matching

### 9.1 Alert budget

Use the calibration split to choose one threshold per system for each registered operating point. Primary point:

\[
\operatorname{FPR}\le 5\text{ alerts per }10{,}000\text{ nominal frames}.
\]

If a system cannot meet the budget, report infeasibility; do not raise its threshold on test data.

### 9.2 Byte budget

Byte cost is total serialized bytes required to reconstruct the registered stream and its anomaly/discovery evidence, divided by canonical raw float bytes. Report payload, index, card, model-state, and manifest bytes separately and together.

Freeze at least four policy operating points from calibration data, intended around byte fractions `{0.10, 0.25, 0.50, 1.00}` when feasible. Compare frontiers rather than forcing an infeasible codec to hit an exact fraction.

### 9.3 Latency budget

Include encoding, detection, scout, and card-generation wall time. Exclude one-time environment creation and dataset download. Report median, p95, and frames/second on the same machine.

## 10. Metrics

Compute per system, domain, scenario, seed, and operating point:

- event precision, recall, \(F_2\), and false negatives;
- false positives per 10,000 nominal frames;
- first-detection delay and attack/outcome-window coverage;
- anomaly-preservation recall (APR);
- normal and anomaly reconstruction RMSE separately;
- compressed/raw byte fraction (NBC) with accounting breakdown;
- evidence completeness (EC) under the common wrapper;
- deterministic replay success (REP);
- memory utility (MU);
- VOI calibration and realized loss reduction;
- representation choice frequency and disagreement;
- runtime, p95 latency, peak RAM/VRAM, crashes, non-finite values;
- uncertainty coverage and unresolved `UNKNOWN` rates.

### 10.1 Memory utility

For matched S7/S8 repeat events, define:

\[
MU=\tfrac12\left(\operatorname{Recall}_{S8,\mathrm{repeat}}
+1-\operatorname{ReviewRate}_{S7,\mathrm{repeat}}\right).
\]

Report both terms. A high aggregate cannot hide failure on dangerous repeats.

### 10.2 Evidence completeness

EC is the fraction of required, truthful fields present and resolvable: observation, score/threshold, time/source range, selected representation, uncertainty, bounded next action, raw references when claimed, config/code identity, and replay command. Unsupported fields score absent, not partially correct.

### 10.3 Primary comparison

The Eidos Value Vector from the companion spec and Pareto dominance under matched budgets are primary. The scalar EVM is secondary and must be reported across the frozen sensitivity grid.

## 11. Statistical analysis

- Use paired differences by seed for synthetic scenarios and paired source windows/events for real domains.
- Report median difference, mean difference, 95% paired bootstrap interval with 10,000 resamples, and exact numerator/denominator.
- Bootstrap whole seed or source-window units, not individual autocorrelated frames.
- Apply Holm correction within each family of primary component comparisons.
- Report all registered systems and ablations, including negative effects and infeasible operating points.
- Do not convert a missing baseline, failed run, or undefined metric to zero.
- Report weight-sensitivity results; a claim that exists only at one scalar weighting fails robustness.

## 12. Acceptance gates

Gates are evaluated in order. A later gain cannot repair an earlier integrity or safety failure.

### G0. Identity and reproducibility

- execution lock verifies;
- code/dataset/config hashes match;
- no unapproved dirty-tree source changes;
- required artifact manifest is complete;
- deterministic replay succeeds on all selected replay cases;
- full targeted tests and the compatible repository suite pass;
- skipped tests/dependencies are explicit.

Failure verdict: `INVALID_RUN`.

### G1. Causality and label isolation

- no future-window feature construction;
- normalization/calibration is past-only or calibration-split-only;
- sealed labels/outcomes are inaccessible to online components;
- a sentinel “poison label” test proves the output is unchanged when test labels are permuted while inputs remain fixed.

Failure verdict: `FAILED_INTEGRITY`.

### G2. Safety invariants

- `C1_nuisance_subspace`: raw escape preserves 100% of registered events;
- `S8_dangerous_repeat`: repeat-event recall is at least 0.95 and no lower than first-occurrence recall by more than 0.05;
- `S6_noise_thrash` and `S0_nominal`: primary operating point remains at or below 5 false alerts per 10,000 nominal frames;
- no card states attack, failure, seizure, compromise, or cause as fact from anomaly evidence alone;
- non-finite values and crashes are zero.

Failure verdict: `FAILED_SAFETY`.

### G3. Mechanism support

On held-out synthetic seeds:

- `A5_no_scout` is worse than full on its targeted S1–S3 joint endpoint with a Holm-adjusted paired 95% lower bound above zero;
- `A1_no_hdc` is worse than full on S8 memory utility without full being worse on S7 review cost;
- `A6_no_raw_escape` exposes at least one C1 miss while full has none;
- `A7_no_voi` is worse on realized decision loss or calibrated review cost;
- geometry, multiscale, and thermodynamic contributions are each reported; any non-contributing component is removed from the supported uniqueness statement.

If safety passes but these targeted effects do not: `MECHANISM_NOT_SUPPORTED`.

### G4. Matched-budget joint value

At the primary alert budget:

- `eidos_ms_full` is non-dominated on the aggregate synthetic EVV frontier;
- it Pareto-dominates the strongest non-Eidos joint baseline in at least four of the seven non-counterexample synthetic scenarios;
- on CICIDS/WebAttacks it is no worse than the strongest baseline by more than 0.02 in event \(F_2\) or APR at matched byte cost, and it improves at least one of NBC, delay, EC, REP, or MU by at least 10% relative without breaking G2;
- the registered scalar sensitivity grid shows positive median EVM delta for at least 80% of weight settings.

If only synthetic conditions pass: `PROMISING_SYNTHETIC_ONLY`.
If synthetic and CICIDS pass but Domain B is absent: `SYNTHETIC_AND_CYBER_ONLY`.

### G5. Cross-domain value

On the locked real non-cyber domain:

- G0–G2 pass;
- full is non-dominated at the primary budget;
- full is no worse than the strongest baseline by more than 0.02 in the primary detection/preservation metric;
- full improves at least one registered benefit or cost by at least 10% relative;
- no domain-specific tuning uses test outcomes.

Passing G0–G5 supports `CROSS_DOMAIN_VALUE_SUPPORTED_BOUNDED`.

### G6. Independent operator evidence

Human usefulness is a separate gate. The implementation team and ChatGPT/Codex are not eligible as the primary operator reviewer.

- Execute the already locked `PICQR-2026-08-26-v2` for the August incident-card claim after resolving its implementation-SHA identity discrepancy.
- For new Grand Proof discovery cards, freeze a blinded, stratified review surface and rubric before an independent operator sees it.
- Hide truth labels, variant names, system names, prior scores, and source receipt from the operator-visible surface.
- Report technical and human verdicts separately.

Passing G0–G5 while G6 is pending yields `TECHNICAL_VALUE_SUPPORTED_HUMAN_PENDING`. Passing the registered independent review permits `UNIQUE_VALUE_SUPPORTED_BOUNDED`.

No verdict in this protocol means production-ready.

## 13. Required artifacts

```text
artifacts/grand_proof_v1_<utc>/
  protocol/
    meaningful_surprise_v1_spec.md
    grand_proof_protocol_v1.md
    execution_prompt.md
    run_lock.json
    lock_hashes.json
  provenance/
    git_commit.txt
    git_status.txt
    environment.txt
    dependency_inventory.json
    commands.jsonl
    dataset_manifest.json
    artifact_manifest.json
  configs/
    engine_config.json
    domain_contracts.json
    scenario_config.json
    baseline_config.json
    metric_config.json
  captures/
    live_frame_observer.jsonl
    shadow_decisions.jsonl
    shadow_tokens.jsonl
  scenarios/<scenario>/<seed>/<system>/
    metrics.json
    events.jsonl
    discovery_cards/
    replay_logs/
  domains/<domain>/<split>/<system>/
    metrics.json
    events.jsonl
    incident_cards/
    replay_logs/
  ablations/
    paired_results.csv
    component_claims.json
  statistics/
    pareto_points.csv
    paired_intervals.csv
    weight_sensitivity.csv
  failures/
    failure_ledger.jsonl
    counterexamples/
  reports/
    benchmark_report.md
    theorem_status.md
    proof_logic_meaning.md
    final_verdict.json
```

Long runs use incremental JSONL, atomic checkpoints, and resumable manifests. Partial results remain visible and are never promoted to a completed verdict.

## 14. Replay requirements

Select at minimum:

- one true consequential event per scenario/domain;
- one false positive per system when any exist;
- one S7 harmless repeat;
- one S8 dangerous repeat;
- the C1 raw-escape case;
- one case where full loses to a baseline;
- one case where a component ablation equals or beats full.

Replay must regenerate source frames and decision fields from the locked command. Hash differences are failures with an explicit field-level diff.

## 15. Stop and invalidation rules

Stop the held-out phase and preserve artifacts if:

- test labels or test-wide statistics enter the online path;
- a protocol/config/dataset hash changes;
- a safety invariant fails;
- a system crashes or produces non-finite output;
- an implementation bug changes decision semantics.

Bug fixes require a new execution lock and rerun of all systems from the beginning. Do not patch only the losing or failing scenario.

The following are prohibited:

- tuning on held-out seeds or windows;
- deleting failed runs;
- replacing unavailable baselines with zeros;
- changing metric weights after results;
- awarding Eidos unique card/replay credit;
- calling a proxy path “full Eidos”;
- treating a synthetic domain as real cross-domain evidence;
- treating familiarity as safety;
- merging to `main` without explicit instruction.

## 16. Negative-result interpretation

| Failure pattern | Likely implication | Next bounded action |
| --- | --- | --- |
| scout ablation equals full | representation selection adds no demonstrated value | simplify scout or redesign lifts |
| HDC ablation equals full on S8 | familiarity/memory is not contributing to consequence-aware recurrence | revise causal memory feedback, not HDC size |
| no-VOI equals full | “meaning” layer is only renamed surprise | improve outcome/loss estimation or reject the claim |
| raw escape fails C1 | safety construction is wrong | fix invariant before any other optimization |
| synthetic passes, real domains fail | generators are too favorable or representation overfits mechanisms | revise claim downward; add natural counterexamples |
| detection improves but bytes/latency collapse | not a joint intelligence codec advantage | optimize policy only after preserving safety |
| cards score poorly | machine result is not operationally usable | improve evidence linkage under a new human protocol |

## 17. Proof Logic + Meaning

**Goal.** Close the claim–engine–proof gap with one experiment that runs the actual Eidos mechanisms, compares them fairly, and can falsify the unique-value claim.

**Logic and math.** The protocol separates mechanism tests from natural-domain tests, matches alert and byte budgets, uses causal splits, compares a Pareto vector before scalar weighting, and demands targeted ablation effects. Raw escape and dangerous repeat are hard safety gates, not optional score contributions.

**Philosophy.** A powerful system should know not only that reality surprised it, but why preserving that surprise may change a future choice—and when its own representation may be hiding evidence. The proof must therefore reward consequence, memory, restraint, and auditability together.

**Why this is better.** It removes the present proxy/full-engine mismatch and the metric asymmetry that can make Eidos win by construction. Baselines receive common card/replay machinery, unavailable metrics remain missing, and every supported component must earn its place through ablation.

**Movement toward the north star.** A passing result would justify a bounded, distinctive statement about a self-monitoring streaming intelligence codec. A failure would still reveal exactly which promised mechanism is not carrying its weight.

**Evidence still required.** A clean local implementation, execution lock, full-engine runs, real non-cyber raw data, reproducible receipts, and an actual independent operator. This protocol precommits the decision rules; it does not predeclare success.
