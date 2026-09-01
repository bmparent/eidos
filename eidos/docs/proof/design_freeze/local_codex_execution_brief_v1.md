# Local Codex Execution Brief — Eidos Meaningful Surprise / Grand Proof v1

**Use:** copy the prompt below into Codex while it is opened at the real Eidos repository.
**Scope:** Eidos Brain / Eidos Sentinel only.
**Authority:** implement, test, run, commit, and push a feature branch if repository credentials permit; do not merge.
**Companion files:** `meaningful_surprise_v1_spec.md`, `grand_proof_protocol_v1.md`, and `design_freeze_manifest_v1.json`.

---

## Copy/paste prompt for local Codex

You are implementing and executing Eidos Brain / Eidos Sentinel Meaningful Surprise v1 and Grand Proof v1 in the current Eidos git repository.

Read these files completely before editing:

1. every applicable `AGENTS.md` or repository instruction file;
2. `eidos/docs/proof/design_freeze/meaningful_surprise_v1_spec.md`;
3. `eidos/docs/proof/design_freeze/grand_proof_protocol_v1.md`;
4. `eidos/docs/proof/design_freeze/design_freeze_manifest_v1.json`;
5. `eidos/docs/proof/incident_card_operator_explanation_v1.md`;
6. `eidos/docs/proof_runs/2026-08-26/eidos_revival_proof_report_2026_08_26.md`;
7. `eidos/docs/proof_runs/2026-08-26/paired_incident_card_review_protocol_v2_2026_08_26.md`;
8. the current live engine, codec, proof runner, labeled runner, controlled regimes, and their tests.

If the three design-freeze files are not present, stop and ask me to add them. Do not reconstruct them from this prompt.

### Mission

Build the smallest safe, full-engine, shadow-only implementation of `EIDOS-MS-v1`; build a fair, reproducible `EIDOS-GP-v1` harness; run every safe stage the machine can support; and return evidence. The goal is to determine whether Eidos preserves decision-relevant deviations under resource constraints—not to manufacture a positive verdict.

### Non-negotiable scope

- Work only on Eidos Brain / Eidos Sentinel.
- Do not touch Kalshi, PolySentinel, trading, market execution, or unrelated applications.
- Do not merge to `main`.
- Do not alter current live detector thresholds, event boundaries, confirmation, calibration, HDC write/freeze logic, thermodynamic control, codec policy, incident-card facts, or domain math for the default-off path.
- Do not implement Leap IV / multiverse, enlarge the default reservoir, or add new physics controllers.
- Do not call `eidos_minimal` or the quantum Sentinel proxy “full Eidos.”
- Do not use test labels, future frames, test-wide normalization, or held-out outcomes online.
- Do not delete, overwrite, stage, or clean unrelated user changes or untracked files.
- Do not weaken gates, shrink held-out requirements, tune on test, or convert missing values to zero.
- Do not claim production readiness, clinical validity, causal diagnosis, attack truth, or universal meaning.

### Git and provenance preflight

1. Print and record `git status --short --branch`, remotes, current HEAD, `origin/main`, and relevant branch history.
2. Identify the exact base containing the August evidence-linkage work. The inspected Work-mode clone had:
   - `origin/main` at `f676fe2342b98886bc04cd8f4b0e943fce77ec9a`;
   - evidence-link code at `6f98eeb`;
   - proof receipts at `00489e358865994ec4b40a5c5bfdfa034560773a`;
   - the August report/protocol citing a different implementation SHA, `4a639cd693701fb764fe30ba672d4811bdbf5a75`.
3. Resolve what each SHA represents from git evidence. Do not rewrite old receipts or declare the PICQR discrepancy resolved without a verifiable mapping. Record the result in a new identity receipt. The discrepancy blocks the human `PICQR-v2` gate, not unrelated engineering work.
4. Create or resume `codex/eidos-meaningful-surprise-grand-proof-v1` from the correct reviewed base. If branch creation would overwrite or strand user work, stop and report the exact conflict.
5. Verify the design-freeze manifest. Do not edit locked v1 documents. If a correction is essential, create v1.1 plus a reasoned diff before opening held-out results.

### Architecture rule: shadow first

The treatment must be off by default. With `meaningful_surprise_enabled=false`, deterministic baseline fixtures must retain identical authoritative outputs except explicitly allowlisted timestamps and absolute paths.

Use a proof observer to capture, causally and without changing decisions:

- exact frame and live `best_pred`;
- raw residual and normalized error;
- current surprise score and threshold;
- Sentinel status and metrics;
- eigen dominance, state entropy/flatness, spectral entropy/flatness;
- HDC bank, similarity, familiarity, write state;
- thermodynamic energy/rho/temperature/lambda when enabled;
- current codec decision and serialized byte count;
- source metadata, frame/step identity, config/code hashes, and replay references.

An optional default-off capture hook in the live engine is permitted only if it is instrumentation-only, deterministic, tested for no state/RNG/decision changes, and isolated behind config. Prefer a narrow writer or callback interface over duplicating engine math. Do not derive “live” proof from the existing minimal scout.

### Suggested file layout

Follow repository conventions if a better existing home is obvious. Otherwise use:

```text
eidos/repo/src/eidos_brain/proof/
  frame_observer.py
  meaningful_surprise.py
  representations.py
  grand_proof_scenarios.py
  grand_proof_metrics.py
  grand_proof_runner.py
eidos/repo/tests/
  test_frame_observer.py
  test_meaningful_surprise.py
  test_grand_proof_scenarios.py
  test_grand_proof_metrics.py
  test_grand_proof_runner.py
eidos/tools/
  run_grand_proof_v1.py
```

Keep one authoritative implementation. Remove or explicitly label mirrored/proxy proof logic; do not create a third independent detector stack.

### Phase 1 — proof observer and compatibility

Implement a versioned observer schema and incremental JSONL writer. Requirements:

- atomic/resumable writes and explicit incomplete-run status;
- finite numeric validation;
- stable field ordering/canonical hashing where identity matters;
- bounded memory use;
- capture after the live decision so observation cannot affect that decision;
- exact source ranges and artifact references;
- deterministic replay command.

Add tests for:

- feature disabled means no observer artifact and no authoritative output change;
- enabled capture does not change engine decisions, HDC writes, model hashes, codec modes, or RNG-dependent outputs;
- a live synthetic run proves prediction source is live consensus, Sentinel source is live analysis, and HDC source is live metrics;
- malformed/non-finite frames fail visibly;
- interrupted capture is marked partial and can resume safely or restart under a new run ID.

### Phase 2 — Meaningful Surprise v1

Implement the companion contract, including:

- causal representation interface for `raw`, `spectral`, `multiscale`, `geometry`, `memory`, and `consensus`;
- lift-specific past-only calibration;
- generalized/quotient residual support;
- persistence;
- representation disagreement with an explicitly reported definition;
- causal phase coherence and common-phase invariance;
- familiarity separate from memory consequence risk;
- delayed consequence feedback with provenance and support/confidence minimums;
- VOI estimate, uncertainty, lower confidence bound, and later realized-loss scoring;
- constrained shadow selection of representation, fidelity, and action;
- mandatory raw-residual escape as a final OR constraint;
- versioned decision JSON and discovery-card integration;
- meaning states `STRUCTURAL_HYPOTHESIS`, `OUTCOME_ESTIMATED`, `OUTCOME_VALIDATED`.

Do not silently invent a universal task loss. Every run must load and hash a domain contract. Missing contracts cap semantic claims as specified.

Required property tests:

- raw escape cannot be suppressed by any lift, cost, familiarity, or negative VOI estimate;
- increasing validated danger recall cannot reduce action/fidelity;
- increasing familiarity alone cannot certify safety;
- `UNKNOWN` memory consequence is not treated as zero risk;
- higher uncertainty at high risk cannot lower retention;
- phase coherence is invariant to a common phase rotation;
- all lifts are causal under a future-frame mutation test;
- projection scaling is documented and tested; do not claim Johnson–Lindenstrauss norm preservation if the implementation uses a shrinking convention;
- canonical decision replay is deterministic.

### Phase 3 — fair Grand Proof harness

Implement the protocol exactly:

- S0, S1, S2, S3, S6, matched S7/S8, and C1 generators;
- smoke, calibration, and held-out seed separation;
- full live capture once per stream, then paired shadow systems/ablations over the identical capture where valid;
- `eidos_ms_full`, current live Eidos, existing minimal scout, rolling z, EWMA, CUSUM, optional Isolation Forest, and causal kNN episode baseline;
- raw/gzip/lzma/available-zstd compression references;
- a common truthful evidence/replay wrapper for all detector baselines;
- matched alert budget and byte-frontier operating points;
- Eidos Value Vector, Pareto analysis, registered EVM sensitivity grid, and missing-value-safe aggregation;
- paired bootstrap by seed/window with Holm correction;
- A0–A7 ablations;
- counterexamples, failure ledger, and replay sample selection;
- exact artifact tree and permissible verdicts from the protocol.

Do not copy the current `compute_metrics` asymmetry. Compression-only baselines are references, not fake detectors. Baselines receive common EC/REP opportunity. Optional unavailable systems are `SKIPPED` with reason, never zero.

### Phase 4 — real-domain adapters

Reuse `eidos/tools/run_labeled_domain_proof.py` and preserve every existing raw/merged/deduped/confirmed/pre-calibration/calibrated/attack-window/latency/FP-taxonomy view.

For CICIDS/WebAttacks:

- verify the expected raw dataset SHA `d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d` before use;
- fail closed on byte, row, label-column, or split mismatch;
- use source-order, non-overlapping frozen splits and keep an attack window in one split;
- prove labels are scoring-only with a label-permutation/poison test.

For the real non-cyber domain:

- first search local approved project data for the raw `epileptic-seizure-recognition` source used in earlier Eidos work;
- old `.bin`, geometry JSON, top-surprise text, and reports are not raw-source substitutes;
- verify license/provenance, hash, causal stream construction, and group-aware split before use;
- treat it only as a signal benchmark, not medical validation;
- if raw data is absent, record the blocker and cap the verdict. Do not download an arbitrary replacement or use synthetic quantum data to satisfy this gate without explicit user authorization.

The quantum/crypto/binary suite is a labeled synthetic transfer stress suite. Replace or clearly separate its proxy row from a new full-engine row.

### Phase 5 — execution lock

Before opening held-out seeds/windows:

1. run the resource-profile selection rule from the protocol using performance only;
2. freeze all exact scenario numbers, domain contracts, transforms, thresholds, budgets, coefficients, normalizers, splits, seeds, dependencies, commands, and artifact paths in `run_lock.json`;
3. hash the lock and copy locked protocol files into the artifact tree;
4. run an automated preflight that fails closed on any mismatch;
5. commit the implementation and lock so the test run has a resolvable code SHA.

Do not use outcome quality to select reservoir size or runtime profile.

### Phase 6 — verification and runs

Run in this order:

1. focused unit/property tests;
2. current live residual-codec integration test;
3. current proof-runner and labeled-runner tests;
4. compatible full repository test suite with JUnit receipt;
5. smoke suite seeds `0,1`;
6. calibration seeds `10..19`;
7. freeze thresholds/operating points;
8. held-out synthetic seeds `100..119`;
9. locked CICIDS/WebAttacks evaluation;
10. locked real non-cyber evaluation if valid raw data exists;
11. transfer stress suite;
12. deterministic replay set;
13. artifact verification and final verdict.

Use incremental JSONL, checkpoints, and manifests for long work. Never hide partial legs. If the estimated full run exceeds the available session or hardware, finish the implementation, tests, smoke run, resource receipt, and execution lock; return `BLOCKED_RESOURCE_BEFORE_HELDOUT`. Do not reduce the protocol to make it finish.

If PyTorch or required runtime support is absent, make the skip/blocker explicit. A proxy cannot replace the full-engine leg.

### Phase 7 — human gate boundary

Do not use ChatGPT, Codex, the implementation author, or a synthetic rubric as the primary operator reviewer.

Prepare but do not self-score:

- the existing `PICQR-2026-08-26-v2` package after the SHA discrepancy is resolved;
- a new blinded, stratified Grand Proof discovery-card review package with truth labels and variant mapping sealed.

Technical and human verdicts remain separate. If technical gates pass but the independent review has not happened, the maximum technical verdict is `TECHNICAL_VALUE_SUPPORTED_HUMAN_PENDING`.

### Required final receipts

Return and commit, at minimum:

- files changed and why;
- exact base, branch, commits, and dirty-tree state;
- August SHA identity receipt;
- tests run, counts, failures, skips, and JUnit path;
- resource-profile receipt;
- design and execution lock hashes;
- dataset identity/split receipts;
- smoke/calibration/held-out commands and statuses;
- per-system and ablation tables;
- Pareto and weight-sensitivity tables;
- replay results and negative examples;
- failure ledger;
- final machine-readable verdict and plain-language report;
- `proof_logic_meaning.md` with goal, exact logic/math, philosophical meaning, why better or not, movement toward the north star, supporting artifacts, limitations, and exactly one next experiment.

### Git handoff

Make small, logical commits. Do not include datasets, secrets, environments, caches, unrelated artifacts, or user files. Push only the feature branch if normal credentials and policy permit. Open or prepare a PR with the bounded claim and evidence, but do not merge it.

End with one of the protocol’s exact verdicts. A negative or blocked verdict is acceptable. Evidence integrity is more important than a favorable result.

---

## Expected first local Codex response

The first response should state:

1. repository/branch/base discovered;
2. whether the design-freeze hashes verify;
3. how the August SHA discrepancy resolves or why it remains open;
4. the proposed instrumentation seam;
5. the exact initial files it will edit;
6. any blocker that prevents a full-engine or real non-cyber run.

It should then proceed unless a genuine permission, identity, data-authority, or destructive-worktree blocker requires the user.
