# Codex Journal — 2026-09-04

## What happened today

Completed a frozen, controlled comparison of reservoir state rounding and residual carry. Decision: inconclusive for adoption; current production policy remains unchanged.

## What was accomplished

- Preserved the dirty original checkout at 3bb11eb; fetched origin/main resolved to 6bb0b980349f94c55a6ca1ca9665737570c0fd01, exactly the historical source. The engine file has no differences between those two checkout revisions either.
- Created isolated branch codex/controlled-memory-benchmark-2026-09-04 and a separate detached baseline checkout for validation.
- Verified the delivered ZIP SHA-256 ebfd677dd541d5fcc10a495955897dbc71cee9b061f2d13f3bbe9d84214a10bf and every delivered file manifest entry. Preserved original receipts and both original-cap partial attempts.
- Completed the extended-budget exact check: 946,680 scalar map points, 18,432 coupled trajectories, 737,280 time steps, 5,898,240 coordinate inequalities, and 351 sharpness checks.
- Completed 24 matched configurations / 192 trajectories / 7,927,584 unique state updates, plus three timing repeats. Eight neurons, three channels, fixed weight seed 42, forcing/initialization seeds 7 and 19, four policies, three leak vectors, and four streams. Tail: 39,990 steps. Six complete 50-digit reference checks passed.
- Saved all policy variants, inputs, readouts, raw traces, block effects, carry-storage checks, environment, figures, and hashes.

## Tests and commands run

- `python -m proof.memory_benchmark calibrate --out artifacts/controlled_memory_2026_09_04/calibration` — passed.
- `python -m proof.memory_benchmark prepare --out artifacts/controlled_memory_2026_09_04/main --calibration artifacts/controlled_memory_2026_09_04/calibration` — frozen before comparisons.
- `python -m proof.memory_benchmark run --out artifacts/controlled_memory_2026_09_04/main` — complete in 1471.047 seconds (24.52 minutes).
- `python -m proof.memory_report --run artifacts/controlled_memory_2026_09_04/main --out artifacts/controlled_memory_2026_09_04/report` — passed; PNG visually inspected.
- `python -m pytest tests/test_memory_benchmark.py tests/test_memory_collection.py -q` — 11 passed.
- `python -m pytest eidos/repo/tests -q` — 78 passed, 4 documented skips.
- `python -m pytest tests -q` — 19 passed, 5 existing failures; clean baseline has 9 passed and the same 5 failures.
- `python -m pytest -q` — initial collection failed; `python -m pytest --collect-only -q` after the test namespace fix has the same six errors as the clean baseline, with all 11 new tests collected.
- `python artifacts/controlled_memory_2026_09_04/research_rerun/verify.py --evaluate` and `python artifacts/controlled_memory_2026_09_04/research_rerun/verify_sharpness.py` — original 240-second cap yielded a partial coupled check and 351 passing sharpness checks.
- The same source commands ran in `research_rerun_py312` with the bundled Python 3.12.14 runtime — original cap again partial.
- The unchanged source evaluator ran with `--calibrate --freeze --evaluate` in `research_extended_budget`, using Python 3.12.14, after explicitly changing only the cap from 240 to 900 seconds — full original finite scope passed in 436.485 seconds; 351 sharpness checks also passed.
- `git diff --check` and Python compilation checks — passed.

All commands ran from their git repository root. XML/log paths and the actual Python versions are preserved with each receipt. No manual PYTHONPATH edit was used.

## Problems encountered

The original 240-second mathematical package cap was too short on this host. The only extended-copy protocol change is documented; original evidence and failures remain intact.

Full root collection has six baseline errors: an existing UTF-16 test report and unavailable separate Sentinel service package. Five existing RNG proof tests fail on an operator_explanation import. These failures were reproduced at the exact clean baseline. The new benchmark initially encountered a proof-package namespace collision; a pytest-only compatibility change and a regression test resolved it for both root and legacy imports. No numerical evaluator code changed after its freeze.

## What changed

Added proof/memory_core.py, memory_benchmark.py, memory_report.py and memory_bundle.py, benchmark tests, a collection regression test, a small tests/conftest.py compatibility change, separate proof dependencies, reproduction docs, and dated evidence. README and .gitignore link/document the benchmark and keep bulky data outside git.

## What did not change

Core model behavior was not changed. Reservoir dynamics, RLS, Sentinel labels/thresholds, anomaly policy, compression, familiarity and incident logic, production defaults, live Lab, deployments and main branch were not changed.

## Proof Logic + Meaning


### Goal reached
Controlled numerical benchmark: **complete**; adoption utility gate: **missing / untested**. Source reproduction and repository test statuses are reported separately in the handoff receipts.

### Previous state
The research package supplied proofs and toy controls but had not measured the current Torch reservoir under matched rounding policies. The working checkout also contained unrelated residue. This work isolates the current fetched baseline and preserves that residue.

### Technical logic utilized
Freeze the real engine matrices; extract the listen recurrence and validate it against the real path; vary only state rounding. Use exact rational controls and certificates, two initializations, long zero-input tails, fixed exploration readouts, and disjoint suffix diagnostics.

### Math / scoring logic
r_next = (I-A)r + A*tanh(Wr + Bu). Pulse map: k_next = round_even((1-alpha)^m*k). Carry: z=F(r)+c, r_next=Q(z), c_next=z-r_next. Corrected-state error obeys |e_next| <= (I-A+AK)|e| + A(I+K)Delta/2; the invariant box gives |r-x| <= (I-K)^-1 Delta. MSE = mean((prediction-target)^2). RMS discrepancy = sqrt(mean((r-x)^2)). Runtime ratio = median(candidate replay time)/median(current replay time). No utility score is invented.

### Philosophical meaning
Reproducibility is truth that can be revisited. Apparent memory is trustworthy only after representation-induced persistence has been checked.

### Why this is better
The project now has a rerunnable isolated evaluator, frozen inputs, explicit precision limits, measured effects/costs, and retained negative controls. This improves auditability and operator trust; it does not establish useful predictive or detection improvement.

### How this moves Eidos closer to the north-star goal
For the self-monitoring streaming intelligence codec goal, this strengthens reproducibility and interpretation of internal-state memory. It prepares reliable evidence-backed detection work without claiming anomaly preservation, compression advantage, or incident explanation has been demonstrated here.

### Evidence
protocol.json, freeze.json, adapter_selected_size.json, calibration.json, raw/*.npz, results/*.json, precision/*.json, benchmark_summary.csv, temporal_blocks.csv, evidence_figure.png, and evidence_manifest.json. The archive preserves the delivered research ZIP and both original and rerun receipts.

### Remaining uncertainty
Default-size reservoir, GPU, adaptive training/thermostat feedback, exact forgetting of the actual coupled engine, detection labels, operational utility, acceptable overhead, full-model compression behavior, and theorem originality remain unproven.



## Artifacts generated

Repo-local root: C:\Users\bmpar\codex-worktrees\eidos-memory-benchmark\artifacts\controlled_memory_2026_09_04

main/protocol.json, freeze.json, run_manifest.json, environment.txt, git_commit.txt, adapter_selected_size.json; inputs/; raw/; results/; precision/; report/decision_report.md; evidence_figure.png and SVG; benchmark_summary.csv; plot_data.csv; temporal_blocks.csv; report/progress/; research_original/; research_rerun/; research_rerun_py312/; research_extended_budget/; all XML/logs; validation.json; artifact_validation.json; this journal and the plain-language analysis. The bundler adds code snapshots, evidence_manifest.json and a reproducibility ZIP.

## Google Drive archive status

At this pre-package journal snapshot, final copying has not yet run. The intended new research subfolder is G:\My Drive\Eidos_Brain_Proof_Phase\2026-09-04\quantized_memory_math\controlled_memory_2026-09-04_210733. The adjacent drive_manifest.json is generated after packaging and records actual copy success, exact files, checksums and skipped reasons. The final repository journal/archive receipt records the later verified result. This timing distinction prevents a premature success claim and avoids self-referential ZIP hashes.

## Thoughts on improvement

The measured worst state discrepancy fell from 0.000516142 for current pulses to 0.0000285443 for carry, but carry replay was about 1.74–1.75 times current runtime in this small harness. Persistent state arrays increased from 64 to 128 bytes. Prediction MSE changes have both signs. Fidelity and useful detection need separate acceptance gates.

## Where to improve next

Define an operational labeled task, minimum useful effect, acceptable overhead and a chronological utility protocol. Then test a candidate with matched trials and temporal blocks. Treat default-size reservoirs, adaptive feedback and GPU execution as separate follow-ups.

## Anything that stands out

The implemented four bands clip the roadmap's 0.5 to 0.2. The actual small matrix lies outside rho(abs(W))<1, certified by exact row-sum arithmetic; that is not proof of instability. Twenty scalar relaxation times did not remove all coupled initialization separation. The signed carry two-cycle remains a counterexample to universal forgetting. Originality, consciousness, new physics and a new mechanism are not claimed.

## End-of-task summary

1. Files changed: benchmark-only helpers, tests/collection shim, proof dependency list, docs and compact receipts.
2. Core behavior changed: no.
3. Tests added or skipped: 11 targeted passes; 78 package passes and four explained skips; baseline gate failures retained.
4. Repo-root commands run: recorded above with XML/log receipts.
5. Artifacts generated: full frozen numerical and research record, plots, metrics, checksums and bundle.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: final post-package drive_manifest.json and archive receipt are authoritative.
9. Known limitations: eight neurons, CPU, frozen feedback; actual W outside sufficient theorem condition; utility untested; full repo gate failures.
10. Follow-up tasks not implemented: labeled utility protocol, default-size/GPU/adaptive comparison, unrelated packaging fixes.
11. Proof Logic + Meaning written: yes.
12. Math/logic explanation included: recurrence, residual identity, scalar pulse condition, matrix comparison, RMS/MSE and runtime ratios.
13. Philosophical meaning included: reproducibility and distinguishing numerical persistence from useful memory.
14. Better than previous state: actual controlled engine measurements and recoverable evidence replace an untested implementation assumption.
15. North-star contribution: stronger reproducibility and interpretation of internal state; no detection/compression value claim.
16. Evidence files cited: protocol/freeze, raw traces, exact controls, precision checks, plots, XMLs and manifests.
17. Remaining uncertainty: utility, full engine, GPU, true coupled forgetting, operational false positives and originality.

## Post-archive addendum

The Google Drive copy succeeded: 18 files copied with matching mounted-file SHA-256 checksums and zero skips. Connector readback confirms the new research folder, report, and the 195,319,150-byte ZIP. Original research files were preserved. See archive_status.md and drive_manifest.json for verified links, destinations and hashes. This addendum postdates the immutable journal snapshot inside the ZIP.
