# Codex Journal — 2026-09-05

## What happened today
Continued from numerical benchmark commit 85e588238a54a5b3feaf9a675917d95ac28b210d on a new isolated branch. Inspected PR #36, the local labeled dataset, existing calibration rules and available timestamped-source routes.

## What was accomplished
Implemented a strict benchmark-only preparation gate, 21 new tests, source audit, label-leakage controls, archived public input and prospective user runtime budget.

## Tests and commands run
32 targeted checks passed; package 78 passed/4 skipped; root 41 passed/5 existing failures; collection 271/6 existing errors. Preparation blocked as expected and its freeze verified.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m pytest tests/test_memory_benchmark.py tests/test_memory_collection.py tests/test_memory_utility_data.py -q --junitxml=artifacts/controlled_memory_utility_2026_09_05/pytest_results.xml
python -m pytest eidos/repo/tests -q --junitxml=artifacts/controlled_memory_utility_2026_09_05/package_pytest.xml
python -m pytest tests -q --junitxml=artifacts/controlled_memory_utility_2026_09_05/root_pytest.xml
python -m pytest --collect-only -q
python -m proof.memory_utility_data prepare --file artifacts/controlled_memory_utility_2026_09_05/data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --out artifacts/controlled_memory_utility_rerun/data_gate
python -m proof.memory_utility_data verify --out artifacts/controlled_memory_utility_2026_09_05/local_dataset_gate
```

## Problems encountered
Timestamp-free CSV; 11 negative durations; whole-sample normalization; ground-truth-dependent calibration; official download error/form; no matching timestamped Drive file; existing import/collection failures. PR #36 also revealed an automatic Vercel preview triggered by its earlier push. The user now explicitly permits deployment if necessary; follow-up draft delivery may trigger another preview.

## What changed
Added proof/memory_utility_data.py, tests/test_memory_utility_data.py, docs/memory_utility_readiness.md and dated compact evidence. Updated no production source.

## What did not change
Reservoir dynamics, RLS, surprise scores, Sentinel thresholds, calibration behavior, compression, memory, forecasting, domains, production settings and the frozen numerical benchmark are untouched. Original checkout residue is preserved.

## Proof Logic + Meaning

### Goal reached
**Passed:** tested data-preparation and evidence gates. **Blocked:** chronological utility experiment. Adoption remains **inconclusive**.

### Previous state
A large labeled file existed, but chronology was unverified. Whole-sample normalization and label-assisted calibration could be mistaken for a held-out detection experiment.

### Technical logic utilized
The new benchmark-only helper verifies schema and source hashes, rejects invalid timing, fits normalization on an exploration prefix, keeps labels separate and requires explicit completion-time ordering, split and gap. Synthetic negative controls execute the existing normalizer and calibration decisions to expose their information dependencies. Production code is untouched.

### Math / scoring logic
`available_i = start_i + duration_i`; `gap >= max(duration_i)`; `x'_i = clip((x_i-mu_prefix)/sigma_prefix,-3,3)/3`. Candidate/current per-frame runtime must be `<= 1.10` by the user's prospective requirement. Later scoring must use `precision = TP_events/(TP_events+FP_events)`, `recall = distinct_detected_attack_windows/all_attack_windows`, and `FP_per_10k_benign = FP_events/benign_frames*10000`. Legacy FP per total frames must remain separately named. These detection metrics are null here because no detector ran. A cryptographic hash proves byte identity, not data correctness or operational utility.

### Philosophical meaning
Honesty before optimization. Useful memory must be evaluated without exposing the answers to the model. Reproducibility is truth that can be revisited, including inconvenient failures.

### Why this is better
The full-file audit distinguishes data volume from valid chronology, records 11 invalid durations, and the preparation path cannot silently proceed. Twenty-one new tests catch leakage, ordering, malformed data and receipt tampering. This replaces a plausible but untrustworthy experiment with explicit prerequisites.

### How this moves Eidos closer to the north-star goal
Eidos Brain is intended to learn live streams, compress predictable behavior, preserve meaningful anomalies, monitor internal state and explain incidents. This milestone strengthens reproducibility and the credibility of future anomaly-preservation evidence. It does not yet establish useful detection, compression improvement, or a better deployed memory policy.

### Evidence
`local_dataset_gate/dataset_audit.json`, `local_dataset_gate/freeze.json`, `duration_quality.json`, `existing_harness_controls.json`, `utility_requirements.json`, `pytest_results.xml`, `package_pytest.xml`, `validation.json`, source snapshots and archive checksums.

### Remaining uncertainty
Timestamped source provenance and export semantics; treatment of invalid durations; effect magnitude; memory budget; actual label-blind Sentinel integration; chronological utility; default-size runtime; GPU and end-to-end adaptive behavior. The original numerical candidate cost is not an end-to-end operational cost measurement. Six collection errors and five root-suite failures remain pre-existing repository limitations.

## Artifacts generated
Local root: artifacts/controlled_memory_utility_2026_09_05/. Input CSV, data gate, hashes, source controls, requirements, environment, XML/logs, synthetic test fixtures, decision report, progress dashboard, journal, analysis and final reproducibility archive.

## Google Drive archive status
Configured root: G:/My Drive. A new run folder under the existing quantized_memory_math research folder is used. This journal is assembled before copying; drive_manifest.json and archive_status.md are the authoritative completed copy receipts. No original research file is replaced.

## Thoughts on improvement
Keep metadata qualification, label-blind inference and scoring separate. Unknown utility and costs must stay explicit rather than inherit a numerical fidelity pass.

## Where to improve next
Recover timestamped labeled source and document invalid-duration treatment. Freeze an actual Sentinel trial, effect requirement and timing under the 10% ceiling before evaluating carry.

## Anything that stands out
More labeled rows are not necessarily a stronger test. The source file's missing chronology and the oracle-assisted accounting matter more to validity than dataset size.

## End-of-task summary
1. Files changed: helper, tests, guide, dated evidence.
2. Core behavior changed: no.
3. Tests added: 21; targeted 32 passed; package 78 passed/4 explicit skips.
4. Repo-root commands: recorded above and in logs.
5. Artifacts generated: yes, in the local root above.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: final drive_manifest.json and archive_status.md record readback checks.
9. Known limitations: missing timestamps, negative durations, legacy leakage, baseline test failures.
10. Follow-up not implemented: actual utility evaluator, new detector policy, production promotion.
11. Proof Logic + Meaning written: yes.
12. Math/logic explanation: availability, gap, prefix transform, metric denominators, runtime ratio.
13. Philosophical meaning: honesty before optimization.
14. Better than previous state: inappropriate data paths now fail explicitly with reproducible evidence.
15. North-star contribution: reproducibility and credible anomaly-detection proof.
16. Evidence cited: audit, freeze, controls, requirements, XML, logs, archive hashes.
17. Remaining uncertainty: utility, effect threshold, memory cost, default-size/GPU/adaptive behavior.
