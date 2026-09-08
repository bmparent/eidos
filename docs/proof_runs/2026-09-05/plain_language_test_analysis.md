# Memory utility readiness — 2026-09-05

The next-step data gate is implemented and tested. The operational utility experiment is **blocked**, so the decision remains **inconclusive** and current rounding defaults remain unchanged.

The available CSV contains **170,366 flows: 168,186 benign and 2,180 attack-labeled** (1,507 Brute Force, 652 XSS, 21 SQL Injection). It has **no timestamp column** and **11 negative durations**, all `-1` microsecond on benign rows. Those rows were preserved, not repaired or silently removed. Source SHA-256: `d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d`. The public feature CSV is included in the local/Drive bundle; the preparation receipt itself saves metadata, not raw rows.

The existing proof runner normalizes the whole selected sample. Its optional calibration uses true attack windows to retain or suppress events. Executed controls confirm both dependencies. These are limitations for unseen utility, not evidence that the historical accounting failed its documented purpose. No prior report or production behavior was changed.

The user specified a **10% maximum runtime overhead** for the future trial. `utility_requirements.json` records this after the first immutable preparation snapshot; that earlier snapshot remains byte-identical. A smallest useful operational improvement is still unspecified. No candidate results were inspected or detector outcomes produced in this follow-up.

The [official CICIDS page](https://www.unb.ca/cic/datasets/ids-2017.html) distinguishes labeled-flow and machine-learning CSV archives. Its linked [download page](https://cicresearch.ca/CICDataset/CIC-IDS-2017/) showed a server error and registration form; no personal data was submitted. Targeted local/Drive searches did not recover a timestamped source. This does not prove that no public mirror exists.

## Validation

- Memory/data checks: **32 passed**, including **21 new tests**.
- Package suite: **78 passed, 4 skipped** (ffprobe unavailable; three missing-dependency controls skipped because those dependencies are installed).
- Root tests: **41 passed, 5 failed**, the same `operator_explanation` import failures recorded on the clean baseline.
- Full collection: **271 collected, 6 errors**, the same UTF-16 doctest and unavailable service-package errors as before.
- Real dataset preparation: expected **exit 2 / blocked**; receipt integrity check passed.
- Core model and the frozen September 4 numerical evaluator: unchanged.

Commands below are repository-root commands. Actual logs retain `--basetemp` fixture destinations under the run folder; use new artifact directories for reruns.

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
