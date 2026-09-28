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
