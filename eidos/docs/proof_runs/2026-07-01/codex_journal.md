# Codex Journal - 2026-07-01

## What happened today

Built a larger labeled Sentinel guardrail scale-matrix harness and resumed it conservatively after long CPU proof legs.

## What was accomplished

- Preserved the previous guardrail branch before creating the scale branch.
- Added dataset discovery and registry receipts for CICIDS/WebAttacks CSV sources.
- Added a bounded normal-only negative control so CPU smoke proof remains finite.
- Reused completed proof receipts and recorded partial receipts instead of silently rerunning or claiming success.

## Tests and commands run

- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 8 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off --suite smoke --sample-mode transition --event-merge-gap 25 --confirmation-mode off --sentinel-calibration-mode off --confirmation-profile-sweep low_noise --confirmation-profile-sweep balanced --confirmation-profile-sweep high_recall --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> returncode `0`, reused `True`, timed_out `False`
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\cicids_webattacks_samples\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode off --sentinel-calibration-mode off --confirmation-profile-sweep low_noise --confirmation-profile-sweep balanced --confirmation-profile-sweep high_recall --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> returncode `0`, reused `True`, timed_out `False`
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\cicids_webattacks_samples\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off --suite full --sample-mode transition --event-merge-gap 25 --confirmation-mode off --sentinel-calibration-mode off --confirmation-profile-sweep low_noise --confirmation-profile-sweep balanced --confirmation-profile-sweep high_recall --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> returncode `0`, reused `True`, timed_out `False`
- `existing partial proof receipts reused as incomplete` -> returncode `None`, reused `True`, timed_out `True`
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/sentinel_guardrail_scale_matrix_2026_07_01/generated/normal_only_negative_control.csv --label-column Label --frames 1000 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off --suite full --sample-mode natural --event-merge-gap 25 --sentinel-calibration-mode off --confirmation-profile-sweep low_noise balanced high_recall --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> returncode `0`, reused `False`, timed_out `False`

## Problems encountered

- `natural_larger_replay_cpu`: existing partial run lacks run_manifest.json or labeled_metrics.json; rerun skipped because --rerun-existing was not set
- `gpu_10k_optional`: CUDA unavailable; torch reports CPU-only runtime

## What changed

Proof-harness and reporting code changed. Generated artifact folders remain ignored by git.

## What did not change

Core model behavior did not change: reservoir dynamics, RLS updates, Sentinel anomaly policy, thresholds, compression behavior, and memory/familiarity behavior were untouched.

## Proof Logic + Meaning

Goal reached: larger labeled Sentinel guardrail scale packaging is `CALIBRATION_ONLY_NEEDS_TUNING`. The package discovered and registered a larger CICIDS/WebAttacks CSV, reused completed CPU proof receipts when present, and kept incomplete legs explicit.

Previous state: the earlier guardrail proof was tiny-fixture-only. It could prove the runner shape, but not whether a larger real labeled CSV existed or how far the proof harness could scale on CPU.

Technical logic utilized: the builder inspects label distributions, resolves CICIDS leading-space label headers, runs or reuses the existing labeled-domain proof runner, preserves raw/merged/deduped/confirmed/calibrated event views, and evaluates crash/core-touch receipts without changing reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, or hippocampus/familiarity behavior.

Math / scoring logic:

```text
FP/10k = false_positive_events / benign_frames * 10000
precision = true_positive_events / max(true_positive_events + false_positive_events, 1)
recall = detected_attack_windows / max(total_attack_windows, 1)
F1 = 2 * precision * recall / max(precision + recall, epsilon)
attack_window_coverage = attack_windows_with_detection / total_attack_windows
```

Philosophical meaning: Sentinel calibration is restraint before alarm, and this matrix is honesty before scale. It records what ran, what timed out, and what remains unproven.

Why this is better: the proof trail now includes a dataset registry, larger-run receipts, explicit skip reasons, a bounded negative control, and Drive status instead of relying on a tiny fixture alone.

How this moves Eidos closer to the north-star goal: Eidos Brain is a self-monitoring streaming intelligence codec. This milestone strengthens the reproducibility and incident-explanation side of that claim by making anomaly preservation and uncertainty visible on larger labeled data.

Evidence:

- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.json`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.csv`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.md`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/proof_logic_meaning.md`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/drive_manifest.json`
- `docs/proof/dataset_registry.json`
- `docs/proof/dataset_registry.md`
- `docs/proof_runs/2026-07-01/sentinel_guardrail_scale_matrix.md`

Remaining uncertainty: the natural larger replay did not produce a completed manifest in the resumed package, optional GPU proof was skipped when CUDA was unavailable, and the result does not prove production readiness or universal CICIDS coverage.

## Artifacts generated

- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.json`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.csv`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/scale_matrix.md`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/proof_logic_meaning.md`
- `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/drive_manifest.json`
- `docs/proof/dataset_registry.json`
- `docs/proof/dataset_registry.md`
- `docs/proof_runs/2026-07-01/sentinel_guardrail_scale_matrix.md`

## Google Drive archive status

- Copy attempted: `True`
- Copy success: `True`
- Drive root: `G:\My Drive`
- Drive run dir: `G:\My Drive\Eidos_Brain_Proof_Phase\2026-07-01\sentinel_guardrail_scale_matrix_2026_07_01`
- Reason/status: `copied allowlisted scale matrix receipts`

## Thoughts on improvement

The next proof-sized improvement is to optimize or window the natural replay so it can complete on CPU, then compare profiles without hiding raw false positives.

## Where to improve next

Add a smaller natural attack-window replay mode or a checkpoint/resume path for long natural-order CICIDS runs.

## Anything that stands out

The larger CSV is available, but full natural replay is expensive enough that the proof gate should remain conservative.

## End-of-task summary

1. Files changed: proof-harness script, tests, dataset registry/docs, and proof-run reports.
2. Whether core behavior changed: no.
3. Tests added or skipped: scale-matrix tests added; optional GPU leg skipped if CUDA unavailable.
4. Repo-root commands run: see command list above.
5. Artifacts generated: `artifacts/sentinel_guardrail_scale_matrix_2026_07_01`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: recorded in drive manifest.
9. Known limitations: natural replay completion and GPU proof remain unproven.
10. Follow-up tasks not implemented: CPU natural replay optimization/checkpointing.
11. Proof Logic + Meaning written: yes.
12. Math/logic explanation included: yes.
13. Philosophical meaning included: yes.
14. Why this is better than previous state: larger dataset registry and bounded scale receipts replace tiny-only evidence.
15. How this moves Eidos closer to the ultimate goal: it strengthens reproducible self-monitoring proof.
16. Evidence files cited: see evidence list above.
17. Remaining uncertainty / unproven claims: natural full replay, CUDA/GPU, and production readiness.
