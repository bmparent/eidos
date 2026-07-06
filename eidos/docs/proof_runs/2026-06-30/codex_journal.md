# Codex Journal - 2026-06-30

## What happened today
Stabilized the post-merge Sentinel calibration guardrail package and reran the feasible proof legs from the Eidos command root.

## What was accomplished
- Final verdict remains `CALIBRATION_ONLY_NEEDS_TUNING`.
- Tiny fixture smoke, natural attack-window replay, and normal-only negative control were measured with raw/merged/deduped/confirmed/calibrated views preserved.
- Dataset availability is now a machine-readable receipt instead of an assumption.
- Drive copying now uses an allowlist-first summary/per-run receipt strategy instead of blind recursion.

## Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 8 --seed 42 --out artifacts/sentinel_calibration_guardrails_2026_06_30/runs/tiny_fixture_smoke/off --suite smoke --sample-mode transition --event-merge-gap 25 --sentinel-calibration-mode off --confirmation-profile-sweep low_noise balanced high_recall --attack-labels "Web Attack - Brute Force"` -> returncode `0`
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 12 --seed 42 --out artifacts/sentinel_calibration_guardrails_2026_06_30/runs/natural_attack_replay_cpu/off --suite smoke --sample-mode natural_attack_windows --event-merge-gap 25 --sentinel-calibration-mode off --confirmation-profile-sweep low_noise balanced high_recall --attack-labels "Web Attack - Brute Force" --natural-window-pre 2 --natural-window-post 2 --natural-window-max-windows 1` -> returncode `0`
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/sentinel_calibration_guardrails_2026_06_30/generated/normal_only_negative_control.csv --label-column Label --frames 8 --seed 42 --out artifacts/sentinel_calibration_guardrails_2026_06_30/runs/normal_only_negative_control/off --suite smoke --sample-mode natural --event-merge-gap 25 --sentinel-calibration-mode off --confirmation-profile-sweep low_noise balanced high_recall --attack-labels "Web Attack - Brute Force"` -> returncode `0`
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_sentinel_calibration_guardrails.py -q` -> `5 passed in 6.05s`
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_proof_baseline_runner.py -q` -> `11 passed in 8.11s`
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_labeled_domain_proof_runner.py -q` -> `28 passed in 17.10s`
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_core_touch_policy.py -q` -> `3 passed in 19.94s`
- `python tools/check_core_touch_policy.py --base origin/main` -> `passed: true`
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest -q` -> `119 passed, 1 skipped, 11 warnings in 231.77s (0:03:51)`

## Problems encountered
- Dataset rows available: `12` total, `8` benign, `4` attack.
- Balanced 250 and transition 1k remain infeasible without a larger labeled CSV.
- GPU 10k remains skipped because CUDA availability is `False`.
- Drive copy status: `copied`; reason: allowlist copy completed.

## What changed
- Proof-side guardrail builder, proof-run generated-prefix hygiene, crash-scan metadata ignore coverage, docs, and tests.

## What did not change
Core behavior did not change: reservoir dynamics, RLS, Sentinel thresholds, anomaly policy, compression behavior, hippocampus/memory behavior, incident-card generation, forecasting, procedural memory, and domain adapter math stayed untouched.

## Proof Logic + Meaning
Goal reached: `CALIBRATION_ONLY_NEEDS_TUNING` is supported by explicit receipts. The package proves useful calibration guardrail evidence, but not enough scale to reopen core behavior.

Previous state: the branch had useful proof artifacts, but the dataset size, normal-only stall, and Drive partial-copy status were not fully stabilized.

Technical logic utilized: the harness calls the existing labeled proof runner, compares raw, merged, deduped, confirmed, calibrated, attack-window, false-positive, incident-card, crash-scan, Git, and Drive receipts side by side, and keeps calibration as postprocessing proof logic only.

Math/scoring logic:

```text
precision = true_positive_events / (true_positive_events + false_positive_events)
recall = detected_attack_windows / total_attack_windows
FP_per_10k = false_positive_events / benign_frames * 10000
precision_lift = calibrated_precision - raw_precision
```

Philosophical meaning: a system that only speaks less is not necessarily wiser. A system that speaks less while preserving the truth is becoming more intelligent.

Why this is better: the branch now separates missing data, skipped scale legs, normal-only behavior, crash cleanliness, and Drive persistence instead of blending them into one ambiguous verdict.

How this moves Eidos closer to the north-star goal: it strengthens reproducibility, anomaly preservation, internal monitoring, and human-readable proof receipts for the claim that Eidos Brain is a self-monitoring streaming intelligence codec.

Evidence:
- artifacts/sentinel_calibration_guardrails_2026_06_30/calibration_guardrail_matrix.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/calibration_guardrail_matrix.md
- artifacts/sentinel_calibration_guardrails_2026_06_30/profile_comparison.csv
- artifacts/sentinel_calibration_guardrails_2026_06_30/attack_window_guardrails.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/false_positive_guardrails.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/normal_only_guardrails.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/dataset_availability_receipt.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/core_touch_policy.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/proof_logic_meaning.md
- artifacts/sentinel_calibration_guardrails_2026_06_30/drive_manifest.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/runs/natural_attack_replay_cpu/off: run manifest, environment, precision ledger, proof digest, crash scan, and Drive manifest
- artifacts/sentinel_calibration_guardrails_2026_06_30/runs/normal_only_negative_control/off: run manifest, environment, precision ledger, proof digest, crash scan, and Drive manifest
- artifacts/sentinel_calibration_guardrails_2026_06_30/runs/tiny_fixture_smoke/off: run manifest, environment, precision ledger, proof digest, crash scan, and Drive manifest

Remaining uncertainty: larger CICIDS/WebAttacks scale, GPU 10k, broader domain performance, and production readiness remain unproven.

## Artifacts generated
- artifacts/sentinel_calibration_guardrails_2026_06_30/calibration_guardrail_matrix.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/calibration_guardrail_matrix.md
- artifacts/sentinel_calibration_guardrails_2026_06_30/profile_comparison.csv
- artifacts/sentinel_calibration_guardrails_2026_06_30/attack_window_guardrails.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/false_positive_guardrails.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/normal_only_guardrails.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/dataset_availability_receipt.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/core_touch_policy.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/proof_logic_meaning.md
- artifacts/sentinel_calibration_guardrails_2026_06_30/drive_manifest.json
- artifacts/sentinel_calibration_guardrails_2026_06_30/runs/natural_attack_replay_cpu/off: run manifest, environment, precision ledger, proof digest, crash scan, and Drive manifest
- artifacts/sentinel_calibration_guardrails_2026_06_30/runs/normal_only_negative_control/off: run manifest, environment, precision ledger, proof digest, crash scan, and Drive manifest
- artifacts/sentinel_calibration_guardrails_2026_06_30/runs/tiny_fixture_smoke/off: run manifest, environment, precision ledger, proof digest, crash scan, and Drive manifest

## Google Drive archive status
- Drive root used: `G:\My Drive`
- Drive folder used: `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-30\sentinel_calibration_guardrails_2026_06_30`
- Copy status: `copied`
- Files copied: `36`
- Files skipped intentionally: `10`
- Failed entries: `0`
- Reason: allowlist copy completed

## Thoughts on improvement
The next proof improvement is not tuning. It is providing a larger labeled CICIDS/WebAttacks sample and rerunning the same matrix without changing core behavior.

## Where to improve next
Run balanced 250 CPU and transition 1k CPU against an explicit larger dataset path; run optional GPU 10k only in a CUDA environment.

## Anything that stands out
Normal-only status is `completed`. The previous stall was not reproduced after child Drive mirroring was disabled.

## End-of-task summary
1. Files changed: proof-side tool, proof-runner generated-prefix hygiene, crash-scan regression test, guardrail tests, proof docs.
2. Whether core behavior changed: no.
3. Tests added or skipped: guardrail unit tests added; scale proof legs skipped only when data/CUDA was unavailable.
4. Repo-root commands run: proof matrix, focused pytest commands, core-touch policy command, and full pytest are recorded above.
5. Artifacts generated: receipts under `artifacts/sentinel_calibration_guardrails_2026_06_30`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: `copied`.
9. Known limitations: larger CPU/GPU evidence remains unproven.
10. Follow-up tasks not implemented: no larger dataset was invented or downloaded; no GPU run was forced.
11. Proof Logic + Meaning written: yes.
12. Math/logic explanation included: yes.
13. Philosophical meaning included: yes.
14. Why this is better than previous state: the proof package now distinguishes restraint from blindness with explicit receipts.
15. How this moves Eidos closer to the ultimate goal: it improves reproducible proof and operator trust without touching core behavior.
16. Evidence files cited: matrix JSON/MD, profile CSV, dataset receipt, normal-only receipt, core-touch policy, per-run proof receipts, Drive manifest.
17. Remaining uncertainty / unproven claims: scale, CUDA, broader domains, and production readiness.
