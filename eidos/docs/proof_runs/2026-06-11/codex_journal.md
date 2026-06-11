## Labeled CICIDS/WebAttacks proof harness -- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610

### What happened today
Built and ran the event-confirmation layer for the labeled/domain proof harness.

### What was accomplished
- Added proof-side candidate scoring and confirmation modes for labeled-domain events.
- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.
- Captured raw, merged, deduped, and confirmed event metrics side by side.
- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 12 --seed 42 --out artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610 --suite smoke --sample-mode natural --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/benchmark_summary.csv
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/benchmark_summary.md
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/calibrated_precision_ledger.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/calibrated_precision_ledger.md
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/candidate_funnel_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/candidate_funnel_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/config.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/confirmation_profile_sweep.csv
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/confirmation_profile_sweep.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/confirmation_profile_sweep.md
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/crash_scan.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/drive_manifest.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260611_000310_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260611_000310_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000306_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000306_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000306_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000306_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000306_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000306_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000306_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/forecast.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260611_000310_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/incident_cards.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/manifest.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260611_000310_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260611_000310_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260611_000310_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260611_000310_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260611_000310_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_reopen_gate.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/engine_reopen_gate.md
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/environment.txt
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/event_confirmation_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/event_confirmation_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/event_summary.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/git_commit.txt
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/incident_cards/engine_card_001.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/labeled_metrics.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/labeled_metrics.md
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/logs/engine_output.log
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/precision_ledger.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/precision_ledger.md
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/proof_digest.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/proof_digest.md
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/run_manifest.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/sentinel_calibration_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/sentinel_calibration_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/sentinel_calibration_v1.json
- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\cicids_webattacks_proof_smoke_seed42_frames11_20260611T000311Z
- Files copied: 49
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 12 --seed 42 --out artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610 --suite smoke --sample-mode natural --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force"`.
5. Artifacts generated: 50 files under `artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.

## Aggregate engine reopen readiness gate -- 2026-06-11

### What happened today
Completed the gated Sentinel Calibration v1 + Engine Reopen Readiness proof pass on the new branch. The work stayed calibration/reporting/test/documentation scoped and did not reopen core engine behavior.

### What was accomplished
- Added `--sentinel-calibration-mode off|low_noise|balanced|high_recall`.
- Added raw, merged, deduped, and calibrated views side by side in `precision_ledger.json` and `precision_ledger.md`.
- Added `sentinel_calibration_report.json/.md` and `engine_reopen_gate.json/.md` to each proof run.
- Added `tools/check_core_touch_policy.py` and tests proving allowed calibration/reporting paths pass while core engine paths fail.
- Ran the required CPU proof matrix and mirrored all run artifacts to Google Drive.

### Tests and commands run
- `python -m py_compile tools/run_labeled_domain_proof.py tools/check_core_touch_policy.py sentinel/calibration.py sentinel/hysteresis.py sentinel/normal_suppression.py proof/event_confirmation.py tools/build_calibration_recall_diagnostics.py` -> passed.
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_core_touch_policy.py tests/test_sentinel_event_confirmation.py tests/test_labeled_domain_proof_runner.py -q` -> `36 passed`.
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/test_labeled_domain_proof_runner.py -q` -> `28 passed`.
- `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest -q` -> `106 passed, 1 skipped, 11 warnings`.
- `python tools/check_core_touch_policy.py --base main --json-out artifacts/proof_runs/2026-06-10/engine_reopen_core_touch_policy_precommit.json --md-out artifacts/proof_runs/2026-06-10/engine_reopen_core_touch_policy_precommit.md` -> passed.

### Proof matrix summary
| run | mode | frames | raw FP/10k | deduped FP/10k | calibrated FP/10k | raw recall | calibrated recall | calibrated F1 | crash hits | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tiny smoke | balanced | 11 | 909.090909 | 909.090909 | 0.0 | 0.0 | 0.0 | NA | 0 | CALIBRATION_ONLY |
| balanced250 | balanced | 225 | 222.222222 | 0.0 | 0.0 | 0.259259 | 0.166667 | 0.285714 | 0 | CALIBRATION_ONLY_NEEDS_TUNING |
| transition1k | balanced | 900 | 188.888889 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0 | CALIBRATION_ONLY_NEEDS_TUNING |
| balanced250 | low_noise | 225 | 222.222222 | 0.0 | 0.0 | 0.259259 | 0.166667 | 0.285714 | 0 | CALIBRATION_ONLY_NEEDS_TUNING |
| balanced250 | high_recall | 225 | 222.222222 | 0.0 | 0.0 | 0.259259 | 0.166667 | 0.285714 | 0 | CALIBRATION_ONLY_NEEDS_TUNING |

### Engine gate verdict
Aggregate verdict: `CALIBRATION_ONLY_NEEDS_TUNING`.

Reason: crash scans were clean, calibrated FP/10k reached `0.0` on the required CPU legs, raw evidence stayed visible, and Drive copies succeeded. However, the balanced 250-row legs still show calibrated recall below raw recall (`0.166667` vs `0.259259`), so this does not justify narrow core experiments yet.

### What changed
Code, tests, and docs changed. Core behavior did not change.

### What did not change
Reservoir dynamics, RLS equations, hippocampus write/freeze behavior, compression codec/ratio accounting, active thermodynamics, prediction semantics, and default `spectral_radius`/`leak`/`forgetting`/`weight_decay` were untouched.

### Artifacts generated
- `artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610`
- `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610`
- `artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610`
- `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610`
- `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610`
- `artifacts/proof_runs/2026-06-10/engine_reopen_core_touch_policy_precommit.json`
- `artifacts/proof_runs/2026-06-10/engine_reopen_core_touch_policy_precommit.md`

### Google Drive archive status
Drive copy succeeded for all five proof runs under `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\...`.

### End-of-task summary
1. Files changed: calibration helpers, proof event confirmation, labeled proof runner, core-touch checker, tests, and proof docs.
2. Whether core behavior changed: no.
3. Tests added or skipped: tests added for core-touch policy, calibration mode alias, isolated spike rejection, stable normal suppression, and high-recall vs low-noise confirmation.
4. Repo-root commands run: py_compile, focused pytest, labeled-runner pytest, full pytest, core-touch policy, and five proof runs.
5. Artifacts generated: yes, locally and in Drive.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: succeeded for proof runs.
9. Known limitations: balanced 250-row recall remains below raw recall; optional GPU 10k was not run in this CPU pass.
10. Follow-up tasks not implemented: no core experiments, no threshold rewrite, no model behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610

### What happened today
Built and ran the event-confirmation layer for the labeled/domain proof harness.

### What was accomplished
- Added proof-side candidate scoring and confirmation modes for labeled-domain events.
- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.
- Captured raw, merged, deduped, and confirmed event metrics side by side.
- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610 --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/benchmark_summary.csv
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/benchmark_summary.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/calibrated_precision_ledger.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/calibrated_precision_ledger.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/candidate_funnel_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/candidate_funnel_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/config.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/confirmation_profile_sweep.csv
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/confirmation_profile_sweep.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/confirmation_profile_sweep.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/crash_scan.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/drive_manifest.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260611_000617_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260611_000617_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000535_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000535_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000535_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000535_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000535_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000535_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000535_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/forecast.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260611_000616_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/incident_cards.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/manifest.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260611_000616_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260611_000616_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260611_000616_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260611_000616_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260611_000617_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_reopen_gate.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/engine_reopen_gate.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/environment.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/event_confirmation_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/event_confirmation_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/event_summary.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/git_commit.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/confirmed_event_001.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/confirmed_event_002.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/confirmed_event_003.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/engine_card_001.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/engine_card_002.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/engine_card_003.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/engine_card_004.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/engine_card_005.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/engine_card_006.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/engine_card_007.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/engine_card_008.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/incident_cards/engine_card_009.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/labeled_metrics.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/labeled_metrics.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/logs/engine_output.log
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/precision_ledger.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/precision_ledger.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/proof_digest.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/proof_digest.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/run_manifest.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/sentinel_calibration_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/sentinel_calibration_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/sentinel_calibration_v1.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\cicids_webattacks_proof_full_seed42_frames225_20260611T000619Z
- Files copied: 60
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610 --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 61 files under `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610

### What happened today
Built and ran the event-confirmation layer for the labeled/domain proof harness.

### What was accomplished
- Added proof-side candidate scoring and confirmation modes for labeled-domain events.
- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.
- Captured raw, merged, deduped, and confirmed event metrics side by side.
- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610 --suite full --sample-mode transition --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/benchmark_summary.csv
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/benchmark_summary.md
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/calibrated_precision_ledger.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/calibrated_precision_ledger.md
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/candidate_funnel_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/candidate_funnel_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/config.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/confirmation_profile_sweep.csv
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/confirmation_profile_sweep.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/confirmation_profile_sweep.md
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/crash_scan.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/drive_manifest.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260611_001303_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260611_001303_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000917_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000917_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000917_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000917_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000917_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000917_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/eidos_brain_archive/20260611_000917_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/forecast.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260611_001303_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/incident_cards.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/manifest.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260611_001303_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260611_001303_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260611_001303_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260611_001303_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260611_001303_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_reopen_gate.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/engine_reopen_gate.md
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/environment.txt
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/event_confirmation_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/event_confirmation_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/event_summary.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/git_commit.txt
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/confirmed_event_001.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_001.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_002.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_003.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_004.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_005.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_006.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_007.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_008.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_009.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_010.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_011.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_012.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_013.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_014.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_015.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_016.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_017.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_018.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_019.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_020.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_021.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_022.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_023.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_024.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_025.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_026.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_027.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_028.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_029.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_030.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_031.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_032.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/incident_cards/engine_card_033.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/labeled_metrics.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/labeled_metrics.md
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/logs/engine_output.log
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/precision_ledger.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/precision_ledger.md
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/proof_digest.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/proof_digest.md
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/run_manifest.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/sentinel_calibration_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/sentinel_calibration_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/sentinel_calibration_v1.json
- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\cicids_webattacks_proof_full_seed42_frames900_20260611T001306Z
- Files copied: 82
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610 --suite full --sample-mode transition --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 83 files under `artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610

### What happened today
Built and ran the event-confirmation layer for the labeled/domain proof harness.

### What was accomplished
- Added proof-side candidate scoring and confirmation modes for labeled-domain events.
- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.
- Captured raw, merged, deduped, and confirmed event metrics side by side.
- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610 --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode low_noise --sentinel-calibration-mode low_noise --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/benchmark_summary.csv
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/benchmark_summary.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/calibrated_precision_ledger.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/calibrated_precision_ledger.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/candidate_funnel_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/candidate_funnel_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/config.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/confirmation_profile_sweep.csv
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/confirmation_profile_sweep.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/confirmation_profile_sweep.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/crash_scan.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/drive_manifest.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260611_001651_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260611_001651_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/eidos_brain_archive/20260611_001540_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/eidos_brain_archive/20260611_001540_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/eidos_brain_archive/20260611_001540_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/eidos_brain_archive/20260611_001540_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/eidos_brain_archive/20260611_001540_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/eidos_brain_archive/20260611_001540_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/eidos_brain_archive/20260611_001540_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/forecast.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260611_001651_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/incident_cards.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/manifest.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260611_001651_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260611_001651_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260611_001651_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260611_001651_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260611_001651_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_reopen_gate.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/engine_reopen_gate.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/environment.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/event_confirmation_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/event_confirmation_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/event_summary.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/git_commit.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/confirmed_event_001.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/confirmed_event_002.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/confirmed_event_003.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/engine_card_001.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/engine_card_002.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/engine_card_003.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/engine_card_004.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/engine_card_005.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/engine_card_006.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/engine_card_007.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/engine_card_008.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/incident_cards/engine_card_009.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/labeled_metrics.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/labeled_metrics.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/logs/engine_output.log
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/precision_ledger.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/precision_ledger.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/proof_digest.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/proof_digest.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/run_manifest.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/sentinel_calibration_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/sentinel_calibration_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/sentinel_calibration_v1.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\cicids_webattacks_proof_full_seed42_frames225_20260611T001654Z
- Files copied: 60
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610 --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode low_noise --sentinel-calibration-mode low_noise --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 61 files under `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610

### What happened today
Built and ran the event-confirmation layer for the labeled/domain proof harness.

### What was accomplished
- Added proof-side candidate scoring and confirmation modes for labeled-domain events.
- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.
- Captured raw, merged, deduped, and confirmed event metrics side by side.
- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610 --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode high_recall --sentinel-calibration-mode high_recall --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/benchmark_summary.csv
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/benchmark_summary.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/calibrated_precision_ledger.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/calibrated_precision_ledger.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/candidate_funnel_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/candidate_funnel_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/config.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/confirmation_profile_sweep.csv
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/confirmation_profile_sweep.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/confirmation_profile_sweep.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/crash_scan.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/drive_manifest.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260611_002040_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260611_002040_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/eidos_brain_archive/20260611_001939_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/eidos_brain_archive/20260611_001939_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/eidos_brain_archive/20260611_001939_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/eidos_brain_archive/20260611_001939_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/eidos_brain_archive/20260611_001939_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/eidos_brain_archive/20260611_001939_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/eidos_brain_archive/20260611_001939_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/forecast.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260611_002039_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/incident_cards.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/manifest.jsonl
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260611_002039_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260611_002039_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260611_002039_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260611_002039_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260611_002040_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_reopen_gate.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/engine_reopen_gate.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/environment.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/event_confirmation_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/event_confirmation_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/event_summary.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/git_commit.txt
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/confirmed_event_001.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/confirmed_event_002.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/confirmed_event_003.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/engine_card_001.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/engine_card_002.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/engine_card_003.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/engine_card_004.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/engine_card_005.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/engine_card_006.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/engine_card_007.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/engine_card_008.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/incident_cards/engine_card_009.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/labeled_metrics.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/labeled_metrics.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/logs/engine_output.log
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/precision_ledger.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/precision_ledger.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/proof_digest.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/proof_digest.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/run_manifest.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/sentinel_calibration_report.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/sentinel_calibration_report.md
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/sentinel_calibration_v1.json
- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\cicids_webattacks_proof_full_seed42_frames225_20260611T002041Z
- Files copied: 60
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610 --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode high_recall --sentinel-calibration-mode high_recall --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 61 files under `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.
