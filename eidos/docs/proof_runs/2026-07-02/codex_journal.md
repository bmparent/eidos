## Labeled CICIDS/WebAttacks proof harness -- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off

### What happened today
Built and ran the event-confirmation layer for the labeled/domain proof harness.

### What was accomplished
- Added proof-side candidate scoring and confirmation modes for labeled-domain events.
- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.
- Captured raw, merged, deduped, and confirmed event metrics side by side.
- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 8 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off --suite smoke --sample-mode transition --event-merge-gap 25 --confirmation-mode off --sentinel-calibration-mode off --confirmation-profile-sweep low_noise --confirmation-profile-sweep balanced --confirmation-profile-sweep high_recall --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: skipped or failed; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/benchmark_summary.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/benchmark_summary.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/calibrated_precision_ledger.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/calibrated_precision_ledger.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/candidate_funnel_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/candidate_funnel_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/config.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/confirmation_profile_sweep.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/confirmation_profile_sweep.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/confirmation_profile_sweep.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/crash_scan.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/drive_manifest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260702_010610_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260702_010610_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/eidos_brain_archive/20260702_010606_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/eidos_brain_archive/20260702_010606_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/eidos_brain_archive/20260702_010606_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/eidos_brain_archive/20260702_010606_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/eidos_brain_archive/20260702_010606_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/forecast.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260702_010610_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/manifest.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260702_010609_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_reopen_gate.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_reopen_gate.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/environment.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/event_confirmation_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/event_confirmation_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/event_summary.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/git_commit.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/incident_cards/README.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/labeled_metrics.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/labeled_metrics.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/logs/engine_output.log
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/precision_ledger.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/precision_ledger.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/proof_digest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/proof_digest.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/run_manifest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/sentinel_calibration_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/sentinel_calibration_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/sentinel_calibration_v1.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: unknown
- Drive folder used: unknown
- Files copied: 0
- Files skipped: 0
- Reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount

### End-of-task summary
1. Files changed: proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 8 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off --suite smoke --sample-mode transition --event-merge-gap 25 --confirmation-mode off --sentinel-calibration-mode off --confirmation-profile-sweep low_noise --confirmation-profile-sweep balanced --confirmation-profile-sweep high_recall --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 43 files under `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: skipped or failed; no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off

### What happened today
Built and ran the event-confirmation layer for the labeled/domain proof harness.

### What was accomplished
- Added proof-side candidate scoring and confirmation modes for labeled-domain events.
- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.
- Captured raw, merged, deduped, and confirmed event metrics side by side.
- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\cicids_webattacks_samples\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode off --sentinel-calibration-mode off --confirmation-profile-sweep low_noise --confirmation-profile-sweep balanced --confirmation-profile-sweep high_recall --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: skipped or failed; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/benchmark_summary.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/benchmark_summary.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/calibrated_precision_ledger.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/calibrated_precision_ledger.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/candidate_funnel_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/candidate_funnel_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/config.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/confirmation_profile_sweep.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/confirmation_profile_sweep.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/confirmation_profile_sweep.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/crash_scan.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/drive_manifest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260702_011259_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260702_011259_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011018_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011018_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011018_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011018_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011018_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011018_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011018_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/forecast.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260702_011258_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/incident_cards.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/manifest.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260702_011258_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260702_011258_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260702_011259_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260702_011259_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260702_011259_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_reopen_gate.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_reopen_gate.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/environment.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/event_confirmation_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/event_confirmation_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/event_summary.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/git_commit.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/confirmed_event_001.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/confirmed_event_002.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/confirmed_event_003.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/engine_card_001.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/engine_card_002.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/engine_card_003.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/engine_card_004.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/engine_card_005.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/engine_card_006.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/engine_card_007.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/engine_card_008.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/incident_cards/engine_card_009.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/labeled_metrics.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/labeled_metrics.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/logs/engine_output.log
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/precision_ledger.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/precision_ledger.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/proof_digest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/proof_digest.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/run_manifest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/sentinel_calibration_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/sentinel_calibration_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/sentinel_calibration_v1.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: unknown
- Drive folder used: unknown
- Files copied: 0
- Files skipped: 0
- Reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount

### End-of-task summary
1. Files changed: proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\cicids_webattacks_samples\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode off --sentinel-calibration-mode off --confirmation-profile-sweep low_noise --confirmation-profile-sweep balanced --confirmation-profile-sweep high_recall --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 61 files under `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: skipped or failed; no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off

### What happened today
Built and ran the event-confirmation layer for the labeled/domain proof harness.

### What was accomplished
- Added proof-side candidate scoring and confirmation modes for labeled-domain events.
- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.
- Captured raw, merged, deduped, and confirmed event metrics side by side.
- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\cicids_webattacks_samples\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off --suite full --sample-mode transition --event-merge-gap 25 --confirmation-mode off --sentinel-calibration-mode off --confirmation-profile-sweep low_noise --confirmation-profile-sweep balanced --confirmation-profile-sweep high_recall --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: skipped or failed; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/benchmark_summary.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/benchmark_summary.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/calibrated_precision_ledger.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/calibrated_precision_ledger.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/candidate_funnel_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/candidate_funnel_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/config.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/confirmation_profile_sweep.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/confirmation_profile_sweep.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/confirmation_profile_sweep.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/crash_scan.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/drive_manifest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260702_013008_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260702_013008_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011728_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011728_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011728_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011728_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011728_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011728_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011728_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/forecast.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260702_013007_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/incident_cards.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/manifest.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260702_013006_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260702_013007_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260702_013008_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260702_013008_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260702_013008_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_reopen_gate.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_reopen_gate.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/environment.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/event_confirmation_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/event_confirmation_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/event_summary.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/git_commit.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/confirmed_event_001.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_001.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_002.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_003.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_004.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_005.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_006.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_007.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_008.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_009.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_010.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_011.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_012.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_013.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_014.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_015.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_016.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_017.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_018.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_019.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_020.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_021.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_022.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_023.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_024.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_025.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_026.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_027.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_028.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_029.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_030.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_031.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_032.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/incident_cards/engine_card_033.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/labeled_metrics.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/labeled_metrics.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/logs/engine_output.log
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/precision_ledger.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/precision_ledger.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/proof_digest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/proof_digest.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/run_manifest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/sentinel_calibration_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/sentinel_calibration_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/sentinel_calibration_v1.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: unknown
- Drive folder used: unknown
- Files copied: 0
- Files skipped: 0
- Reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount

### End-of-task summary
1. Files changed: proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\cicids_webattacks_samples\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off --suite full --sample-mode transition --event-merge-gap 25 --confirmation-mode off --sentinel-calibration-mode off --confirmation-profile-sweep low_noise --confirmation-profile-sweep balanced --confirmation-profile-sweep high_recall --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 83 files under `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: skipped or failed; no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off

### What happened today
Built and ran the event-confirmation layer for the labeled/domain proof harness.

### What was accomplished
- Added proof-side candidate scoring and confirmation modes for labeled-domain events.
- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.
- Captured raw, merged, deduped, and confirmed event metrics side by side.
- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/sentinel_guardrail_scale_matrix_2026_07_01/generated/normal_only_negative_control.csv --label-column Label --frames 1000 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off --suite full --sample-mode natural --event-merge-gap 25 --confirmation-mode off --sentinel-calibration-mode off --confirmation-profile-sweep low_noise --confirmation-profile-sweep balanced --confirmation-profile-sweep high_recall --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: skipped or failed; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/benchmark_summary.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/benchmark_summary.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/calibrated_precision_ledger.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/calibrated_precision_ledger.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/candidate_funnel_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/candidate_funnel_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/config.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/confirmation_profile_sweep.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/confirmation_profile_sweep.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/confirmation_profile_sweep.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/crash_scan.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/drive_manifest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260702_023259_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260702_023259_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/eidos_brain_archive/20260702_022854_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/eidos_brain_archive/20260702_022854_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/eidos_brain_archive/20260702_022854_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/eidos_brain_archive/20260702_022854_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/eidos_brain_archive/20260702_022854_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/eidos_brain_archive/20260702_022854_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/eidos_brain_archive/20260702_022854_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/forecast.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260702_023258_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/incident_cards.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/manifest.jsonl
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260702_023258_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260702_023258_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260702_023259_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260702_023259_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260702_023259_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_reopen_gate.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/engine_reopen_gate.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/environment.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/event_confirmation_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/event_confirmation_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/event_summary.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/git_commit.txt
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/confirmed_event_001.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/confirmed_event_002.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/confirmed_event_003.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/confirmed_event_004.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/confirmed_event_005.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/confirmed_event_006.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_001.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_002.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_003.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_004.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_005.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_006.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_007.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_008.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_009.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_010.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_011.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_012.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_013.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_014.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_015.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_016.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_017.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_018.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_019.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_020.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_021.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_022.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_023.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_024.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_025.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_026.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_027.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_028.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/incident_cards/engine_card_029.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/labeled_metrics.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/labeled_metrics.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/logs/engine_output.log
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/precision_ledger.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/precision_ledger.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/proof_digest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/proof_digest.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/run_manifest.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/sentinel_calibration_report.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/sentinel_calibration_report.md
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/sentinel_calibration_v1.json
- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: unknown
- Drive folder used: unknown
- Files copied: 0
- Files skipped: 0
- Reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount

### End-of-task summary
1. Files changed: proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/sentinel_guardrail_scale_matrix_2026_07_01/generated/normal_only_negative_control.csv --label-column Label --frames 1000 --seed 42 --out artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off --suite full --sample-mode natural --event-merge-gap 25 --confirmation-mode off --sentinel-calibration-mode off --confirmation-profile-sweep low_noise --confirmation-profile-sweep balanced --confirmation-profile-sweep high_recall --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 84 files under `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: skipped or failed; no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.
