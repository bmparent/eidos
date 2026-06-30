## Labeled CICIDS/WebAttacks proof harness -- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250

### What happened today
Built and ran the event-confirmation layer for the labeled/domain proof harness.

### What was accomplished
- Added proof-side candidate scoring and confirmation modes for labeled-domain events.
- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.
- Captured raw, merged, deduped, and confirmed event metrics side by side.
- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250 --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode balanced --confirmation-profile-sweep balanced --confirmation-profile-sweep recall_guarded --confirmation-profile-sweep high_recall --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/benchmark_summary.csv
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/benchmark_summary.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/calibrated_precision_ledger.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/calibrated_precision_ledger.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/candidate_funnel_report.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/candidate_funnel_report.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/config.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/confirmation_profile_sweep.csv
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/confirmation_profile_sweep.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/confirmation_profile_sweep.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/crash_scan.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/drive_manifest.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260608_001124_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260608_001124_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/eidos_brain_archive/20260608_001025_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/eidos_brain_archive/20260608_001025_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/eidos_brain_archive/20260608_001025_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/eidos_brain_archive/20260608_001025_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/eidos_brain_archive/20260608_001025_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/eidos_brain_archive/20260608_001025_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/eidos_brain_archive/20260608_001025_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/forecast.jsonl
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260608_001124_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/incident_cards.jsonl
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/manifest.jsonl
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260608_001123_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260608_001124_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260608_001124_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260608_001124_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260608_001124_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/environment.txt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/event_confirmation_report.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/event_confirmation_report.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/event_summary.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/git_commit.txt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/confirmed_event_001.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/confirmed_event_002.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/confirmed_event_003.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/engine_card_001.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/engine_card_002.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/engine_card_003.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/engine_card_004.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/engine_card_005.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/engine_card_006.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/engine_card_007.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/engine_card_008.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/incident_cards/engine_card_009.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/labeled_metrics.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/labeled_metrics.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/logs/engine_output.log
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/precision_ledger.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/precision_ledger.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/proof_digest.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/proof_digest.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/run_manifest.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/sentinel_calibration_v1.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-08\cicids_webattacks_proof_full_seed42_frames225_20260608T001126Z
- Files copied: 56
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250 --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode balanced --confirmation-profile-sweep balanced --confirmation-profile-sweep recall_guarded --confirmation-profile-sweep high_recall --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 57 files under `artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced250`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000

### What happened today
Built and ran the event-confirmation layer for the labeled/domain proof harness.

### What was accomplished
- Added proof-side candidate scoring and confirmation modes for labeled-domain events.
- Added optional Sentinel calibration v1 as a proof-stage false-positive suppression layer around confirmed events.
- Captured raw, merged, deduped, and confirmed event metrics side by side.
- Added reason codes, suppression examples, confirmation examples, calibration guardrails, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000 --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode balanced --confirmation-profile-sweep balanced --confirmation-profile-sweep recall_guarded --confirmation-profile-sweep high_recall --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/benchmark_summary.csv
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/benchmark_summary.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/calibrated_precision_ledger.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/calibrated_precision_ledger.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/candidate_funnel_report.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/candidate_funnel_report.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/config.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/confirmation_profile_sweep.csv
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/confirmation_profile_sweep.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/confirmation_profile_sweep.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/crash_scan.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/drive_manifest.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260608_001639_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260608_001639_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/eidos_brain_archive/20260608_001333_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/eidos_brain_archive/20260608_001333_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/eidos_brain_archive/20260608_001333_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/eidos_brain_archive/20260608_001333_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/eidos_brain_archive/20260608_001333_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/eidos_brain_archive/20260608_001333_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/eidos_brain_archive/20260608_001333_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/forecast.jsonl
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260608_001638_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/incident_cards.jsonl
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/manifest.jsonl
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260608_001638_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260608_001638_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260608_001639_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260608_001639_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260608_001639_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/environment.txt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/event_confirmation_report.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/event_confirmation_report.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/event_summary.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/git_commit.txt
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_001.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_002.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_003.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_004.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_005.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_006.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_007.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_008.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_009.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_010.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_011.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_012.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_013.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_014.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_015.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_016.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_017.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/confirmed_event_018.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_001.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_002.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_003.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_004.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_005.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_006.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_007.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_008.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_009.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_010.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_011.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_012.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_013.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_014.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_015.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_016.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_017.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_018.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_019.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_020.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_021.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_022.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_023.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_024.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_025.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_026.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_027.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_028.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_029.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_030.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_031.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_032.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_033.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_034.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_035.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/incident_cards/engine_card_036.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/labeled_metrics.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/labeled_metrics.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/logs/engine_output.log
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/precision_ledger.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/precision_ledger.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/proof_digest.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/proof_digest.md
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/run_manifest.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/sentinel_calibration_v1.json
- artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-08\cicids_webattacks_proof_full_seed42_frames900_20260608T001644Z
- Files copied: 98
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000 --suite full --sample-mode balanced --event-merge-gap 25 --confirmation-mode balanced --confirmation-profile-sweep balanced --confirmation-profile-sweep recall_guarded --confirmation-profile-sweep high_recall --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 99 files under `artifacts/proof_runs/2026-06-07/calibration_recall_diagnostics/runs/balanced1000`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: event confirmation is proof-side postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.
