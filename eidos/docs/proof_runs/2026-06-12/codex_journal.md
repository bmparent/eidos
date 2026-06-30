## Labeled CICIDS/WebAttacks proof harness -- artifacts/calibration_ledger_tiny_fixture_cpu

### What happened today
Built and ran the calibration ledger, calibration gate, and baseline scaffold receipts for the labeled/domain proof harness.

### What was accomplished
- Captured raw, merged, deduped, calibrated, and attack-context event metrics side by side.
- Added calibration gate verdicts, sampling semantics, calibration ratchet status, baseline scaffold entries, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 6 --seed 42 --out artifacts/calibration_ledger_tiny_fixture_cpu --suite smoke --sample-mode balanced --event-merge-gap 25 --device auto --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/baseline_scaffold.py
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- docs/proof/calibration_ledger_baseline_scaffold.md
- .gitignore
- artifacts/calibration_ledger_tiny_fixture_cpu

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/calibration_ledger_tiny_fixture_cpu/baseline_scaffold.json
- artifacts/calibration_ledger_tiny_fixture_cpu/baseline_scaffold.md
- artifacts/calibration_ledger_tiny_fixture_cpu/benchmark_summary.csv
- artifacts/calibration_ledger_tiny_fixture_cpu/benchmark_summary.md
- artifacts/calibration_ledger_tiny_fixture_cpu/calibrated_precision_ledger.json
- artifacts/calibration_ledger_tiny_fixture_cpu/calibrated_precision_ledger.md
- artifacts/calibration_ledger_tiny_fixture_cpu/calibration_gate.json
- artifacts/calibration_ledger_tiny_fixture_cpu/calibration_gate.md
- artifacts/calibration_ledger_tiny_fixture_cpu/candidate_funnel_report.json
- artifacts/calibration_ledger_tiny_fixture_cpu/candidate_funnel_report.md
- artifacts/calibration_ledger_tiny_fixture_cpu/config.json
- artifacts/calibration_ledger_tiny_fixture_cpu/confirmation_profile_sweep.csv
- artifacts/calibration_ledger_tiny_fixture_cpu/confirmation_profile_sweep.json
- artifacts/calibration_ledger_tiny_fixture_cpu/confirmation_profile_sweep.md
- artifacts/calibration_ledger_tiny_fixture_cpu/crash_scan.json
- artifacts/calibration_ledger_tiny_fixture_cpu/drive_manifest.json
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260612_014734_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260612_014734_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_artifacts/eidos_brain_archive/20260612_014733_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_artifacts/eidos_brain_archive/20260612_014733_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_artifacts/eidos_brain_archive/20260612_014733_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_artifacts/eidos_brain_archive/20260612_014733_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_artifacts/eidos_brain_archive/20260612_014733_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_artifacts/forecast.jsonl
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260612_014734_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_artifacts/manifest.jsonl
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260612_014734_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_reopen_gate.json
- artifacts/calibration_ledger_tiny_fixture_cpu/engine_reopen_gate.md
- artifacts/calibration_ledger_tiny_fixture_cpu/environment.txt
- artifacts/calibration_ledger_tiny_fixture_cpu/event_confirmation_report.json
- artifacts/calibration_ledger_tiny_fixture_cpu/event_confirmation_report.md
- artifacts/calibration_ledger_tiny_fixture_cpu/event_summary.json
- artifacts/calibration_ledger_tiny_fixture_cpu/git_commit.txt
- artifacts/calibration_ledger_tiny_fixture_cpu/incident_cards/README.md
- artifacts/calibration_ledger_tiny_fixture_cpu/labeled_metrics.json
- artifacts/calibration_ledger_tiny_fixture_cpu/labeled_metrics.md
- artifacts/calibration_ledger_tiny_fixture_cpu/logs/engine_output.log
- artifacts/calibration_ledger_tiny_fixture_cpu/precision_ledger.json
- artifacts/calibration_ledger_tiny_fixture_cpu/precision_ledger.md
- artifacts/calibration_ledger_tiny_fixture_cpu/proof_digest.json
- artifacts/calibration_ledger_tiny_fixture_cpu/proof_digest.md
- artifacts/calibration_ledger_tiny_fixture_cpu/run_manifest.json
- artifacts/calibration_ledger_tiny_fixture_cpu/sampling_semantics.json
- artifacts/calibration_ledger_tiny_fixture_cpu/sampling_semantics.md
- artifacts/calibration_ledger_tiny_fixture_cpu/sentinel_calibration_report.json
- artifacts/calibration_ledger_tiny_fixture_cpu/sentinel_calibration_report.md
- artifacts/calibration_ledger_tiny_fixture_cpu/sentinel_calibration_v1.json
- artifacts/calibration_ledger_tiny_fixture_cpu/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\cicids_webattacks_proof_smoke_seed42_frames1_20260612T014735Z
- Files copied: 48
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/baseline_scaffold.py, proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, docs/proof/calibration_ledger_baseline_scaffold.md, .gitignore, artifacts/calibration_ledger_tiny_fixture_cpu
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests cover calibration gate, precision ledger v1.1, sampling semantics, baseline skip reasons, device receipts, and raw visibility; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 6 --seed 42 --out artifacts/calibration_ledger_tiny_fixture_cpu --suite smoke --sample-mode balanced --event-merge-gap 25 --device auto --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force"`.
5. Artifacts generated: 49 files under `artifacts/calibration_ledger_tiny_fixture_cpu`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: calibration and baseline scaffold receipts are proof-side only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: full Week 3 baseline competitor matrix or core behavior changes.

## Skipped proof leg -- artifacts/calibration_ledger_natural10k_cpu_skip

### What happened today
The natural-order 10k CPU run was skipped with an explicit receipt because the completed local CPU runs showed that a 10k local proof would require a longer run window.

### Tests and commands run
- `not run; skip receipt emitted`.

### What did not change
Core behavior changed: no.

### Artifacts generated
- artifacts/calibration_ledger_natural10k_cpu_skip/skip_receipt.json
- artifacts/calibration_ledger_natural10k_cpu_skip/skip_receipt.md
- artifacts/calibration_ledger_natural10k_cpu_skip/calibration_gate.json
- artifacts/calibration_ledger_natural10k_cpu_skip/calibration_gate.md
- artifacts/calibration_ledger_natural10k_cpu_skip/run_manifest.json
- artifacts/calibration_ledger_natural10k_cpu_skip/drive_manifest.json

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\calibration_ledger_natural10k_cpu_skip_20260612
- Reason: copy completed

## Skipped proof leg -- artifacts/calibration_ledger_natural_attack_windows_10k_gpu_skip

### What happened today
The optional natural_attack_windows 10k GPU run was skipped because CUDA is unavailable in this environment (`torch 2.6.0+cpu`, `cuda_available=false`).

### Tests and commands run
- `python -c "import json; from tools.run_labeled_domain_proof import collect_device_receipt; print(json.dumps(collect_device_receipt(requested_device='cuda'), sort_keys=True))"` -> CUDA unavailable; optional GPU proof skipped.

### What did not change
Core behavior changed: no.

### Artifacts generated
- artifacts/calibration_ledger_natural_attack_windows_10k_gpu_skip/skip_receipt.json
- artifacts/calibration_ledger_natural_attack_windows_10k_gpu_skip/skip_receipt.md
- artifacts/calibration_ledger_natural_attack_windows_10k_gpu_skip/calibration_gate.json
- artifacts/calibration_ledger_natural_attack_windows_10k_gpu_skip/calibration_gate.md
- artifacts/calibration_ledger_natural_attack_windows_10k_gpu_skip/run_manifest.json
- artifacts/calibration_ledger_natural_attack_windows_10k_gpu_skip/drive_manifest.json

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\calibration_ledger_natural_attack_windows_10k_gpu_skip_20260612
- Reason: copy completed

## Labeled CICIDS/WebAttacks proof harness -- artifacts/calibration_ledger_balanced250_cpu

### What happened today
Built and ran the calibration ledger, calibration gate, and baseline scaffold receipts for the labeled/domain proof harness.

### What was accomplished
- Captured raw, merged, deduped, calibrated, and attack-context event metrics side by side.
- Added calibration gate verdicts, sampling semantics, calibration ratchet status, baseline scaffold entries, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/calibration_ledger_balanced250_cpu --suite smoke --sample-mode balanced --event-merge-gap 25 --device auto --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/baseline_scaffold.py
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- docs/proof/calibration_ledger_baseline_scaffold.md
- .gitignore
- artifacts/calibration_ledger_balanced250_cpu

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/calibration_ledger_balanced250_cpu/baseline_scaffold.json
- artifacts/calibration_ledger_balanced250_cpu/baseline_scaffold.md
- artifacts/calibration_ledger_balanced250_cpu/benchmark_summary.csv
- artifacts/calibration_ledger_balanced250_cpu/benchmark_summary.md
- artifacts/calibration_ledger_balanced250_cpu/calibrated_precision_ledger.json
- artifacts/calibration_ledger_balanced250_cpu/calibrated_precision_ledger.md
- artifacts/calibration_ledger_balanced250_cpu/calibration_gate.json
- artifacts/calibration_ledger_balanced250_cpu/calibration_gate.md
- artifacts/calibration_ledger_balanced250_cpu/candidate_funnel_report.json
- artifacts/calibration_ledger_balanced250_cpu/candidate_funnel_report.md
- artifacts/calibration_ledger_balanced250_cpu/config.json
- artifacts/calibration_ledger_balanced250_cpu/confirmation_profile_sweep.csv
- artifacts/calibration_ledger_balanced250_cpu/confirmation_profile_sweep.json
- artifacts/calibration_ledger_balanced250_cpu/confirmation_profile_sweep.md
- artifacts/calibration_ledger_balanced250_cpu/crash_scan.json
- artifacts/calibration_ledger_balanced250_cpu/drive_manifest.json
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260612_015039_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260612_015039_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/eidos_brain_archive/20260612_014934_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/eidos_brain_archive/20260612_014934_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/eidos_brain_archive/20260612_014934_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/eidos_brain_archive/20260612_014934_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/eidos_brain_archive/20260612_014934_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/eidos_brain_archive/20260612_014934_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/eidos_brain_archive/20260612_014934_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/forecast.jsonl
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260612_015039_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/incident_cards.jsonl
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/manifest.jsonl
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260612_015039_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260612_015039_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260612_015039_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260612_015039_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_balanced250_cpu/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260612_015039_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/calibration_ledger_balanced250_cpu/engine_reopen_gate.json
- artifacts/calibration_ledger_balanced250_cpu/engine_reopen_gate.md
- artifacts/calibration_ledger_balanced250_cpu/environment.txt
- artifacts/calibration_ledger_balanced250_cpu/event_confirmation_report.json
- artifacts/calibration_ledger_balanced250_cpu/event_confirmation_report.md
- artifacts/calibration_ledger_balanced250_cpu/event_summary.json
- artifacts/calibration_ledger_balanced250_cpu/git_commit.txt
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/confirmed_event_001.json
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/confirmed_event_002.json
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/confirmed_event_003.json
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/engine_card_001.json
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/engine_card_002.json
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/engine_card_003.json
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/engine_card_004.json
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/engine_card_005.json
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/engine_card_006.json
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/engine_card_007.json
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/engine_card_008.json
- artifacts/calibration_ledger_balanced250_cpu/incident_cards/engine_card_009.json
- artifacts/calibration_ledger_balanced250_cpu/labeled_metrics.json
- artifacts/calibration_ledger_balanced250_cpu/labeled_metrics.md
- artifacts/calibration_ledger_balanced250_cpu/logs/engine_output.log
- artifacts/calibration_ledger_balanced250_cpu/precision_ledger.json
- artifacts/calibration_ledger_balanced250_cpu/precision_ledger.md
- artifacts/calibration_ledger_balanced250_cpu/proof_digest.json
- artifacts/calibration_ledger_balanced250_cpu/proof_digest.md
- artifacts/calibration_ledger_balanced250_cpu/run_manifest.json
- artifacts/calibration_ledger_balanced250_cpu/sampling_semantics.json
- artifacts/calibration_ledger_balanced250_cpu/sampling_semantics.md
- artifacts/calibration_ledger_balanced250_cpu/sentinel_calibration_report.json
- artifacts/calibration_ledger_balanced250_cpu/sentinel_calibration_report.md
- artifacts/calibration_ledger_balanced250_cpu/sentinel_calibration_v1.json
- artifacts/calibration_ledger_balanced250_cpu/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\cicids_webattacks_proof_smoke_seed42_frames225_20260612T015041Z
- Files copied: 66
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/baseline_scaffold.py, proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, docs/proof/calibration_ledger_baseline_scaffold.md, .gitignore, artifacts/calibration_ledger_balanced250_cpu
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests cover calibration gate, precision ledger v1.1, sampling semantics, baseline skip reasons, device receipts, and raw visibility; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/calibration_ledger_balanced250_cpu --suite smoke --sample-mode balanced --event-merge-gap 25 --device auto --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 67 files under `artifacts/calibration_ledger_balanced250_cpu`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: calibration and baseline scaffold receipts are proof-side only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: full Week 3 baseline competitor matrix or core behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/calibration_ledger_balanced1k_cpu

### What happened today
Built and ran the calibration ledger, calibration gate, and baseline scaffold receipts for the labeled/domain proof harness.

### What was accomplished
- Captured raw, merged, deduped, calibrated, and attack-context event metrics side by side.
- Added calibration gate verdicts, sampling semantics, calibration ratchet status, baseline scaffold entries, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/calibration_ledger_balanced1k_cpu --suite smoke --sample-mode balanced --event-merge-gap 25 --device auto --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/baseline_scaffold.py
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- docs/proof/calibration_ledger_baseline_scaffold.md
- .gitignore
- artifacts/calibration_ledger_balanced1k_cpu

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/calibration_ledger_balanced1k_cpu/baseline_scaffold.json
- artifacts/calibration_ledger_balanced1k_cpu/baseline_scaffold.md
- artifacts/calibration_ledger_balanced1k_cpu/benchmark_summary.csv
- artifacts/calibration_ledger_balanced1k_cpu/benchmark_summary.md
- artifacts/calibration_ledger_balanced1k_cpu/calibrated_precision_ledger.json
- artifacts/calibration_ledger_balanced1k_cpu/calibrated_precision_ledger.md
- artifacts/calibration_ledger_balanced1k_cpu/calibration_gate.json
- artifacts/calibration_ledger_balanced1k_cpu/calibration_gate.md
- artifacts/calibration_ledger_balanced1k_cpu/candidate_funnel_report.json
- artifacts/calibration_ledger_balanced1k_cpu/candidate_funnel_report.md
- artifacts/calibration_ledger_balanced1k_cpu/config.json
- artifacts/calibration_ledger_balanced1k_cpu/confirmation_profile_sweep.csv
- artifacts/calibration_ledger_balanced1k_cpu/confirmation_profile_sweep.json
- artifacts/calibration_ledger_balanced1k_cpu/confirmation_profile_sweep.md
- artifacts/calibration_ledger_balanced1k_cpu/crash_scan.json
- artifacts/calibration_ledger_balanced1k_cpu/drive_manifest.json
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260612_015454_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260612_015454_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015247_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015247_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015247_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015247_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015247_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015247_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015247_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/forecast.jsonl
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260612_015454_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/incident_cards.jsonl
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/manifest.jsonl
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260612_015454_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260612_015454_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260612_015454_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260612_015454_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_balanced1k_cpu/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260612_015454_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/calibration_ledger_balanced1k_cpu/engine_reopen_gate.json
- artifacts/calibration_ledger_balanced1k_cpu/engine_reopen_gate.md
- artifacts/calibration_ledger_balanced1k_cpu/environment.txt
- artifacts/calibration_ledger_balanced1k_cpu/event_confirmation_report.json
- artifacts/calibration_ledger_balanced1k_cpu/event_confirmation_report.md
- artifacts/calibration_ledger_balanced1k_cpu/event_summary.json
- artifacts/calibration_ledger_balanced1k_cpu/git_commit.txt
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_001.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_002.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_003.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_004.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_005.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_006.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_007.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_008.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_009.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_010.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_011.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_012.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_013.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_014.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_015.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_016.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_017.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/confirmed_event_018.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_001.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_002.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_003.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_004.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_005.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_006.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_007.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_008.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_009.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_010.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_011.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_012.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_013.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_014.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_015.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_016.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_017.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_018.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_019.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_020.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_021.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_022.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_023.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_024.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_025.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_026.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_027.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_028.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_029.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_030.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_031.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_032.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_033.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_034.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_035.json
- artifacts/calibration_ledger_balanced1k_cpu/incident_cards/engine_card_036.json
- artifacts/calibration_ledger_balanced1k_cpu/labeled_metrics.json
- artifacts/calibration_ledger_balanced1k_cpu/labeled_metrics.md
- artifacts/calibration_ledger_balanced1k_cpu/logs/engine_output.log
- artifacts/calibration_ledger_balanced1k_cpu/precision_ledger.json
- artifacts/calibration_ledger_balanced1k_cpu/precision_ledger.md
- artifacts/calibration_ledger_balanced1k_cpu/proof_digest.json
- artifacts/calibration_ledger_balanced1k_cpu/proof_digest.md
- artifacts/calibration_ledger_balanced1k_cpu/run_manifest.json
- artifacts/calibration_ledger_balanced1k_cpu/sampling_semantics.json
- artifacts/calibration_ledger_balanced1k_cpu/sampling_semantics.md
- artifacts/calibration_ledger_balanced1k_cpu/sentinel_calibration_report.json
- artifacts/calibration_ledger_balanced1k_cpu/sentinel_calibration_report.md
- artifacts/calibration_ledger_balanced1k_cpu/sentinel_calibration_v1.json
- artifacts/calibration_ledger_balanced1k_cpu/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\cicids_webattacks_proof_smoke_seed42_frames900_20260612T015459Z
- Files copied: 108
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/baseline_scaffold.py, proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, docs/proof/calibration_ledger_baseline_scaffold.md, .gitignore, artifacts/calibration_ledger_balanced1k_cpu
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests cover calibration gate, precision ledger v1.1, sampling semantics, baseline skip reasons, device receipts, and raw visibility; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/calibration_ledger_balanced1k_cpu --suite smoke --sample-mode balanced --event-merge-gap 25 --device auto --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 109 files under `artifacts/calibration_ledger_balanced1k_cpu`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: calibration and baseline scaffold receipts are proof-side only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: full Week 3 baseline competitor matrix or core behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/calibration_ledger_transition1k_cpu

### What happened today
Built and ran the calibration ledger, calibration gate, and baseline scaffold receipts for the labeled/domain proof harness.

### What was accomplished
- Captured raw, merged, deduped, calibrated, and attack-context event metrics side by side.
- Added calibration gate verdicts, sampling semantics, calibration ratchet status, baseline scaffold entries, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/calibration_ledger_transition1k_cpu --suite smoke --sample-mode transition --event-merge-gap 25 --device auto --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/baseline_scaffold.py
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- docs/proof/calibration_ledger_baseline_scaffold.md
- .gitignore
- artifacts/calibration_ledger_transition1k_cpu

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/calibration_ledger_transition1k_cpu/baseline_scaffold.json
- artifacts/calibration_ledger_transition1k_cpu/baseline_scaffold.md
- artifacts/calibration_ledger_transition1k_cpu/benchmark_summary.csv
- artifacts/calibration_ledger_transition1k_cpu/benchmark_summary.md
- artifacts/calibration_ledger_transition1k_cpu/calibrated_precision_ledger.json
- artifacts/calibration_ledger_transition1k_cpu/calibrated_precision_ledger.md
- artifacts/calibration_ledger_transition1k_cpu/calibration_gate.json
- artifacts/calibration_ledger_transition1k_cpu/calibration_gate.md
- artifacts/calibration_ledger_transition1k_cpu/candidate_funnel_report.json
- artifacts/calibration_ledger_transition1k_cpu/candidate_funnel_report.md
- artifacts/calibration_ledger_transition1k_cpu/config.json
- artifacts/calibration_ledger_transition1k_cpu/confirmation_profile_sweep.csv
- artifacts/calibration_ledger_transition1k_cpu/confirmation_profile_sweep.json
- artifacts/calibration_ledger_transition1k_cpu/confirmation_profile_sweep.md
- artifacts/calibration_ledger_transition1k_cpu/crash_scan.json
- artifacts/calibration_ledger_transition1k_cpu/drive_manifest.json
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260612_015906_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260612_015906_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015649_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015649_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015649_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015649_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015649_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015649_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/eidos_brain_archive/20260612_015649_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/forecast.jsonl
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260612_015906_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/incident_cards.jsonl
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/manifest.jsonl
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260612_015906_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260612_015906_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260612_015906_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260612_015906_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_transition1k_cpu/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260612_015906_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/calibration_ledger_transition1k_cpu/engine_reopen_gate.json
- artifacts/calibration_ledger_transition1k_cpu/engine_reopen_gate.md
- artifacts/calibration_ledger_transition1k_cpu/environment.txt
- artifacts/calibration_ledger_transition1k_cpu/event_confirmation_report.json
- artifacts/calibration_ledger_transition1k_cpu/event_confirmation_report.md
- artifacts/calibration_ledger_transition1k_cpu/event_summary.json
- artifacts/calibration_ledger_transition1k_cpu/git_commit.txt
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/confirmed_event_001.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_001.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_002.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_003.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_004.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_005.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_006.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_007.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_008.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_009.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_010.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_011.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_012.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_013.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_014.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_015.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_016.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_017.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_018.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_019.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_020.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_021.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_022.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_023.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_024.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_025.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_026.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_027.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_028.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_029.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_030.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_031.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_032.json
- artifacts/calibration_ledger_transition1k_cpu/incident_cards/engine_card_033.json
- artifacts/calibration_ledger_transition1k_cpu/labeled_metrics.json
- artifacts/calibration_ledger_transition1k_cpu/labeled_metrics.md
- artifacts/calibration_ledger_transition1k_cpu/logs/engine_output.log
- artifacts/calibration_ledger_transition1k_cpu/precision_ledger.json
- artifacts/calibration_ledger_transition1k_cpu/precision_ledger.md
- artifacts/calibration_ledger_transition1k_cpu/proof_digest.json
- artifacts/calibration_ledger_transition1k_cpu/proof_digest.md
- artifacts/calibration_ledger_transition1k_cpu/run_manifest.json
- artifacts/calibration_ledger_transition1k_cpu/sampling_semantics.json
- artifacts/calibration_ledger_transition1k_cpu/sampling_semantics.md
- artifacts/calibration_ledger_transition1k_cpu/sentinel_calibration_report.json
- artifacts/calibration_ledger_transition1k_cpu/sentinel_calibration_report.md
- artifacts/calibration_ledger_transition1k_cpu/sentinel_calibration_v1.json
- artifacts/calibration_ledger_transition1k_cpu/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\cicids_webattacks_proof_smoke_seed42_frames900_20260612T015909Z
- Files copied: 88
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/baseline_scaffold.py, proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, docs/proof/calibration_ledger_baseline_scaffold.md, .gitignore, artifacts/calibration_ledger_transition1k_cpu
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests cover calibration gate, precision ledger v1.1, sampling semantics, baseline skip reasons, device receipts, and raw visibility; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/calibration_ledger_transition1k_cpu --suite smoke --sample-mode transition --event-merge-gap 25 --device auto --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 89 files under `artifacts/calibration_ledger_transition1k_cpu`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: calibration and baseline scaffold receipts are proof-side only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: full Week 3 baseline competitor matrix or core behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/calibration_ledger_natural_attack_windows_cpu

### What happened today
Built and ran the calibration ledger, calibration gate, and baseline scaffold receipts for the labeled/domain proof harness.

### What was accomplished
- Captured raw, merged, deduped, calibrated, and attack-context event metrics side by side.
- Added calibration gate verdicts, sampling semantics, calibration ratchet status, baseline scaffold entries, false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/calibration_ledger_natural_attack_windows_cpu --suite smoke --sample-mode natural_attack_windows --event-merge-gap 25 --device auto --confirmation-mode balanced --sentinel-calibration-mode balanced --natural-window-pre 250 --natural-window-post 250 --natural-window-max-windows 3 --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: copied; reason: copy completed.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- proof/baseline_scaffold.py
- proof/event_confirmation.py
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- docs/proof/calibration_ledger_baseline_scaffold.md
- .gitignore
- artifacts/calibration_ledger_natural_attack_windows_cpu

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/calibration_ledger_natural_attack_windows_cpu/baseline_scaffold.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/baseline_scaffold.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/benchmark_summary.csv
- artifacts/calibration_ledger_natural_attack_windows_cpu/benchmark_summary.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/calibrated_precision_ledger.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/calibrated_precision_ledger.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/calibration_gate.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/calibration_gate.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/candidate_funnel_report.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/candidate_funnel_report.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/config.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/confirmation_profile_sweep.csv
- artifacts/calibration_ledger_natural_attack_windows_cpu/confirmation_profile_sweep.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/confirmation_profile_sweep.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/crash_scan.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/drive_manifest.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260612_020117_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260612_020117_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/eidos_brain_archive/20260612_020043_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/eidos_brain_archive/20260612_020043_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/eidos_brain_archive/20260612_020043_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/eidos_brain_archive/20260612_020043_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/eidos_brain_archive/20260612_020043_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/eidos_brain_archive/20260612_020043_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/eidos_brain_archive/20260612_020043_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/forecast.jsonl
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260612_020117_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/incident_cards.jsonl
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/manifest.jsonl
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260612_020117_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260612_020117_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260612_020117_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260612_020117_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260612_020117_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_reopen_gate.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/engine_reopen_gate.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/environment.txt
- artifacts/calibration_ledger_natural_attack_windows_cpu/event_confirmation_report.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/event_confirmation_report.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/event_summary.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/git_commit.txt
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_001.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_002.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_003.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_004.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_005.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_006.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_007.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_008.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_009.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_010.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_011.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/confirmed_event_012.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_001.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_002.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_003.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_004.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_005.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_006.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_007.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_008.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_009.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_010.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_011.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_012.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_013.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_014.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_015.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_016.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_017.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_018.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_019.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_020.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/incident_cards/engine_card_021.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/labeled_metrics.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/labeled_metrics.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/logs/engine_output.log
- artifacts/calibration_ledger_natural_attack_windows_cpu/precision_ledger.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/precision_ledger.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/proof_digest.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/proof_digest.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/run_manifest.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/sampling_semantics.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/sampling_semantics.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/sentinel_calibration_report.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/sentinel_calibration_report.md
- artifacts/calibration_ledger_natural_attack_windows_cpu/sentinel_calibration_v1.json
- artifacts/calibration_ledger_natural_attack_windows_cpu/sentinel_calibration_v1.md

### Google Drive archive status
- Drive root used: G:\My Drive
- Drive folder used: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\cicids_webattacks_proof_smoke_seed42_frames508_20260612T020118Z
- Files copied: 87
- Files skipped: 0
- Reason: copy completed

### End-of-task summary
1. Files changed: proof/baseline_scaffold.py, proof/event_confirmation.py, tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, docs/proof/calibration_ledger_baseline_scaffold.md, .gitignore, artifacts/calibration_ledger_natural_attack_windows_cpu
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests cover calibration gate, precision ledger v1.1, sampling semantics, baseline skip reasons, device receipts, and raw visibility; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/calibration_ledger_natural_attack_windows_cpu --suite smoke --sample-mode natural_attack_windows --event-merge-gap 25 --device auto --confirmation-mode balanced --sentinel-calibration-mode balanced --natural-window-pre 250 --natural-window-post 250 --natural-window-max-windows 3 --calibration-enabled --calibration-benign-context-grace 0 --calibration-attack-window-guard 0 --calibration-min-confirmed-span 2 --calibration-min-evidence-count 2 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 88 files under `artifacts/calibration_ledger_natural_attack_windows_cpu`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: copied; copy completed.
9. Known limitations: calibration and baseline scaffold receipts are proof-side only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: full Week 3 baseline competitor matrix or core behavior changes.
