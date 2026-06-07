## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_calibration_v1_transition1k_cpu_balanced_20260603

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 900
- Crash hits: 0
- Incident cards: 34
- Confirmation mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 34 / 3 / 3 / 3

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_transition1k_cpu_balanced_20260603`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_proof_full_seed42_frames900_20260604T010022Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_calibration_v1_transition2k_cpu_balanced_20260603

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 1800
- Crash hits: 0
- Incident cards: 68
- Confirmation mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 1
- Raw / merged / deduped / confirmed events: 68 / 2 / 2 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_transition2k_cpu_balanced_20260603`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_proof_full_seed42_frames1800_20260604T010848Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_calibration_v1_transition4360_cpu_balanced_20260603

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 3924
- Crash hits: 0
- Incident cards: 166
- Confirmation mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 2
- Raw / merged / deduped / confirmed events: 166 / 4 / 4 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_transition4360_cpu_balanced_20260603`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_proof_full_seed42_frames3924_20260604T012057Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_calibration_v1_natural2k_cpu_balanced_20260603

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 1800
- Crash hits: 0
- Incident cards: 79
- Confirmation mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 2
- Raw / merged / deduped / confirmed events: 79 / 5 / 5 / 0

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_natural2k_cpu_balanced_20260603`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_proof_full_seed42_frames1800_20260604T012618Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Sentinel calibration v1 acceptance -- artifacts/cicids_webattacks_calibration_v1_acceptance_rerun_20260604

### What the task attempted
This task turned Sentinel calibration v1 into a reviewable acceptance gate instead of a loose proof-run success claim.

### Why the test matters
The 90-day plan says Month 1 should make Eidos trustworthy by reducing false positives without destroying anomaly recall.

### What was tested
The acceptance package checked FP/10k reduction, recall preservation, attack-window coverage, first latency, missed windows, crash scans, raw/pre/post metric visibility, and Drive status.

### What passed
- Decision: approved
- Recommended baseline: balanced + sentinel_calibration_v1

### What remains uncertain
- This is still CICIDS/WebAttacks receipt evidence, not proof across every domain.
- The natural2k sample is benign-only pressure evidence.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_acceptance_rerun_20260604`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_calibration_v1_acceptance_rerun_20260604; reason: copy completed.

### What should happen next
Move to broader Month 1 baselines and proof-report automation before adding Month 2 features.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_calibration_v1_generalization_normal3600_cpu_balanced_20260604

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 3240
- Crash hits: 0
- Incident cards: 156
- Confirmation mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 7
- Raw / merged / deduped / confirmed events: 156 / 11 / 11 / 0

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_generalization_normal3600_cpu_balanced_20260604`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_proof_full_seed42_frames3240_20260604T234308Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_calibration_v1_generalization_natural_attack1600_cpu_balanced_20260604

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 1440
- Crash hits: 0
- Incident cards: 89
- Confirmation mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 89 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_generalization_natural_attack1600_cpu_balanced_20260604`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_proof_full_seed42_frames1440_20260604T234647Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_calibration_v1_generalization_bruteforce600_cpu_balanced_20260604

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 540
- Crash hits: 0
- Incident cards: 23
- Confirmation mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 23 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_generalization_bruteforce600_cpu_balanced_20260604`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_proof_full_seed42_frames540_20260604T235105Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_calibration_v1_generalization_xss400_cpu_balanced_20260604

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 360
- Crash hits: 0
- Incident cards: 16
- Confirmation mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 16 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_generalization_xss400_cpu_balanced_20260604`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_proof_full_seed42_frames360_20260604T235523Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_calibration_v1_generalization_sql40_cpu_balanced_20260604

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 36
- Crash hits: 0
- Incident cards: 3
- Confirmation mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 3 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_generalization_sql40_cpu_balanced_20260604`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_proof_full_seed42_frames36_20260604T235730Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_calibration_v1_generalization_balanced200_seed0_cpu_balanced_20260604

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 180
- Crash hits: 0
- Incident cards: 9
- Confirmation mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 9 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_generalization_balanced200_seed0_cpu_balanced_20260604`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_proof_full_seed0_frames180_20260604T235935Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_calibration_v1_generalization_balanced200_seed1_cpu_balanced_20260604

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 180
- Crash hits: 0
- Incident cards: 10
- Confirmation mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 10 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_generalization_balanced200_seed1_cpu_balanced_20260604`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_proof_full_seed1_frames180_20260605T000100Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Sentinel calibration v1 generalization -- artifacts/cicids_webattacks_calibration_v1_generalization_20260604

### What the task attempted
The task checked whether Sentinel calibration v1 still controls false positives outside the first acceptance receipts.

### Why the test matters
The Month 1 proof goal is to reduce alert pressure without hiding raw evidence or losing attack-window visibility.

### What was tested
Acceptance rerun consistency, generalization matrix metrics, suppression audit details, attack-window coverage, crash scans, runtime/FPS/device receipts, and leakage risks.

### What passed
- Recommendation: approve.
- Baseline candidate: balanced + sentinel_calibration_v1.

### What failed
- No gate failures recorded.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_generalization_20260604`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-04\cicids_webattacks_calibration_v1_generalization_20260604; reason: copy completed.

### What remains uncertain
This remains a proof-stage CICIDS/WebAttacks harness baseline, not a production or cross-domain claim.

### What should happen next
Keep widening proof baselines and report automation before any Sentinel behavior changes.
