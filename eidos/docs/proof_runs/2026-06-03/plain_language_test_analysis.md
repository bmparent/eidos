## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_confirmation_transition2k_cpu_balanced_20260602

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
- Raw / merged / deduped / confirmed events: 68 / 2 / 2 / 2

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_proof_confirmation_transition2k_cpu_balanced_20260602`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_confirmation_natural2k_cpu_balanced_20260602

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
- Raw / merged / deduped / confirmed events: 79 / 5 / 5 / 2

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_proof_confirmation_natural2k_cpu_balanced_20260602`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_confirmation_transition4360_cpu_balanced_20260602

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
- Raw / merged / deduped / confirmed events: 166 / 4 / 4 / 3

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_proof_confirmation_transition4360_cpu_balanced_20260602`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled confirmation comparison -- artifacts/cicids_webattacks_calibration_v1_transition1k_comparison_20260603

### What the task attempted
The task compared saved CICIDS/WebAttacks labeled proof runs across event-confirmation modes.

### Why the test matters
The comparison turns separate proof runs into one decision surface for choosing a later calibration direction.

### What was tested
The tool read labeled metrics, event summaries, confirmation reports, precision ledgers, run manifests, crash scans, proof digests, and benchmark summaries when present.

### What passed
- Recommended mode: balanced (uncalibrated)
- Policy: balanced_f1
- Compared modes: off (uncalibrated), high_recall (uncalibrated), low_noise (uncalibrated), balanced (uncalibrated), balanced + sentinel_calibration_v1

### What failed or remains uncertain
- Missing optional artifacts are disclosed in the reports instead of being silently ignored.
- The comparison cannot prove a mode outside the receipts it was given.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_transition1k_comparison_20260603`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-03\cicids_webattacks_calibration_v1_transition1k_comparison_20260603; reason: copy completed.

### What should happen next
Use the comparison report to choose whether a separately gated Sentinel calibration task is worth doing.

## Sentinel calibration v1 acceptance -- artifacts/cicids_webattacks_calibration_v1_acceptance_20260604

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
Artifacts were saved under `artifacts/cicids_webattacks_calibration_v1_acceptance_20260604`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-03\cicids_webattacks_calibration_v1_acceptance_20260604; reason: copy completed.

### What should happen next
Move to broader Month 1 baselines and proof-report automation before adding Month 2 features.
