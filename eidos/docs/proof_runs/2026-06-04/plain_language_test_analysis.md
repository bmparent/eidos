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
