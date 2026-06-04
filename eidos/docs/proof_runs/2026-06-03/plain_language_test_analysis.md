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
