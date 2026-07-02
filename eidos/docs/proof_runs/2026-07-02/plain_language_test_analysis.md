## Labeled CICIDS/WebAttacks proof harness -- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 1
- Crash hits: 0
- Incident cards: 0
- Confirmation mode: off
- Sentinel calibration mode: off
- Calibration enabled: False
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 0 / 0 / 0 / 0

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 225
- Crash hits: 0
- Incident cards: 12
- Confirmation mode: off
- Sentinel calibration mode: off
- Calibration enabled: False
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 12 / 1 / 1 / 12

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off

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
- Confirmation mode: off
- Sentinel calibration mode: off
- Calibration enabled: False
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 34 / 3 / 3 / 34

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 900
- Crash hits: 0
- Incident cards: 35
- Confirmation mode: off
- Sentinel calibration mode: off
- Calibration enabled: False
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 35 / 4 / 4 / 35

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/normal_only_negative_control/off`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.
