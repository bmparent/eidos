## Labeled CICIDS/WebAttacks proof harness -- artifacts/calibration_ledger_tiny_fixture_cpu

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/calibrated/attack-context proof events to those windows, and wrote crash, compression, device, precision, calibration, and baseline scaffold receipts.

### What passed
- Frames processed: 1
- Crash hits: 0
- Incident cards: 0
- Confirmation mode: balanced
- Sentinel calibration mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / calibrated events: 0 / 0 / 0 / 0

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/calibration_ledger_tiny_fixture_cpu`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\cicids_webattacks_proof_smoke_seed42_frames1_20260612T014735Z; reason: copy completed.

### What should happen next
Compare balanced, balanced_blocks, transition, and natural_attack_windows receipts before deciding whether any separately gated calibration work is warranted.

## Skipped proof leg -- artifacts/calibration_ledger_natural10k_cpu_skip

### What the task attempted
This leg would have tested the natural-order 10k CPU sample.

### Why it was skipped
The completed local CPU proof runs were enough to validate the new ledger and gate receipts, and a natural-order 10k run would require a longer local run window.

### What was saved locally
Skip artifacts were saved under `artifacts/calibration_ledger_natural10k_cpu_skip`.

### What was saved to Google Drive
The skip receipt was copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\calibration_ledger_natural10k_cpu_skip_20260612`.

### What remains uncertain
Natural-order 10k metrics remain unmeasured in this PR.

## Skipped proof leg -- artifacts/calibration_ledger_natural_attack_windows_10k_gpu_skip

### What the task attempted
This leg would have tested natural_attack_windows at 10k frames on CUDA.

### Why it was skipped
CUDA is not available in this environment (`torch 2.6.0+cpu`, `cuda_available=false`), so a GPU proof would be misleading.

### What was saved locally
Skip artifacts were saved under `artifacts/calibration_ledger_natural_attack_windows_10k_gpu_skip`.

### What was saved to Google Drive
The skip receipt was copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\calibration_ledger_natural_attack_windows_10k_gpu_skip_20260612`.

### What remains uncertain
The optional CUDA 10k result remains unmeasured until a CUDA-enabled environment is available.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/calibration_ledger_balanced250_cpu

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/calibrated/attack-context proof events to those windows, and wrote crash, compression, device, precision, calibration, and baseline scaffold receipts.

### What passed
- Frames processed: 225
- Crash hits: 0
- Incident cards: 12
- Confirmation mode: balanced
- Sentinel calibration mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / calibrated events: 12 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/calibration_ledger_balanced250_cpu`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\cicids_webattacks_proof_smoke_seed42_frames225_20260612T015041Z; reason: copy completed.

### What should happen next
Compare balanced, balanced_blocks, transition, and natural_attack_windows receipts before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/calibration_ledger_balanced1k_cpu

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/calibrated/attack-context proof events to those windows, and wrote crash, compression, device, precision, calibration, and baseline scaffold receipts.

### What passed
- Frames processed: 900
- Crash hits: 0
- Incident cards: 54
- Confirmation mode: balanced
- Sentinel calibration mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / calibrated events: 54 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/calibration_ledger_balanced1k_cpu`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\cicids_webattacks_proof_smoke_seed42_frames900_20260612T015459Z; reason: copy completed.

### What should happen next
Compare balanced, balanced_blocks, transition, and natural_attack_windows receipts before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/calibration_ledger_transition1k_cpu

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/calibrated/attack-context proof events to those windows, and wrote crash, compression, device, precision, calibration, and baseline scaffold receipts.

### What passed
- Frames processed: 900
- Crash hits: 0
- Incident cards: 34
- Confirmation mode: balanced
- Sentinel calibration mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / calibrated events: 34 / 3 / 3 / 3

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/calibration_ledger_transition1k_cpu`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\cicids_webattacks_proof_smoke_seed42_frames900_20260612T015909Z; reason: copy completed.

### What should happen next
Compare balanced, balanced_blocks, transition, and natural_attack_windows receipts before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/calibration_ledger_natural_attack_windows_cpu

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/calibrated/attack-context proof events to those windows, and wrote crash, compression, device, precision, calibration, and baseline scaffold receipts.

### What passed
- Frames processed: 508
- Crash hits: 0
- Incident cards: 33
- Confirmation mode: balanced
- Sentinel calibration mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / calibrated events: 33 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/calibration_ledger_natural_attack_windows_cpu`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-12\cicids_webattacks_proof_smoke_seed42_frames508_20260612T020118Z; reason: copy completed.

### What should happen next
Compare balanced, balanced_blocks, transition, and natural_attack_windows receipts before deciding whether any separately gated calibration work is warranted.
