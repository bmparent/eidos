## Labeled CICIDS/WebAttacks proof harness -- artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped/confirmed proof events to those windows, and wrote crash, compression, device, precision, and confirmation receipts.

### What passed
- Frames processed: 11
- Crash hits: 0
- Incident cards: 1
- Confirmation mode: balanced
- Sentinel calibration mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 1 / 1 / 1 / 0

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\cicids_webattacks_proof_smoke_seed42_frames11_20260611T000311Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610

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
- Confirmation mode: balanced
- Sentinel calibration mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 12 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\cicids_webattacks_proof_full_seed42_frames225_20260611T000619Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610

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
- Sentinel calibration mode: balanced
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 34 / 3 / 3 / 3

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\cicids_webattacks_proof_full_seed42_frames900_20260611T001306Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610

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
- Confirmation mode: low_noise
- Sentinel calibration mode: low_noise
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 12 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\cicids_webattacks_proof_full_seed42_frames225_20260611T001654Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610

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
- Confirmation mode: high_recall
- Sentinel calibration mode: high_recall
- Calibration enabled: True
- Calibration suppressed events: 0
- Raw / merged / deduped / confirmed events: 12 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610`.

### What was saved to Google Drive
Drive status: copied; folder: G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\cicids_webattacks_proof_full_seed42_frames225_20260611T002041Z; reason: copy completed.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Aggregate engine reopen readiness gate -- 2026-06-11

### What the task attempted
This task tested whether Sentinel Calibration v1 is strong enough to reopen narrow future core engine experiments. It was a gate, not a core rewrite.

### Why the test matters
The goal was to reduce false-positive pressure on normal streams while preserving raw evidence and recall. A core experiment should only be considered if calibration receipts are strong and clean.

### What was tested
The branch added calibration/reporting/gate tooling, then ran focused tests, full pytest, core-touch policy, and five proof runs: tiny smoke, balanced250 balanced, transition1k balanced, balanced250 low_noise, and balanced250 high_recall.

### What passed
- Full pytest passed: `106 passed, 1 skipped, 11 warnings`.
- Core-touch policy passed with no forbidden core-path changes.
- Crash scans were clean on all proof runs.
- Raw, merged, deduped, and calibrated metrics were emitted side by side.
- `engine_reopen_gate.json/.md` and `sentinel_calibration_report.json/.md` were emitted.
- Google Drive copy succeeded for all five proof runs.
- Calibrated FP/10k was `0.0` on the required CPU sample legs.

### What failed or remains uncertain
Balanced 250-row recall still needs tuning. Raw recall was `0.259259`, while calibrated recall was `0.166667` for balanced, low_noise, and high_recall. That means the gate should not reopen core behavior yet.

### Gate verdict
`CALIBRATION_ONLY_NEEDS_TUNING`.

### What was saved locally
- `artifacts/engine_reopen_sentinel_calibration_v1_tiny_smoke_cpu_balanced_20260610`
- `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_balanced_20260610`
- `artifacts/engine_reopen_sentinel_calibration_v1_transition1k_cpu_balanced_20260610`
- `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_low_noise_20260610`
- `artifacts/engine_reopen_sentinel_calibration_v1_balanced250_cpu_high_recall_20260610`

### What was saved to Google Drive
All five proof runs copied successfully under `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-11\...`.

### What should happen next
Do another calibration-only tuning pass focused on balanced-sample recall before considering any narrow core experiment.
