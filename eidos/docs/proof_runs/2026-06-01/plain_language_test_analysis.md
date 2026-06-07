## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_precision_tiny_20260601

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped proof events to those windows, and wrote crash, compression, device, and precision receipts.

### What passed
- Frames processed: 11
- Crash hits: 0
- Incident cards: 1
- Raw / merged / deduped events: 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_proof_precision_tiny_20260601`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Use the precision ledger to compare alert pressure across samples before deciding whether any separately gated calibration work is warranted.

## Precision Ledger v1 implementation summary

### What the task attempted
This task added a precision discipline layer to the labeled CICIDS/WebAttacks proof harness. The layer measures raw events separately from merged and deduped proof events, so alert pressure can be explained without changing the Eidos engine.

### Why the test matters
The prior proof showed Eidos behaving like a sensitive tripwire. This work keeps that raw behavior visible while adding transparent postprocessing receipts that explain duplicate pressure, transition-adjacent false positives, attack-window timing, and precision lift.

### What was tested
The tests covered attack-label parsing, raw CICIDS label normalization, binary proof labels, natural/balanced/transition sample construction, event merging, duplicate event collapse, false-positive classification, attack-window latency, CPU-only device receipts, manifest hygiene, crash-scan cleanliness, and generated artifact handling.

### What passed
- Focused labeled-runner pytest passed: 13 tests.
- Full pytest passed: 77 passed and 1 skipped.
- Tiny fixture proof passed with clean crash scan and precision ledger artifacts.
- Balanced 250 CPU sample passed with clean crash scan and precision ledger artifacts.
- Transition 1k CPU sample passed with clean crash scan and precision ledger artifacts.

### What failed
No acceptance test failed after fixes. Google Drive copy was skipped because no verified writable Drive root was available. Optional GPU 10k was skipped because CUDA is unavailable on this machine.

### What artifacts were generated
- Tiny fixture: `artifacts/cicids_webattacks_proof_precision_tiny_20260601/`
- Balanced CPU: `artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/`
- Transition CPU: `artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/`
- Downloaded WebAttacks CSV source: `artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`

### What was saved locally
Each proof folder includes `precision_ledger.json`, `precision_ledger.md`, `run_manifest.json`, `labeled_metrics.json`, `benchmark_summary.csv`, `benchmark_summary.md`, `crash_scan.json`, `drive_manifest.json`, `environment.txt`, incident cards, logs, and engine artifacts.

### What was saved to Google Drive
Nothing was copied to Google Drive. Each run wrote a `drive_manifest.json` explaining that no writable configured or mounted Drive root was available.

### What remains uncertain
The precision ledger is a postprocessing/accounting layer. It does not prove production readiness, and it does not tune thresholds or change anomaly policy.

### What should happen next
Review these precision-ledger receipts across more labeled samples before considering a separate gated calibration change.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped proof events to those windows, and wrote crash, compression, device, and precision receipts.

### What passed
- Frames processed: 225
- Crash hits: 0
- Incident cards: 12
- Raw / merged / deduped events: 12 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Use the precision ledger to compare alert pressure across samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601

### What the task attempted
The task connected Eidos Brain to a labeled cyber anomaly CSV so the system can be scored against known benign and attack rows.

### Why the test matters
The official GPU 10k proof established that the engine could run cleanly. This proof starts measuring labeled domain behavior, which is the next evidence step.

### What was tested
The runner processed a labeled CICIDS/WebAttacks-style sample, grouped attack labels into windows, compared raw/merged/deduped proof events to those windows, and wrote crash, compression, device, and precision receipts.

### What passed
- Frames processed: 900
- Crash hits: 0
- Incident cards: 34
- Raw / merged / deduped events: 34 / 3 / 3

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Use the precision ledger to compare alert pressure across samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_confirmation_tiny_off_20260601

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
- Confirmation mode: off
- Raw / merged / deduped / confirmed events: 1 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_proof_confirmation_tiny_off_20260601`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_confirmation_tiny_balanced_20260601

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
- Raw / merged / deduped / confirmed events: 1 / 1 / 1 / 0

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_proof_confirmation_tiny_balanced_20260601`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_confirmation_balanced250_cpu_off_20260601

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
- Raw / merged / deduped / confirmed events: 12 / 1 / 1 / 12

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_proof_confirmation_balanced250_cpu_off_20260601`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_confirmation_balanced250_cpu_high_recall_20260601

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
- Raw / merged / deduped / confirmed events: 12 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_proof_confirmation_balanced250_cpu_high_recall_20260601`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_confirmation_balanced250_cpu_balanced_20260601

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
- Raw / merged / deduped / confirmed events: 12 / 1 / 1 / 1

### What failed or remains uncertain
- Any false positives and false negatives are recorded in the metrics instead of being tuned away.
- Real-data coverage depends on the caller-provided CICIDS/WebAttacks CSV path.

### What was saved locally
Artifacts were saved under `artifacts/cicids_webattacks_proof_confirmation_balanced250_cpu_balanced_20260601`.

### What was saved to Google Drive
Drive status: skipped or failed; folder: unknown; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.

### What should happen next
Compare confirmation modes across balanced and transition samples before deciding whether any separately gated calibration work is warranted.
