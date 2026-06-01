## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_precision_tiny_20260601

### What happened today
Built and ran the precision-ledger layer for the labeled/domain proof harness.

### What was accomplished
- Added first-class binary proof-label accounting and sample receipts.
- Captured raw, merged, and deduped event metrics in a precision ledger.
- Added false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 12 --seed 42 --out artifacts/cicids_webattacks_proof_precision_tiny_20260601 --suite smoke --sample-mode natural --event-merge-gap 25 --attack-labels "Web Attack - Brute Force"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: skipped or failed; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/cicids_webattacks_proof_precision_tiny_20260601

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/benchmark_summary.csv
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/benchmark_summary.md
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/config.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/crash_scan.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/drive_manifest.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260601_015712_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260601_015712_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/eidos_brain_archive/20260601_015706_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/eidos_brain_archive/20260601_015706_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/eidos_brain_archive/20260601_015706_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/eidos_brain_archive/20260601_015706_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/eidos_brain_archive/20260601_015706_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/eidos_brain_archive/20260601_015706_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/eidos_brain_archive/20260601_015706_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/forecast.jsonl
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260601_015712_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/incident_cards.jsonl
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/manifest.jsonl
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260601_015712_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260601_015712_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260601_015712_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260601_015712_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260601_015712_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/environment.txt
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/event_summary.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/git_commit.txt
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/incident_cards/engine_card_001.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/labeled_metrics.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/labeled_metrics.md
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/logs/engine_output.log
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/precision_ledger.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/precision_ledger.md
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/proof_digest.json
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/proof_digest.md
- artifacts/cicids_webattacks_proof_precision_tiny_20260601/run_manifest.json

### Google Drive archive status
- Drive root used: unknown
- Drive folder used: unknown
- Files copied: 0
- Files skipped: 0
- Reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount

### End-of-task summary
1. Files changed: tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/cicids_webattacks_proof_precision_tiny_20260601
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 12 --seed 42 --out artifacts/cicids_webattacks_proof_precision_tiny_20260601 --suite smoke --sample-mode natural --event-merge-gap 25 --attack-labels "Web Attack - Brute Force"`.
5. Artifacts generated: 35 files under `artifacts/cicids_webattacks_proof_precision_tiny_20260601`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: skipped or failed; no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
9. Known limitations: precision ledger is postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.

## Precision Ledger v1 implementation summary

### What happened today
Implemented Eidos Precision Ledger v1 for the CICIDS/WebAttacks labeled proof harness and validated it against unit tests, the tiny fixture, and real labeled WebAttacks CPU samples.

### What was accomplished
- Added robust `--attack-labels` parsing for single labels, comma-separated labels, repeated flags, and raw CICIDS Web Attack labels with replacement-character punctuation.
- Added binary proof-label normalization with `OriginalLabel` preserved and `EidosProofLabel` recorded.
- Added natural, balanced, and transition sample modes with sample receipts.
- Added `precision_ledger.json` and `precision_ledger.md` with raw, merged, and deduped event views.
- Added attack-window timing diagnostics, false-positive taxonomy, incident-card accounting, CPU/GPU receipts, and git hygiene receipts.

### Tests and commands run
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_labeled_domain_proof_runner.py -q` -> passed, 13 tests.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` -> passed, 77 passed and 1 skipped.
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 12 --seed 42 --out artifacts/cicids_webattacks_proof_precision_tiny_20260601 --suite smoke --sample-mode natural --event-merge-gap 25 --attack-labels "Web Attack - Brute Force"` -> passed, crash scan clean.
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601 --suite full --sample-mode balanced --event-merge-gap 25 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> passed, crash scan clean.
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601 --suite full --sample-mode transition --event-merge-gap 25 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> passed, crash scan clean.
- `python -c "from tools.run_labeled_domain_proof import collect_device_receipt; import json; print(json.dumps(collect_device_receipt(), indent=2))"` -> CUDA unavailable; optional GPU 10k proof skipped.

### Problems encountered
- Google Drive copy was skipped because no writable `EIDOS_PROOF_DRIVE_DIR`, `EIDOS_ARTIFACT_ROOT`, or mounted Colab Drive path was available.
- The local machine has CPU-only Torch (`2.6.0+cpu`), so the optional GPU run was not attempted.
- Eidos reports post-warmup processed frame counts, so the 250-row sample produced 225 processed frames and the 1k transition sample produced 900 processed frames.

### What changed
- `.gitignore`
- `eidos_domain_adapters.py`
- `tools/run_labeled_domain_proof.py`
- `tests/test_labeled_domain_proof_runner.py`
- `docs/proof_runs/2026-06-01/codex_journal.md`
- `docs/proof_runs/2026-06-01/plain_language_test_analysis.md`

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, hippocampus memory behavior, and core incident-card generation behavior were not changed.

### Artifacts generated
- `artifacts/cicids_webattacks_proof_precision_tiny_20260601/`
- `artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/`
- `artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/`
- `artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`

### Google Drive archive status
Drive copy was skipped or failed in each run because no verified writable Drive root was available. Each run wrote `drive_manifest.json` with the skip reason.

### Thoughts on improvement
The ledger is intentionally a transparent accounting layer. It shows that merged/deduped views can reduce alert pressure while preserving the raw event trail, but it should not be treated as threshold tuning or production-readiness evidence.

### Where to improve next
Review precision-ledger outputs across more labeled datasets before deciding whether a separate, explicitly gated calibration PR is warranted.

### Anything that stands out
The transition sample preserved raw recall visibility (`recall=1.0`) while the precision ledger separated raw alert pressure from merged/deduped accounting.

### End-of-task summary
1. Files changed: `.gitignore`, `eidos_domain_adapters.py`, `tools/run_labeled_domain_proof.py`, `tests/test_labeled_domain_proof_runner.py`, `docs/proof_runs/2026-06-01/codex_journal.md`, `docs/proof_runs/2026-06-01/plain_language_test_analysis.md`.
2. Whether core behavior changed: no.
3. Tests added or skipped: focused precision-ledger tests added; optional GPU run skipped because CUDA is unavailable.
4. Repo-root commands run: focused pytest, full pytest, tiny fixture proof, balanced 250 CPU proof, transition 1k CPU proof, CUDA/device receipt check.
5. Artifacts generated: three local proof folders plus the ignored downloaded WebAttacks CSV sample source.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: skipped or failed; no verified writable Drive path was available.
9. Known limitations: postprocessing only; no core threshold or anomaly-policy tuning.
10. Follow-up tasks not implemented: threshold calibration, production claims, or GPU 10k run.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601

### What happened today
Built and ran the precision-ledger layer for the labeled/domain proof harness.

### What was accomplished
- Added first-class binary proof-label accounting and sample receipts.
- Captured raw, merged, and deduped event metrics in a precision ledger.
- Added false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601 --suite full --sample-mode balanced --event-merge-gap 25 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: skipped or failed; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/benchmark_summary.csv
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/benchmark_summary.md
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/config.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/crash_scan.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/drive_manifest.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260601_020110_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260601_020110_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020039_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020039_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020039_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020039_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020039_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020039_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020039_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/forecast.jsonl
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260601_020109_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/incident_cards.jsonl
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/manifest.jsonl
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260601_020109_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260601_020109_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260601_020109_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260601_020109_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260601_020110_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/environment.txt
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/event_summary.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/git_commit.txt
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/confirmed_event_001.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/confirmed_event_002.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/confirmed_event_003.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/engine_card_001.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/engine_card_002.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/engine_card_003.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/engine_card_004.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/engine_card_005.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/engine_card_006.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/engine_card_007.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/engine_card_008.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/incident_cards/engine_card_009.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/labeled_metrics.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/labeled_metrics.md
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/logs/engine_output.log
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/precision_ledger.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/precision_ledger.md
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/proof_digest.json
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/proof_digest.md
- artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601/run_manifest.json

### Google Drive archive status
- Drive root used: unknown
- Drive folder used: unknown
- Files copied: 0
- Files skipped: 0
- Reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount

### End-of-task summary
1. Files changed: tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601 --suite full --sample-mode balanced --event-merge-gap 25 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 46 files under `artifacts/cicids_webattacks_proof_precision_balanced250_cpu_20260601`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: skipped or failed; no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
9. Known limitations: precision ledger is postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.

## Labeled CICIDS/WebAttacks proof harness -- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601

### What happened today
Built and ran the precision-ledger layer for the labeled/domain proof harness.

### What was accomplished
- Added first-class binary proof-label accounting and sample receipts.
- Captured raw, merged, and deduped event metrics in a precision ledger.
- Added false-positive context, attack-window timing diagnostics, device receipts, and artifact hygiene receipts.
- Kept core Eidos model behavior untouched.

### Tests and commands run
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601 --suite full --sample-mode transition --event-merge-gap 25 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` -> labeled smoke proof artifacts written.

### Problems encountered
- Google Drive status: skipped or failed; reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
- This is not threshold tuning, so misses or false positives are reported rather than optimized away.

### What changed
- tools/run_labeled_domain_proof.py
- tests/test_labeled_domain_proof_runner.py
- .gitignore
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601

### What did not change
Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, and architecture layers were not changed.

### Artifacts generated
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/benchmark_summary.csv
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/benchmark_summary.md
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/config.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/crash_scan.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/drive_manifest.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260601_020412_bicameral_stream_cicids_webattacks_labeled_proof.bin
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/compression/cicids_webattacks_labeled_proof/20260601_020412_bicameral_stream_meta_cicids_webattacks_labeled_proof.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020252_cicids_webattacks_labeled_proof_seed42/anomalies.jsonl
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020252_cicids_webattacks_labeled_proof_seed42/clusters.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020252_cicids_webattacks_labeled_proof_seed42/report.txt
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020252_cicids_webattacks_labeled_proof_seed42/session_meta.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020252_cicids_webattacks_labeled_proof_seed42/state_capsule.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020252_cicids_webattacks_labeled_proof_seed42/steps.csv
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/eidos_brain_archive/20260601_020252_cicids_webattacks_labeled_proof_seed42/summary.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/forecast.jsonl
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/hippocampus/cicids_webattacks_labeled_proof/20260601_020412_hippocampus_snapshot_cicids_webattacks_labeled_proof.pt
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/incident_cards.jsonl
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/manifest.jsonl
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/reservoir_checkpoints/cicids_webattacks_labeled_proof/20260601_020412_reservoir_checkpoint_cicids_webattacks_labeled_proof.pt
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260601_020412_reservoir_geom_cicids_webattacks_labeled_proof.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/reservoir_geometry/cicids_webattacks_labeled_proof/20260601_020412_reservoir_states_cicids_webattacks_labeled_proof.npy
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260601_020412_top_100_surprises_cicids_webattacks_labeled_proof.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/engine_artifacts/sentinel_forensics/cicids_webattacks_labeled_proof/20260601_020412_top_100_surprises_text_cicids_webattacks_labeled_proof.txt
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/environment.txt
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/event_summary.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/git_commit.txt
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/confirmed_event_001.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_001.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_002.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_003.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_004.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_005.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_006.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_007.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_008.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_009.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_010.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_011.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_012.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_013.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_014.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_015.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_016.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_017.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_018.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_019.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_020.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_021.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_022.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_023.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_024.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_025.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_026.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_027.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_028.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_029.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_030.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_031.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_032.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/incident_cards/engine_card_033.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/labeled_metrics.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/labeled_metrics.md
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/logs/engine_output.log
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/precision_ledger.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/precision_ledger.md
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/proof_digest.json
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/proof_digest.md
- artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601/run_manifest.json

### Google Drive archive status
- Drive root used: unknown
- Drive folder used: unknown
- Files copied: 0
- Files skipped: 0
- Reason: no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount

### End-of-task summary
1. Files changed: tools/run_labeled_domain_proof.py, tests/test_labeled_domain_proof_runner.py, .gitignore, artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601
2. Whether core behavior changed: no.
3. Tests added or skipped: focused labeled runner tests added; full pytest run handled outside this runner.
4. Repo-root commands run: `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601 --suite full --sample-mode transition --event-merge-gap 25 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"`.
5. Artifacts generated: 68 files under `artifacts/cicids_webattacks_proof_precision_transition1k_cpu_20260601`.
6. Plain-language analysis written: yes.
7. Journal entry written: yes.
8. Google Drive copy status: skipped or failed; no writable Colab Drive root found among: \content\drive\MyDrive, \content\drive\My Drive; local Google Drive auto-discovery skipped; set EIDOS_PROOF_DRIVE_DIR or EIDOS_ARTIFACT_ROOT to a verified Drive mount.
9. Known limitations: precision ledger is postprocessing only; no threshold tuning was attempted.
10. Follow-up tasks not implemented: threshold calibration or core behavior changes.
