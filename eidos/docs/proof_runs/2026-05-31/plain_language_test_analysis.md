# Plain-Language Test Analysis -- 2026-05-31

## What the task attempted

This task turned a CUDA tensor/config hotfix into a cleaner proof branch. It also made the proof runner more useful by adding external compression baselines and a compact digest that records the most important proof numbers and crash-scan result.

Follow-up packaging note: the earlier local proof in this file was a 1200-frame CPU-only validation. The official frozen proof receipt is now the clean Colab GPU 10k run from commit `3eca182f7351c382a0f47b6b5b5e3bee5c956f49`, recorded in `official_colab_gpu_10000_summary.md` and `official_colab_gpu_10000_receipt.json`.

## Why the test matters

The original Colab failure happened when CUDA tensors entered CPU/NumPy similarity code. The new tests and validation script make sure tensor-like signatures are safely detached, moved to CPU, and converted before similarity logic runs. The proof-runner additions make future receipts easier to compare because they now include external compression baselines and a small JSON/Markdown digest.

## What was tested

- CPU-safe conversion for Python lists and NumPy arrays.
- CPU torch tensor conversion when torch is installed.
- CUDA torch tensor conversion when CUDA is available, with a clean skip when it is not.
- Incident-card similarity with tensor signatures.
- Procedural-memory prototype/update/ranking with tensor signatures.
- Forecast update/risk/similarity with tensor signatures.
- Packaged engine `validate_config` success and failure cases.
- The Colab/local validation script.
- Compression baseline utility behavior and optional dependency skip reporting.
- Proof digest/crash scan behavior.
- Full repository pytest.
- A 1200-frame proof smoke run.

## What passed

- Focused hotfix/config tests passed: 17 passed, 1 skipped.
- Proof-runner and GPU-smoke wrapper tests passed: 10 passed.
- Standalone validation script passed on CPU fallback.
- Full pytest passed: 64 passed, 1 skipped.
- 1200-frame proof passed and wrote `artifacts/hotfix_official_1200`.
- Crash scan found no `CRASH IN INCIDENT LOGIC`, `can't convert cuda`, or `Traceback` strings in the generated proof logs/text files.

## What failed

No test or proof command failed in the local CPU-only verification.

## What artifacts were generated

- `artifacts/hotfix_official_1200`

Important files inside that folder include:

- `benchmark_summary.csv`
- `benchmark_summary.md`
- `config.json`
- `run_manifest.json`
- `drive_manifest.json`
- `event_summary.json`
- `proof_digest.json`
- `proof_digest.md`
- `pytest_results.xml`
- `logs/false_positive_control.jsonl`
- `incident_cards/`
- `scenarios/`

## What was saved locally

The official local proof artifacts were saved under `artifacts/hotfix_official_1200`. Generated artifact folders are intentionally left out of the commit.

## What was saved to Google Drive

Nothing was copied to Google Drive from this local run. No verified Drive mount or configured `EIDOS_PROOF_DRIVE_DIR` / `EIDOS_ARTIFACT_ROOT` was available.

## Key 1200-frame proof numbers

- Frames: 1200
- Runtime seconds: recorded in `proof_digest.json` for the generated run
- Eidos compression ratio: 7.3438164045898855
- Raw bytes: 614400
- Eidos bytes: 83662
- Best external baseline: lzma at 1.130027
- zlib ratio: 1.042393
- lzma ratio: 1.130027
- delta+zlib ratio: 1.084397
- zstandard: skipped because `zstandard` is not installed
- lz4: skipped because `lz4` is not installed
- Normal-only confirmed false positives per 10k synthetic frames: 0
- Legacy raw-spike alerts: 4
- Confirmed events: 4
- Candidate events: 9
- Suppressed candidates: 5
- Incident cards: 4
- Crash scan: clean, 0 hits

## What remains uncertain

CUDA was not available during the local 1200-frame validation, but the later official Colab receipt exercised CUDA on a Tesla T4 and completed the 10k proof. What remains uncertain is performance on labeled production-like telemetry rather than synthetic smoke fixtures.

## What should happen next

Run the first real labeled/domain proof using CICIDS2017/WebAttacks or controlled system telemetry. Track true positives, false positives, compression ratio, incident-card quality, runtime/memory, and crash scan.
