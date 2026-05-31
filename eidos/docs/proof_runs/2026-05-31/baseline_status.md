# Baseline Status -- 2026-05-31

## Current Official Baseline

The official hotfix baseline is the clean Colab GPU 10k proof receipt:

- Summary: `docs/proof_runs/2026-05-31/official_colab_gpu_10000_summary.md`
- Machine-readable receipt: `docs/proof_runs/2026-05-31/official_colab_gpu_10000_receipt.json`
- Source branch: `codex/eidos-hotfix-gpu-config-2026-05-31`
- Source commit: `3eca182f7351c382a0f47b6b5b5e3bee5c956f49`
- Origin status: branch and commit reachable on `origin`
- GPU status: CUDA exercised in clean Colab on a Tesla T4
- Proof status: 10k frames completed, pytest passed, crash scan clean

## Baseline Records

| Record | Frames | Environment | Status | Purpose |
| --- | ---: | --- | --- | --- |
| Local CPU proof | 1200 | Local CPU-only machine | Passed | Pre-Colab validation of the hotfix branch and proof-runner changes. |
| Official Colab GPU proof | 10000 | Clean Colab GPU, Tesla T4 | Passed | Official frozen GPU validation receipt for the hotfix branch. |

The 1200-frame CPU/local proof remains useful as a local validation receipt, but it is not the frozen official GPU baseline.

## Dirty State Interpretation

The official proof digest recorded `git_dirty=true` after the Colab proof run. The dirty state was caused by generated or updated receipt files after the proof source commit was checked out:

- `docs/proof_runs/2026-05-31/codex_journal.md`
- `docs/proof_runs/2026-05-31/plain_language_test_analysis.md`
- `artifacts/hotfix_official_10000_colab_gpu/`
- `full_pytest_output.txt`

The generated runtime artifacts are not committed in the docs packaging update. The important receipt facts are frozen in the summary and JSON receipt.

## Source Cleanliness After Packaging

The committed source state after this packaging update is intended to be clean for tracked files. The generated Colab runtime folder and full pytest output are intentionally left uncommitted and covered by targeted ignore rules.

## Known Limitations

- The official proof still uses synthetic smoke fixtures.
- The next proof should use labeled production-like or domain-specific telemetry.
- Detection quality should not be claimed beyond the tested synthetic fixture scope.

## Recommended Next Experiment

Run the first real labeled/domain proof using CICIDS2017/WebAttacks or controlled system telemetry.

Track these metrics:

- True positives
- False positives
- Compression ratio
- Incident-card quality
- Runtime and memory
- Crash scan
