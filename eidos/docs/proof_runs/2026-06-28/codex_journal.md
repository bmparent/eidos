# Codex Journal - 2026-06-28

## What happened today

I verified the current Eidos Brain / Sentinel labeled-domain proof harness on a new local branch, kept the work proof-side, ran the requested tests, completed the CPU proof matrix, refreshed crash-scan receipts, and prepared the next Sentinel calibration prompt without changing core engine behavior.

## What was accomplished

- Created branch `codex/eidos-sentinel-proof-verification-2026-06-28` from current `main`.
- Wrote `artifacts/eidos_current_state_audit.md`.
- Confirmed the older CICIDS/WebAttacks proof branch is already an ancestor of `main`.
- Verified Precision Ledger support for label parsing, normalized labels, sample modes, raw/merged/deduped event views, false-positive taxonomy, attack-window diagnostics, incident-card accounting, device receipts, Drive manifests, and artifact hygiene.
- Added a portable ignore rule for `artifacts/proof_runs/`.
- Bounded the Colab GPU bridge dry-run receipt so it no longer performs heavyweight `pip freeze` during bridge-only dry runs.
- Expanded crash scanning to include `RuntimeError`, `ValueError`, `NaN`, and `Inf`.
- Prevented generated scan/digest receipts from self-triggering crash scans.
- Recorded known nonfatal `HIPP ... sim=NaN` telemetry as a warning instead of hiding it or counting it as a runtime crash.
- Completed tiny fixture, balanced 250 CPU, transition 1k CPU, and natural-order CPU proof runs.
- Wrote an optional GPU 10k skip receipt because CUDA was unavailable.
- Mirrored proof artifacts and refreshed crash/digest receipts to Google Drive.

## Tests and commands run

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_labeled_domain_proof_runner.py -q` - passed, 24 tests.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_colab_gpu_bridge.py -q` - passed, 5 tests after bridge dry-run receipt patch.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_proof_baseline_runner.py tests/test_labeled_domain_proof_runner.py -q` - passed, 34 tests after crash-scan patch.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` - final pass, 175 passed, 2 skipped, 11 warnings.
- `powershell -ExecutionPolicy Bypass -File tools/configure_proof_drive_env.ps1 -CheckOnly` - found writable `G:\My Drive`.
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 12 --seed 42 --out artifacts/proof_runs/tiny_fixture_smoke --suite smoke --sample-mode natural --event-merge-gap 25 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` - passed.
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/proof_runs/cicids_webattacks_balanced_250_cpu --suite full --sample-mode balanced --event-merge-gap 25 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` - passed.
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/proof_runs/cicids_webattacks_transition_1k_cpu --suite full --sample-mode transition --event-merge-gap 25 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` - passed.
- `python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 13000 --seed 42 --out artifacts/proof_runs/cicids_webattacks_natural_cpu --suite full --sample-mode natural --event-merge-gap 25 --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"` - passed.

## Problems encountered

- The worktree was already dirty before my changes because of a modified parent `README.md` and many pre-existing untracked OneDrive duplicate files. I did not clean or stage those unrelated files.
- Full pytest initially failed because `tools/colab_gpu_bridge.py --dry-run` timed out in the full-suite subprocess. The bridge now writes a lightweight bridge environment receipt; child proof runs still write full `environment.txt`.
- The previous crash scanner did not include every pattern requested in this task. I expanded it and added tests.
- A naive expanded scanner self-triggered on generated `crash_scan.json` metadata. I updated the scanner to ignore generated scan/digest receipts.
- The natural run logs contain 5 expected nonfatal `HIPP bank=INCIDENT sim=NaN` telemetry lines. They are recorded as warnings, not crash hits.
- CUDA was unavailable: torch is installed as `2.6.0+cpu`, so the optional GPU 10k run was skipped with a receipt.

## What changed

- `.gitignore` now ignores `artifacts/proof_runs/`.
- `tools/colab_gpu_bridge.py` now writes bounded bridge-only environment receipts.
- `tools/run_proof_baseline.py` now scans for the full requested crash pattern set and separates known nonfatal HIPP NaN telemetry warnings from crash hits.
- `tests/test_colab_gpu_bridge.py` and `tests/test_proof_baseline_runner.py` cover those proof-tooling changes.
- `docs/proof_runs/2026-06-28/` records this verification task.

## What did not change

Core model behavior was untouched. Reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression codec behavior, hippocampus memory behavior, and core incident-card generation were not changed.

## Artifacts generated

- `artifacts/eidos_current_state_audit.md`
- `artifacts/eidos_sentinel_proof_verification_summary.md`
- `artifacts/next_codex_prompt_sentinel_calibration_v1.md`
- `artifacts/proof_runs/tiny_fixture_smoke/`
- `artifacts/proof_runs/cicids_webattacks_balanced_250_cpu/`
- `artifacts/proof_runs/cicids_webattacks_transition_1k_cpu/`
- `artifacts/proof_runs/cicids_webattacks_natural_cpu/`
- `artifacts/proof_runs/cicids_webattacks_gpu_10k/`

## Google Drive archive status

Google Drive was available at `G:\My Drive` after setting `EIDOS_PROOF_DRIVE_DIR` in-process. Proof artifacts were copied to `G:\My Drive\Eidos_Brain_Proof_Phase\2026-06-28\...`.

- Tiny fixture: copied.
- Balanced 250 CPU: copied.
- Transition 1k CPU: copied.
- Natural CPU: copied.
- Optional GPU 10k skip receipt: copied.

## Thoughts on improvement

The proof harness is now strong enough to show both the value and the risk of postprocessed views. Merged/deduped views sharply reduce FP pressure, but balanced mode loses recall after merging. The next calibration step needs strict recall and attack-window guardrails.

## Where to improve next

Run Sentinel calibration v1 as a proof-harness-first postprocessing comparison with candidate confirmation windows, persistence requirements, merge/cooldown tuning, mode-specific profiles, and normal-only negative controls.

## Anything that stands out

Natural order exposed the clearest raw false-positive pressure: 604 raw events, 602 raw false positives, and 514.53 raw FP/10k frames. Merged/deduped views reduced that to 2 events and 0.85 FP/10k while preserving a merged/deduped recall view of 1.0, but raw attack-window diagnostics still showed only 2 of 5 windows detected before interpretation.

## End-of-task summary

1. Files changed: `.gitignore`, `tools/colab_gpu_bridge.py`, `tools/run_proof_baseline.py`, `tests/test_colab_gpu_bridge.py`, `tests/test_proof_baseline_runner.py`, `docs/proof_runs/2026-06-28/codex_journal.md`, `docs/proof_runs/2026-06-28/plain_language_test_analysis.md`.
2. Whether core behavior changed: no.
3. Tests added or skipped: added bridge environment and crash-scan tests; no requested tests skipped.
4. Repo-root commands run: targeted pytest, full pytest, Drive check, and four labeled-domain proof runs listed above.
5. Artifacts generated: audit, summary, next prompt, four CPU proof folders, one GPU skip folder.
6. Plain-language analysis written: `docs/proof_runs/2026-06-28/plain_language_test_analysis.md`.
7. Journal entry written: `docs/proof_runs/2026-06-28/codex_journal.md`.
8. Google Drive copy status: succeeded for completed proof runs and GPU skip receipt.
9. Known limitations: natural replay processed 11,700 of 13,000 requested frames; CUDA unavailable; pre-existing unrelated worktree dirt remains.
10. Follow-up tasks not implemented: Sentinel calibration v1 itself, core engine changes, threshold changes, GPU 10k execution.
