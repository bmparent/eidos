# Plain-Language Test Analysis - 2026-05-27

## What the task attempted

This task moved the Sentinel V3 controlled-regime calibration patch into repo code. The new code tests whether Sentinel can stay quiet on normal streams while escalating known abnormal regimes: periodic/backdoor, noise crash, and frozen low-variance collapse.

## Why the test matters

The prior controlled notebook run showed that residual spikes alone were too quiet. The new proof checks a broader policy: residual evidence is only one input, and confirmed AMBER/RED states now also depend on persistence, input-stat deviation, geometry collapse, novelty, hysteresis, and cooldown behavior.

## What was tested

The tests covered finite residual statistics, normal suppression, frozen RED detection, noise crash alerting, periodic/backdoor alerting, and the rule that detector APIs do not need regime labels. A smoke runner also executed a two-reservoir controlled grid and checked that required CSV/JSON artifacts were written.

## What passed

- `tests/test_sentinel_v3_calibration.py` passed, 6 tests.
- `tests/test_controlled_regimes_smoke.py` passed, 1 test.
- The combined focused suite passed, 7 tests.
- The acceptance artifact has `pass_all=True` for both reservoir rows tested.

## What failed

An extra compatibility check against the older `tests/test_sentinel.py` failed because the current legacy Sentinel returns `GREEN: NOMINAL` while that test expects exactly `NOMINAL`. This patch did not change the legacy Sentinel engine behavior, so that mismatch is recorded as a pre-existing or separate legacy-test issue rather than fixed here.

## What artifacts were generated

Artifacts were saved under:

```text
artifacts/sentinel_v3_patch/
```

The key files are:

```text
summary_v3_patch.csv
summary_v3_patch_full.json
run_manifest_v3_patch.json
per_regime_summary_v3_patch.csv
acceptance_v3_patch.csv
eidos_v3_steps_1400_reservoir_64.csv
eidos_v3_steps_1400_reservoir_128.csv
config_v3_patch.json
drive_manifest.json
```

## What was saved locally

All required proof artifacts were saved locally in `artifacts/sentinel_v3_patch/`.

## What was saved to Google Drive

No files were copied to Google Drive. The run wrote `drive_manifest.json` with the skip reason: no writable `EIDOS_PROOF_DRIVE_DIR`, `EIDOS_ARTIFACT_ROOT`, or mounted Colab Drive path was available in this local environment.

## What remains uncertain

The CI-sized proof used reservoirs 64 and 128 with 16 features and 300 frames per regime. The CLI defaults support the larger requested grid of 256, 768, and 1536 reservoirs, but that larger manual run was not executed in this pass to keep local validation bounded.

## What should happen next

The next PR-sized step is to run the larger manual grid and compare the V3 controlled acceptance outputs against the earlier Colab failure artifacts.
