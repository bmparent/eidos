# Eidos Brain Baseline Proof Run — 2026-05

## Exact command used

```bash
python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 1200 --out artifacts/proof_baseline_2026_05
```

- Artifact directory: `artifacts/proof_baseline_2026_05`
- Git commit: `a99f9751f04eb07f7086f32029057d8fe7c19390`
- Git branch: `main`
- Git dirty: `True`
- Config hash: `b29aae8afac95ba63408ce04c8f90fc4e8885f1b5258c9d275dffd0a6e16b5ab`
- Seed: `42`
- Frames: `1200`
- Suite: `smoke`
- Scenario list: synthetic_smoke
- Pytest command: `python -m pytest -m smoke --junitxml C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\proof_baseline_2026_05\pytest_results.xml`
- Pytest status: `passed` (pytest completed successfully)
- Frame-count note: `1200` frames were used for this smoke receipt because the corrected 10000-frame Windows smoke rerun exceeded the available execution window before artifact writing. The seed/frame/suite values are still captured in config, manifest, CSV, and Markdown outputs.

## Summary table

| scenario | status | frames | eidos compression ratio | anomaly f1 | runtime seconds | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| synthetic_smoke | passed | 1200 | 7.3438164045898855 |  | 409.928303 | no ground-truth labels; anomaly precision/recall/f1 left blank |

## Skipped baselines and reasons

- `baseline_compression_ratio`: Week 1 runner records Eidos ratio only; external compression baselines are scheduled for later proof work.
- `detection_ground_truth_metrics`: One or more scenarios did not provide labels, so anomaly precision/recall/f1 are blank for those rows.

## Known limitations

- Week 1 freezes the baseline package and does not tune Sentinel thresholds.
- External compression baselines are not implemented in this runner, so baseline compression ratio is blank.
- Smoke synthetic scenarios do not provide ground-truth anomaly labels, so detection precision/recall/f1 can be blank.
- No plots were produced for this smoke baseline unless a later plotting task adds them.

## Next step

Week 2 false-positive suppression is the next proof-plan step and was not implemented today.
