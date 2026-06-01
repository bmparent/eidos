# Eidos Brain Baseline Proof Run — 2026-05

## Exact command used

```bash
python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 96 --out artifacts/readiness_smoke_2026_05_29
```

- Artifact directory: `artifacts/readiness_smoke_2026_05_29`
- Git commit: `bb1d09d4041687583632bd4d08469bf2c3081e28`
- Git branch: `codex/eidos-readiness-2026-05-29`
- Git dirty: `True`
- Config hash: `9ae00987ed7ea8c10b705e89f4f7597eab1a8fa5696653bd882906ebe153266d`
- Seed: `42`
- Frames: `96`
- Suite: `smoke`
- Scenario list: synthetic_smoke
- Pytest command: `python -m pytest -m smoke --junitxml C:\Users\bmpar\OneDrive\Documents\eidos-brain-readiness\eidos\artifacts\readiness_smoke_2026_05_29\pytest_results.xml`
- Pytest status: `passed` (pytest completed successfully)
- Frame-count note: `96` frames were used for this smoke receipt because the corrected 10000-frame Windows smoke rerun exceeded the available execution window before artifact writing. The seed/frame/suite values are still captured in config, manifest, CSV, and Markdown outputs.

## Summary table

| scenario | status | frames | eidos compression ratio | anomaly f1 | runtime seconds | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| synthetic_smoke | passed | 96 | 2.977341086825747 |  | 42.502151 | no ground-truth labels; anomaly precision/recall/f1 left blank |

## Skipped baselines and reasons

- `baseline_compression_ratio`: Week 1 runner records Eidos ratio only; external compression baselines are scheduled for later proof work.
- `detection_ground_truth_metrics`: One or more scenarios did not provide labels, so anomaly precision/recall/f1 are blank for those rows.

## Sentinel false-positive control

- Normal-only confirmed false positives per 10k frames: `0`
- Legacy raw-spike alerts on normal-only stream: `4`
- Confirmed events: `4`
- Candidate events: `9`
- Suppressed candidates: `5`
- Merged events: `0`
- Cooldown suppressions: `0`
- RED count: `2`
- AMBER count: `1`
- Mode confirmed-event counts: `{'low_noise': 1, 'balanced': 2, 'high_recall': 3}`
- Recall preservation note: Synthetic sustained burst confirmed
- Incident-card policy: cards are written for confirmed events, not every raw spike.
- Eidos Life lifecycle bridge: generation 63 collapse/recovery is treated as lifecycle events, with post-recovery nominal frames suppressed.

## Known limitations

- This runner still wraps existing engine behavior and does not tune core SentinelMonitor thresholds.
- External compression baselines are not implemented in this runner, so baseline compression ratio is blank.
- Smoke synthetic scenarios do not provide ground-truth anomaly labels, so detection precision/recall/f1 can be blank.
- No plots were produced for this smoke baseline unless a later plotting task adds them.
- False-positive control uses deterministic synthetic policy checks; broader labeled real-world validation remains future work.

## Next step

Broaden false-positive control to labeled real-world streams and compare against the checked-in smoke receipt.
