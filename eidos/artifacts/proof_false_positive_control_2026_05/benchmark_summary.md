# Eidos Brain Baseline Proof Run — 2026-05

## Exact command used

```bash
python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 1200 --out artifacts/proof_false_positive_control_2026_05
```

- Artifact directory: `artifacts/proof_false_positive_control_2026_05`
- Git commit: `89155b57f9a5486c9c586a0c9facfaf29fce30b0`
- Git branch: `codex/real-world-corpus-v0`
- Git dirty: `True`
- Config hash: `9c2d9a6bc5bb8b2f5a117d13daf51e5b264383d9e3fc79624106a323d9597a05`
- Seed: `42`
- Frames: `1200`
- Suite: `smoke`
- Scenario list: synthetic_smoke
- Pytest command: `python -m pytest tests/test_sentinel.py tests/test_proof_artifacts.py tests/test_proof_baseline_runner.py tests/test_sentinel_false_positive_control.py tests/test_sentinel_event_confirmation.py tests/test_sentinel_modes.py tests/test_incident_card_confirmation.py --junitxml C:\Users\bmpar\OneDrive\Documents\eidos-brain-real-world-corpus\eidos\artifacts\proof_false_positive_control_2026_05\pytest_results.xml`
- Pytest status: `passed` (pytest completed successfully)
- Frame-count note: `1200` frames were used for this smoke receipt because the corrected 10000-frame Windows smoke rerun exceeded the available execution window before artifact writing. The seed/frame/suite values are still captured in config, manifest, CSV, and Markdown outputs.

## Summary table

| scenario | status | frames | eidos compression ratio | anomaly f1 | runtime seconds | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| synthetic_smoke | passed | 1200 | 7.3438164045898855 |  | 95.793405 | no ground-truth labels; anomaly precision/recall/f1 left blank |

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
