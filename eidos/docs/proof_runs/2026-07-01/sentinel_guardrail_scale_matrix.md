# Sentinel Guardrail Scale Matrix

- Final verdict: `CALIBRATION_ONLY_NEEDS_TUNING`
- Selected dataset: `C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\cicids_webattacks_samples\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- Branch pushed before run: `True`
- Core behavior changed: `False`
- Core-touch policy: `True`

## Metrics And Formulas

```text
FP/10k = false_positive_events / benign_frames * 10000
precision = true_positive_events / max(true_positive_events + false_positive_events, 1)
recall = detected_attack_windows / max(total_attack_windows, 1)
F1 = 2 * precision * recall / max(precision + recall, epsilon)
attack_window_coverage = attack_windows_with_detection / total_attack_windows
```

## Scale Matrix

| run | profile | frames | benign | attack | raw | merged | deduped | confirmed | calibrated | FP/10k | precision | recall | F1 | coverage | latency | crash | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tiny_fixture_smoke | off | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | 0 | NA | 0 | NA | 0 | APPROVE |
| tiny_fixture_smoke | low_noise | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | 0 | NA | 0 | NA | 0 | APPROVE |
| tiny_fixture_smoke | balanced | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | 0 | NA | 0 | NA | 0 | APPROVE |
| tiny_fixture_smoke | high_recall | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | 0 | NA | 0 | NA | 0 | APPROVE |
| balanced_250_cpu | off | 225 | 125 | 125 | 12 | 1 | 1 | 12 | 12 | 222.222 | 0.583333 | 0.259259 | 0.358974 | 65.5172 | 0 | 0 | APPROVE |
| balanced_250_cpu | low_noise | 225 | 125 | 125 | 12 | 1 | 1 | 1 | 1 | 0 | 1 | 0.166667 | 0.285714 | 91.3793 | 0 | 0 | APPROVE |
| balanced_250_cpu | balanced | 225 | 125 | 125 | 12 | 1 | 1 | 1 | 1 | 0 | 1 | 0.166667 | 0.285714 | 91.3793 | 0 | 0 | APPROVE |
| balanced_250_cpu | high_recall | 225 | 125 | 125 | 12 | 1 | 1 | 1 | 1 | 0 | 1 | 0.166667 | 0.285714 | 91.3793 | 0 | 0 | APPROVE |
| transition_1k_cpu | off | 900 | 500 | 500 | 34 | 3 | 3 | 34 | 34 | 188.889 | 0.5 | 1 | 0.666667 | 100 | 0 | 0 | APPROVE |
| transition_1k_cpu | low_noise | 900 | 500 | 500 | 34 | 3 | 3 | 3 | 3 | 0 | 1 | 1 | 1 | 100 | 0 | 0 | APPROVE |
| transition_1k_cpu | balanced | 900 | 500 | 500 | 34 | 3 | 3 | 3 | 3 | 0 | 1 | 1 | 1 | 100 | 0 | 0 | APPROVE |
| transition_1k_cpu | high_recall | 900 | 500 | 500 | 34 | 3 | 3 | 3 | 3 | 0 | 1 | 1 | 1 | 100 | 0 | 0 | APPROVE |
| normal_only_negative_control | off | 900 | 1000 | NA | 35 | 4 | 4 | 35 | 35 | 388.889 | 0 | NA | NA | NA | NA | 0 | APPROVE |
| normal_only_negative_control | low_noise | 900 | 1000 | NA | 35 | 4 | 4 | 3 | 3 | 33.3333 | 0 | NA | NA | NA | NA | 0 | APPROVE |
| normal_only_negative_control | balanced | 900 | 1000 | NA | 35 | 4 | 4 | 3 | 3 | 33.3333 | 0 | NA | NA | NA | NA | 0 | APPROVE |
| normal_only_negative_control | high_recall | 900 | 1000 | NA | 35 | 4 | 4 | 4 | 4 | 44.4444 | 0 | NA | NA | NA | NA | 0 | APPROVE |

## Skipped Legs

- `natural_larger_replay_cpu`: existing partial run lacks run_manifest.json or labeled_metrics.json; rerun skipped because --rerun-existing was not set
- `gpu_10k_optional`: CUDA unavailable; torch reports CPU-only runtime

## Proof Logic + Meaning

Goal reached: larger labeled CICIDS/WebAttacks availability was turned into a registry and, where data allowed, a scale matrix with raw, merged, deduped, confirmed, and calibrated views side by side.

Specific logic/math used: the proof compares event funnel counts, FP/10k, precision, recall, F1, attack-window coverage, first detection latency, crash-hit count, runtime, FPS, and core-touch policy without changing core Eidos behavior.

Why this is better than the previous state: the earlier guardrail run was limited to the tiny fixture. This package records whether a larger real labeled source exists, what it contains, and what the proof harness did with it.

Evidence/artifacts: `scale_matrix.json`, `scale_matrix.csv`, `scale_matrix.md`, per-run proof receipts, dataset registry, plots, core-touch receipt, and Drive manifest when available.

What it proves: proof-side reproducibility and larger-data guardrail accounting improved. It proves only the rows and profiles actually run.

What it does not prove: production readiness, every CICIDS/WebAttacks variant, GPU behavior when CUDA is unavailable, or any core behavior improvement.

How this moves Eidos closer to the ultimate goal: Eidos is not becoming more intelligent because it speaks less. It is becoming more intelligent only if it speaks less while preserving truth, preserving anomaly visibility, and making uncertainty auditable.
