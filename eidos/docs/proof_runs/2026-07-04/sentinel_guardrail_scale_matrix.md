# Sentinel Guardrail Scale Matrix

- Final verdict: `MERGE_READY_LARGER_LABELED_GUARDRAILS`
- Selected dataset: `artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- Dataset rows: `170366`
- Benign / attack rows: `168186` / `2180`
- Raw label distribution: `{'BENIGN': 168186, 'Web Attack � Brute Force': 1507, 'Web Attack � XSS': 652, 'Web Attack � Sql Injection': 21}`
- Normalized label distribution: `{'BENIGN': 168186, 'ATTACK': 2180}`
- Branch pushed before run: `True`
- Core behavior changed: `False`
- Core-touch policy: `True`
- Normal-only FP/10k target: `<=10`; stretch target: `<=5`
- Completed proof legs: `balanced_250_cpu, natural_larger_replay_cpu, normal_only_negative_control, tiny_fixture_smoke, transition_1k_cpu`
- Exact merge-readiness reason: `All required larger labeled CPU guardrails completed without FP, recall, crash, or core-touch holds.`

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
| tiny_fixture_smoke | off | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | 0 | NA | 0 | NA | 0 | ROW_PASS |
| tiny_fixture_smoke | low_noise | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | 0 | NA | 0 | NA | 0 | ROW_PASS |
| tiny_fixture_smoke | balanced | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | 0 | NA | 0 | NA | 0 | ROW_PASS |
| tiny_fixture_smoke | high_recall | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | 0 | NA | 0 | NA | 0 | ROW_PASS |
| tiny_fixture_smoke | strict | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | NA | 0 | NA | 0 | NA | 0 | ROW_PASS |
| balanced_250_cpu | off | 225 | 125 | 125 | 35 | 1 | 1 | 35 | 35 | 844.444 | 0.457143 | 0.444444 | 0.450704 | 65.5172 | 0 | 0 | ROW_PASS |
| balanced_250_cpu | low_noise | 225 | 125 | 125 | 35 | 1 | 1 | 1 | 1 | 0 | 1 | 0.166667 | 0.285714 | 91.3793 | 0 | 0 | ROW_PASS |
| balanced_250_cpu | balanced | 225 | 125 | 125 | 35 | 1 | 1 | 1 | 1 | 0 | 1 | 0.166667 | 0.285714 | 91.3793 | 0 | 0 | ROW_PASS |
| balanced_250_cpu | high_recall | 225 | 125 | 125 | 35 | 1 | 1 | 1 | 1 | 0 | 1 | 0.166667 | 0.285714 | 91.3793 | 0 | 0 | ROW_PASS |
| balanced_250_cpu | strict | 225 | 125 | 125 | 35 | 1 | 1 | 1 | 1 | 0 | 1 | 0.166667 | 0.285714 | 91.3793 | 0 | 0 | ROW_PASS |
| transition_1k_cpu | off | 900 | 500 | 500 | 80 | 3 | 3 | 80 | 80 | 533.333 | 0.4 | 1 | 0.571429 | 100 | 0 | 0 | ROW_PASS |
| transition_1k_cpu | low_noise | 900 | 500 | 500 | 80 | 3 | 3 | 3 | 3 | 0 | 1 | 1 | 1 | 100 | 0 | 0 | ROW_PASS |
| transition_1k_cpu | balanced | 900 | 500 | 500 | 80 | 3 | 3 | 3 | 3 | 0 | 1 | 1 | 1 | 100 | 0 | 0 | ROW_PASS |
| transition_1k_cpu | high_recall | 900 | 500 | 500 | 80 | 3 | 3 | 3 | 3 | 0 | 1 | 1 | 1 | 100 | 0 | 0 | ROW_PASS |
| transition_1k_cpu | strict | 900 | 500 | 500 | 80 | 3 | 3 | 3 | 3 | 0 | 1 | 1 | 1 | 100 | 0 | 0 | ROW_PASS |
| natural_larger_replay_cpu | off | 451 | 496 | 5 | 29 | 1 | 1 | 29 | 29 | 598.67 | 0.0689655 | 0.4 | 0.117647 | 40 | 0 | 0 | ROW_PASS |
| natural_larger_replay_cpu | low_noise | 451 | 496 | 5 | 29 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 100 | 0 | 0 | ROW_PASS |
| natural_larger_replay_cpu | balanced | 451 | 496 | 5 | 29 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 100 | 0 | 0 | ROW_PASS |
| natural_larger_replay_cpu | high_recall | 451 | 496 | 5 | 29 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 100 | 0 | 0 | ROW_PASS |
| natural_larger_replay_cpu | strict | 451 | 496 | 5 | 29 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 100 | 0 | 0 | ROW_PASS |
| normal_only_negative_control | off | 900 | 1000 | NA | 35 | 4 | 4 | 35 | 35 | 388.889 | 0 | NA | NA | NA | NA | 0 | ROW_PASS |
| normal_only_negative_control | low_noise | 900 | 1000 | NA | 35 | 4 | 4 | 3 | 0 | 0 | NA | NA | NA | NA | NA | 0 | ROW_PASS |
| normal_only_negative_control | balanced | 900 | 1000 | NA | 35 | 4 | 4 | 3 | 0 | 0 | NA | NA | NA | NA | NA | 0 | ROW_PASS |
| normal_only_negative_control | high_recall | 900 | 1000 | NA | 35 | 4 | 4 | 4 | 0 | 0 | NA | NA | NA | NA | NA | 0 | ROW_PASS |
| normal_only_negative_control | strict | 900 | 1000 | NA | 35 | 4 | 4 | 3 | 0 | 0 | NA | NA | NA | NA | NA | 0 | ROW_PASS |

## Skipped Legs

- `gpu_10k_optional`: CUDA unavailable; torch reports CPU-only runtime

## Proof Logic + Meaning

Goal reached: larger labeled CICIDS/WebAttacks availability was turned into a registry and, where data allowed, a scale matrix with raw, merged, deduped, confirmed, and calibrated views side by side. Gate status is `MERGE_READY_LARGER_LABELED_GUARDRAILS`.

Specific logic/math used: the proof compares event funnel counts, FP/10k, precision, recall, F1, attack-window coverage, first detection latency, crash-hit count, runtime, FPS, and core-touch policy without changing core Eidos behavior.

Why this is better than the previous state: the earlier guardrail run was limited to the tiny fixture. This package records whether a larger real labeled source exists, what it contains, and what the proof harness did with it.

Evidence/artifacts: `scale_matrix.json`, `scale_matrix.csv`, `scale_matrix.md`, `scale_matrix_before_after.json`, `scale_matrix_before_after.csv`, `scale_matrix_before_after.md`, per-run proof receipts, dataset registry, plots, core-touch receipt, and Drive manifest when available.

What it proves: proof-side reproducibility and larger-data guardrail accounting improved. It proves only the rows and profiles actually run.

What it does not prove: production readiness, every CICIDS/WebAttacks variant, GPU behavior when CUDA is unavailable, or any core behavior improvement.

## Remaining Uncertainty

- All required larger labeled CPU guardrails completed without FP, recall, crash, or core-touch holds.

How this moves Eidos closer to the ultimate goal: Eidos is not becoming more intelligent because it speaks less. It is becoming more intelligent only if it speaks less while preserving truth, preserving anomaly visibility, and making uncertainty auditable.
