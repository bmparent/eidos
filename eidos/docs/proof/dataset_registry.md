# Eidos CICIDS/WebAttacks Dataset Registry

- Generated at UTC: `2026-07-02T02:25:08Z`
- Verdict: `larger_labeled_dataset_available`
- Selected dataset: `C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\cicids_webattacks_samples\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`

## Search Roots

- `C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\cicids_webattacks_samples\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `tests\fixtures`
- `artifacts`
- `C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\cicids_webattacks_samples`
- `\tmp\eidos_proof_data`
- `E:\agent data`

## Candidates

| path | type | rows | benign | attack | label column | balanced 250 | transition 1k | GPU 10k rows | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| artifacts/sentinel_calibration_guardrails_2026_06_30/runs/natural_attack_replay_cpu/off/engine_artifacts/eidos_brain_archive/20260701_005428_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_calibration_guardrails_2026_06_30/runs/natural_attack_replay_cpu/off/engine_artifacts/eidos_brain_archive/20260701_025801_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_calibration_guardrails_2026_06_30/runs/natural_attack_replay_cpu/off/engine_artifacts/eidos_brain_archive/20260701_030346_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_calibration_guardrails_2026_06_30/runs/normal_only_negative_control/off/engine_artifacts/eidos_brain_archive/20260701_024606_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_calibration_guardrails_2026_06_30/runs/normal_only_negative_control/off/engine_artifacts/eidos_brain_archive/20260701_025900_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_calibration_guardrails_2026_06_30/runs/normal_only_negative_control/off/engine_artifacts/eidos_brain_archive/20260701_030456_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_calibration_guardrails_2026_06_30/runs/tiny_fixture_smoke/off/engine_artifacts/eidos_brain_archive/20260701_003257_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_calibration_guardrails_2026_06_30/runs/tiny_fixture_smoke/off/engine_artifacts/eidos_brain_archive/20260701_004355_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_calibration_guardrails_2026_06_30/runs/tiny_fixture_smoke/off/engine_artifacts/eidos_brain_archive/20260701_005141_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_calibration_guardrails_2026_06_30/runs/tiny_fixture_smoke/off/engine_artifacts/eidos_brain_archive/20260701_025656_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_calibration_guardrails_2026_06_30/runs/tiny_fixture_smoke/off/engine_artifacts/eidos_brain_archive/20260701_030249_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/balanced_250_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011018_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/tiny_fixture_smoke/off/engine_artifacts/eidos_brain_archive/20260702_010606_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| artifacts/sentinel_guardrail_scale_matrix_2026_07_01/runs/transition_1k_cpu/off/engine_artifacts/eidos_brain_archive/20260702_011728_cicids_webattacks_labeled_proof_seed42/steps.csv | csv | 0 | 0 | 0 | NA | False | False | False | label column 'Label' not found |
| C:\Users\bmpar\OneDrive\Documents\eidos-brain\eidos\artifacts\cicids_webattacks_samples\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | csv | 170366 | 168186 | 2180 |  Label | True | True | False | ok |
| tests/fixtures/cicids_webattacks_tiny.csv | csv | 12 | 8 | 4 | Label | False | False | False | ok |

## Proof Logic + Meaning

The registry turns local dataset availability into a receipt. This prevents the proof harness from quietly falling back to a tiny fixture when a larger labeled CSV is missing or when a discovered file has a label-column problem.

Known limits: the current proof runner consumes CSV input. Parquet candidates are recorded but not used unless converted or supported by a future proof-side adapter.
