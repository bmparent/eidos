# Eidos CICIDS/WebAttacks Dataset Registry

- Generated at UTC: `2026-07-04T19:13:51Z`
- Verdict: `larger_labeled_dataset_available`
- Selected dataset: `artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`

## Search Roots

- `artifacts\cicids_webattacks_samples\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `tests\fixtures`
- `artifacts\cicids_webattacks_samples`

## Candidates

| path | type | rows | benign | attack | label column | balanced 250 | transition 1k | GPU 10k rows | reason |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | csv | 170366 | 168186 | 2180 |  Label | True | True | False | ok |
| tests/fixtures/cicids_webattacks_tiny (1).csv | csv | 12 | 8 | 4 | Label | False | False | False | ok |
| tests/fixtures/cicids_webattacks_tiny.csv | csv | 12 | 8 | 4 | Label | False | False | False | ok |

## Proof Logic + Meaning

The registry turns local dataset availability into a receipt. This prevents the proof harness from quietly falling back to a tiny fixture when a larger labeled CSV is missing or when a discovered file has a label-column problem.

Known limits: the current proof runner consumes CSV input. Parquet candidates are recorded but not used unless converted or supported by a future proof-side adapter.
