# Calibration Ledger Baseline Scaffold

This proof-phase PR adds reporting and audit receipts around the labeled-domain
proof harness. It does not change Eidos reservoir dynamics, RLS updates,
Sentinel anomaly policy, compression behavior, hippocampus memory, incident-card
core generation, or domain-adapter math.

## What This Adds

- `calibration_gate.json` and `calibration_gate.md` for every labeled proof run.
- `precision_ledger.json` schema `precision_ledger_v1.1`, with raw, merged,
  deduped, calibrated, and attack-context event views visible side by side.
- Sampling semantics receipts explaining what each sample mode proves and what
  it does not prove.
- Attack-window diagnostics with first-detection latency, coverage, and
  calibration-suppression warnings.
- False-positive taxonomy records for each event view.
- Incident-card accounting across engine cards, Sentinel confirmations, proof
  events, duplicates, and event/card coverage.
- A lightweight calibration ratchet with Tier A, Tier B, and Tier C status.
- A minimal baseline scaffold that registers required compression and detector
  baselines and records explicit skip reasons when a baseline is not executed.

## What This Does Not Change

- No reservoir, RLS, Sentinel threshold, anomaly policy, compression codec,
  memory, incident-card generation, or thermodynamic/plasticity controller logic
  is changed.
- This is not residual codec v2.
- This is not GPU or grid-search tuning.
- This is not the full Week 3 baseline competitor matrix.
- This is not familiarity-aware memory.

## Why Every View Is Shown

FP reduction is only meaningful if reviewers can see where alerts went. The
ledger keeps these views visible:

| view | meaning |
| --- | --- |
| `raw_events` | Engine cards and Sentinel confirmations before proof postprocessing. |
| `merged_events` | Overlapping or near-overlapping event windows merged by `--event-merge-gap`. |
| `deduped_events` | Repeated cards collapsed inside broader event regions. |
| `calibrated_events` | Events that survive the active proof calibration/profile settings. |
| `attack_context_events` | Events that intersect or sit near configured attack windows. |

Reports may prefer merged, deduped, or calibrated interpretation, but raw events
must remain visible. If calibrated results are present and raw results are
hidden, the calibration gate fails.

## Proof Commands

Run from the Eidos repo root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_labeled_domain_proof_runner.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q

python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file tests/fixtures/cicids_webattacks_tiny.csv --label-column Label --frames 6 --seed 42 --out artifacts/calibration_ledger_tiny_fixture_cpu --suite smoke --sample-mode balanced --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --attack-labels "Web Attack - Brute Force"

python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 250 --seed 42 --out artifacts/calibration_ledger_balanced250_cpu --suite smoke --sample-mode balanced --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"

python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/calibration_ledger_balanced1k_cpu --suite smoke --sample-mode balanced --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"

python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/calibration_ledger_transition1k_cpu --suite smoke --sample-mode transition --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"

python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 1000 --seed 42 --out artifacts/calibration_ledger_natural_attack_windows_cpu --suite smoke --sample-mode natural_attack_windows --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"
```

The natural-order 10k CPU leg can be run with the same WebAttacks file and
`--sample-mode natural --frames 10000`. In this local proof pass it was skipped
with a receipt because the completed CPU runs showed it would require a longer
run window.

Only run the CUDA leg when `torch.cuda.is_available()` is true:

```bash
python tools/run_labeled_domain_proof.py --dataset cicids_webattacks --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --frames 10000 --seed 42 --out artifacts/calibration_ledger_natural_attack_windows_10k_gpu --suite smoke --sample-mode natural_attack_windows --device cuda --event-merge-gap 25 --confirmation-mode balanced --sentinel-calibration-mode balanced --calibration-enabled --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection"
```

## Verdicts

| verdict | meaning |
| --- | --- |
| `APPROVE` | Crash scan is clean, raw visibility is intact, FP pressure is controlled or not worse, and attack-window coverage does not collapse. |
| `HOLD` | The run is not a hard failure, but a safety condition needs review, such as FP improvement with coverage collapse. |
| `CALIBRATION_ONLY_NEEDS_TUNING` | Proof-side calibration/reporting can continue, but the evidence is not strong enough to reopen broad core changes. |
| `FAIL` | A hard receipt failed, such as crash hits, missing raw results, hidden raw views, or missing device reporting. |

## Why This Comes Before Week 3 Baselines

The current weakness is balanced row-shuffled recall, while calibrated FP/10k is
controlled on key proof legs. Before expanding competitor baselines, the proof
harness needs clear accounting that shows whether FP reduction came from
legitimate event accounting or from hiding raw alerts. This PR locks that
readout first, then leaves the broader baseline matrix as a clean follow-up.
