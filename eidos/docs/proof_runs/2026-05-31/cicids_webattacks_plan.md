# CICIDS/WebAttacks Labeled Proof Plan -- 2026-05-31

## Why this follows the official GPU baseline

The official Colab GPU 10k proof baseline showed that the current Eidos Brain proof runner can complete a clean 10k smoke run with CUDA exercised, artifacts written, pytest passed, and crash scan clean. That baseline was still synthetic.

This CICIDS/WebAttacks harness is the first labeled/domain proof step after that baseline. It does not try to improve or tune behavior. It gives Eidos a cyber dataset path with known benign and attack labels, then records what the existing engine and Sentinel confirmation layer do against those labels.

## How to run in Colab

Upload or mount the CICIDS2017 WebAttacks CSV, then run from the repo root:

```bash
python tools/run_labeled_domain_proof.py \
  --dataset cicids_webattacks \
  --file /content/drive/MyDrive/path/to/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
  --label-column Label \
  --attack-labels "Web Attack - Brute Force,Web Attack - XSS,Web Attack - Sql Injection" \
  --frames 10000 \
  --seed 42 \
  --suite full \
  --max-rows 10000 \
  --out artifacts/cicids_webattacks_proof_colab_10000
```

For a tiny local smoke fixture, use the same command shape with a small CSV and `--suite smoke`.

## Metrics that matter

- Frames processed
- Labels detected and label distribution
- Candidate events, confirmed events, and suppressed candidates
- True positives, false positives, and false negatives over label windows
- Precision, recall, and F1 when label windows are available
- False positives per 10k frames
- Incident-card count
- Eidos compression ratio
- External compression baselines when optional packages are available
- Runtime seconds
- Crash hit count for `CRASH IN INCIDENT LOGIC`, `can't convert cuda`, and `Traceback`

## Known limitations

- This is a harness and adapter task, not a threshold tuning task.
- The runner does not download CICIDS. The dataset file must be mounted, uploaded, or otherwise available at `--file`.
- Event metrics depend on row order and frame-aligned labels.
- A tiny smoke fixture proves the artifact and metric path, not real-world cyber detection quality.
- False positives and false negatives should be reported honestly and reviewed before any tuning.

## What is explicitly not changing

This task does not change reservoir dynamics, RLS updates, Sentinel thresholds, anomaly policy, compression behavior, or architecture layers.
