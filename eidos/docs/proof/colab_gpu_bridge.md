# Eidos Colab GPU Bridge

This bridge is the standard way to use Google Colab GPU runtime for Eidos Brain
and Eidos Sentinel proof work while keeping Codex in charge of repo edits,
review, and receipt interpretation.

The bridge is intentionally narrow. It does not add model features and does not
change core behavior. It only runs existing proof commands, captures Colab GPU
context, and writes receipts that Codex can inspect after the run.

## Files

- `tools/colab_gpu_bridge.py` - repo-root command wrapper for Colab GPU proof runs.
- `notebooks/eidos_colab_gpu_bridge.ipynb` - Colab notebook template.
- `docs/proof/colab_gpu_bridge.md` - this operating note.

## Supported Modes

- `tensor-smoke` - runs `scripts/verify_colab_gpu_hotfix.py` to confirm CUDA-safe tensor paths.
- `proof-baseline` - runs `tools/run_proof_baseline.py`.
- `labeled-domain` - runs `tools/run_labeled_domain_proof.py`.

## Recommended Colab Flow

1. Open `notebooks/eidos_colab_gpu_bridge.ipynb` in Google Colab.
2. Set Colab runtime type to GPU.
3. Set `REPO_URL`, `BRANCH`, `MODE`, `FRAMES`, and `OUT`.
4. Run all cells.
5. Download or inspect the artifact folder named in the final cell.
6. Bring the receipt files back to Codex for packaging or interpretation.

The bridge should be run with `--require-cuda` for official GPU proof runs. If
CUDA is not available, it fails before running the proof command and writes a
preflight receipt explaining why.

## Example Proof Baseline GPU Run

```bash
python tools/colab_gpu_bridge.py --mode proof-baseline --suite smoke --seed 42 --frames 10000 --out artifacts/proof_runs/2026-06-23/colab_gpu_proof_baseline_10k --mount-drive --require-cuda
```

This runs:

```bash
python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 10000 --out artifacts/proof_runs/2026-06-23/colab_gpu_proof_baseline_10k
```

## Example Labeled Domain GPU Run

```bash
python tools/colab_gpu_bridge.py --mode labeled-domain --suite full --seed 42 --frames 10000 --out artifacts/proof_runs/2026-06-23/colab_gpu_labeled_cicids_10k --file artifacts/cicids_webattacks_samples/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --label-column Label --sample-mode natural --confirmation-mode balanced --calibration-enabled --attack-labels "Web Attack - Brute Force" --attack-labels "Web Attack - XSS" --attack-labels "Web Attack - Sql Injection" --mount-drive --require-cuda
```

## Receipts Written

The bridge writes these additional files into the run artifact folder:

- `colab_gpu_bridge_receipt.json`
- `colab_gpu_bridge_receipt.md`
- `colab_gpu_bridge_environment.txt`
- `colab_gpu_bridge_git_commit.txt`

The existing proof runners still write their normal receipts, including:

- `config.json`
- `run_manifest.json`
- `benchmark_summary.csv`
- `benchmark_summary.md`
- `environment.txt`
- `git_commit.txt`
- `proof_digest.json`
- `proof_digest.md`
- `drive_manifest.json`

For labeled-domain runs, the existing runner also writes the precision ledger,
event confirmation report, Sentinel calibration report, incident cards, and
plain-language proof notes.

## Drive Behavior

When `--mount-drive` is used inside Colab, the bridge asks Colab to mount
Google Drive at `/content/drive`. If `/content/drive/MyDrive` exists, it sets
`EIDOS_PROOF_DRIVE_DIR` for the child proof command.

Drive copy is still receipt-driven:

- If Drive is available, artifacts are copied under
  `Eidos_Brain_Proof_Phase/YYYY-MM-DD/<run_id>/`.
- If Drive is unavailable, the proof run remains valid locally and
  `drive_manifest.json` records the skip reason.
- Drive unavailability should not be hidden or treated as proof success.

## Codex Handoff

After a Colab run, Codex should inspect at least:

- `colab_gpu_bridge_receipt.json`
- `run_manifest.json`
- `proof_digest.json`
- `benchmark_summary.csv`
- `drive_manifest.json`
- `environment.txt`
- `git_commit.txt`

For official claims, distinguish the validated runtime commit from later docs or
packaging commits. Preserve CPU/local and Colab/GPU receipts as separate records.

## Core Behavior

This bridge must remain orchestration-only. Do not use it to change:

- reservoir dynamics
- RLS updates
- surprise scoring
- Sentinel labels or thresholds
- anomaly policy
- compression behavior
- memory/familiarity behavior
- incident logic
- forecasting logic
- domain-profile behavior

Any future behavior change must be a separate, explicitly flagged Eidos task
with before/after regression receipts.
