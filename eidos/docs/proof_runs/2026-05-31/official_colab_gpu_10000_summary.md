# Official Colab GPU 10k Proof Summary -- 2026-05-31

## Status

This is the official GPU validation receipt for the hotfix branch `codex/eidos-hotfix-gpu-config-2026-05-31`.

The proof source commit was `3eca182f7351c382a0f47b6b5b5e3bee5c956f49`. The branch and commit were verified as reachable on `origin` through:

```bash
git ls-remote --heads origin codex/eidos-hotfix-gpu-config-2026-05-31
```

At packaging start, that command returned `3eca182f7351c382a0f47b6b5b5e3bee5c956f49` for `refs/heads/codex/eidos-hotfix-gpu-config-2026-05-31`. If a later docs-packaging commit is pushed to the same branch, the proof source commit should be read as the validated runtime commit and remains reachable as an ancestor of the hotfix branch.

This receipt packages the clean Colab GPU proof result. It does not commit the bulky generated runtime artifact folder.

## Official Proof Receipt

- Repo: `bmparent/eidos`
- Branch: `codex/eidos-hotfix-gpu-config-2026-05-31`
- Proof source commit: `3eca182f7351c382a0f47b6b5b5e3bee5c956f49`
- Command: `python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 10000 --out artifacts/hotfix_official_10000_colab_gpu`
- Environment: clean Google Colab GPU runtime
- GPU: Tesla T4
- CUDA exercised: yes
- Suite: `smoke`
- Seed: `42`
- Frames: `10000`
- Runtime seconds: `66.608078`
- Pytest: passed
- Crash scan: clean, 0 hits
- Normal-only confirmed false positives per 10k frames: `0`
- Confirmed events: `4`
- Candidate events: `9`
- Suppressed candidates: `5`
- Incident cards: `4`
- Eidos compression ratio: `16.192510963679297`
- External compression baselines: present
- Best external baseline: `lzma`, ratio `1.130315`
- Git commit recorded in proof digest: `3eca182f7351c382a0f47b6b5b5e3bee5c956f49`
- Branch recorded in proof digest: `codex/eidos-hotfix-gpu-config-2026-05-31`

## Relationship To The 1200-Frame CPU/Local Proof

The existing `codex_journal.md` and `plain_language_test_analysis.md` in this folder record the earlier local CPU proof run:

- Local command: `python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 1200 --out artifacts/hotfix_official_1200`
- Local runtime context: CPU-only, CUDA unavailable locally
- Local purpose: validate the hotfix branch and proof-runner changes before the clean Colab GPU run

That 1200-frame CPU/local receipt remains useful as a local validation record. It is not the official GPU baseline. The official hotfix proof baseline is the 10k Colab GPU receipt summarized here and captured in `official_colab_gpu_10000_receipt.json`.

## Git Dirty Finding From The Proof Digest

The official Colab proof digest recorded `git_dirty=true`. The exact dirty causes after proof generation were:

- Two tracked docs modified during or after proof generation:
  - `docs/proof_runs/2026-05-31/codex_journal.md`
  - `docs/proof_runs/2026-05-31/plain_language_test_analysis.md`
- Generated runtime artifact folder:
  - `artifacts/hotfix_official_10000_colab_gpu/`
- Generated pytest output file:
  - `full_pytest_output.txt`

Those dirty files are understood as post-run receipt/doc generation effects, not evidence that the proof source commit was different from the commit recorded in the digest.

## Source Packaging Policy

The generated Colab runtime artifact folder and `full_pytest_output.txt` are intentionally not committed in this packaging update. The committed source package preserves the important receipt facts in docs and keeps bulky generated runtime outputs out of Git.

Targeted ignore rules were added for:

- `artifacts/hotfix_official_10000_colab_gpu/`
- `full_pytest_output.txt`

The final committed source state after packaging is intended to be clean for tracked files. The generated runtime artifacts remain uncommitted by design.

## Known Limitations

- The 10k proof uses synthetic smoke fixtures, not labeled production telemetry.
- This receipt does not prove production detection quality on real-world traffic.
- External compression baselines were present, but the only recorded best external baseline in this package is `lzma` at ratio `1.130315`.
- Incident-card quality was confirmed by count and clean crash scan, not by a labeled analyst scoring set.

## Recommended Next Experiment

Run the first real labeled/domain proof using CICIDS2017/WebAttacks or controlled system telemetry.

Recommended metrics:

- True positives
- False positives
- Compression ratio
- Incident-card quality
- Runtime and memory
- Crash scan
