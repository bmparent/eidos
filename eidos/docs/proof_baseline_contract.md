# Proof Baseline Artifact Contract

The proof phase baseline folder is frozen at:

```text
artifacts/proof_baseline_2026_05/
  config.json
  benchmark_summary.csv
  benchmark_summary.md
  pytest_results.xml
  environment.txt
  git_commit.txt
  run_manifest.json
  scenarios/
  plots/
```

This contract names the expected layout only. Missing files should remain visible until a later population task writes real contents.

## Required Files

| Path | Purpose | Later population task |
| --- | --- | --- |
| `config.json` | Exact baseline configuration used for the proof run, including suite, seed, frame limits, specs, and any explicit skip policy. | Proof runner wrapper captures the normalized invocation config before running benchmarks. |
| `benchmark_summary.csv` | Machine-readable per-scenario benchmark rows, with `NA` or empty nullable fields plus skip reasons when a metric is unavailable. | Benchmark aggregation task maps existing runner output into the baseline schema. |
| `benchmark_summary.md` | Human-readable summary of observed results, skipped baselines, caveats, and links to scenario artifacts. | Report writer task summarizes the CSV and manifest after benchmark execution. |
| `pytest_results.xml` | JUnit XML output from the repo-root pytest run used for the proof baseline. | Test runner task runs pytest with `--junitxml=artifacts/proof_baseline_2026_05/pytest_results.xml`. |
| `environment.txt` | Python version, OS, package versions, relevant environment variables, and command availability needed to reproduce the run. | Environment capture task records runtime details before the benchmark starts. |
| `git_commit.txt` | Git commit, branch, dirty-tree status, and any untracked proof files present at run time. | Git capture task records the working tree state before and after the run. |
| `run_manifest.json` | Machine-readable manifest tying together config hash, git state, environment capture, commands, artifact paths, and skip reasons. | Manifest writer task combines outputs from the proof runner, pytest runner, environment capture, and report writer. |

## Required Directories

| Path | Purpose | Later population task |
| --- | --- | --- |
| `scenarios/` | Per-scenario receipts, copied specs, input metadata, failure notes, and scenario-level logs. | Proof runner wrapper creates one child folder per scenario or records a skip note. |
| `plots/` | Optional plots generated from proof metrics. Empty is allowed only when plotting is skipped with an explicit manifest reason. | Plotting/report task writes figures or records a skip reason. |

## Current Command Audit

Repo-root tests are currently driven by `pytest.ini`:

```bash
pytest
```

The existing `scripts/run_tests.ps1` is not the proof command because it hardcodes `d:\eidos\tests` instead of the current repository root.

For the proof baseline, the later test task should emit JUnit XML explicitly:

```bash
pytest --junitxml=artifacts/proof_baseline_2026_05/pytest_results.xml
```

The closest current benchmark-like command is the domain tuner:

```bash
python tools/domain_tuner.py --domain cyber --spec specs/cyber_small.json --trials 120 --seed 7 --seeds 0 1 2
```

That command runs from the repository root and writes the existing tuning artifacts:

```text
artifacts/trials_cyber.csv
artifacts/best_profile_cyber.json
artifacts/recommendation_cyber.md
```

It is not the frozen proof-baseline command because it does not accept an `--out` directory, does not emit the required baseline filenames, and does not capture pytest XML, environment details, git state, scenario receipts, plots, or a combined manifest.

The Week 1 frozen baseline wrapper is a repo-root command shaped like:

```bash
python tools/run_proof_baseline.py --suite smoke --seed 42 --frames 10000 --out artifacts/proof_baseline_2026_05
```

That wrapper calls `create_proof_artifact_dir(out_dir)`, invokes the existing benchmark/session logic, and populates the contract above without changing model behavior.
