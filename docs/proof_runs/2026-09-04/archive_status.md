# Controlled memory archive receipt

Decision: **inconclusive for adoption**. Production state rounding remains unchanged.

- [Research run folder](https://drive.google.com/drive/folders/1jNdue93xhf0uPdo1GdszxVSHPp3vY2ch)
- [Reproducibility ZIP](https://drive.google.com/file/d/1iMBDyLnTdpTVhxoaVBzKoAhEF7NuME2y/view)
- [Decision report](https://drive.google.com/file/d/1renNwSc182AJd_4Lijb8h8psww6np-Qh/view)
- [Evidence figure](https://drive.google.com/file/d/1ZzglUbpIkFHR-rKBrXK-F4VjFkCpfUJl/view)
- [Underlying plot data](https://drive.google.com/file/d/1WD1v2tXIzUZRNg1bYiziwf1mZXdl6f82/view)
- [Machine-readable copy receipt](https://drive.google.com/file/d/1XBxg4I2jueE4bs_incw0SrbcmhfNPhrR/view)

The connector returned the new folder and file links and confirmed the 195,319,150-byte ZIP. All 18 copied files passed SHA-256 comparison against the mounted Drive copies; zero files were skipped. All 268 ZIP manifest entries were independently hash-checked inside the completed archive. Cloud metadata/readback validates existence and size; an independent cloud-byte SHA-256 download was not performed.

ZIP SHA-256:

```text
6707e80634d603d935dd7e4916c2f15075f730e05b8f99031b735c5612f64443
```

The ZIP contains the frozen evaluator, exact inputs, full numerical traces, precision references, all research-package attempts, plots/data, environment, tests, pre-package journal/analysis, and replay instructions. The manifest describes the immutable ZIP snapshot. Post-package logs and copy receipts are separate to avoid self-referential hashes. The original research folder/files were not replaced.

The local run is `artifacts/controlled_memory_2026_09_04/` under the isolated checkout. Bulky data are excluded from git; this dated folder contains compact receipts. Start with [decision_report.md](decision_report.md), [validation.json](validation.json), [codex_journal.md](codex_journal.md), and [protocol.json](protocol.json).

Reproduction commands are in [the benchmark guide](../../controlled_memory_benchmark.md). Required dependencies and per-run versions are recorded separately from production dependencies. Remaining unrelated root-suite failures were reproduced on the clean starting revision; the targeted suite and packaged-engine suite passed.
