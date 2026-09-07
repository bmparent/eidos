# Sentinel audit evidence commit - September 7, 2026

This package makes the remaining 97 audit files available from Git. The application fixes and initial closeout documents were already merged in PRs #43-45. No application, engine, configuration, threshold, label, split, or workflow behavior changes here.

Start with [the completed live report](../sentinel-production-live-20260906/final-report.md). The [earlier audit report](../sentinel-production-audit-20260906/final-report.md) is a historical checkpoint: its access blocker was resolved in the live closeout. Deployment identities, CI status and Drive destinations in both archives describe their recorded timestamps, not an automatically refreshed current state. The optional CI patch is retained as an unapplied proposal; it is not an installed workflow.

The 97 files total 5,521,527 bytes. They include reports, sanitized service receipts, numeric engine/evaluation traces, controlled browser screenshots and test logs. No raw dataset input or held-out rows are included. All 89 entries in the two original file manifests matched before staging. Seven screenshots were visually reviewed. A targeted text scan found no private-key, provider-token, JWT, credential-bearing URL, bearer-token or literal secret-assignment patterns; this is a bounded scan, not a universal secret-detection guarantee.

The original files remain byte-for-byte unchanged. Scoped `.gitattributes` rules disable text conversion in these archives. `run_manifest.json` indexes every original file, including files added after the earlier archive manifests were made. New records for this commit live here, separately from the frozen evidence.

Whitespace checks are disabled only for the two original archives: their raw logs contain original CRLF endings, terminal padding and final blank lines. Normalizing those would destroy hash correspondence. New documentation and verification files retain normal whitespace checks.

Run from repository root:

```sh
node artifacts/sentinel-evidence-commit-20260907/verify-archives.mjs --worktree
node artifacts/sentinel-evidence-commit-20260907/verify-archives.mjs --index
node artifacts/sentinel-evidence-commit-20260907/verify-archives.mjs --head
```

Each command must report 97 files, 5,521,527 bytes and `allHashesMatch: true`. Index verification checks the exact staged Git blobs; HEAD verification checks the committed blobs. No credentials, dataset download or live compute are needed. Full app and runner suites were not rerun for an evidence-only commit; the prior 46 app tests and 26 runner tests remain timestamped evidence, not new test results.

## Proof Logic + Meaning

- Goal reached: preserve the complete engineering evidence in Git with exact-byte reproducibility.
- Previous state: application changes were merged, but 97 supporting files existed only locally and in Drive archives.
- Technical logic: compare the SHA-256 and byte length of each file to a frozen inventory, including staged and committed Git blobs; parse all JSON and JSONL records.
- Math: for every file f, SHA256(GitBlob(f)) = recorded SHA256(f), and length(GitBlob(f)) = recorded bytes(f). Detection results remain FP/(FP+TN) = 301/600 = 50.1667%.
- Philosophical meaning: reproducibility is truth that can be revisited, including negative results.
- Why this is better: a checkout can now reproduce the evidence-integrity check without depending on an operator session or expiring provider snapshot.
- North-star connection: strengthens reproducible operation and inspection of internal state; it does not add proof of detector or codec utility.
- Evidence: this inventory and validation receipt; the original source, dataset, prediction, evaluation and artifact-verification receipts in the live archive.
- Remaining uncertainty: the benign-only slice cannot establish recall, AUC or detection utility; held-out rows remain excluded and zero scientific gates advanced. Historical controlled failure tests are not induced production incidents.

## Archive and journal

Existing Drive archives are preserved. `drive_manifest.json` records the new complete commit-evidence mirror. The supplemental journal and plain-language analysis are under `docs/proof_runs/2026-09-07/` with the `artifact_commit_` prefix. No new experiment was run.
