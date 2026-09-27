# Plain-language analysis - remaining Sentinel evidence commit

The application fixes were already released. This task preserves the remaining 97 supporting files in Git so they are available with the code, rather than relying only on this computer, Drive or a provider snapshot.

All 89 entries declared in the earlier manifests matched their original files. A new inventory covers all 97 files, and the verifier compares their sizes and SHA-256 fingerprints with the actual staged and committed Git contents. JSON and JSONL records must also parse successfully. Git attributes prevent Windows line-ending conversion from changing those fingerprints. Text scanning found no credential patterns, and seven screenshots were reviewed visually.

No runtime code, engine behavior or production settings changed, so the runtime test suites were not rerun. The earlier passing test logs remain archived with their original dates. No experiment was launched. The new package README describes exact commands and the validation receipt records the packaging checks. The new Drive manifest identifies the separate archive copy; older copies are preserved.

## Proof Logic + Meaning

The goal is reproducible evidence preservation. Before this task, the supporting files were untracked. The logic is to require matching file fingerprints and lengths before and after Git storage: SHA256(committed bytes) = SHA256(original bytes). That strengthens the ability to revisit the evidence and inspect internal-state receipts in Eidos's streaming-codec goal. It does not improve detection quality by itself: the previous all-benign slice still has 301 false positives out of 600 rows (50.1667%), with recall and AUC unavailable. Holdout data remains excluded and zero scientific proof gates advance.

The concrete evidence is in `artifacts/sentinel-evidence-commit-20260907/` and the two original audit folders it indexes. The next useful practice is to apply exact-byte Git preservation when future evidence is first generated; further detector research requires its own authorized protocol.
