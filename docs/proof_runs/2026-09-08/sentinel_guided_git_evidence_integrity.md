# Sentinel evidence byte-preservation receipt

UTC date: 2026-09-09. Local work date: 2026-09-08.

The final Git integrity check found that automatic Windows line-ending conversion changed 81 of the 165 staged review artifacts relative to their original local bytes. The complete Drive ZIP retained the original bytes. A narrowly scoped `.gitattributes` rule now disables text conversion for `artifacts/sentinel-guided-20260908/**`, and the tracked evidence was restaged. The repeated check passed: 165 files checked, zero byte mismatches. Raw logs were preserved without whitespace edits.

Files changed: `.gitattributes` and the byte representation of affected evidence blobs. No runtime or legacy core behavior changed. The prior source candidate is `24e00500431a654e167618fa6ca7543405865c45`; the evidence commit before this correction is `53b701b65114f61b29b1815814fc3ff6f3f9345c`. Its runner, verify and Vercel checks passed. Runtime tests were not repeated solely for this Git attribute change; the meaningful validation is exact blob equality.

Run this from the repository root with Python to reproduce the check:

```python
import hashlib
import pathlib
import subprocess

rows = subprocess.check_output(
    ['git', 'ls-files', '-s', '--', 'artifacts/sentinel-guided-20260908'],
    text=True,
).splitlines()
for row in rows:
    metadata, name = row.split('\t', 1)
    data = pathlib.Path(name).read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert actual == metadata.split()[1], name
print(f'{len(rows)} tracked evidence files match their Git blobs byte for byte')
```

## Proof Logic + Meaning

- Goal reached: the Git evidence byte-preservation gate passed.
- Previous state: Windows checkout conversion could invalidate otherwise correct artifact hashes after cloning.
- Technical logic: preserve original bytes with a path-scoped Git attribute and compare each file with its indexed Git blob.
- Math: `SHA1("blob " + byte_length + NUL + original_bytes) = indexed_blob_id`; the evidence manifest separately records SHA-256 content hashes.
- Philosophical meaning: reproducibility is truth that can be revisited.
- Why this is better: a clean checkout can retain the same evidence bytes as the original run and archive.
- North-star contribution: strengthens reproducible execution and auditable incident receipts; it does not establish improved detection or compression.
- Evidence: this reproducible check, `.gitattributes`, the tracked artifact files and existing SHA-256 manifest.
- Remaining uncertainty: detector qualification remains failed and hosted compute completion remains blocked by Vercel HTTP 402. This check does not change those outcomes or advance a research gate.

## Artifact and journal status

This report is an addendum to the Sentinel Codex journal and implementation ledger. It is kept beside those reports and mirrored separately to the existing task Drive folder after the immutable evidence ZIP; it is not represented as an entry inside that earlier ZIP. Existing artifact and Drive manifests describe their timestamped snapshots. No historical archive or original dirty checkout was changed.
