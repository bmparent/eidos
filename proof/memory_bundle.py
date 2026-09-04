"""Package evidence and byte-identical evaluator sources, optionally mirror to Drive."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .memory_benchmark import ROOT, sha, save, stamp


def build(run_root, drive_dir=None):
    bundle=run_root/'controlled_memory_reproducibility.zip'
    if bundle.exists():
        raise FileExistsError('Bundle already exists; preserve the prior package and use a new run directory.')
    code=run_root/'code'
    code.mkdir(exist_ok=False)
    paths=['proof/__init__.py','proof/memory_core.py','proof/memory_benchmark.py','proof/memory_report.py',
           'proof/memory_bundle.py','tests/test_memory_benchmark.py','tests/test_memory_collection.py','tests/conftest.py',
           'eidos/proof/__init__.py','eidos/proof/sentinel_calibration_v1.py',
           'eidos/repo/src/eidos_brain/engine/eidos_v0_4_7_02.py','eidos/verify_data/incident_test_data.csv',
           'requirements-memory-benchmark.txt','docs/controlled_memory_benchmark.md']
    for rel in paths:
        target=code/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,target)
    frozen=json.loads((run_root/'main/freeze.json').read_text())
    for rel,digest in frozen['source_hashes'].items():
        if sha(code/rel)!=digest:
            raise ValueError(f'Frozen evaluator bytes changed: {rel}')
    (run_root/'REPRODUCE.md').write_text('''# Reproduce the preserved experiment

The `research_original` directory is the delivered record. Both `research_rerun` and
`research_rerun_py312` retain attempts under its original 240-second cap.
`research_extended_budget` explicitly increases only the cap to 900 seconds;
the evaluator is unchanged and its new protocol was calibrated/frozen before that run.

The numerical `main` directory is immutable. `code` contains byte-identical evaluator
sources and the compatible engine/fixture. Installed dependency versions are in
`main/environment.txt`. The source code and all inputs are hashed in `main/freeze.json`.

To replay the exact frozen inputs, first copy `main/protocol.json`, `main/freeze.json`,
and the entire `main/inputs` directory into a NEW sibling directory called `replay`.
From the `code` root, run:

```text
python -m pip install -r requirements-memory-benchmark.txt
python -m pytest tests/test_memory_benchmark.py tests/test_memory_collection.py -q
python -m proof.memory_benchmark run --out ../replay
python -m proof.memory_report --run ../replay --out ../replay_report
```

Do not copy `main/raw`, `results`, `curves` or `precision` into `replay`; existing output
directories are rejected. Same stored coefficients/inputs and the recorded environment
support close numerical reproduction; cross-platform bitwise equivalence is not claimed.
Run the fresh calibration/prepare commands in docs/controlled_memory_benchmark.md from
a git checkout to create an independently frozen run. Do not overwrite delivered outputs.

`evidence_manifest.json` hashes every included evidence file except itself and the ZIP.
`drive_manifest.json` is an external copy receipt written after packaging and is not
recursively included in the ZIP. Its bundle checksum verifies the archived package.
''',encoding='utf-8')
    paths=sorted(p for p in run_root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('evidence_manifest.json','drive_manifest.json',bundle.name))
    manifest=dict(utc=stamp(),files=[dict(path=p.relative_to(run_root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in paths],
                  exclusions=['__pycache__','self manifest hash','ZIP self hash','post-packaging Drive receipt'])
    save(run_root/'evidence_manifest.json',manifest)
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for p in paths+[run_root/'evidence_manifest.json']:
            archive.write(p,p.relative_to(run_root).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        bad=archive.testzip()
        if bad:
            raise ValueError(f'ZIP CRC verification failed: {bad}')
    receipt=dict(drive_copy_attempted=False,drive_copy_success=False,drive_root='unknown',drive_run_dir='unknown',
                 reason='No writable configured Drive path available',files_considered=[],files_copied=[],files_skipped=[],
                 timestamp_utc=stamp(),bundle_sha256=sha(bundle),bundle_bytes=bundle.stat().st_size)
    if drive_dir is None:
        date=datetime.now(timezone.utc).date().isoformat()
        for candidate in (os.environ.get('EIDOS_PROOF_DRIVE_DIR'),os.environ.get('EIDOS_ARTIFACT_ROOT'),
                          '/content/drive/MyDrive' if 'google.colab' in __import__('sys').modules else None):
            if candidate and Path(candidate).is_dir() and os.access(candidate,os.W_OK):
                drive_dir=Path(candidate)/'Eidos_Brain_Proof_Phase'/date/run_root.name
                break
    chosen=[bundle,run_root/'evidence_manifest.json',run_root/'report/decision_report.md',run_root/'report/evidence_figure.png',
            run_root/'report/evidence_figure.svg',run_root/'report/benchmark_summary.csv',run_root/'report/plot_data.csv',
            run_root/'report/temporal_blocks.csv',run_root/'report/precision_summary.json',run_root/'codex_journal.md',
            run_root/'plain_language_test_analysis.md']
    chosen += sorted((run_root/'report/progress').glob('*'))
    receipt['files_considered']=[p.name for p in chosen]
    if drive_dir:
        receipt.update(drive_copy_attempted=True,drive_root=str(drive_dir.parent),drive_run_dir=str(drive_dir))
        try:
            drive_dir.mkdir(parents=True,exist_ok=False)
            for p in chosen:
                if not p.exists():
                    receipt['files_skipped'].append(dict(file=p.name,reason='not generated'))
                    continue
                destination=drive_dir/'progress'/p.name if p.parent.name=='progress' else drive_dir/p.name
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,destination)
                digest=sha(p)
                if sha(destination)!=digest:
                    raise ValueError(f'Copied checksum mismatch: {p.name}')
                receipt['files_copied'].append(dict(file=destination.relative_to(drive_dir).as_posix(),sha256=digest,bytes=p.stat().st_size))
            receipt.update(drive_copy_success=True,reason='Copied new files to the requested research subfolder; all copied hashes verified locally on the mounted Drive.')
        except Exception as exc:
            receipt['reason']=f'{type(exc).__name__}: {exc}'
    save(run_root/'drive_manifest.json',receipt)
    if receipt['drive_copy_success']:
        shutil.copy2(run_root/'drive_manifest.json',drive_dir/'drive_manifest.json')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--drive-dir',type=Path)
    args=parser.parse_args()
    build(args.run_root,args.drive_dir)
