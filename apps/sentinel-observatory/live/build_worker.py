"""Bundle dependency-free live sources for the Sites Worker ESM starter.

The optional recorded artifact is supplied explicitly and preserved verbatim.
This build never creates telemetry, probes a target or embeds a credential.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(output: Path, recorded: Path | None = None) -> dict:
    sources = {
        'html': ROOT / 'web' / 'index.html',
        'app': ROOT / 'web' / 'app.js',
        'css': ROOT / 'web' / 'styles.css',
        'brain': ROOT / 'brain.mjs',
    }
    if recorded:
        sources['recorded'] = recorded
    assets = {key: path.read_text(encoding='utf-8') for key, path in sources.items()}
    collector = (ROOT / 'collector.mjs').read_text(encoding='utf-8')
    entry = (ROOT / 'worker-entry.mjs').read_text(encoding='utf-8')
    required_import = "import {handleObservation} from './collector.mjs';"
    marker = '/* OBSERVATORY_ASSETS */ Object.freeze({})'
    if entry.count(required_import) != 1 or entry.count(marker) != 1:
        raise ValueError('Unexpected Worker entry contract')
    entry = entry.replace(required_import, '').replace(marker, json.dumps(assets, ensure_ascii=True))
    combined = collector + '\n\n' + entry
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(combined, encoding='utf-8')
    return {'output': str(output.resolve()), 'bytes': output.stat().st_size,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'sources': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recorded', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.recorded), indent=2))
