"""Bundle dependency-free live sources for the Sites Worker ESM starter.

The optional recorded artifact is supplied explicitly and preserved verbatim.
This build never creates telemetry, probes a target or embeds a credential.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(output: Path, recorded: Path | None = None) -> dict:
    sources = {
        'html': ROOT / 'web' / 'index.html',
        'app': ROOT / 'web' / 'app.js',
        'css': ROOT / 'web' / 'styles.css',
        'brain': ROOT / 'brain.mjs',
        'interpreter': ROOT / 'web' / 'browser-interpreter.mjs',
        'model_worker': ROOT / 'web' / 'browser-model-worker.mjs',
        'security': ROOT / 'security-events.mjs',
        'network_panel': ROOT / 'web' / 'network-panel.mjs',
        'target_panel': ROOT / 'web' / 'target-panel.mjs',
        'probe_explanation': ROOT / 'probe-explanation.mjs',
    }
    if recorded:
        sources['recorded'] = recorded
    assets = {key: path.read_text(encoding='utf-8') for key, path in sources.items()}
    collector = (ROOT / 'collector.mjs').read_text(encoding='utf-8')
    sensor = (ROOT / 'hosted-sensor.mjs').read_text(encoding='utf-8')
    sensor_import = "import {normalizeTarget} from './collector.mjs';"
    if sensor.count(sensor_import) != 1:
        raise ValueError('Unexpected hosted sensor import contract')
    sensor = sensor.replace(sensor_import, '')
    sol = (ROOT / 'sol-interpreter.mjs').read_text(encoding='utf-8')
    sol_import = "import {resolveSensorEvidence} from './hosted-sensor.mjs';"
    if sol.count(sol_import) != 1:
        raise ValueError('Unexpected Sol interpreter import contract')
    sol = sol.replace(sol_import, '')
    def module(name, exports, imports=()):
        source = (ROOT / name).read_text(encoding='utf-8')
        for statement in imports:
            if source.count(statement) != 1:
                raise ValueError('Unexpected module import: ' + statement)
            source = source.replace(statement, '')
        source = re.sub(r'^export (?=(?:const|class|function|async function)\b)', '', source, flags=re.M)
        names = ', '.join(exports)
        return 'const {' + names + '} = (() => {\n' + source + '\nreturn {' + names + '};\n})();\n'
    modules = module('brain.mjs', ['SentinelPort', 'ENGINE_PROVENANCE'])
    modules += module('target-brain.mjs', ['TargetBrain', 'TARGET_PROFILE'], ["import {SentinelPort, ENGINE_PROVENANCE} from './brain.mjs';"])
    modules += module('target-store.mjs', ['retainTargetObservation', 'targetHistory', 'resolveTargetEvidence'], ["import {TargetBrain, TARGET_PROFILE} from './target-brain.mjs';"])
    modules += module('target-relay.mjs', ['targetProbeConfiguration', 'collectTargetRelayObservation'], ["import {normalizeTarget, ipCollectorConfiguration} from './collector.mjs';"])
    modules += module('probe-explanation.mjs', ['probeExplanation'])
    modules += module('target-interpretation.mjs', ['prepareTargetInterpretation'], ["import {probeExplanation} from './probe-explanation.mjs';"])
    sol = sol.replace("import {resolveTargetEvidence} from './target-store.mjs';", '').replace("import {prepareTargetInterpretation} from './target-interpretation.mjs';", '')
    entry = (ROOT / 'worker-entry.mjs').read_text(encoding='utf-8')
    required_import = "import {handleObservation, ipCollectorConfiguration} from './collector.mjs';"
    required_sensor_import = "import {capability, snapshot, recordObservation} from './hosted-sensor.mjs';"
    required_sol_import = "import {hostedInterpretationCapability, handleInterpretation} from './sol-interpreter.mjs';"
    entry = entry.replace("import {targetProbeConfiguration, collectTargetRelayObservation} from './target-relay.mjs';", '').replace("import {retainTargetObservation, targetHistory} from './target-store.mjs';", '')
    marker = '/* OBSERVATORY_ASSETS */ Object.freeze({})'
    if entry.count(required_import) != 1 or entry.count(required_sensor_import) != 1 or entry.count(required_sol_import) != 1 or entry.count(marker) != 1:
        raise ValueError('Unexpected Worker entry contract')
    entry = entry.replace(required_import, '').replace(required_sensor_import, '').replace(required_sol_import, '').replace(marker, json.dumps(assets, ensure_ascii=True))
    combined = collector + '\n\n' + sensor + '\n\n' + modules + '\n\n' + sol + '\n\n' + entry
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(combined, encoding='utf-8', newline='\n')
    return {'output': str(output.resolve()), 'bytes': output.stat().st_size,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'sources': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recorded', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.recorded), indent=2))
