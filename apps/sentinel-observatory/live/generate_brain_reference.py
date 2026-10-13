"""Generate deterministic port constants + parity fixtures from canonical Python.

Fixtures are explicitly test-only. Recorded latency cases retain the source hash.
This does not run a collector, train on historical data, or issue live forecasts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "eidos/repo/src"
sys.path.insert(0, str(SRC))
from eidos_brain.sentinel.sentinel_v3 import SentinelV3, SentinelV3Config
from eidos_brain.sentinel.input_detectors import InputEvidenceDetector


class CadenceNeutral(InputEvidenceDetector):
    def _period_power(self):
        return None


def engine():
    cfg = SentinelV3Config(features=1, reservoir_size=32, warmup=12, rolling_window=8, seed=42)
    detector = SentinelV3(cfg)
    detector.input_detector = CadenceNeutral(cfg.detector_config(), features=1)
    return detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "brain-reference.json")
    parser.add_argument("--module", type=Path, default=HERE / "brain.mjs")
    args = parser.parse_args()
    sentinel = engine()
    source_files = ["sentinel_v3.py", "calibration.py", "input_detectors.py", "hysteresis.py", "normal_suppression.py", "event_merge.py"]
    hashes = {name: hashlib.sha256((SRC / "eidos_brain/sentinel" / name).read_bytes()).hexdigest() for name in source_files}
    constants = {"wIn": sentinel.reservoir.w_in[:, 0].tolist(),
                 "recurrentSigns": sentinel.reservoir.recurrent_signs.tolist(),
                 "probe": float(sentinel.input_detector.probe[0]), "sourceHashes": hashes,
                 "config": {"features": 1, "reservoir_size": 32, "warmup": 12,
                            "rolling_window": 8, "seed": 42, "periodic_probe": "disabled"}}
    module = args.module.read_text()
    begin, end = "// BEGIN GENERATED BRAIN CONSTANTS", "// END GENERATED BRAIN CONSTANTS"
    prefix, rest = module.split(begin, 1)
    _, suffix = rest.split(end, 1)
    args.module.write_text(prefix + begin + "\nconst BRAIN_CONSTANTS = Object.freeze(" + json.dumps(constants, indent=2) + ");\n" + end + suffix)
    cases = []
    if args.snapshot:
        raw = args.snapshot.read_bytes()
        recorded = json.loads(raw)
        sources = sorted({event["source_id"] for event in recorded["events"]})
        for source in sources:
            selected = sorted([e for e in recorded["events"] if e["source_id"] == source and e["status"] is not None], key=lambda e: e["completed_at"])
            cases.append({"id": "recorded-" + source, "provenance": "recorded real response latencies; parity-only replay",
                          "source_sha256": hashlib.sha256(raw).hexdigest(),
                          "frames": [math.log1p(e["latency_ms"]) for e in selected]})
    rng = np.random.default_rng(819)
    cases.extend([
        {"id": "synthetic-regime-changes", "provenance": "TEST FIXTURE ONLY; never telemetry", "frames": np.concatenate([
            rng.normal(4, .4, 100), rng.normal(7, 2, 180), np.full(120, 1.4), rng.normal(3, .4, 100)
        ]).tolist()},
        {"id": "synthetic-zero-and-extreme", "provenance": "TEST FIXTURE ONLY; never telemetry", "frames":
         [0.0] * 20 + [math.log1p(86_400_000), 0.0] * 50 + [1e-6] * 40},
        {"id": "synthetic-stationary", "provenance": "TEST FIXTURE ONLY; never telemetry", "frames": rng.normal(2, .2, 180).tolist()},
    ])
    for case in cases:
        detector = engine()
        case["expected"] = [detector.step([frame]) for frame in case["frames"]]
    args.output.write_text(json.dumps({"schema": "eidos.sentinel.js-parity.v1", "sourceHashes": hashes,
                                      "cases": cases}, separators=(",", ":")) + "\n")
    print(json.dumps({"cases": len(cases), "frames": sum(len(c["frames"]) for c in cases), "output": str(args.output), "module": str(args.module)}))


if __name__ == "__main__":
    main()
