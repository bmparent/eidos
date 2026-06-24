#!/usr/bin/env python
"""CLI for Eidos RNG Null Proof v1."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proof.rng_null_proof import run_proof


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run Eidos RNG null-proof suites")
    p.add_argument("--suite", choices=["smoke", "controls", "null", "full"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=5000)
    p.add_argument("--out", required=True)
    p.add_argument("--predictor", choices=["baseline", "eidos_brain"], default="baseline", help="Predictor label for compatibility with Brain-backed proof commands; the existing proof gate still decides official readiness.")
    args = p.parse_args(argv)
    result = run_proof(args.suite, args.seed, args.frames, args.out)
    print(f"wrote RNG null proof artifacts to {args.out}")
    for verdict in result["verdicts"]:
        print(f"{verdict['source_name']}: {verdict['verdict']} top1={verdict['top1_accuracy']:.4f} chance={verdict['chance_top1']:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
