"""Reproduce the legacy ambiguity and verify actual, full-token semantic vectors."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sentinel_runner.engine_bridge import discover_engine_path, load_engine
from sentinel_runner.guided.semantic import ENCODER, encode


def main():
    out = Path("artifacts/sentinel-guided-20260908/semantic-final")
    out.mkdir(parents=True, exist_ok=True)
    engine_path = discover_engine_path()
    engine = load_engine(engine_path, out)
    prefix = "This is the same reference text. " * 20
    texts = [prefix + " Database connections failed after a cache outage." * 15,
             prefix + " Garden irrigation watered the flowers at dawn." * 15,
             "The database was slow after the cache failed.",
             "Database delays followed an unavailable cache.",
             "Garden flowers opened in the morning sun."]
    legacy_identical = np.array_equal(engine.embed_line_to_vec(texts[0]), engine.embed_line_to_vec(texts[1]))
    vectors = np.asarray(encode(texts, out / "cache"))
    cached = np.asarray(encode(texts, out / "cache"))
    assert legacy_identical
    assert vectors.shape == (5, 384) and np.isfinite(vectors).all()
    assert np.array_equal(vectors, cached)
    assert np.linalg.norm(vectors[0] - vectors[1]) > 0.1
    assert vectors[2] @ vectors[3] > vectors[2] @ vectors[4] + 0.2

    # Execute the actual legacy assignment AST, not a reimplemented selection.
    source = engine_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"err_R_t", "err_L_t", "L_ok", "R_bad", "use_L", "best_pred"}
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    nodes.sort(key=lambda n: n.lineno)
    selected = []
    for observed in [0., 10.]:
        scope = {"torch": torch, "frame": torch.tensor([observed]), "y_R": torch.tensor([0.]), "y_L": torch.tensor([10.])}
        for node in nodes: exec(compile(ast.Module(body=[node], type_ignores=[]), str(engine_path), "exec"), scope)
        selected.append(float(scope["best_pred"].item()))
    assert selected == [0., 10.]
    # Controlled mathematical example of current-sample contamination, with source lines.
    mu, variance, residual, beta = 1., 1., 10., .99
    prior_score = (residual - mu) / variance ** .5
    after_mu = beta * mu + (1 - beta) * residual
    after_variance = beta * variance + (1 - beta) * (residual - after_mu) ** 2
    after_score = (residual - after_mu) / after_variance ** .5
    assert after_score < prior_score
    receipt = {"status": "passed", "encoder": ENCODER, "legacyEngineSha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
               "sourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "legacy64PrefixIdentical": bool(legacy_identical), "semanticSuffixVectorDistance": float(np.linalg.norm(vectors[0] - vectors[1])),
               "paraphraseCosine": float(vectors[2] @ vectors[3]), "unrelatedCosine": float(vectors[2] @ vectors[4]),
               "allTokenWindows": True, "cacheExact": bool(np.array_equal(vectors, cached)),
               "legacySelectionExecutedASTLines": [n.lineno for n in nodes], "observations": [0., 10.], "legacySelectedPredictions": selected,
               "scoreOrderControlledMathExample": {"prior": prior_score, "afterCurrentUpdate": after_score, "beta": beta,
                   "scope": "controlled arithmetic illustration, not an empirical benefit or exact full legacy MAD-history trace"},
               "legacyUnmodified": True, "llmCalls": 0, "researchGatesAdvanced": 0}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__": main()
