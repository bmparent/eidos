"""Reproduce the legacy ambiguity and verify actual, full-token semantic vectors."""
import ast
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import torch
from sentinel_runner.engine_bridge import discover_engine_path, load_engine
from sentinel_runner.guided.semantic import ENCODER, encode
from sentinel_runner.guided.causal import baseline


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
    # Execute the actual legacy EMA/history/MAD/score block on controlled history.
    start = source.index("        ema_count += 1")
    end = source.index("\n", source.index("        z_score = abs(best_err - ema_err) / sigma", start))
    block = textwrap.dedent(source[start:end])
    history = [.8, 1., 1.2] * 10
    center, scale = baseline(history)
    causal_prior_score = (10. - center) / scale
    legacy_scope = {"np": np, "math": __import__("math"), "ema_count": 30, "ema_err": 1., "ema_alpha": .01,
                    "best_err": 10., "residual_history": history.copy(), "MAD_WINDOW": 128, "EIDOS_BRAIN_CONFIG": {"trace_seal_sigma_taper": "none"}}
    exec(compile(block, str(engine_path), "exec"), legacy_scope)
    assert len(legacy_scope["residual_history"]) == 31
    assert legacy_scope["z_score"] < causal_prior_score
    bridge = Path("services/sentinel-runner/sentinel_runner/engine_bridge.py").read_text()
    assert '"domain": "cicids_webattacks"' in bridge and 'features=64' in bridge
    receipt = {"status": "passed", "encoder": ENCODER, "legacyEngineSha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
               "sourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "legacy64PrefixIdentical": bool(legacy_identical), "semanticSuffixVectorDistance": float(np.linalg.norm(vectors[0] - vectors[1])),
               "paraphraseCosine": float(vectors[2] @ vectors[3]), "unrelatedCosine": float(vectors[2] @ vectors[4]),
               "allTokenWindows": True, "cacheExact": bool(np.array_equal(vectors, cached)),
               "legacySelectionExecutedASTLines": [n.lineno for n in nodes], "observations": [0., 10.], "legacySelectedPredictions": selected,
               "scoreOrderControlledMathExample": {"prior": prior_score, "afterCurrentUpdate": after_score, "beta": beta,
                   "scope": "controlled arithmetic illustration, not an empirical benefit or exact full legacy MAD-history trace"},
               "actualLegacyBaselineBlock": {"firstLine": source[:start].count("\n") + 1, "lastLine": source[:end].count("\n") + 1,
                   "causalPriorScore": causal_prior_score, "legacyAfterCurrentScore": legacy_scope["z_score"],
                   "historyBefore": 30, "historyAfter": len(legacy_scope["residual_history"]), "legacyEmaAfter": legacy_scope["ema_err"],
                   "meaning": "actual old score block versus actual new prior baseline, controlled input; interpretability audit, not utility evidence"},
               "legacyBridge": {"domain": "cicids_webattacks", "features": 64, "sha256": hashlib.sha256(bridge.encode()).hexdigest(),
                   "preserved": True, "newGuidedProfile": "confirmed_named_measurements with direct selected feature count"},
               "legacyUnmodified": True, "llmCalls": 0, "researchGatesAdvanced": 0}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__": main()
