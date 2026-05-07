from pathlib import Path

from eidos_brain.benchmarks.quantum_compression_benchmark import run_benchmark


def test_quantum_compression_benchmark_smoke(tmp_path):
    results = run_benchmark(tmp_path, n_frames=24, seed=4, include_optional=False)
    scenarios = {row["scenario"] for row in results}
    methods = {row["method"] for row in results}

    assert "quantum_decoherence_drift" in scenarios
    assert "crypto_downgrade_exfiltration_risk" in scenarios
    assert "binary_high_entropy_blob" in scenarios
    assert "eidos_residual_codec_gzip" in methods
    assert "gzip_raw_frames" in methods
    assert (Path(tmp_path) / "quantum_compression_comparison.json").exists()
    assert (Path(tmp_path) / "quantum_compression_comparison.md").exists()
    assert (Path(tmp_path) / "quantum_compression_comparison.csv").exists()
