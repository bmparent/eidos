# Quantum-Aware Anomaly-Preserving Compression

This adds a defensive compression and benchmarking layer to Eidos Brain. It treats Eidos as a predictive, anomaly-preserving compression and detection engine: ordinary frames can collapse into references or small residuals, while unusual frames are preserved as structured residuals or anomaly capsules with replay metadata.

It is not a magic quantum detector. For classical systems, the adapters look for indicators consistent with quantum-era cryptographic risk: legacy RSA/ECC dependency, downgrade pressure, PQC negotiation failures, harvest-now-decrypt-later risk proxies, entropy/RNG anomalies, and unusual encrypted egress patterns.

## What Changed

- `eidos_brain.compression.policy` maps Sentinel regimes and configured residual/surprise thresholds to compression modes.
- `eidos_brain.compression.residual_codec` predicts each frame, encodes residuals first, and preserves AMBER/RED/VIOLET context.
- `eidos_brain.compression.entropy_pack` writes readable JSONL tokens and optional binary/gzip/lzma/zstd-packed streams.
- `eidos_brain.compression.ffmpeg_ingest` wraps optional `ffmpeg`/`ffprobe` media ingestion without making those tools required.
- `eidos_brain.adapters.quantum_syndrome_adapter` converts quantum telemetry into fixed-dimensional Eidos frames.
- `eidos_brain.adapters.crypto_agility_adapter` converts defensive crypto-agility events into fixed-dimensional Eidos frames.
- `eidos_brain.adapters.binary_river_adapter` converts byte windows into entropy, sketch, and header-marker features.
- `eidos_brain.benchmarks.quantum_compression_benchmark` compares Eidos against conventional compression and anomaly baselines.
- The live Eidos run loop now emits `ResidualFirstCodec` token streams using the engine's real consensus
  predictor, Sentinel status/metrics, and hippocampus familiarity/write metrics.

## Compression Policy

The default policy is configurable through `CompressionPolicyConfig`:

| Sentinel Status | Default Mode | Preservation Intent |
| --- | --- | --- |
| GREEN | `reference_or_null` | Aggressive compression for predictable frames |
| BLUE | `low_residual` | Quantized residual for mild departures |
| VIOLET | `structured_residual` | Preserve residual structure and top drivers |
| AMBER | `anomaly_capsule` | Preserve raw frame, residual, drivers, and replay context |
| RED | `raw_frame_plus_full_context` | Preserve raw frame, residual, model summary, Sentinel metrics, and replay context |

Thresholds such as `violet_residual_norm`, `amber_residual_norm`, and `red_surprise_z` are configuration fields, so deployments can tune them per domain.

## Running Tests

From `eidos/repo`:

```bash
python -m pytest tests/test_residual_codec_roundtrip.py tests/test_quantum_syndrome_adapter.py tests/test_crypto_agility_adapter.py tests/test_binary_river_adapter.py
python -m pytest tests/test_quantum_compression_benchmark_smoke.py
```

FFmpeg tests skip when `ffmpeg` or `ffprobe` is unavailable. Optional zstd, scikit-learn, and River baselines are also skipped when dependencies are not installed.

## Running The Benchmark

From `eidos/repo`:

```bash
python -m eidos_brain.benchmarks.quantum_compression_benchmark --output-dir artifacts/benchmarks
```

The benchmark writes:

- `artifacts/benchmarks/quantum_compression_comparison.json`
- `artifacts/benchmarks/quantum_compression_comparison.md`
- `artifacts/benchmarks/quantum_compression_comparison.csv`

Synthetic scenarios include:

- normal quantum stream
- decoherence drift stream
- syndrome burst stream
- readout bias stream
- crypto downgrade/exfiltration risk stream
- binary river high-entropy blob stream

## Live Run Artifacts

During `run_session`, the residual codec writes artifacts under:

```text
compression/<profile>/
```

The live token metadata records:

- `prediction_source = live_eidos_consensus_best_pred`
- `sentinel_source = live_sentinel_analyze`
- `hdc_source = live_hippocampus_metrics`
- mode and Sentinel status distributions
- packed token bytes and compression ratio
- reconstruction RMSE when prediction storage is enabled

The default config stores per-token predictions so JSONL tokens can be replayed offline. Deployments that prefer
smaller token streams can set `residual_codec_store_prediction` to `False` and reconstruct only with a matching model
replay path.

## Interpreting Metrics

- `compression_ratio`: raw feature bytes divided by emitted or packed bytes. Higher is smaller output.
- `reconstruction_error`: RMSE between original feature frames and decoded frames where reconstruction applies.
- `anomaly_preservation_score`: fraction of labeled anomaly frames emitted as structured residuals, anomaly capsules, or raw-context tokens.
- `anomaly_capsule_count`: count of frames preserved with anomaly capsule or full raw context modes.
- `precision`, `recall`, `F1`, `false positives`, and `false negatives`: detection quality against deterministic synthetic labels.
- `detection_delay`: number of frames between the first labeled anomaly and first detection after that point.
- `compression_collapse_score`: how much anomaly-frame compression intentionally collapses relative to normal-frame compression. Higher means anomalies are being preserved rather than over-compressed.

## Known Limitations

- The benchmark uses deterministic synthetic telemetry, not calibrated production quantum hardware or enterprise network data.
- The standalone benchmark still uses a lightweight Sentinel proxy; live `run_session` artifacts use the real Eidos
  consensus predictor, Sentinel outputs, and HDC metrics.
- Conventional compression baselines operate over feature-frame bytes, not every possible raw source representation.
- The crypto-agility adapter is defensive monitoring only and does not perform exploit generation, key recovery, traffic interception, or offensive analysis.
- FFmpeg support depends on external `ffmpeg` and `ffprobe` binaries being present on `PATH`.

## Future Work

- Wire the residual codec directly into the live Eidos run loop so it can use reservoir predictions and real Sentinel metrics per frame.
- Calibrate policy thresholds by domain profile.
- Add golden replay fixtures for anomaly capsules.
- Add production telemetry adapters for concrete quantum-control and crypto-inventory schemas.
- Track HDC familiarity/recurrence fields when live hippocampus metrics are available.
