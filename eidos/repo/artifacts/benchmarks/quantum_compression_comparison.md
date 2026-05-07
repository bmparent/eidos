# Quantum-Aware Anomaly-Preserving Compression Benchmark

Synthetic, deterministic local benchmark. Eidos is evaluated as a predictive compression and anomaly-preservation layer, not as a magic quantum detector.

| Scenario | Method | Compression Ratio | Reconstruction Error | Precision | Recall | F1 | Detection Delay | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| quantum_normal | eidos_residual_codec_gzip | 1.585 | 0.004591 | 0 | 1 | 0 | - | Predictive Eidos token stream with Sentinel proxy; proxy warmup=8. |
| quantum_normal | raw_uncompressed | 1 | 0 | - | - | - | - | Raw feature bytes. |
| quantum_normal | gzip_raw_frames | 4.914 | 0 | - | - | - | - | gzip over raw frames. |
| quantum_normal | lzma_raw_frames | 5.399 | 0 | - | - | - | - | lzma over raw frames. |
| quantum_normal | naive_delta_int16 | 1.979 | 0.0006844 | - | - | - | - | First frame plus int16 quantized deltas; defensive baseline, not anomaly-aware. |
| quantum_normal | rolling_zscore_detector | 1 | 0 | 0 | 1 | 0 | - | Simple rolling z-score detector over feature energy. |
| quantum_decoherence_drift | eidos_residual_codec_gzip | 1.564 | 0.003999 | 0.9412 | 0.6667 | 0.7805 | 0 | Predictive Eidos token stream with Sentinel proxy; proxy warmup=8. |
| quantum_decoherence_drift | raw_uncompressed | 1 | 0 | - | - | - | - | Raw feature bytes. |
| quantum_decoherence_drift | gzip_raw_frames | 4.861 | 0 | - | - | - | - | gzip over raw frames. |
| quantum_decoherence_drift | lzma_raw_frames | 5.352 | 0 | - | - | - | - | lzma over raw frames. |
| quantum_decoherence_drift | naive_delta_int16 | 1.979 | 0.001013 | - | - | - | - | First frame plus int16 quantized deltas; defensive baseline, not anomaly-aware. |
| quantum_decoherence_drift | rolling_zscore_detector | 1 | 0 | 1 | 0.5 | 0.6667 | 5 | Simple rolling z-score detector over feature energy. |
| quantum_syndrome_burst | eidos_residual_codec_gzip | 1.26 | 0.004154 | 1 | 1 | 1 | 0 | Predictive Eidos token stream with Sentinel proxy; proxy warmup=8. |
| quantum_syndrome_burst | raw_uncompressed | 1 | 0 | - | - | - | - | Raw feature bytes. |
| quantum_syndrome_burst | gzip_raw_frames | 4.975 | 0 | - | - | - | - | gzip over raw frames. |
| quantum_syndrome_burst | lzma_raw_frames | 5.585 | 0 | - | - | - | - | lzma over raw frames. |
| quantum_syndrome_burst | naive_delta_int16 | 1.979 | 0.0006153 | - | - | - | - | First frame plus int16 quantized deltas; defensive baseline, not anomaly-aware. |
| quantum_syndrome_burst | rolling_zscore_detector | 1 | 0 | 1 | 1 | 1 | 0 | Simple rolling z-score detector over feature energy. |
| quantum_readout_bias_shift | eidos_residual_codec_gzip | 1.202 | 0.00278 | 0.7302 | 0.9583 | 0.8288 | 0 | Predictive Eidos token stream with Sentinel proxy; proxy warmup=8. |
| quantum_readout_bias_shift | raw_uncompressed | 1 | 0 | - | - | - | - | Raw feature bytes. |
| quantum_readout_bias_shift | gzip_raw_frames | 5.092 | 0 | - | - | - | - | gzip over raw frames. |
| quantum_readout_bias_shift | lzma_raw_frames | 5.642 | 0 | - | - | - | - | lzma over raw frames. |
| quantum_readout_bias_shift | naive_delta_int16 | 1.979 | 0.0006421 | - | - | - | - | First frame plus int16 quantized deltas; defensive baseline, not anomaly-aware. |
| quantum_readout_bias_shift | rolling_zscore_detector | 1 | 0 | 0.8333 | 0.625 | 0.7143 | 1 | Simple rolling z-score detector over feature energy. |
| crypto_downgrade_exfiltration_risk | eidos_residual_codec_gzip | 1.376 | 0.001581 | 0.5647 | 1 | 0.7218 | 0 | Predictive Eidos token stream with Sentinel proxy; proxy warmup=8. |
| crypto_downgrade_exfiltration_risk | raw_uncompressed | 1 | 0 | - | - | - | - | Raw feature bytes. |
| crypto_downgrade_exfiltration_risk | gzip_raw_frames | 9.052 | 0 | - | - | - | - | gzip over raw frames. |
| crypto_downgrade_exfiltration_risk | lzma_raw_frames | 10.17 | 0 | - | - | - | - | lzma over raw frames. |
| crypto_downgrade_exfiltration_risk | naive_delta_int16 | 1.979 | 0.0006982 | - | - | - | - | First frame plus int16 quantized deltas; defensive baseline, not anomaly-aware. |
| crypto_downgrade_exfiltration_risk | rolling_zscore_detector | 1 | 0 | 0.6154 | 1 | 0.7619 | 0 | Simple rolling z-score detector over feature energy. |
| binary_high_entropy_blob | eidos_residual_codec_gzip | 1.14 | 0.007848 | 0.9796 | 1 | 0.9897 | 0 | Predictive Eidos token stream with Sentinel proxy; proxy warmup=8. |
| binary_high_entropy_blob | raw_uncompressed | 1 | 0 | - | - | - | - | Raw feature bytes. |
| binary_high_entropy_blob | gzip_raw_frames | 3.69 | 0 | - | - | - | - | gzip over raw frames. |
| binary_high_entropy_blob | lzma_raw_frames | 4.77 | 0 | - | - | - | - | lzma over raw frames. |
| binary_high_entropy_blob | naive_delta_int16 | 1.979 | 0.001022 | - | - | - | - | First frame plus int16 quantized deltas; defensive baseline, not anomaly-aware. |
| binary_high_entropy_blob | rolling_zscore_detector | 1 | 0 | 1 | 1 | 1 | 0 | Simple rolling z-score detector over feature energy. |

## Plain-Language Conclusion

- Eidos beat conventional gzip/lzma/zstd compression on: none in this run.
- Conventional entropy codecs beat Eidos on: quantum_normal (lzma_raw_frames), quantum_decoherence_drift (lzma_raw_frames), quantum_syndrome_burst (lzma_raw_frames), quantum_readout_bias_shift (lzma_raw_frames), crypto_downgrade_exfiltration_risk (lzma_raw_frames), binary_high_entropy_blob (lzma_raw_frames).
- Eidos anomaly preservation averaged 0.9375 across labeled scenarios by emitting structured residuals or capsules.
- Eidos false positives across all scenarios: 87; tune surprise thresholds if this is too high for the deployment domain.
- Next tuning target: calibrate Sentinel proxy thresholds per domain and connect the codec to the live reservoir predictor for lower normal-frame token cost.
