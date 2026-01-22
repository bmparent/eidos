# Spectral / Regime Spec

## Implementation
- `SpectralMonitor` (`repo/src/eidos_brain/engine/eidos_v0_4_7_02.py`).
- Operates on a scalar summary of the stream (mean of frame).

## Definition
- Sliding window buffer collects recent scalar samples.
- Compute FFT magnitude spectrum.
- Spectral entropy:
  - Normalize power spectrum `p_i = S_i / sum(S)`
  - `Hs = -sum(p_i * log(p_i)) / log(n)`
- Spectral flatness:
  - `flatness = geometric_mean(S) / arithmetic_mean(S)`

## Outputs
- `spectral_entropy` (0..1)
- `spectral_flatness` (0..1)

## Failure Modes
- Short windows reduce frequency resolution.
- Constant signals can yield near-zero power (guarded by eps).
