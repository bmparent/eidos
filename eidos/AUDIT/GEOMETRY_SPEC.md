# Geometry / Fractal Metrics

## Implementation
- Geometry sampling: reservoir states are sampled during run.
- `build_and_store_geometry()` computes box-count dimension and summary stats.

## Definition (Box Counting)
- Given state samples, for each scale `eps`:
  - Count occupied boxes in embedding space.
  - Estimate slope of `log(N(eps))` vs `log(1/eps)`.

## Outputs
- `geometry.json` artifact with estimated dimension and sample metadata.

## Failure Modes
- Too few samples yields unstable estimates.
- Highly correlated states can bias dimension low.
