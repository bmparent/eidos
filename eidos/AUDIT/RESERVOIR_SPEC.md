# Reservoir Spec

## Implementation
- Class: `RLSReservoir` in `repo/src/eidos_brain/engine/eidos_v0_4_7_02.py`.
- Initialization creates:
  - Input weights `W_in`
  - Reservoir weights `W` (spectral radius scaled)
  - Output weights `W_out`
  - State vector `state`

## State Update
- Core update (conceptual):
  - `h_{t+1} = (1 - leak) * h_t + leak * tanh(W_in x_t + W h_t + b)`
- `listen()` updates state without learning.
- `adapt()` updates `W_out` using RLS and updates state.

## Determinism / Dtype
- Random seeds are set for numpy/torch in the engine configuration.
- CUDA determinism flags optional (`deterministic_cuda`).
- Dtype governed by `precision` config (float32 or float64).

## Failure Modes
- Poor scaling can lead to saturation or divergence.
- High spectral radius + leak can cause unstable activations.
