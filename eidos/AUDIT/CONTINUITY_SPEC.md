# Continuity Spec

## Operational Definition
Continuity means a run can resume with the same internal state and verify identity via hashes.

## State Capsule
A minimal capsule is stored per session:
- `state_capsule.json` contains:
  - `session_id`, `engine_version`, `config_hash`
  - `synaptic_hash_initial` and `synaptic_hash_final`
  - `continuity_hash` (hash of config + reservoir state/weights + synaptic hash)
  - `gate_state` (z-threshold, ema_err, sigma, surprise_ema, fatigue)
  - `checkpoint_path` (reservoir checkpoint)
  - `hippocampus_snapshot_path`

## Continuity Hash
- Hash material: `config_hash`, `reservoir_state_hash`, `reservoir_weights_hash`, `synaptic_hash_final`.
- If `continuity_expected_hash` is provided in config, the engine emits a warning on mismatch.

## Persistence Paths
- Reservoir checkpoint: `reservoir_checkpoints/<profile>/reservoir_checkpoint_...pt`.
- Hippocampus snapshot: `hippocampus/<profile>/hippocampus_snapshot_...pt`.
- Capsule: `<session_dir>/state_capsule.json`.

## Known Gaps
- Normalizer state and quantile gate state are not yet serialized separately.
- Resume-from-capsule requires manual wiring (optional future work).
