# Receipts Status Map

| Receipt Field | Location / Source | Status |
| --- | --- | --- |
| `session_id` | `SessionRecorder.session_id` | Implemented |
| `run_version` | `meta["engine_version"]` set in `run_eidos_sentinel()` | Implemented |
| `config_hash` | `meta["config_hash"]` from `get_config_hash()` | Implemented |
| `synaptic_hash_initial` | `initial_hash` captured in run loop | Implemented |
| `synaptic_hash_current` | `record_anomaly(... synaptic_hash_current=right_brain.get_synaptic_hash())` | Implemented |
| `t` / `timestamp` | `record_anomaly(... step, timestamp)` | Implemented |
| `source` (dataset/path/row/snippet) | `context_meta` from adapters/meta | Implemented |
| `err`, `ema_err`, `sigma`, `z` | Computed per-step in `run_eidos_sentinel()` | Implemented |
| `thresholds.z_thresh_eff` | `eff_z_thresh` | Implemented |
| `thresholds.abs_threshold` | `current_threshold` passed to recorder | Implemented |
| `attrib.topk` | `_residual_payload()` in run loop | Implemented |
| `fingerprint_topk` | Derived from `attrib.topk_features` | Implemented |
| `baseline` | `context_meta.norm_mean_head/std` + `ema_err/sigma` | Implemented |
| `recent_window` | Computed in `SessionRecorder.record_anomaly()` | Implemented |
| `evidence_paths` | Constructed in `SessionRecorder.record_anomaly()` | Implemented |

## Notes
- Attribution is only populated on surprise frames; receipts now include empty objects if missing.
- Evidence paths reference session artifacts; not all artifacts exist for every run (e.g., geometry).
