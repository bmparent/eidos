import numpy as np

from eidos_brain.adapters import QuantumSyndromeAdapter, generate_quantum_telemetry_stream


def test_quantum_adapter_emits_fixed_dimension_and_metadata():
    adapter = QuantumSyndromeAdapter(features=64, hash_seed=5)
    event = generate_quantum_telemetry_stream("normal", n_frames=1, seed=1)[0]
    frame, metadata = adapter.transform(event)

    assert frame.shape == (64,)
    assert len(metadata["feature_names"]) == 64
    assert metadata["source_id"] == "quantum.synthetic"
    assert np.all(np.isfinite(frame))


def test_syndrome_burst_generator_increases_burst_feature():
    normal_adapter = QuantumSyndromeAdapter(features=64, hash_seed=5)
    burst_adapter = QuantumSyndromeAdapter(features=64, hash_seed=5)
    normal_events = generate_quantum_telemetry_stream("normal", n_frames=24, seed=2)
    burst_events = generate_quantum_telemetry_stream("syndrome_burst", n_frames=24, seed=2)

    normal_frames, _ = normal_adapter.transform_many(normal_events)
    burst_frames, burst_meta = burst_adapter.transform_many(burst_events)

    assert any(meta["is_anomaly"] for meta in burst_meta)
    assert burst_frames[12:, 1].mean() > normal_frames[12:, 1].mean()
