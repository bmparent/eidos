import numpy as np

from eidos_brain.adapters import BinaryRiverAdapter, generate_binary_stream


def test_binary_adapter_flags_high_entropy_blob():
    adapter = BinaryRiverAdapter(features=64)
    normal_event = generate_binary_stream("normal", n_windows=1, seed=3)[0]
    blob_event = generate_binary_stream("high_entropy_blob", n_windows=4, seed=3)[-1]

    normal_frame, _ = adapter.transform(normal_event)
    blob_frame, metadata = adapter.transform(blob_event)

    assert normal_frame.shape == (64,)
    assert blob_frame.shape == (64,)
    assert blob_frame[0] > normal_frame[0]
    assert blob_frame[12] == 1.0
    assert metadata["is_anomaly"] is True
    assert np.all(np.isfinite(blob_frame))


def test_binary_stream_is_deterministic_for_same_seed():
    left = generate_binary_stream("high_entropy_blob", n_windows=6, seed=44)
    right = generate_binary_stream("high_entropy_blob", n_windows=6, seed=44)

    assert [item["bytes"] for item in left] == [item["bytes"] for item in right]
