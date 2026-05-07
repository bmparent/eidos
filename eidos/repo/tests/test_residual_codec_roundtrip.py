import numpy as np

from eidos_brain.compression import (
    ResidualFirstCodec,
    anomaly_preservation_score,
    pack_tokens,
    reconstruction_error,
    tokens_from_jsonl,
    tokens_to_jsonl,
    unpack_tokens,
)


def test_residual_codec_roundtrip_preserves_anomaly_context():
    frames = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [3.5, -2.0, 1.0],
            [3.5, -2.0, 1.0],
        ],
        dtype=float,
    )
    metadata = [
        {"frame_id": 0, "sentinel_status": "GREEN", "surprise_z": 0.0},
        {"frame_id": 1, "sentinel_status": "GREEN", "surprise_z": 0.0},
        {"frame_id": 2, "sentinel_status": "BLUE", "surprise_z": 1.4},
        {
            "frame_id": 3,
            "sentinel_status": "RED",
            "surprise_z": 10.0,
            "model_state_summary": {"reservoir_state_hash": "abc"},
            "sentinel_metrics": {"dominance": 0.8},
        },
        {"frame_id": 4, "sentinel_status": "GREEN", "surprise_z": 0.0},
    ]

    codec = ResidualFirstCodec(feature_names=["a", "b", "c"], source_id="unit")
    tokens = codec.encode_stream(frames, metadata)
    reconstructed = ResidualFirstCodec(feature_names=["a", "b", "c"]).decode_stream(tokens)

    assert tokens[1]["compression_mode"] == "reference_or_null"
    assert tokens[2]["compression_mode"] == "low_residual"
    assert tokens[3]["compression_mode"] == "raw_frame_plus_full_context"
    assert "raw_frame" in tokens[3]["payload"]
    assert reconstruction_error(frames, reconstructed) < 0.02
    assert anomaly_preservation_score(tokens, [False, False, False, True, False]) == 1.0


def test_token_jsonl_and_entropy_pack_roundtrip():
    frames = np.zeros((3, 4), dtype=float)
    codec = ResidualFirstCodec(feature_names=["a", "b", "c", "d"])
    tokens = codec.encode_stream(frames)

    jsonl = tokens_to_jsonl(tokens)
    assert tokens_from_jsonl(jsonl) == tokens

    for packer in ("binary", "gzip", "lzma"):
        packed = pack_tokens(tokens, codec=packer)
        unpacked = unpack_tokens(packed)
        assert unpacked == tokens
