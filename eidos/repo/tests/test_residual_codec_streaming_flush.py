import json

import numpy as np

from eidos_brain.compression import JSONLTokenWriter, PackedTokenWriter, ResidualFirstCodec, unpack_tokens


def _frames(count):
    for idx in range(count):
        base = np.array([idx * 0.01, 0.0, 0.0], dtype=float)
        if idx == 6:
            base += np.array([3.0, -2.0, 1.0], dtype=float)
        yield idx, base


def test_residual_codec_streams_tokens_to_writers_without_token_accumulation(tmp_path):
    jsonl_path = tmp_path / "residual-tokens.jsonl"
    packed_path = tmp_path / "residual-tokens.gz"
    codec = ResidualFirstCodec(feature_names=["a", "b", "c"], store_prediction_on_tokens=True)

    with JSONLTokenWriter(jsonl_path, flush_every=3) as jsonl_writer, PackedTokenWriter(
        packed_path,
        codec="gzip",
        flush_every=2,
    ) as packed_writer:
        for idx, frame in _frames(9):
            token = codec.encode_frame(
                frame,
                {
                    "frame_id": idx,
                    "prediction": np.zeros(3, dtype=float),
                    "sentinel_status": "RED" if idx == 6 else "GREEN",
                    "surprise_z": 9.0 if idx == 6 else 0.0,
                },
            )
            jsonl_writer.write(token)
            packed_writer.write(token)

        jsonl_summary = jsonl_writer.finalize()
        packed_summary = packed_writer.finalize()

    assert not hasattr(jsonl_writer, "tokens")
    assert jsonl_summary["tokens_written"] == 9
    assert jsonl_summary["chunks_written"] == 3
    assert jsonl_summary["anomaly_capsule_count"] == 1
    assert packed_summary["tokens_written"] == 9
    assert packed_summary["chunks_written"] == 5

    jsonl_tokens = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    packed_tokens = unpack_tokens(packed_path.read_bytes(), codec="gzip")
    assert jsonl_tokens == packed_tokens
    assert jsonl_tokens[6]["compression_mode"] == "raw_frame_plus_full_context"
