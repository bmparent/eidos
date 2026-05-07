import json

import pytest

from eidos_brain.compression import JSONLTokenWriter


def _token(frame_id, mode="low_residual"):
    return {
        "frame_id": frame_id,
        "compression_mode": mode,
        "payload": {"q_residual": [frame_id]},
    }


def test_jsonl_token_writer_flushes_final_buffer_on_exception(tmp_path):
    path = tmp_path / "exception-safe-tokens.jsonl"

    with pytest.raises(RuntimeError):
        with JSONLTokenWriter(path, flush_every=100, flush_bytes=1_000_000) as writer:
            writer.write(_token(1))
            writer.write(_token(2, "anomaly_capsule"))
            raise RuntimeError("simulated stream interruption")

    lines = path.read_text(encoding="utf-8").splitlines()
    tokens = [json.loads(line) for line in lines]
    assert [token["frame_id"] for token in tokens] == [1, 2]
    assert tokens[1]["compression_mode"] == "anomaly_capsule"


def test_jsonl_token_writer_finalize_is_idempotent(tmp_path):
    path = tmp_path / "tokens.jsonl"
    writer = JSONLTokenWriter(path, flush_every=10)
    writer.write(_token(10))

    first = writer.finalize()
    second = writer.finalize()

    assert first == second
    assert first["tokens_written"] == 1
    assert first["first_frame_id"] == 10
    assert first["last_frame_id"] == 10
    assert path.read_text(encoding="utf-8").count("\n") == 1
