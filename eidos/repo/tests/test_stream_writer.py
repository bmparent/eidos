import gzip
import json

from eidos_brain.compression import JSONLTokenWriter, PackedTokenWriter, unpack_tokens
from eidos_brain.compression.stream_writer import InMemoryTokenWriter, NullTokenWriter


def _token(frame_id: int, mode: str = "low_residual"):
    return {
        "frame_id": frame_id,
        "compression_mode": mode,
        "payload": {"q_residual": [frame_id]},
    }


def test_jsonl_token_writer_flushes_bounded_chunks(tmp_path):
    path = tmp_path / "tokens.jsonl"
    with JSONLTokenWriter(path, flush_every=2, flush_bytes=10_000) as writer:
        writer.write(_token(1))
        writer.write(_token(2, "anomaly_capsule"))
        writer.write(_token(3))
        summary = writer.finalize()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["frame_id"] for line in lines] == [1, 2, 3]
    assert summary["tokens_written"] == 3
    assert summary["chunks_written"] == 2
    assert summary["first_frame_id"] == 1
    assert summary["last_frame_id"] == 3
    assert summary["anomaly_capsule_count"] == 1
    assert summary["compression_mode_counts"]["anomaly_capsule"] == 1
    assert summary["bytes_written"] == path.stat().st_size


def test_packed_token_writer_roundtrips_gzip(tmp_path):
    path = tmp_path / "tokens.gz"
    tokens = [_token(1), _token(2, "raw_frame_plus_full_context"), _token(3)]
    with PackedTokenWriter(path, codec="gzip", flush_every=1) as writer:
        for token in tokens:
            writer.write(token)
        summary = writer.finalize()

    assert unpack_tokens(path.read_bytes(), codec="gzip") == tokens
    assert gzip.decompress(path.read_bytes()).count(b"\n") == 3
    assert summary["tokens_written"] == 3
    assert summary["chunks_written"] == 3
    assert summary["anomaly_capsule_count"] == 1
    assert summary["bytes_written"] == path.stat().st_size
    assert summary["input_bytes"] > 0


def test_packed_token_writer_roundtrips_binary(tmp_path):
    path = tmp_path / "tokens.bin"
    tokens = [_token(10), _token(11)]
    writer = PackedTokenWriter(path, codec="binary", flush_every=10)
    for token in tokens:
        writer.write(token)
    summary = writer.finalize()

    assert unpack_tokens(path.read_bytes(), codec="binary") == tokens
    assert summary["tokens_written"] == 2
    assert summary["chunks_written"] == 1


def test_packed_token_writer_roundtrips_jsonl(tmp_path):
    path = tmp_path / "tokens.jsonl"
    tokens = [_token(20), _token(21)]
    with PackedTokenWriter(path, codec="jsonl", flush_every=1) as writer:
        for token in tokens:
            writer.write(token)
        summary = writer.finalize()

    assert unpack_tokens(path.read_bytes(), codec="jsonl") == tokens
    assert summary["tokens_written"] == 2
    assert summary["chunks_written"] == 2


def test_memory_and_null_writers_track_without_files():
    memory = InMemoryTokenWriter()
    null = NullTokenWriter()
    for writer in (memory, null):
        writer.write(_token(1))
        writer.write(_token(2, "anomaly_capsule"))
        summary = writer.finalize()
        assert summary["tokens_written"] == 2
        assert summary["anomaly_capsule_count"] == 1

    assert len(memory.tokens) == 2
