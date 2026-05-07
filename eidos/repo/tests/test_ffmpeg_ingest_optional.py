import pytest

from eidos_brain.compression import ffmpeg_ingest


def test_ffmpeg_detection_helpers_do_not_require_tools():
    assert isinstance(ffmpeg_ingest.ffmpeg_available(), bool)
    assert isinstance(ffmpeg_ingest.ffprobe_available(), bool)


def test_ffprobe_optional_skip_when_unavailable(tmp_path):
    if not ffmpeg_ingest.ffprobe_available():
        pytest.skip("ffprobe is not installed; optional media metadata test skipped")

    missing = tmp_path / "missing.mp4"
    with pytest.raises(Exception):
        ffmpeg_ingest.probe_media(missing)
