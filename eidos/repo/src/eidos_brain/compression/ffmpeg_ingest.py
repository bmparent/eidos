"""Optional FFmpeg/ffprobe ingestion helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class FFmpegUnavailable(RuntimeError):
    """Raised when FFmpeg/ffprobe helpers are used without the external tools installed."""


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    return ffmpeg_path() is not None


def ffprobe_available() -> bool:
    return ffprobe_path() is not None


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegUnavailable(f"{name} is not installed or is not on PATH")
    return path


def probe_media(path: str | Path) -> dict[str, Any]:
    """Return compact ffprobe metadata for audio/video files."""

    ffprobe = _require_tool("ffprobe")
    media_path = str(path)
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", media_path],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams", [])
    format_info = payload.get("format", {})
    first_audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    first_video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    primary = first_video or first_audio or (streams[0] if streams else {})
    return {
        "duration": _maybe_float(format_info.get("duration") or primary.get("duration")),
        "codec": primary.get("codec_name"),
        "bitrate": _maybe_int(format_info.get("bit_rate") or primary.get("bit_rate")),
        "sample_rate": _maybe_int(first_audio.get("sample_rate")),
        "channels": _maybe_int(first_audio.get("channels")),
        "fps": _parse_rate(first_video.get("avg_frame_rate") or first_video.get("r_frame_rate")),
        "streams": len(streams),
    }


def audio_windows(
    path: str | Path,
    sample_rate: int = 16_000,
    window_size: int = 1024,
    max_windows: int | None = None,
) -> Iterator[np.ndarray]:
    """Decode audio to mono float32 PCM windows."""

    ffmpeg = _require_tool("ffmpeg")
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    limit = len(samples) // window_size
    if max_windows is not None:
        limit = min(limit, int(max_windows))
    for idx in range(limit):
        start = idx * window_size
        yield samples[start : start + window_size].copy()


def video_frame_windows(
    path: str | Path,
    fps: float = 1.0,
    size: tuple[int, int] = (64, 64),
    max_frames: int | None = None,
) -> Iterator[np.ndarray]:
    """Decode sampled RGB frames as uint8 arrays shaped ``(height, width, 3)``."""

    ffmpeg = _require_tool("ffmpeg")
    width, height = size
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={fps},scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    frame_bytes = width * height * 3
    total = len(proc.stdout) // frame_bytes
    if max_frames is not None:
        total = min(total, int(max_frames))
    for idx in range(total):
        start = idx * frame_bytes
        frame = np.frombuffer(proc.stdout[start : start + frame_bytes], dtype=np.uint8)
        yield frame.reshape((height, width, 3)).copy()


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_rate(value: Any) -> float | None:
    if not value:
        return None
    text = str(value)
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            denominator = float(den)
            return None if denominator == 0 else float(num) / denominator
        except ValueError:
            return None
    return _maybe_float(text)
