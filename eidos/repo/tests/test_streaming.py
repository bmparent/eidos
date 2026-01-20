"""Streaming adapters with mocked inputs."""

import itertools

import pytest

pytest.importorskip("torch")

from eidos_brain.engine import eidos_v0_4_7_02 as engine


def test_stream_http_lines_mocked(monkeypatch):
    def fake_stream_http_lines(url, headers, timeout):
        yield "1,2,3"
        yield "4,5,6"

    monkeypatch.setattr(engine, "stream_http_lines", fake_stream_http_lines)

    frames = list(
        itertools.islice(
            engine.stream_live_frames(
                features=3,
                kind="URL",
                url="http://example.com/stream",
                headers={},
                timeout=(1, 1),
                ip_endpoint=None,
                max_frames=2,
                normalize_online=False,
                project_seed=1,
                text_embed=True,
            ),
            2,
        )
    )

    assert len(frames) == 2
    assert frames[0][1]["stream_kind"] == "http"


def test_stream_websocket_mocked(monkeypatch):
    def fake_ws_lines(url, headers, out_q, stop_evt):
        out_q.put("1,2,3")
        out_q.put("4,5,6")

    monkeypatch.setattr(engine, "_require_websockets", lambda: object())
    monkeypatch.setattr(engine, "websocket_lines_to_queue", fake_ws_lines)

    frames = list(
        itertools.islice(
            engine.stream_live_frames(
                features=3,
                kind="URL",
                url="ws://example.com/socket",
                headers={},
                timeout=(1, 1),
                ip_endpoint=None,
                max_frames=2,
                normalize_online=False,
                project_seed=1,
                text_embed=True,
            ),
            2,
        )
    )

    assert len(frames) == 2
    assert frames[0][1]["stream_kind"] == "ws"


def test_stream_websocket_failure_emits_error(monkeypatch):
    def fake_ws_lines(url, headers, out_q, stop_evt):
        out_q.put(f"{engine.WS_ERROR_SENTINEL}:RuntimeError:boom")

    monkeypatch.setattr(engine, "_require_websockets", lambda: object())
    monkeypatch.setattr(engine, "websocket_lines_to_queue", fake_ws_lines)

    with pytest.raises(RuntimeError, match="Websocket stream error"):
        next(
            engine.stream_live_frames(
                features=3,
                kind="URL",
                url="ws://example.com/socket",
                headers={},
                timeout=(1, 1),
                ip_endpoint=None,
                max_frames=2,
                normalize_online=False,
                project_seed=1,
                text_embed=True,
            )
        )


@pytest.mark.parametrize("proto", ["tcp", "udp"])
def test_stream_ip_lines_mocked(monkeypatch, proto):
    def fake_stream_lines(host, port):
        yield "1,2,3"

    if proto == "tcp":
        monkeypatch.setattr(engine, "stream_tcp_lines", fake_stream_lines)
    else:
        monkeypatch.setattr(engine, "stream_udp_lines", fake_stream_lines)

    frames = list(
        itertools.islice(
            engine.stream_live_frames(
                features=3,
                kind="IP",
                url=None,
                headers={},
                timeout=(1, 1),
                ip_endpoint=f"{proto}://127.0.0.1:9000",
                max_frames=1,
                normalize_online=False,
                project_seed=1,
                text_embed=True,
            ),
            1,
        )
    )

    assert len(frames) == 1
    assert frames[0][1]["stream_kind"] == proto
