"""Optional dependency guards."""

import importlib
import importlib.util
import sys

import pytest

pytest.importorskip("torch")

from eidos_brain.engine import eidos_v0_4_7_02 as engine


def test_missing_websockets_guard():
    if importlib.util.find_spec("websockets") is not None:
        pytest.skip("websockets installed")

    with pytest.raises(ImportError, match="websockets"):
        engine.stream_live_frames(
            features=3,
            kind="URL",
            url="ws://example.com/socket",
            headers={},
            timeout=(1, 1),
            ip_endpoint=None,
            max_frames=1,
            normalize_online=False,
            project_seed=1,
            text_embed=True,
        )


def test_missing_pubsub_guard():
    if importlib.util.find_spec("google.cloud.pubsub_v1") is not None:
        pytest.skip("google-cloud-pubsub installed")

    with pytest.raises(ImportError, match="google-cloud-pubsub"):
        engine._stream_pubsub_generator("project", "sub", features=3, max_frames=1)


def test_missing_gcs_backend_guard(monkeypatch):
    if importlib.util.find_spec("google.cloud.storage") is not None:
        pytest.skip("google-cloud-storage installed")

    module_name = "eidos_brain.engine.eidos_v0_4_7_02"
    sys.modules.pop(module_name, None)
    monkeypatch.setenv("HIVE_BACKEND", "GCS")

    with pytest.raises(RuntimeError, match="google-cloud-storage"):
        importlib.import_module(module_name)
