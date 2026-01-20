"""Preflight checks."""

import pytest

pytest.importorskip("torch")

from eidos_brain.engine import eidos_v0_4_7_02 as engine


def test_preflight_manual_local_missing_path_fails(monkeypatch):
    monkeypatch.setattr(engine, "DATA_SOURCE_TYPE", "LOCAL")
    monkeypatch.setattr(engine, "LOCAL_MODE", "ARCHIVE")
    monkeypatch.setattr(engine, "LOCAL_TARGET", "/no/such/dir")
    monkeypatch.setattr(engine, "CONFIG_MODE", "MANUAL")

    with pytest.raises(FileNotFoundError):
        engine._preflight_inputs()


def test_preflight_nl_mode_does_not_block(monkeypatch):
    monkeypatch.setattr(engine, "DATA_SOURCE_TYPE", "LOCAL")
    monkeypatch.setattr(engine, "LOCAL_MODE", "ARCHIVE")
    monkeypatch.setattr(engine, "LOCAL_TARGET", "/no/such/dir")
    monkeypatch.setattr(engine, "CONFIG_MODE", "NL_GEMINI")

    engine._preflight_inputs()
