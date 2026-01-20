"""Multi-session isolation checks."""

from eidos_brain.engine import eidos_v0_4_7_02 as engine


def test_deep_merge_dict():
    dst = {"a": {"b": 1, "c": 2}, "d": 1}
    src = {"a": {"b": 3}, "e": {"f": 4}}

    merged = engine.deep_merge_dict(dst, src)

    assert merged["a"]["b"] == 3
    assert merged["a"]["c"] == 2
    assert merged["d"] == 1
    assert merged["e"]["f"] == 4


def test_multi_session_isolation(tmp_path):
    import pytest

    pytest.importorskip("torch")
    from eidos_brain.engine.adapters import run_session

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"

    config_a = {
        "source_type": "LOCAL",
        "artifact_root": str(root_a),
        "profile_label": "profile_a",
        "source_params": {
            "local": {
                "mode": "SYNTHETIC",
                "max_frames": 10,
                "target": "a",
            }
        },
        "engine_config": {
            "steps": 10,
            "warmup_cap": 2,
            "reservoir": 32,
        },
    }

    config_b = {
        "source_type": "LOCAL",
        "artifact_root": str(root_b),
        "profile_label": "profile_b",
        "source_params": {
            "local": {
                "mode": "SYNTHETIC",
                "max_frames": 20,
                "target": "b",
            }
        },
        "engine_config": {
            "steps": 20,
            "warmup_cap": 2,
            "reservoir": 96,
        },
    }

    result_a = run_session(config_a)
    result_b = run_session(config_b)

    assert result_a["artifact_root"] == str(root_a)
    assert result_b["artifact_root"] == str(root_b)
    assert engine.PROFILE_LABEL == "profile_b"
    assert engine.LOCAL_MAX_FRAMES == 20
    assert engine.EIDOS_BRAIN_CONFIG["reservoir"] == 96
