from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "repo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eidos_brain.experiments.controlled_regimes import generate_controlled_regime_stream
from eidos_brain.sentinel.input_detectors import InputEvidenceDetector
from eidos_brain.sentinel.sentinel_v3 import SentinelV3, SentinelV3Config


def _run_sequence(*, warmup: int, regime_frames: dict[str, int], features: int = 16, reservoir: int = 64):
    sentinel = SentinelV3(SentinelV3Config(features=features, reservoir_size=reservoir, warmup=warmup, seed=42))
    rows = []
    for item in generate_controlled_regime_stream(
        features=features,
        warmup=warmup,
        seed=42,
        regime_frames=regime_frames,
    ):
        rows.append((item.regime, sentinel.step(item.frame)))
    return rows


def _alert_rate(rows, regime: str) -> float:
    selected = [row for label, row in rows if label == regime]
    assert selected
    return sum(1 for row in selected if row["status"] in {"AMBER", "RED"}) / len(selected)


def _red_rate(rows, regime: str) -> float:
    selected = [row for label, row in rows if label == regime]
    assert selected
    return sum(1 for row in selected if row["status"] == "RED") / len(selected)


def test_residual_stats_are_finite():
    rows = _run_sequence(
        warmup=120,
        regime_frames={"NORMAL": 80, "BACKDOOR_PERIODIC": 0, "NOISE_CRASH": 0, "FROZEN_LOW_VARIANCE": 0},
    )
    post_warmup = [row for label, row in rows if label != "WARMUP"]
    assert post_warmup
    assert all(math.isfinite(float(row["ema_err"])) for row in post_warmup)
    assert all(math.isfinite(float(row["sigma"])) and float(row["sigma"]) > 0.0 for row in post_warmup)


def test_normal_suppression_low_false_positive():
    rows = _run_sequence(
        warmup=180,
        regime_frames={"NORMAL": 320, "BACKDOOR_PERIODIC": 0, "NOISE_CRASH": 0, "FROZEN_LOW_VARIANCE": 0},
    )
    assert _alert_rate(rows, "NORMAL") <= 0.05


def test_frozen_low_variance_goes_red():
    rows = _run_sequence(
        warmup=180,
        regime_frames={"NORMAL": 0, "BACKDOOR_PERIODIC": 0, "NOISE_CRASH": 0, "FROZEN_LOW_VARIANCE": 320},
    )
    assert _red_rate(rows, "FROZEN_LOW_VARIANCE") >= 0.50
    assert any(row["adaptation_frozen"] for label, row in rows if label == "FROZEN_LOW_VARIANCE")


def test_noise_crash_goes_amber_or_red():
    rows = _run_sequence(
        warmup=180,
        regime_frames={"NORMAL": 0, "BACKDOOR_PERIODIC": 0, "NOISE_CRASH": 320, "FROZEN_LOW_VARIANCE": 0},
    )
    assert _alert_rate(rows, "NOISE_CRASH") >= 0.80


def test_backdoor_periodic_detector():
    rows = _run_sequence(
        warmup=180,
        regime_frames={"NORMAL": 0, "BACKDOOR_PERIODIC": 360, "NOISE_CRASH": 0, "FROZEN_LOW_VARIANCE": 0},
    )
    assert _alert_rate(rows, "BACKDOOR_PERIODIC") >= 0.80
    assert max(float(row["period47_score"]) for label, row in rows if label == "BACKDOOR_PERIODIC") > 0.0


def test_no_regime_label_leakage():
    assert "regime" not in inspect.signature(SentinelV3.step).parameters
    assert "regime" not in inspect.signature(InputEvidenceDetector.update).parameters
