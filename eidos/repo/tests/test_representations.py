import copy

import numpy as np

from eidos_brain.proof.representations import (
    GaussianProjector,
    RepresentationPipeline,
    common_phase_coherence,
)


def live_record(frame_id: int, frame: np.ndarray) -> dict:
    return {
        "frame_id": frame_id,
        "frame": frame.tolist(),
        "raw_residual": (frame * 0.2).tolist(),
        "sentinel_metrics": {
            "eigen_dominance": 0.2,
            "state_entropy": 0.8,
            "state_flatness": 0.7,
            "spectral_entropy": 0.6,
            "spectral_flatness": 0.5,
        },
        "hdc_metrics": {"similarity": 0.3, "familiarity": 0.4, "write": False},
        "thermodynamic_metrics": {"rho": 1.0, "temperature": 0.1, "energy": 0.2},
    }


def test_phase_coherence_is_common_rotation_invariant():
    phases = np.asarray([0.1, 0.7, 1.4, 2.2])
    before = common_phase_coherence(phases)
    after = common_phase_coherence(phases + 1.234)
    assert np.isclose(before, after, atol=1e-12)


def test_all_lifts_are_causal_under_future_mutation():
    rng = np.random.default_rng(4)
    frames = rng.normal(size=(20, 8))
    mutated = frames.copy()
    mutated[12:] += 1000.0
    left = RepresentationPipeline(spectral_window=16, min_calibration=2)
    right = RepresentationPipeline(spectral_window=16, min_calibration=2)
    for index in range(12):
        a = left.observe(live_record(index, frames[index]))
        b = right.observe(live_record(index, mutated[index]))
        assert a == b


def test_projection_uses_documented_jl_scaling_in_expectation():
    vector = np.linspace(-1.0, 1.0, 128)
    ratios = []
    for seed in range(64):
        projected = GaussianProjector(128, 64, seed=seed).transform(vector)
        ratios.append(np.linalg.norm(projected) ** 2 / np.linalg.norm(vector) ** 2)
    assert 0.8 < float(np.mean(ratios)) < 1.2


def test_lift_calibration_count_is_past_only():
    pipeline = RepresentationPipeline(spectral_window=8, min_calibration=2)
    first = pipeline.observe(live_record(0, np.ones(8)))
    second = pipeline.observe(live_record(1, np.ones(8) * 2))
    assert first["lifts"]["raw"]["calibration_count"] == 0
    assert second["lifts"]["raw"]["calibration_count"] == 1

