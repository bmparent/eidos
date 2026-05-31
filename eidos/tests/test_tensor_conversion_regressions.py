import numpy as np
import pytest

from eidos_forecast import ForecastEngine, TrajectoryRecord
from eidos_incident_cards import EpisodeIndex, EpisodeRecord
from eidos_procedural_memory import ProceduralMemory
from eidos_tensor_utils import to_cpu_numpy_1d


def _tensor(values):
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(values, dtype=torch.float32, device=device, requires_grad=True)


@pytest.mark.regression
def test_tensor_conversion_handles_cuda_when_available_and_cpu_otherwise():
    arr = to_cpu_numpy_1d(_tensor([1.0, 2.0, 3.0]))

    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, np.array([1.0, 2.0, 3.0]))


@pytest.mark.regression
def test_incident_episode_index_accepts_tensor_signatures():
    index = EpisodeIndex(maxlen=5)
    index.add(
        EpisodeRecord(
            step=7,
            ts=1.0,
            regime="RED",
            z=3.0,
            err=0.25,
            signature=_tensor([1.0, 0.0, 0.0]),
            entities={},
            exemplars=[],
            top_drivers=[],
        )
    )

    results = index.topk(_tensor([1.0, 0.0, 0.0]), regime="RED", k=1)

    assert results[0]["step"] == 7
    assert results[0]["sim"] == pytest.approx(1.0)


@pytest.mark.regression
def test_procedural_memory_accepts_tensor_signatures_for_prototypes_and_ranking():
    memory = ProceduralMemory(domain="generic", enabled=True)
    memory.update_prototype("ALERT_HUMAN", _tensor([0.25, 0.5, 1.0]))

    scores = memory.rank_actions(_tensor([0.25, 0.5, 1.0]), regime="AMBER")

    assert isinstance(memory.proto["ALERT_HUMAN"], np.ndarray)
    assert scores[0]["action"] == "ALERT_HUMAN"
    assert scores[0]["sim"] == pytest.approx(1.0)


@pytest.mark.regression
def test_forecast_similarity_accepts_tensor_signatures():
    forecast = ForecastEngine(window=3, horizons=[10], temp=1.0, enabled=True)
    forecast.trajectories = [
        TrajectoryRecord(
            domain="generic",
            outcome="RED",
            horizon=10,
            sig_seq=[[1.0, 0.0]],
            z_seq=[1.0],
            err_seq=[0.1],
        )
    ]

    forecast.update(_tensor([1.0, 0.0]), z=2.0, err=0.2, regime="AMBER", domain="generic")
    risk = forecast.risk(domain="generic", regime="AMBER")

    assert risk["likely_mode"] == "RED"
    assert risk["evidence"][0]["sim"] == pytest.approx(1.0)
