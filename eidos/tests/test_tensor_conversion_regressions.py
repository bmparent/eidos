import numpy as np
import pytest

from eidos_forecast import ForecastEngine, TrajectoryRecord
from eidos_incident_cards import EpisodeIndex, EpisodeRecord
from eidos_procedural_memory import ProceduralMemory
from eidos_tensor_utils import to_cpu_numpy, to_cpu_numpy_1d


def _tensor(values):
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(values, dtype=torch.float32, device=device, requires_grad=True)


@pytest.mark.regression
def test_tensor_conversion_accepts_python_lists():
    arr = to_cpu_numpy([1, 2, 3], dtype=np.float32)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, np.array([1.0, 2.0, 3.0], dtype=np.float32))


@pytest.mark.regression
def test_tensor_conversion_accepts_numpy_arrays_without_mutating_source():
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    arr = to_cpu_numpy(source, dtype=np.float32, flatten=True)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert source.shape == (2, 2)
    assert source.dtype == np.float64


@pytest.mark.regression
def test_tensor_conversion_accepts_cpu_torch_tensors_when_torch_is_installed():
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32, device="cpu", requires_grad=True)

    arr = to_cpu_numpy_1d(tensor)

    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, np.array([1.5, 2.5, 3.5]))


@pytest.mark.regression
def test_tensor_conversion_handles_cuda_when_available_and_cpu_otherwise():
    arr = to_cpu_numpy_1d(_tensor([1.0, 2.0, 3.0]))

    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, np.array([1.0, 2.0, 3.0]))


@pytest.mark.regression
def test_tensor_conversion_cuda_path_skips_cleanly_when_unavailable():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")

    cuda_tensor = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device="cuda")
    arr = to_cpu_numpy_1d(cuda_tensor)

    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, np.array([4.0, 5.0, 6.0]))


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
