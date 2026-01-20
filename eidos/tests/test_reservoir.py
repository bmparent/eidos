import torch
import pytest

class TestRLSReservoir:
    @pytest.fixture
    def reservoir(self, brain_module):
        return brain_module.RLS_Reservoir(
            n_inputs=64,
            n_reservoir=100,
            spectral_radius=1.2,
            leak_rate=0.1
        )

    def test_initialization(self, reservoir):
        assert reservoir.state.shape[0] == 100
        assert reservoir.W_in.shape == (100, 64)
        
    def test_forward_pass(self, reservoir):
        u = torch.randn(64, device=reservoir.device)
        state_old = reservoir.state.clone()
        reservoir.listen(u)
        assert not torch.equal(state_old, reservoir.state)
        
    def test_thermodynamics(self, reservoir):
        metrics = {
            "surprise_score": 2.0,
            "plasticity": 1.0, 
            "eigen_dominance": 0.5,
            "spectral_entropy": 0.8
        }
        stats = reservoir.update_thermodynamics(metrics)
        assert "thermo_energy" in stats
