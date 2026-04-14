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
        
    def test_forward_pass(self, reservoir, brain_module):
        """Listening to input should change reservoir state."""
        u = torch.randn(64, device=brain_module.device)
        state_old = reservoir.state.clone()
        reservoir.listen(u)
        assert not torch.equal(state_old, reservoir.state)
        
    def test_adapt_learning(self, reservoir, brain_module):
        """Adapt should update readout weights W_out."""
        u = torch.randn(64, device=brain_module.device)
        reservoir.listen(u)
        w_before = reservoir.W_out.clone()
        target = torch.randn(64, device=brain_module.device)
        reservoir.adapt(target)
        assert not torch.equal(w_before, reservoir.W_out)

    def test_synaptic_hash(self, reservoir):
        """Hash should be deterministic for same weights."""
        h1 = reservoir.get_synaptic_hash()
        h2 = reservoir.get_synaptic_hash()
        assert h1 == h2
        assert len(h1) == 16  # SHA256[:16]

    def test_thermodynamics(self, reservoir, brain_module):
        brain_module.EIDOS_BRAIN_CONFIG["thermo_enabled"] = True
        reservoir.thermo_enabled = True
        metrics = {
            "surprise_score": 2.0,
            "error_rms": 0.5,
            "dominance": 0.5,
            "state_entropy": 0.8,
        }
        stats = reservoir.update_thermodynamics(metrics)
        assert "thermo_energy" in stats
        assert "thermo_rho" in stats
        assert "thermo_temp" in stats
        assert "thermo_lambda" in stats
