import pytest

class TestSentinel:
    @pytest.fixture
    def sentinel(self, brain_module):
        return brain_module.SentinelMonitor(window=10)

    def test_initial_state(self, sentinel):
        assert sentinel.analyze() == "CALIBRATING"

    def test_nominal_transition(self, sentinel):
        for _ in range(15):
             sentinel.update(
                ratio=5.0, plasticity=0.01, eigen_dominance=0.1, 
                surprise_score=0.5, error_norm=0.1, error_rms=0.01
            )
        assert sentinel.analyze() == "NOMINAL"
        
    def test_red_state(self, sentinel):
        # Fill buffer (exit calibration)
        for _ in range(40):
             sentinel.update(ratio=5.0, plasticity=0.01, eigen_dominance=0.1)
        
        # Inject chaos (Plasticity Fever) - Fill the window
        for _ in range(60):
            sentinel.update(
                ratio=0.5, plasticity=200.0, eigen_dominance=0.9,
                surprise_score=5.0
            )
        status = sentinel.analyze()
        assert "RED" in status or "AMBER" in status
