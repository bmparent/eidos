import pytest
import numpy as np

class TestSentinel:
    @pytest.fixture
    def sentinel(self, brain_module):
        return brain_module.SentinelMonitor(window=10)

    def test_initial_state(self, sentinel):
        assert sentinel.analyze() == "CALIBRATING"

    def test_nominal_transition(self, sentinel):
        # Fill window (10 frames) then a few more to stabilize
        for _ in range(15):
             sentinel.update(
                ratio=5.0, plasticity=0.01, eigen_dominance=0.1, 
                surprise_score=0.5, error_norm=0.1, error_rms=0.01
            )
        status = sentinel.analyze()
        # SentinelMonitor returns "GREEN: NOMINAL" for healthy state
        assert status.startswith("GREEN"), f"Expected GREEN state, got: {status}"
        
    def test_amber_state(self, sentinel):
        """High plasticity should trigger AMBER."""
        # Fill buffer to exit calibration
        for _ in range(15):
             sentinel.update(ratio=5.0, plasticity=0.01, eigen_dominance=0.1)
        
        # Inject high plasticity to fill the window
        for _ in range(15):
            sentinel.update(
                ratio=5.0, plasticity=200.0, eigen_dominance=0.1,
                surprise_score=5.0
            )
        status = sentinel.analyze()
        assert "AMBER" in status or "RED" in status, f"Expected AMBER/RED, got: {status}"

    def test_red_representation_collapse(self, sentinel):
        """High ratio + high eigen dominance + low plasticity = RED collapse."""
        # Fill buffer to exit calibration with extreme collapse metrics
        for _ in range(15):
            sentinel.update(
                ratio=200.0, plasticity=1.0, eigen_dominance=0.99,
                state_entropy=0.8  # Not collapsed entropy, so geo_ok=True
            )
        status = sentinel.analyze()
        assert "RED" in status, f"Expected RED, got: {status}"
