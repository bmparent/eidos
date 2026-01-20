import pytest
import numpy as np

@pytest.mark.regression
def test_hippocampus_none_crash(brain_module, temp_artifact_root):
    """
    Simulates the condition where is_surprise=False and 
    hippocampus_compute_on_surprise_only=True.
    Previous behavior: Crashed with TypeError.
    New behavior: Should run smooth.
    """
    brain_module.EIDOS_BRAIN_CONFIG["hippocampus_compute_on_surprise_only"] = True
    
    # Generate 10 very boring frames -> is_surprise=False
    def gen():
        for i in range(10):
            yield np.zeros(64, dtype=np.float32), {}
            
    # Should not raise exception
    try:
        brain_module.run_sentinel_stream(
            gen_factory=gen,
            est_frames=10,
            features=64,
            profile_label="reg_test",
            session_label="reg_sess",
            warmup=2,
            save_surprise_artifacts=False
        )
    except TypeError as e:
        pytest.fail(f"Regression Triggered: {e}")

@pytest.mark.regression
def test_warmup_boundary(brain_module, temp_artifact_root):
    """Ensure no crash exactly at warmup boundary."""
    # Run 50 frames, warmup 20
    def gen():
        for i in range(50):
            yield np.random.randn(64).astype(np.float32), {}
            
    brain_module.run_sentinel_stream(
        gen_factory=gen, est_frames=50, features=64,
        profile_label="warmup_test", session_label="warmup_sess",
        warmup=20, sample_geometry=False
    )
