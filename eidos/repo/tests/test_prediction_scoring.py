from eidos_brain.prediction.scoring import brier, finite_or_zero

def test_scoring_finite():
    assert brier(0.7,1)>=0
    assert finite_or_zero(float('inf'))==0.0
