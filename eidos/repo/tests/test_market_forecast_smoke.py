import subprocess, sys

def test_market_smoke(tmp_path):
    out=tmp_path/'market'; subprocess.check_call([sys.executable,'-m','eidos_brain.prediction.run_market_forecast','--fixture','--out',str(out)])
    assert any(out.iterdir())
