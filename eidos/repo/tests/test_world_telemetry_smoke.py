import subprocess, sys

def test_world_smoke(tmp_path):
    out=tmp_path/'world'; subprocess.check_call([sys.executable,'-m','eidos_brain.prediction.run_world_telemetry','--fixture','--out',str(out)])
    assert any(out.iterdir())
