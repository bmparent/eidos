import subprocess
import sys
from pathlib import Path


def test_colab_gpu_hotfix_script_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/verify_colab_gpu_hotfix.py"],
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "VALIDATION_OK" in result.stdout
    assert "cuda_available=" in result.stdout
