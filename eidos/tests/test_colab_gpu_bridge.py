import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import colab_gpu_bridge


def test_proof_baseline_bridge_command_uses_existing_runner():
    args = colab_gpu_bridge.parse_args(
        [
            "--mode",
            "proof-baseline",
            "--suite",
            "smoke",
            "--seed",
            "42",
            "--frames",
            "10000",
            "--out",
            "artifacts/proof_runs/2026-06-23/colab_gpu_10k",
            "--dry-run",
            "--skip-drive-copy",
        ]
    )

    cmd = colab_gpu_bridge.build_child_command(args)

    assert cmd[1:] == [
        "tools/run_proof_baseline.py",
        "--suite",
        "smoke",
        "--seed",
        "42",
        "--frames",
        "10000",
        "--out",
        "artifacts/proof_runs/2026-06-23/colab_gpu_10k",
    ]
    assert "PYTHONPATH" not in " ".join(cmd)


def test_labeled_domain_requires_explicit_label_policy():
    args = colab_gpu_bridge.parse_args(
        [
            "--mode",
            "labeled-domain",
            "--suite",
            "full",
            "--seed",
            "42",
            "--frames",
            "100",
            "--out",
            "artifacts/proof_runs/2026-06-23/labeled",
        ]
    )

    with pytest.raises(ValueError, match="requires --attack-labels"):
        colab_gpu_bridge.build_child_command(args)


def test_bridge_environment_skips_heavy_pip_freeze():
    text = colab_gpu_bridge.collect_bridge_environment(
        {
            "device": {
                "torch_available": True,
                "torch_version": "test-torch",
                "cuda_available": False,
                "device_count": 0,
            }
        }
    )

    assert "torch_version: test-torch" in text
    assert "pip_freeze: skipped for bridge receipt" in text


def test_dry_run_writes_bridge_receipts(tmp_path):
    out_dir = tmp_path / "bridge_dry_run"
    result = subprocess.run(
        [
            sys.executable,
            "tools/colab_gpu_bridge.py",
            "--mode",
            "proof-baseline",
            "--suite",
            "smoke",
            "--seed",
            "42",
            "--frames",
            "1",
            "--out",
            str(out_dir),
            "--dry-run",
            "--skip-drive-copy",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((out_dir / "colab_gpu_bridge_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "dry_run"
    assert receipt["core_behavior_changed"] is False
    assert "tools/run_proof_baseline.py" in receipt["command"]
    assert (out_dir / "colab_gpu_bridge_receipt.md").exists()
    assert (out_dir / "colab_gpu_bridge_environment.txt").exists()
    assert (out_dir / "colab_gpu_bridge_git_commit.txt").exists()
    assert not (out_dir / "drive_manifest.json").exists()


def test_notebook_template_has_gpu_and_drive_guardrails():
    notebook = json.loads(
        (REPO_ROOT / "notebooks" / "eidos_colab_gpu_bridge.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )

    assert "tools/colab_gpu_bridge.py" in source
    assert "--require-cuda" in source
    assert "--mount-drive" in source
    assert "MODE = \"proof-baseline\"" in source
    assert "core Eidos behavior unchanged" in source
