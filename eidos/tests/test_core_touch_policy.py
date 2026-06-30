import subprocess
from pathlib import Path

from tools import check_core_touch_policy


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, text=True, capture_output=True)


def init_repo(repo: Path) -> None:
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    git(repo, "add", "README.md")
    git(repo, "commit", "-m", "baseline")
    git(repo, "switch", "-c", "work")


def test_core_touch_policy_allows_calibration_reports_tests_and_docs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    for path, text in {
        "sentinel/calibration.py": "MODE_CONFIGS = {'balanced': {}}\n",
        "tools/run_labeled_domain_proof.py": "def proof_runner():\n    return 'calibration report'\n",
        "tests/test_gate.py": "def test_gate():\n    assert True\n",
        "docs/proof_runs/2026-06-10/codex_journal.md": "# journal\n",
    }.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    report = check_core_touch_policy.evaluate("main", cwd=repo)

    assert report["passed"] is True
    assert report["failures"] == []


def test_core_touch_policy_allows_colab_bridge_boundary_text(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    target = repo / "tools/colab_gpu_bridge.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "BOUNDARY = 'No reservoir, RLS, Sentinel threshold, anomaly policy, or compression behavior is changed.'\n",
        encoding="utf-8",
    )

    report = check_core_touch_policy.evaluate("main", cwd=repo)

    assert report["passed"] is True
    assert report["failures"] == []


def test_core_touch_policy_fails_for_reservoir_core_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    target = repo / "repo/src/eidos_brain/engine/eidos_v0_4_7_02.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def update_reservoir_state():\n    return 'changed'\n", encoding="utf-8")

    report = check_core_touch_policy.evaluate("main", cwd=repo)

    assert report["passed"] is False
    assert report["failures"][0]["path"] == "repo/src/eidos_brain/engine/eidos_v0_4_7_02.py"
    assert "forbidden core path" in report["failures"][0]["reason"]
