from pathlib import Path

import pytest

from eidos_agents.config import LabConfig


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def lab_config(tmp_path: Path, repo_root: Path) -> LabConfig:
    config = LabConfig.from_env(repo_root, dry_run=True)
    config.artifact_root = tmp_path / "artifacts" / "agent_lab"
    config.pricing_path = repo_root / "config" / "agent_lab" / "model_pricing.yaml"
    return config
