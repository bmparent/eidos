import pytest

from eidos_agents.guardrails import GuardrailBlock
from eidos_agents.repo_tools import RepositoryTools


def test_safe_command_rejects_destructive_git(repo_root):
    tools = RepositoryTools(repo_root)
    with pytest.raises(GuardrailBlock):
        tools.command_run_safe(["git", "reset", "--hard"], task_id="T", actor="forge")


def test_path_escape_rejected(repo_root):
    with pytest.raises(GuardrailBlock):
        RepositoryTools(repo_root).repo_read_file("../outside", actor="archivist")


def test_repo_state_readable(repo_root):
    state = RepositoryTools(repo_root).repo_status("archivist")
    assert "branch_line" in state
