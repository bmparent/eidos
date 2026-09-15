"""Small typed repository tool surface with complete command journaling."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .guardrails import (
    ForbiddenScopeGuard,
    GuardrailBlock,
    MainBranchGuard,
    ScopeGuard,
    SecretGuard,
)
from .permissions import Capability, PermissionPolicy
from .persistence import ArtifactStore


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


class RepositoryTools:
    SAFE_PROGRAMS: ClassVar[set[str]] = {"python", "python.exe", "pytest", "pytest.exe", "git"}
    FORBIDDEN_GIT: ClassVar[set[str]] = {
        "push", "merge", "rebase", "reset", "clean", "checkout", "switch", "branch", "worktree", "tag"
    }
    DESTRUCTIVE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?i)(rm\s+-rf|remove-item|del\s+/|format\s+|git\s+reset|git\s+clean|force-with-lease|--force)"
    )

    def __init__(self, repo_root: Path, store: ArtifactStore | None = None) -> None:
        self.repo_root = repo_root.resolve()
        self.store = store
        self.permissions = PermissionPolicy()
        self.secrets = SecretGuard()

    def _run(self, args: Sequence[str], *, task_id: str | None, actor: str, timeout: int = 300) -> CommandResult:
        if not args:
            raise ValueError("empty command")
        program = Path(args[0]).name.lower()
        if program not in self.SAFE_PROGRAMS:
            raise GuardrailBlock(f"program not allowlisted: {program}")
        joined = " ".join(args)
        if self.DESTRUCTIVE.search(joined):
            raise GuardrailBlock("destructive command rejected")
        read_only_branch = len(args) >= 3 and args[1:3] == ["branch", "--show-current"]
        if program == "git" and len(args) > 1 and args[1].lower() in self.FORBIDDEN_GIT and not read_only_branch:
            raise GuardrailBlock(f"git {args[1]} is unavailable through safe command runner")
        completed = subprocess.run(
            list(args), cwd=self.repo_root, text=True, capture_output=True, timeout=timeout, check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = completed.stdout + "\n" + completed.stderr
        digest = hashlib.sha256(output.encode("utf-8", errors="replace")).hexdigest()
        if self.store:
            self.store.record_command(task_id, actor, joined, completed.returncode, digest)
        return CommandResult(list(args), completed.returncode, completed.stdout, completed.stderr)

    def repo_status(self, actor: str = "archivist") -> dict[str, object]:
        self.permissions.require(actor, Capability.READ_REPO)
        result = self._run(["git", "status", "--porcelain=v1", "--branch"], task_id=None, actor=actor)
        lines = result.stdout.splitlines()
        return {"branch_line": lines[0] if lines else "", "dirty": len(lines) > 1, "entries": lines[1:]}

    def repo_current_branch(self) -> str:
        result = self._run(["git", "branch", "--show-current"], task_id=None, actor="director")
        return result.stdout.strip()

    def repo_current_commit(self) -> str:
        return self._run(["git", "rev-parse", "HEAD"], task_id=None, actor="director").stdout.strip()

    def repo_diff(self, actor: str = "auditor") -> str:
        self.permissions.require(actor, Capability.READ_REPO)
        return self._run(["git", "diff", "--no-ext-diff"], task_id=None, actor=actor).stdout

    def repo_search(self, query: str, paths: list[str] | None = None, actor: str = "archivist") -> list[str]:
        self.permissions.require(actor, Capability.READ_REPO)
        matches: list[str] = []
        roots = [self._safe_path(item) for item in (paths or ["."])]
        regex = re.compile(query, re.IGNORECASE)
        for root in roots:
            files = [root] if root.is_file() else root.rglob("*")
            for path in files:
                if not path.is_file() or path.stat().st_size > 2_000_000 or ".git" in path.parts:
                    continue
                try:
                    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if regex.search(line):
                            matches.append(f"{path.relative_to(self.repo_root).as_posix()}:{number}:{line[:300]}")
                            if len(matches) >= 200:
                                return matches
                except OSError:
                    continue
        return matches

    def repo_read_file(self, path: str, *, max_bytes: int = 100_000, actor: str = "archivist") -> str:
        self.permissions.require(actor, Capability.READ_REPO)
        target = self._safe_path(path)
        data = target.read_bytes()
        if len(data) > max_bytes:
            raise GuardrailBlock(f"file exceeds bounded read limit ({len(data)} bytes)")
        text = data.decode("utf-8", errors="replace")
        self.secrets.check_text(text)
        return text

    def repo_list_tree(self, path: str = ".", *, limit: int = 500, actor: str = "archivist") -> list[str]:
        self.permissions.require(actor, Capability.READ_REPO)
        root = self._safe_path(path)
        return [item.relative_to(self.repo_root).as_posix() for item in sorted(root.rglob("*")) if ".git" not in item.parts][:limit]

    def artifact_list(self, path: str = "artifacts", actor: str = "archivist") -> list[str]:
        return self.repo_list_tree(path, actor=actor)

    def artifact_read(self, path: str, actor: str = "archivist") -> str:
        if not path.replace("\\", "/").startswith("artifacts/"):
            raise GuardrailBlock("artifact_read is restricted to artifacts/")
        return self.repo_read_file(path, actor=actor)

    def pytest_run(self, args: list[str], *, task_id: str, actor: str = "bench", timeout: int = 900) -> CommandResult:
        self.permissions.require(actor, Capability.RUN_TESTS)
        before = self._source_snapshot()
        result = self._run([sys.executable, "-m", "pytest", *args], task_id=task_id, actor=actor, timeout=timeout)
        if actor == "bench" and before != self._source_snapshot():
            raise GuardrailBlock("Bench altered tracked source while running tests")
        return result

    def command_run_safe(self, args: list[str], *, task_id: str, actor: str, timeout: int = 300) -> CommandResult:
        self.permissions.require(actor, Capability.SAFE_SHELL)
        return self._run(args, task_id=task_id, actor=actor, timeout=timeout)

    def calculate_file_hash(self, path: str) -> str:
        return hashlib.sha256(self._safe_path(path).read_bytes()).hexdigest()

    def calculate_config_hash(self, config: dict[str, object]) -> str:
        raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def collect_environment(self) -> dict[str, str]:
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

    def scan_for_crashes(self, text: str) -> list[str]:
        patterns = ("traceback", "segmentation fault", "nan", "inf", "fatal", "out of memory")
        return [line[:500] for line in text.splitlines() if any(token in line.lower() for token in patterns)]

    def create_feature_branch(self, name: str, *, actor: str = "forge") -> None:
        self.permissions.require(actor, Capability.COMMIT_FEATURE_BRANCH)
        current = self.repo_current_branch()
        MainBranchGuard().check_write(current)
        if current != name:
            raise GuardrailBlock(f"workspace is on {current}; automatic branch switching is disabled")

    def validate_changes(self, allowed: list[str], forbidden: list[str], *, actor: str = "forge") -> list[str]:
        self.permissions.require(actor, Capability.WRITE_SOURCE)
        status = self._run(["git", "status", "--porcelain=v1"], task_id=None, actor=actor).stdout.splitlines()
        changed = [line[3:].replace("\\", "/") for line in status if len(line) > 3]
        ScopeGuard().check(changed, allowed)
        ForbiddenScopeGuard().check(changed, forbidden)
        return changed

    def commit_allowed_changes(
        self,
        files: list[str],
        message: str,
        *,
        allowed: list[str],
        forbidden: list[str],
        task_id: str,
        actor: str = "forge",
    ) -> str:
        self.permissions.require(actor, Capability.COMMIT_FEATURE_BRANCH)
        MainBranchGuard().check_write(self.repo_current_branch())
        normalized = [self._safe_path(path).relative_to(self.repo_root).as_posix() for path in files]
        ScopeGuard().check(normalized, allowed)
        ForbiddenScopeGuard().check(normalized, forbidden)
        for path in normalized:
            target = self._safe_path(path)
            if target.exists() and target.is_file() and target.stat().st_size <= 2_000_000:
                self.secrets.check_text(target.read_text(encoding="utf-8", errors="replace"))
        add = self._run(["git", "add", "--", *normalized], task_id=task_id, actor=actor)
        if add.exit_code:
            raise RuntimeError(add.stderr or add.stdout)
        commit = self._run(["git", "commit", "-m", message, "--", *normalized], task_id=task_id, actor=actor)
        if commit.exit_code:
            raise RuntimeError(commit.stderr or commit.stdout)
        return self.repo_current_commit()

    def _safe_path(self, path: str) -> Path:
        target = (self.repo_root / path).resolve()
        if target != self.repo_root and self.repo_root not in target.parents:
            raise GuardrailBlock("path escapes repository")
        return target

    def _source_snapshot(self) -> str:
        result = self._run(["git", "status", "--porcelain=v1", "--untracked-files=no"], task_id=None, actor="bench")
        return hashlib.sha256(result.stdout.encode()).hexdigest()
