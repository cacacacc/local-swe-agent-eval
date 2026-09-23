"""Prepare reproducible, isolated Git worktrees for SWE-bench tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Sequence

from .task import SWEbenchTask


class RepositoryError(RuntimeError):
    """Raised when a repository cannot be prepared reproducibly."""


_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class PreparedRepository:
    """A task worktree verified to be at the requested base commit."""

    task: SWEbenchTask
    path: Path
    resolved_commit: str


class RepositoryManager:
    """Manage one shared clone per repository and one worktree per task."""

    def __init__(
        self,
        cache_root: Path | str,
        workspace_root: Path | str,
        *,
        command_timeout_seconds: int = 600,
    ) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        self.command_timeout_seconds = command_timeout_seconds

    def prepare(
        self,
        task: SWEbenchTask,
        *,
        allow_network: bool,
    ) -> PreparedRepository:
        """Prepare a clean worktree without overwriting an existing task run."""

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        cache_path = self._cache_path(task.repo)
        task_path = self._task_path(task.instance_id)

        if task_path.exists():
            raise RepositoryError(
                f"task workspace already exists; refusing to overwrite it: {task_path}"
            )

        if not cache_path.exists():
            if not allow_network:
                raise RepositoryError(
                    f"repository is not cached and network is disabled: {task.repo}"
                )
            self._clone(task.repo, cache_path)
        else:
            self._verify_cache(cache_path, task.repo)

        if not self._commit_exists(cache_path, task.base_commit):
            if not allow_network:
                raise RepositoryError(
                    f"base commit {task.base_commit} is absent from the offline cache"
                )
            self._run_git(cache_path, "fetch", "--all", "--tags", "--prune")

        if not self._commit_exists(cache_path, task.base_commit):
            raise RepositoryError(
                f"base commit {task.base_commit} does not exist in {task.repo}"
            )

        self._run_git(
            cache_path,
            "worktree",
            "add",
            "--detach",
            str(task_path),
            task.base_commit,
        )
        resolved_commit = self._run_git(task_path, "rev-parse", "HEAD").strip()
        expected_commit = self._run_git(
            cache_path, "rev-parse", f"{task.base_commit}^{{commit}}"
        ).strip()
        if resolved_commit != expected_commit:
            raise RepositoryError(
                f"checkout verification failed: expected {expected_commit}, "
                f"got {resolved_commit}"
            )

        return PreparedRepository(task, task_path, resolved_commit)

    def _clone(self, repo: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{repo}.git"
        self._run(
            ["git", "clone", "--no-checkout", url, str(destination)],
            cwd=destination.parent,
        )

    def _verify_cache(self, cache_path: Path, repo: str) -> None:
        if not (cache_path / ".git").is_dir():
            raise RepositoryError(f"cache path is not a Git clone: {cache_path}")
        actual_url = self._run_git(
            cache_path, "remote", "get-url", "origin"
        ).strip().removesuffix("/")
        accepted_urls = {
            f"https://github.com/{repo}.git",
            f"https://github.com/{repo}",
            f"git@github.com:{repo}.git",
        }
        if actual_url not in accepted_urls:
            raise RepositoryError(
                f"cached origin mismatch for {repo}: {actual_url}"
            )

    def _commit_exists(self, cache_path: Path, commit: str) -> bool:
        result = subprocess.run(
            ["git", "-C", str(cache_path), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.command_timeout_seconds,
            check=False,
        )
        return result.returncode == 0

    def _cache_path(self, repo: str) -> Path:
        return self.cache_root / repo.replace("/", "__")

    def _task_path(self, instance_id: str) -> Path:
        safe_name = _SAFE_PATH_COMPONENT.sub("_", instance_id)
        return self.workspace_root / safe_name

    def _run_git(self, repository: Path, *arguments: str) -> str:
        return self._run(
            ["git", "-C", str(repository), *arguments],
            cwd=repository,
        )

    def _run(self, command: Sequence[str], *, cwd: Path) -> str:
        try:
            result = subprocess.run(
                list(command),
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.command_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RepositoryError(
                f"command could not run: {' '.join(command)}: {error}"
            ) from error

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            raise RepositoryError(
                f"command failed ({result.returncode}): {' '.join(command)}\n{detail}"
            )
        return result.stdout

