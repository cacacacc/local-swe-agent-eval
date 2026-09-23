from pathlib import Path
import subprocess

import pytest

from benchmark.repo_manager import RepositoryError, RepositoryManager
from benchmark.task import SWEbenchTask


def run_git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def make_local_cache(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    run_git(source, "init")
    run_git(source, "config", "user.name", "Phase One Test")
    run_git(source, "config", "user.email", "phase-one@example.invalid")
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    run_git(source, "add", "module.py")
    run_git(source, "commit", "-m", "initial")
    commit = run_git(source, "rev-parse", "HEAD")

    cache_root = tmp_path / "cache"
    cache_path = cache_root / "example__project"
    cache_root.mkdir()
    subprocess.run(
        ["git", "clone", "--no-checkout", str(source), str(cache_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    run_git(cache_path, "remote", "set-url", "origin", "https://github.com/example/project.git")
    return cache_root, commit


def make_task(commit: str) -> SWEbenchTask:
    return SWEbenchTask(
        instance_id="example__project-123",
        repo="example/project",
        base_commit=commit,
        problem_statement="Fix the example.",
    )


def test_prepare_creates_verified_detached_worktree_offline(tmp_path) -> None:
    cache_root, commit = make_local_cache(tmp_path)
    workspace_root = tmp_path / "workspaces"
    manager = RepositoryManager(cache_root, workspace_root)

    prepared = manager.prepare(make_task(commit), allow_network=False)

    assert prepared.resolved_commit == commit
    assert run_git(prepared.path, "rev-parse", "HEAD") == commit
    symbolic_ref = subprocess.run(
        ["git", "-C", str(prepared.path), "symbolic-ref", "-q", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert symbolic_ref.returncode == 1
    assert (prepared.path / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_prepare_refuses_to_overwrite_existing_workspace(tmp_path) -> None:
    cache_root, commit = make_local_cache(tmp_path)
    workspace_root = tmp_path / "workspaces"
    manager = RepositoryManager(cache_root, workspace_root)
    task = make_task(commit)
    manager.prepare(task, allow_network=False)

    with pytest.raises(RepositoryError, match="refusing to overwrite"):
        manager.prepare(task, allow_network=False)


def test_offline_prepare_requires_cached_repository(tmp_path) -> None:
    manager = RepositoryManager(tmp_path / "cache", tmp_path / "workspaces")

    with pytest.raises(RepositoryError, match="network is disabled"):
        manager.prepare(make_task("0123456789abcdef"), allow_network=False)
