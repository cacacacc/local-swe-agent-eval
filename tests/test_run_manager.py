"""Verify the structured classification of generic run artifacts by timeout and patch size."""

import json
from pathlib import Path
import subprocess

from benchmark.task import SWEbenchTask
from tracking.run_manager import RunManager


def run_git(path: Path, *arguments: str) -> str:
    """Run Git in a temporary repo to build a realistic scenario where the Agent has committed changes."""

    result = subprocess.run(
        ["git", "-C", str(path), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_collect_patch_includes_agent_commits_and_untracked_files(tmp_path: Path) -> None:
    """After the Agent commits on its own, the full patch relative to base_commit must still be exported so valid fixes are not lost."""

    repository = tmp_path / "repository"
    repository.mkdir()
    run_git(repository, "init")
    run_git(repository, "config", "user.name", "Run Manager Test")
    run_git(repository, "config", "user.email", "run-manager@example.invalid")
    tracked = repository / "module.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    run_git(repository, "add", "module.py")
    run_git(repository, "commit", "-m", "base")
    base_commit = run_git(repository, "rev-parse", "HEAD")

    task = SWEbenchTask(
        instance_id="owner__repo-committed",
        repo="owner/repo",
        base_commit=base_commit,
        problem_statement="Fix it.",
    )
    session = RunManager(tmp_path / "runs").start(
        task,
        phase="dev",
        agent="claude-code",
        model="qwen3.5:9b",
        prompt="prompt\n",
    )

    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    run_git(repository, "add", "module.py")
    run_git(repository, "commit", "-m", "agent fix")
    (repository / "new_test.py").write_text("assert True\n", encoding="utf-8")

    patch = session.collect_patch(repository)

    assert "-VALUE = 1" in patch
    assert "+VALUE = 2" in patch
    assert "new_test.py" in patch
    assert "+assert True" in patch


def test_collect_patch_excludes_generated_environments_but_keeps_source(
    tmp_path: Path,
) -> None:
    """In-task virtualenvs and caches must not pollute the prediction, while real source changes must still be kept."""

    repository = tmp_path / "repository"
    repository.mkdir()
    run_git(repository, "init")
    run_git(repository, "config", "user.name", "Run Manager Test")
    run_git(repository, "config", "user.email", "run-manager@example.invalid")
    source = repository / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    run_git(repository, "add", "module.py")
    run_git(repository, "commit", "-m", "base")
    base_commit = run_git(repository, "rev-parse", "HEAD")

    task = SWEbenchTask(
        instance_id="owner__repo-artifacts",
        repo="owner/repo",
        base_commit=base_commit,
        problem_statement="Fix it.",
    )
    session = RunManager(tmp_path / "runs").start(
        task,
        phase="dev",
        agent="claude-code",
        model="qwen3.5:9b",
        prompt="prompt\n",
    )

    source.write_text("VALUE = 2\n", encoding="utf-8")
    generated = repository / "test_env" / "lib" / "site-packages"
    generated.mkdir(parents=True)
    (generated / "dependency.py").write_text("GENERATED = True\n", encoding="utf-8")
    cache = repository / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"compiled")

    patch = session.collect_patch(repository)

    assert "module.py" in patch
    assert "+VALUE = 2" in patch
    assert "test_env" not in patch
    assert "__pycache__" not in patch


def test_finalize_classifies_timeout_and_counts_changed_lines(tmp_path: Path) -> None:
    """Exit code 124 must be classified as a timeout, and the patch size must not count the diff header."""

    task = SWEbenchTask(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit="0123456789abcdef0123456789abcdef01234567",
        problem_statement="Fix it.",
    )
    session = RunManager(tmp_path / "runs").start(
        task,
        phase="dev",
        agent="claude-code",
        model="qwen3.5:9b",
        prompt="prompt\n",
    )
    session.finalize(
        exit_code=124,
        agent_log="",
        test_output="",
        events=[],
        patch="--- a/file.py\n+++ b/file.py\n-old\n+new\n",
        metrics={"timed_out": True},
    )

    result = json.loads((session.path / "result.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (session.path / "metadata.json").read_text(encoding="utf-8")
    )

    assert result["run_status"] == "timeout"
    assert result["patch_line_count"] == 2
    assert metadata["status"] == "timeout"
    assert metadata["metrics"]["timed_out"] is True
