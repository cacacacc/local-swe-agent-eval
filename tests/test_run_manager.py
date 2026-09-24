"""验证通用运行产物对超时和补丁规模的结构化分类。"""

import json
from pathlib import Path
import subprocess

from benchmark.task import SWEbenchTask
from tracking.run_manager import RunManager


def run_git(path: Path, *arguments: str) -> str:
    """在临时仓库执行 Git，用于构造 Agent 已提交变更的真实场景。"""

    result = subprocess.run(
        ["git", "-C", str(path), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_collect_patch_includes_agent_commits_and_untracked_files(tmp_path: Path) -> None:
    """Agent 自行 commit 后仍须导出相对 base_commit 的完整 patch，防止有效修复丢失。"""

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


def test_finalize_classifies_timeout_and_counts_changed_lines(tmp_path: Path) -> None:
    """退出码 124 必须归类为 timeout，且补丁规模不能统计 diff header。"""

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
