"""验证任务专属可见测试入口的顺序检测与审计持久化。"""

import json
from pathlib import Path
import subprocess

from scripts.run_visible_tests import (
    _append_audit,
    _repository_changed,
    _test_command_error,
)


def _git(repository: Path, *arguments: str) -> str:
    """执行测试仓库中的确定性 Git 命令并返回标准输出。"""

    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return completed.stdout.strip()


def test_repository_change_detection_covers_clean_dirty_and_committed_edits(
    tmp_path: Path,
) -> None:
    """首次修改前必须为 clean；工作区修改和模型自行 commit 都应被识别。"""

    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    source = repository / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    _git(repository, "add", "module.py")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    base_commit = _git(repository, "rev-parse", "HEAD")

    assert _repository_changed(repository, base_commit) is False
    source.write_text("value = 2\n", encoding="utf-8")
    assert _repository_changed(repository, base_commit) is True
    _git(repository, "add", "module.py")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "agent edit",
    )
    assert _repository_changed(repository, base_commit) is True


def test_audit_append_preserves_every_helper_attempt(tmp_path: Path) -> None:
    """重复调用必须全部保留，才能判定“恰好一次”的阶段协议。"""

    audit_path = tmp_path / "audit.json"
    _append_audit(audit_path, {"command_started": False, "exit_code": 2})
    _append_audit(audit_path, {"command_started": True, "exit_code": 1})

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["attempts"] == [
        {"command_started": False, "exit_code": 2},
        {"command_started": True, "exit_code": 1},
    ]


def test_test_command_validation_rejects_bare_file_but_accepts_real_runners() -> None:
    """防止再次把测试源码当可执行文件，同时兼容 pytest 与 Django runner。"""

    assert _test_command_error([]) == "missing test command argv"
    assert "bare Python test path" in (
        _test_command_error(["tests/test_module.py::test_case"]) or ""
    )
    assert _test_command_error(
        ["python", "-m", "pytest", "tests/test_module.py::test_case"]
    ) is None
    assert _test_command_error(["./tests/runtests.py", "forms_tests"]) is None
