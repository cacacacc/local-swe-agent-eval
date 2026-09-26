"""验证结构化测试计划的消费、清理和命令边界。"""

import json
from pathlib import Path

from agent.test_plan import TEST_PLAN_FILENAME, consume_test_plan


def write_plan(repository: Path, value: object) -> Path:
    """在临时仓库写入单次计划，保持测试准备步骤清晰。"""

    path = repository / TEST_PLAN_FILENAME
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_consume_accepts_argv_and_removes_control_file(tmp_path: Path) -> None:
    """合法 argv 应原样保留，同时控制文件不得进入最终 patch。"""

    path = write_plan(
        tmp_path,
        {"argv": ["python", "-m", "pytest", "tests/test_one.py"]},
    )

    request = consume_test_plan(tmp_path)

    assert request.accepted is True
    assert request.argv == ("python", "-m", "pytest", "tests/test_one.py")
    assert not path.exists()


def test_consume_rejects_shell_and_unknown_fields(tmp_path: Path) -> None:
    """计划不能把 shell 命令字符串或未识别控制字段交给调度器。"""

    path = write_plan(
        tmp_path,
        {"argv": ["bash", "-lc", "pytest"], "network": True},
    )

    request = consume_test_plan(tmp_path)

    assert request.status == "rejected"
    assert "only the argv" in request.error
    assert not path.exists()


def test_consume_reports_missing_without_creating_file(tmp_path: Path) -> None:
    """未提交计划应与无效计划区分，便于报告真实遵循率。"""

    request = consume_test_plan(tmp_path)

    assert request.status == "missing"
    assert request.requested is False
    assert list(tmp_path.iterdir()) == []


def test_consume_rejects_shell_executable_even_with_exact_schema(tmp_path: Path) -> None:
    """argv 虽无 shell 插值，仍不得把 shell 本身作为测试入口。"""

    write_plan(tmp_path, {"argv": ["bash", "-lc", "pytest"]})

    request = consume_test_plan(tmp_path)

    assert request.status == "rejected"
    assert request.error == "test executable is forbidden: bash"
