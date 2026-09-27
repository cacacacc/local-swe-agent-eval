"""验证结构化测试计划的消费、清理和命令边界。"""

import json
from pathlib import Path

from agent.test_plan import (
    TEST_PLAN_FILENAME,
    consume_test_plan,
    generate_repository_test_plan,
)


def write_plan(repository: Path, value: object) -> Path:
    """在临时仓库写入单次计划，保持测试准备步骤清晰。"""

    path = repository / TEST_PLAN_FILENAME
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_consume_accepts_target_and_regression_argv(tmp_path: Path) -> None:
    """两类合法 argv 应原样保留，同时控制文件不得进入最终 patch。"""

    path = write_plan(
        tmp_path,
        {
            "target_argv": [
                "python", "-m", "pytest", "tests/test_one.py::test_bug"
            ],
            "regression_argv": [
                "python", "-m", "pytest", "tests/test_one.py"
            ],
        },
    )

    request = consume_test_plan(tmp_path)

    assert request.accepted is True
    assert request.target_argv[-1] == "tests/test_one.py::test_bug"
    assert request.regression_argv[-1] == "tests/test_one.py"
    assert [label for label, _ in request.commands] == ["target", "regression"]
    assert not path.exists()


def test_consume_rejects_shell_and_unknown_fields(tmp_path: Path) -> None:
    """计划不能把 shell 命令字符串或未识别控制字段交给调度器。"""

    path = write_plan(
        tmp_path,
        {"target_argv": ["bash", "-lc", "pytest"], "network": True},
    )

    request = consume_test_plan(tmp_path)

    assert request.status == "rejected"
    assert "target_argv and regression_argv" in request.error
    assert not path.exists()


def test_consume_reports_missing_without_creating_file(tmp_path: Path) -> None:
    """未提交计划应与无效计划区分，便于报告真实遵循率。"""

    request = consume_test_plan(tmp_path)

    assert request.status == "missing"
    assert request.requested is False
    assert list(tmp_path.iterdir()) == []


def test_consume_rejects_shell_executable_even_with_exact_schema(tmp_path: Path) -> None:
    """argv 虽无 shell 插值，仍不得把 shell 本身作为测试入口。"""

    write_plan(
        tmp_path,
        {
            "target_argv": ["bash", "-lc", "pytest"],
            "regression_argv": ["python", "-m", "pytest", "tests"],
        },
    )

    request = consume_test_plan(tmp_path)

    assert request.status == "rejected"
    assert request.error == "test executable is forbidden: bash"


def test_consume_rejects_duplicate_target_and_regression(tmp_path: Path) -> None:
    """两条相同命令不能冒充目标测试与相邻回归测试。"""

    command = ["python", "-m", "pytest", "tests/test_one.py"]
    write_plan(
        tmp_path,
        {"target_argv": command, "regression_argv": command},
    )

    request = consume_test_plan(tmp_path)

    assert request.status == "rejected"
    assert request.error == "target and regression argv must be different"


def test_parent_generates_pytest_plan_from_modified_source(tmp_path: Path) -> None:
    """父进程应把修改模块映射到现有 pytest 模块及其相邻回归目录。"""

    target = tmp_path / "testing" / "test_reports.py"
    target.parent.mkdir(parents=True)
    target.write_text("def test_report(): pass\n", encoding="utf-8")
    patch = (
        "diff --git a/src/_pytest/reports.py b/src/_pytest/reports.py\n"
        "--- a/src/_pytest/reports.py\n+++ b/src/_pytest/reports.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )

    request = generate_repository_test_plan(
        tmp_path,
        repo="pytest-dev/pytest",
        patch=patch,
    )

    assert request.status == "generated"
    assert request.origin == "parent"
    assert request.target_argv == (
        "python", "-m", "pytest", "-x", "testing/test_reports.py"
    )
    assert request.regression_argv == (
        "python", "-m", "pytest", "testing/test_reports.py"
    )


def test_parent_uses_django_runner_and_dotted_labels(tmp_path: Path) -> None:
    """Django 不能套用通用 pytest 命令，必须生成项目 runner 的 dotted label。"""

    target = tmp_path / "tests" / "forms_tests" / "tests" / "test_forms.py"
    target.parent.mkdir(parents=True)
    target.write_text("class FormsTests: pass\n", encoding="utf-8")
    patch = (
        "diff --git a/django/forms/forms.py b/django/forms/forms.py\n"
        "--- a/django/forms/forms.py\n+++ b/django/forms/forms.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )

    request = generate_repository_test_plan(
        tmp_path,
        repo="django/django",
        patch=patch,
    )

    assert request.target_argv == (
        "python",
        "tests/runtests.py",
        "forms_tests.tests.test_forms",
        "--failfast",
        "--verbosity",
        "0",
    )
    assert request.regression_argv == (
        "python",
        "tests/runtests.py",
        "forms_tests.tests.test_forms",
        "--verbosity",
        "0",
    )


def test_parent_missing_plan_is_diagnostic_for_unmatched_patch(tmp_path: Path) -> None:
    """找不到相邻测试时应记录 missing，而不是伪造命令或阻断验证。"""

    request = generate_repository_test_plan(
        tmp_path,
        repo="owner/project",
        patch=(
            "diff --git a/src/widget.py b/src/widget.py\n"
            "--- a/src/widget.py\n+++ b/src/widget.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
    )

    assert request.status == "missing"
    assert request.origin == "parent"
    assert "no repository-visible test" in request.error


def test_parent_ignores_new_reproduction_when_selecting_tests(tmp_path: Path) -> None:
    """模型新建的复现脚本不能被父进程误认为既有产品源码或可信测试。"""

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_widget.py").write_text("", encoding="utf-8")
    request = generate_repository_test_plan(
        tmp_path,
        repo="owner/project",
        patch=(
            "diff --git a/reproduce_widget.py b/reproduce_widget.py\n"
            "new file mode 100644\n--- /dev/null\n+++ b/reproduce_widget.py\n"
            "@@ -0,0 +1 @@\n+print(1)\n"
        ),
    )

    assert request.status == "missing"
    assert request.error == "parent planner found no modified existing source path"
