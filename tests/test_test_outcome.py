"""验证跨仓库测试输出的失败签名解析与可信度边界。"""

from agent.test_outcome import parse_test_outcome
from agent.test_sandbox import VisibleTestResult


def _failed(output: str) -> VisibleTestResult:
    """构造已真实启动、以非零状态结束的测试结果。"""

    return VisibleTestResult(
        exit_code=1,
        output=output,
        image="swebench/example:latest",
        timed_out=False,
        command_started=True,
    )


def test_parse_pytest_failure_signatures() -> None:
    """pytest 的 FAILED/ERROR 摘要必须归一化为稳定测试 ID。"""

    outcome = parse_test_outcome(
        ("python", "-m", "pytest", "tests/test_widget.py"),
        _failed(
            "FAILED tests/test_widget.py::test_value - AssertionError\n"
            "ERROR tests/test_widget.py::test_setup - RuntimeError\n"
        ),
    )

    assert outcome.parser == "pytest"
    assert outcome.reliable is True
    assert outcome.failure_signatures == frozenset(
        {
            "tests/test_widget.py::test_value",
            "tests/test_widget.py::test_setup",
        }
    )


def test_parse_django_unittest_failure_signatures() -> None:
    """Django runtests.py 的 unittest 标题必须保留测试方法和类名。"""

    outcome = parse_test_outcome(
        ("python", "tests/runtests.py", "admin_views"),
        _failed(
            "FAIL: test_permissions_error (admin_views.tests.AdminViewTests)\n"
            "ERROR: test_simple_tag (template_tests.test_custom.CustomTests)\n"
        ),
    )

    assert outcome.parser == "django-unittest"
    assert outcome.reliable is True
    assert outcome.failure_signatures == frozenset(
        {
            "test_permissions_error (admin_views.tests.AdminViewTests)",
            "test_simple_tag (template_tests.test_custom.CustomTests)",
        }
    )


def test_parse_sympy_failure_signature() -> None:
    """SymPy bin/test 的下划线失败标题必须提取文件与测试函数。"""

    outcome = parse_test_outcome(
        ("python", "bin/test", "sympy/core/tests/test_numbers.py"),
        _failed(
            "________________ sympy/core/tests/test_numbers.py:test_mod ________________\n"
        ),
    )

    assert outcome.parser == "sympy-bin-test"
    assert outcome.reliable is True
    assert outcome.failure_signatures == frozenset(
        {"sympy/core/tests/test_numbers.py:test_mod"}
    )


def test_unparsed_nonzero_result_is_not_reliable() -> None:
    """没有测试 ID 的非零退出不得伪装成可比较的既有测试失败。"""

    outcome = parse_test_outcome(
        ("python", "-m", "pytest", "tests/test_widget.py"),
        _failed("collection crashed before a test summary was printed\n"),
    )

    assert outcome.status == "failed"
    assert outcome.failure_signatures == frozenset()
    assert outcome.reliable is False
