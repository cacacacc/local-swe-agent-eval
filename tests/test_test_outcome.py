"""Verify failure-signature parsing and reliability bounds across repository test output."""

from agent.test_outcome import parse_test_outcome
from agent.test_sandbox import VisibleTestResult


def _failed(output: str) -> VisibleTestResult:
    """Build a test result that actually started and ended with a non-zero status."""

    return VisibleTestResult(
        exit_code=1,
        output=output,
        image="swebench/example:latest",
        timed_out=False,
        command_started=True,
    )


def test_parse_pytest_failure_signatures() -> None:
    """pytest's FAILED/ERROR summary must be normalized into a stable test ID."""

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
    """Django runtests.py unittest headings must preserve the test method and class name."""

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


def test_django_signature_removes_process_specific_object_address() -> None:
    """An object-address change in the same subTest must not be misreported as a new candidate failure."""

    baseline = parse_test_outcome(
        ("python", "tests/runtests.py", "utils_tests"),
        _failed(
            "ERROR: test_strip_tags_files (utils_tests.test_html.TestUtilsHtml) "
            "[<object object at 0x71d623463b40>] (filename='strip_tags1.html')\n"
        ),
    )
    candidate = parse_test_outcome(
        ("python", "tests/runtests.py", "utils_tests"),
        _failed(
            "ERROR: test_strip_tags_files (utils_tests.test_html.TestUtilsHtml) "
            "[<object object at 0x7bc37109fb40>] (filename='strip_tags1.html')\n"
        ),
    )

    expected = frozenset(
        {
            "test_strip_tags_files (utils_tests.test_html.TestUtilsHtml) "
            "[<object object at 0xADDR>] (filename='strip_tags1.html')"
        }
    )
    assert baseline.failure_signatures == expected
    assert candidate.failure_signatures == expected
    assert candidate.failure_signatures - baseline.failure_signatures == frozenset()


def test_parse_sympy_failure_signature() -> None:
    """SymPy bin/test underscore failure headings must extract the file and test function."""

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
    """A non-zero exit without a test ID must not masquerade as a comparable existing test failure."""

    outcome = parse_test_outcome(
        ("python", "-m", "pytest", "tests/test_widget.py"),
        _failed("collection crashed before a test summary was printed\n"),
    )

    assert outcome.status == "failed"
    assert outcome.failure_signatures == frozenset()
    assert outcome.reliable is False
