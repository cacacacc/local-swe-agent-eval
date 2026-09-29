"""Verify the consumption, cleanup, and command boundaries of the structured test plan."""

import json
from pathlib import Path

from agent.test_plan import (
    TEST_PLAN_FILENAME,
    consume_test_plan,
    generate_repository_test_plan,
)


def write_plan(repository: Path, value: object) -> Path:
    """Write a single plan into a temporary repo to keep the test setup steps clear."""

    path = repository / TEST_PLAN_FILENAME
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_consume_accepts_target_and_regression_argv(tmp_path: Path) -> None:
    """Both kinds of legal argv should be preserved verbatim, while the control file must not enter the final patch."""

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
    """The plan must not hand shell command strings or unrecognized control fields to the scheduler."""

    path = write_plan(
        tmp_path,
        {"target_argv": ["bash", "-lc", "pytest"], "network": True},
    )

    request = consume_test_plan(tmp_path)

    assert request.status == "rejected"
    assert "target_argv and regression_argv" in request.error
    assert not path.exists()


def test_consume_reports_missing_without_creating_file(tmp_path: Path) -> None:
    """An unsubmitted plan should be distinguished from an invalid plan, so the real adherence rate can be reported."""

    request = consume_test_plan(tmp_path)

    assert request.status == "missing"
    assert request.requested is False
    assert list(tmp_path.iterdir()) == []


def test_consume_rejects_shell_executable_even_with_exact_schema(tmp_path: Path) -> None:
    """Although argv has no shell interpolation, the shell itself must still not be the test entry point."""

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
    """Two identical commands must not masquerade as the target test and the adjacent regression test."""

    command = ["python", "-m", "pytest", "tests/test_one.py"]
    write_plan(
        tmp_path,
        {"target_argv": command, "regression_argv": command},
    )

    request = consume_test_plan(tmp_path)

    assert request.status == "rejected"
    assert request.error == "target and regression argv must be different"


def test_parent_generates_pytest_plan_from_modified_source(tmp_path: Path) -> None:
    """The parent should map a modified module to the existing pytest module and its adjacent regression directory."""

    target = tmp_path / "testing" / "test_reports.py"
    target.parent.mkdir(parents=True)
    target.write_text("def test_report(): pass\n", encoding="utf-8")
    regression = tmp_path / "testing" / "test_terminal.py"
    regression.write_text("def test_terminal(): pass\n", encoding="utf-8")
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
        "python", "-m", "pytest", "testing/test_terminal.py"
    )


def test_parent_uses_django_runner_and_dotted_labels(tmp_path: Path) -> None:
    """Django cannot reuse the generic pytest command; it must generate a dotted label for the project runner."""

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
        "forms_tests",
        "--verbosity",
        "0",
    )


def test_parent_uses_sympy_native_runner_without_pytest(tmp_path: Path) -> None:
    """The official SymPy image may not have pytest installed, so the parent must call the repository's own ``bin/test``."""

    test_directory = tmp_path / "sympy" / "matrices" / "expressions" / "tests"
    test_directory.mkdir(parents=True)
    target = test_directory / "test_blockmatrix.py"
    target.write_text("def test_blockmatrix(): pass\n", encoding="utf-8")
    regression = test_directory / "test_adjoint.py"
    regression.write_text("def test_adjoint(): pass\n", encoding="utf-8")
    request = generate_repository_test_plan(
        tmp_path,
        repo="sympy/sympy",
        patch=(
            "diff --git a/sympy/matrices/expressions/blockmatrix.py "
            "b/sympy/matrices/expressions/blockmatrix.py\n"
            "--- a/sympy/matrices/expressions/blockmatrix.py\n"
            "+++ b/sympy/matrices/expressions/blockmatrix.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
    )

    prefix = ("python", "bin/test", "--no-colors", "-C")
    assert request.status == "generated"
    assert request.target_argv == (*prefix, target.relative_to(tmp_path).as_posix())
    assert request.regression_argv == (
        *prefix,
        regression.relative_to(tmp_path).as_posix(),
    )


def test_parent_adapts_django_root_test_to_distinct_neighbor(tmp_path: Path) -> None:
    """When the Django root-level test has no app, another module must still be chosen as the adjacent regression."""

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sqlite.py").write_text("", encoding="utf-8")
    (tests / "test_runner.py").write_text("", encoding="utf-8")
    request = generate_repository_test_plan(
        tmp_path,
        repo="django/django",
        patch=(
            "diff --git a/django/db/backends/sqlite3/base.py "
            "b/django/db/backends/sqlite3/base.py\n"
            "--- a/django/db/backends/sqlite3/base.py\n"
            "+++ b/django/db/backends/sqlite3/base.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
    )

    assert request.target_argv[2] == "test_sqlite"
    assert request.regression_argv[2] == "test_runner"


def test_parent_missing_plan_is_diagnostic_for_unmatched_patch(tmp_path: Path) -> None:
    """When no adjacent test is found, record missing instead of fabricating a command or blocking verification."""

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
    """A reproduction script newly created by the model must not be mistaken by the parent for existing product source or a trusted test."""

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
