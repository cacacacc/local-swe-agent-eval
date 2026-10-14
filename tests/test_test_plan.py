"""Verify the consumption, cleanup, and command boundaries of the structured test plan."""

import json
from pathlib import Path

from agent.test_plan import (
    TEST_FILE_SUGGESTION_FILENAME,
    TEST_FILE_SUGGESTION_PLACEHOLDER,
    TEST_PLAN_FILENAME,
    consume_test_file_suggestion,
    consume_test_plan,
    generate_repository_test_plan,
)


def test_parent_accepts_only_an_existing_model_suggested_test_module(tmp_path: Path) -> None:
    """A path hint may select an existing target, but cannot supply commands or a new test."""

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_widget.py").write_text("def test_widget(): pass\n", encoding="utf-8")
    (tests / "test_edge.py").write_text("def test_edge(): pass\n", encoding="utf-8")
    patch = "diff --git a/src/widget.py b/src/widget.py\n--- a/src/widget.py\n+++ b/src/widget.py\n"
    hint = tmp_path / TEST_FILE_SUGGESTION_FILENAME
    hint.write_text("tests/test_edge.py\n", encoding="utf-8")
    suggestion = consume_test_file_suggestion(tmp_path)
    assert suggestion == "tests/test_edge.py"
    assert not hint.exists()

    chosen = generate_repository_test_plan(
        tmp_path, repo="owner/project", patch=patch,
        suggested_test_file=suggestion,
    )
    assert chosen.suggested_test_file_used
    assert chosen.target_argv[-1] == "tests/test_edge.py"
    assert chosen.regression_argv[-1] == "tests/test_widget.py"

    for bad_path in ("../outside.py", "tests/missing.py", "tests/test_edge.py; echo bad"):
        rejected = generate_repository_test_plan(
            tmp_path, repo="owner/project", patch=patch,
            suggested_test_file=bad_path,
        )
        assert not rejected.suggested_test_file_used
        assert rejected.target_argv[-1] == "tests/test_widget.py"

    (tests / "test_link.py").symlink_to(tests / "test_edge.py")
    linked = generate_repository_test_plan(
        tmp_path, repo="owner/project", patch=patch,
        suggested_test_file="tests/test_link.py",
    )
    assert not linked.suggested_test_file_used


def test_suggestion_consumer_rejects_symlink_without_following_it(tmp_path: Path) -> None:
    """An untrusted control-path symlink must be removed without reading its target."""

    outside = tmp_path / "outside.txt"
    outside.write_text("tests/test_edge.py\n", encoding="utf-8")
    hint = tmp_path / TEST_FILE_SUGGESTION_FILENAME
    hint.symlink_to(outside)
    assert consume_test_file_suggestion(tmp_path) is None
    assert outside.exists()
    assert not hint.exists()


def test_unedited_suggestion_placeholder_is_missing(tmp_path: Path) -> None:
    """An optional path hint left untouched must not affect parent test planning."""

    hint = tmp_path / TEST_FILE_SUGGESTION_FILENAME
    hint.write_text(TEST_FILE_SUGGESTION_PLACEHOLDER, encoding="utf-8")
    assert consume_test_file_suggestion(tmp_path) is None
    assert not hint.exists()


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


def test_django_specific_source_maps_to_correct_test_app(tmp_path: Path) -> None:
    """Checks, server handling, and index references must not be trapped in adjacent Django apps."""

    cases = (
        ("django/contrib/admin/checks.py", "admin_checks", "admin_views"),
        ("django/core/servers/basehttp.py", "servers", "builtin_server"),
        ("django/db/backends/ddl_references.py", "indexes", "backends"),
    )
    for source, expected_app, wrong_app in cases:
        for app in (expected_app, wrong_app):
            test = tmp_path / "tests" / app / "tests.py"
            test.parent.mkdir(parents=True, exist_ok=True)
            test.write_text("def test_case(): pass\n", encoding="utf-8")
        patch = (
            f"diff --git a/{source} b/{source}\n"
            f"--- a/{source}\n+++ b/{source}\n@@ -1 +1 @@\n-old\n+new\n"
        )
        request = generate_repository_test_plan(
            tmp_path, repo="django/django", patch=patch,
        )
        assert request.target_argv[2].startswith(f"{expected_app}.")
        assert request.regression_argv[2] == expected_app


def test_issue_keyword_selects_relevant_existing_test_module(tmp_path: Path) -> None:
    """A pickle edge case should select the existing pickle test without accepting a test command from the model."""

    tests = tmp_path / "lib" / "matplotlib" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_figure.py").write_text(
        "from matplotlib.figure import Figure\n", encoding="utf-8"
    )
    (tests / "test_pickle.py").write_text(
        "from matplotlib.figure import Figure\n", encoding="utf-8"
    )
    patch = (
        "diff --git a/lib/matplotlib/figure.py b/lib/matplotlib/figure.py\n"
        "--- a/lib/matplotlib/figure.py\n+++ b/lib/matplotlib/figure.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    request = generate_repository_test_plan(
        tmp_path, repo="matplotlib/matplotlib", patch=patch,
        problem_statement="Figure fails to pickle after label alignment.",
    )
    assert request.target_argv[-1] == "lib/matplotlib/tests/test_pickle.py"


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


def test_parent_prefers_direct_import_over_misleading_test_name(tmp_path: Path) -> None:
    """An exact source import must outrank a lexically similar but unrelated test file."""

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_domain_c.py").write_text(
        "from sphinx.domains.c import CDomain\n", encoding="utf-8"
    )
    (tests / "test_domain_py.py").write_text(
        "from sphinx.domains.python import PythonDomain\n", encoding="utf-8"
    )
    request = generate_repository_test_plan(
        tmp_path,
        repo="sphinx-doc/sphinx",
        patch=(
            "diff --git a/sphinx/domains/python.py b/sphinx/domains/python.py\n"
            "--- a/sphinx/domains/python.py\n+++ b/sphinx/domains/python.py\n"
            "@@ -1 +1 @@ def _parse_annotation():\n-old\n+new\n"
            "diff --git a/tests/test_domain_py.py b/tests/test_domain_py.py\n"
            "--- a/tests/test_domain_py.py\n+++ b/tests/test_domain_py.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
    )

    assert request.target_argv[-1] == "tests/test_domain_py.py"


def test_parent_uses_changed_symbol_with_package_level_import(tmp_path: Path) -> None:
    """A test importing the changed class from a package facade outranks a generic neighbor."""

    tests = tmp_path / "tests" / "forms_tests" / "field_tests"
    tests.mkdir(parents=True)
    (tests / "test_fields.py").write_text(
        "from django.forms import CharField\n", encoding="utf-8"
    )
    (tests / "test_multivaluefield.py").write_text(
        "from django.forms import MultiValueField\n"
        "def test_field():\n    assert MultiValueField\n",
        encoding="utf-8",
    )
    request = generate_repository_test_plan(
        tmp_path,
        repo="django/django",
        patch=(
            "diff --git a/django/forms/fields.py b/django/forms/fields.py\n"
            "--- a/django/forms/fields.py\n+++ b/django/forms/fields.py\n"
            "@@ -20 +20 @@ class MultiValueField:\n-old\n+new\n"
        ),
    )

    assert request.target_argv[2] == "forms_tests.field_tests.test_multivaluefield"


def test_parent_includes_django_app_tests_py_with_direct_import(tmp_path: Path) -> None:
    """Django's conventional tests.py modules must participate in dependency ranking."""

    tests = tmp_path / "tests" / "mail"
    tests.mkdir(parents=True)
    (tests / "test_sendtestemail.py").write_text(
        "def test_command(): pass\n", encoding="utf-8"
    )
    (tests / "tests.py").write_text(
        "from django.core.mail.message import EmailMessage\n"
        "def test_email():\n    assert EmailMessage\n",
        encoding="utf-8",
    )
    request = generate_repository_test_plan(
        tmp_path,
        repo="django/django",
        patch=(
            "diff --git a/django/core/mail/message.py b/django/core/mail/message.py\n"
            "--- a/django/core/mail/message.py\n+++ b/django/core/mail/message.py\n"
            "@@ -1 +1 @@ class EmailMessage:\n-old\n+new\n"
        ),
    )

    assert request.target_argv[2] == "mail.tests"


def test_parent_does_not_select_model_added_tests(tmp_path: Path) -> None:
    """A newly added reproduction cannot become the parent's allegedly independent target test."""

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_widget.py").write_text("def test_old(): pass\n", encoding="utf-8")
    (tests / "test_widget_new.py").write_text(
        "from package.widget import Widget\n", encoding="utf-8"
    )
    request = generate_repository_test_plan(
        tmp_path,
        repo="owner/project",
        patch=(
            "diff --git a/package/widget.py b/package/widget.py\n"
            "--- a/package/widget.py\n+++ b/package/widget.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
            "diff --git a/tests/test_widget_new.py b/tests/test_widget_new.py\n"
            "new file mode 100644\n--- /dev/null\n+++ b/tests/test_widget_new.py\n"
            "@@ -0,0 +1 @@\n+from package.widget import Widget\n"
        ),
    )

    assert request.target_argv[-1] == "tests/test_widget.py"
