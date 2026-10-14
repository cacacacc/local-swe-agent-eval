"""Protect the one-shot issue reproduction boundary and its offline comparison."""

from pathlib import Path

from agent.issue_reproduction import (
    REPRODUCTION_FILENAME,
    REPRODUCTION_PLACEHOLDER,
    IssueReproductionRequest,
    consume_issue_reproduction,
)
from agent.test_sandbox import VisibleTestResult
from benchmark.task import SWEbenchTask
from scripts.run_claude import _TaskBudget, _execute_issue_reproduction


def test_consume_accepts_short_assertion_and_removes_control_file(tmp_path: Path) -> None:
    """A short issue assertion is returned once but never remains in the product patch."""

    path = tmp_path / REPRODUCTION_FILENAME
    path.write_text("from widget import value\nassert value() == 2\n", encoding="utf-8")
    request = consume_issue_reproduction(tmp_path)
    assert request.accepted
    assert "assert value() == 2" in request.code
    assert not path.exists()
    assert consume_issue_reproduction(tmp_path).status == "missing"


def test_unedited_parent_placeholder_counts_as_missing(tmp_path: Path) -> None:
    """An untouched Edit target must not be counted as a rejected model submission."""

    path = tmp_path / REPRODUCTION_FILENAME
    path.write_text(REPRODUCTION_PLACEHOLDER, encoding="utf-8")
    assert consume_issue_reproduction(tmp_path).status == "missing"
    assert not path.exists()


def test_consume_rejects_invalid_or_symlinked_reproduction(tmp_path: Path) -> None:
    """Malformed and symlinked control paths are removed without reading outside the task."""

    path = tmp_path / REPRODUCTION_FILENAME
    path.write_text("print('no assertion')\n", encoding="utf-8")
    assert consume_issue_reproduction(tmp_path).status == "rejected"
    assert not path.exists()
    outside = tmp_path / "outside.py"
    outside.write_text("assert True\n", encoding="utf-8")
    path.symlink_to(outside)
    assert consume_issue_reproduction(tmp_path).status == "rejected"
    assert outside.exists()
    assert not path.exists()


def test_consume_rejects_oversized_reproduction(tmp_path: Path) -> None:
    """A large model-generated script cannot expand Docker argv or prompt context without bound."""

    path = tmp_path / REPRODUCTION_FILENAME
    path.write_text("assert True\n" + "#" * 8192, encoding="utf-8")
    request = consume_issue_reproduction(tmp_path)
    assert request.status == "rejected"
    assert request.error == "reproduction exceeds 8192 bytes"
    assert not path.exists()


def test_consume_accepts_a_realistic_boundary_reproduction(tmp_path: Path) -> None:
    """A 45-line focused script must not be rejected by the earlier 30-line limit."""

    path = tmp_path / REPRODUCTION_FILENAME
    path.write_text("# setup\n" * 44 + "assert True\n", encoding="utf-8")
    request = consume_issue_reproduction(tmp_path)
    assert request.accepted
    assert not path.exists()


def test_rejected_submission_keeps_reason_without_running_docker(tmp_path: Path) -> None:
    """A rejected script carries its precise validation reason into parent evidence."""

    class NoDockerSandbox:
        """Any execution of a rejected model script would cross the trust boundary."""

        def run(self, *args, **kwargs):
            raise AssertionError("rejected reproduction must not run")

    task = SWEbenchTask(
        instance_id="owner__project-rejected", repo="owner/project",
        base_commit="abcdef0", problem_statement="Fix it.",
    )
    evidence = _execute_issue_reproduction(
        tmp_path, task=task, base_commit="abcdef0",
        request=IssueReproductionRequest(
            "rejected", error="reproduction exceeds 30 lines",
        ),
        sandbox=NoDockerSandbox(), task_budget=_TaskBudget.start(0),
    )
    assert evidence.status == "rejected"
    assert evidence.error == "reproduction exceeds 30 lines"


def test_parent_compares_issue_assertion_in_docker_not_on_host(tmp_path: Path) -> None:
    """The parent owns argv and labels a fix only when baseline fails and candidate passes."""

    calls = []

    class FakeSandbox:
        """Simulate the two isolated sides while preserving their actual invocation options."""

        timeout_seconds = 120

        def run(self, repository, **kwargs):
            calls.append(kwargs)
            return VisibleTestResult(
                exit_code=0 if kwargs["apply_patch"] else 1,
                output="" if kwargs["apply_patch"] else "AssertionError",
                image="swebench/example:latest", timed_out=False,
                command_started=True,
            )

    task = SWEbenchTask(
        instance_id="owner__project-1", repo="owner/project",
        base_commit="abcdef0", problem_statement="Value should be two.",
    )
    evidence = _execute_issue_reproduction(
        tmp_path, task=task, base_commit="abcdef0",
        request=IssueReproductionRequest("accepted", "assert value() == 2\n"),
        sandbox=FakeSandbox(), task_budget=_TaskBudget.start(0),
    )
    assert evidence.status == "reproduced_fix"
    assert [call["apply_patch"] for call in calls] == [False, True]
    assert all(call["command"] == ("python", "-c", "assert value() == 2\n") for call in calls)
    assert all(call["timeout_seconds"] <= 45 for call in calls)


def test_import_error_is_not_claimed_as_issue_reproduction(tmp_path: Path) -> None:
    """A broken baseline environment must not masquerade as a reproduced assertion."""

    class FakeSandbox:
        """Return a baseline import failure and a passing patched candidate."""

        timeout_seconds = 120

        def run(self, repository, **kwargs):
            return VisibleTestResult(
                exit_code=0 if kwargs["apply_patch"] else 1,
                output="" if kwargs["apply_patch"] else "ModuleNotFoundError: widget",
                image="swebench/example:latest", timed_out=False,
                command_started=True,
            )

    task = SWEbenchTask(
        instance_id="owner__project-2", repo="owner/project",
        base_commit="abcdef0", problem_statement="Widget should work.",
    )
    evidence = _execute_issue_reproduction(
        tmp_path, task=task, base_commit="abcdef0",
        request=IssueReproductionRequest("accepted", "assert value() == 2\n"),
        sandbox=FakeSandbox(), task_budget=_TaskBudget.start(0),
    )
    assert evidence.status == "baseline_error"


def test_documented_exception_can_reproduce_issue_before_assert(tmp_path: Path) -> None:
    """An issue's documented TypeError can be a valid baseline failure before the assertion executes."""

    class FakeSandbox:
        """Model a pickling failure that disappears after the candidate patch."""

        timeout_seconds = 120

        def run(self, repository, **kwargs):
            return VisibleTestResult(
                exit_code=0 if kwargs["apply_patch"] else 1,
                output="" if kwargs["apply_patch"] else "TypeError: cannot pickle weakref",
                image="swebench/example:latest", timed_out=False,
                command_started=True,
            )

    task = SWEbenchTask(
        instance_id="owner__project-3", repo="owner/project",
        base_commit="abcdef0",
        problem_statement="Pickling fails with TypeError: cannot pickle weakref.",
    )
    evidence = _execute_issue_reproduction(
        tmp_path, task=task, base_commit="abcdef0",
        request=IssueReproductionRequest("accepted", "assert pickle.dumps(value)\n"),
        sandbox=FakeSandbox(), task_budget=_TaskBudget.start(0),
    )
    assert evidence.status == "reproduced_fix"
