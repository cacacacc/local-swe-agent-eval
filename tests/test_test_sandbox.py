"""Verify the image rules, isolation parameters, and output bounds of the visible-test Docker sandbox."""

from pathlib import Path
import subprocess

import pytest

from agent.test_sandbox import (
    TestSandboxError as SandboxError,
    VisibleTestSandbox,
    image_candidates,
    truncate_output,
)


def test_published_image_name_matches_swebench_convention() -> None:
    """Double underscores must be encoded per official Docker Hub rules to avoid pulling the wrong image."""

    assert image_candidates("Django__Django-11951") == (
        "sweb.eval.x86_64.django__django-11951:latest",
        "swebench/sweb.eval.x86_64.django_1776_django-11951:latest",
    )


def test_truncate_output_keeps_start_and_final_failure() -> None:
    """Hard truncation must preserve both the initialization error and the final test summary."""

    output = "BEGIN\n" + "x" * 20000 + "\nFAILED final assertion\n"
    truncated = truncate_output(output, 12000)

    assert len(truncated) == 12000
    assert truncated.startswith("BEGIN")
    assert truncated.endswith("FAILED final assertion\n")
    assert "tool output truncated" in truncated


def test_image_digest_requires_nonempty_immutable_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run metadata may only record the non-empty image ID Docker returns, not masquerade a mutable tag."""

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="sha256:abc123\n",
            stderr="",
        )

    monkeypatch.setattr("agent.test_sandbox.subprocess.run", fake_run)
    sandbox = VisibleTestSandbox(timeout_seconds=30, max_output_chars=12000)

    assert sandbox.image_digest("swebench/example:latest") == "sha256:abc123"


def test_ensure_image_pulls_only_when_preparation_allows_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing image may only be pulled during the network-allowed preparation phase, and re-inspected after pulling."""

    pulled = False
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        nonlocal pulled
        command = [str(item) for item in command]
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            is_remote = command[-1].startswith("swebench/")
            return subprocess.CompletedProcess(
                command,
                0 if pulled and is_remote else 1,
                stdout="",
                stderr="",
            )
        if command[:2] == ["docker", "pull"]:
            pulled = True
            return subprocess.CompletedProcess(command, 0, stdout="pulled\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("agent.test_sandbox.subprocess.run", fake_run)
    sandbox = VisibleTestSandbox(timeout_seconds=30, max_output_chars=12000)

    image = sandbox.ensure_image("owner__repo-7", allow_pull=True)

    assert image == "swebench/sweb.eval.x86_64.owner_1776_repo-7:latest"
    assert sum(command[:2] == ["docker", "pull"] for command in calls) == 1


def test_ensure_image_refuses_pull_during_offline_solving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When network preparation is not authorized, a missing image must fail immediately without running docker pull."""

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        command = [str(item) for item in command]
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr("agent.test_sandbox.subprocess.run", fake_run)
    sandbox = VisibleTestSandbox(timeout_seconds=30, max_output_chars=12000)

    with pytest.raises(SandboxError, match="not cached"):
        sandbox.ensure_image("owner__repo-7", allow_pull=False)

    assert not any(command[:2] == ["docker", "pull"] for command in calls)


def test_sandbox_applies_only_current_patch_and_disables_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The container may only receive the Agent patch and run the test argv in a one-shot, network-free, resource-limited way."""

    repository = tmp_path / "repo"
    repository.mkdir()
    calls: list[list[str]] = []
    docker_timeouts: list[float] = []

    def fake_run(command, **kwargs):
        command = [str(item) for item in command]
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:4] == ["git", "-C", str(repository), "add"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:4] == ["git", "-C", str(repository), "diff"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="diff --git a/a.py b/a.py\n-old\n+new\n",
                stderr="",
            )
        if command[:2] == ["docker", "run"]:
            docker_timeouts.append(kwargs["timeout"])
            shell_command = command[command.index("bash") + 2]
            marker = shell_command.split("printf '%s\\n' ", 1)[1].split(" &&", 1)[0]
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{marker}\n1 passed\n", stderr=""
            )
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("agent.test_sandbox.subprocess.run", fake_run)
    result = VisibleTestSandbox(
        timeout_seconds=30,
        max_output_chars=12000,
    ).run(
        repository,
        instance_id="owner__repo-7",
        base_commit="0123456789abcdef0123456789abcdef01234567",
        command=("python", "-m", "pytest", "tests/test_one.py"),
        timeout_seconds=5,
    )

    docker_run = next(command for command in calls if command[:2] == ["docker", "run"])
    assert docker_run[docker_run.index("--network") + 1] == "none"
    assert docker_run[docker_run.index("--memory") + 1] == "8g"
    assert docker_run[-4:] == ["python", "-m", "pytest", "tests/test_one.py"]
    volume = docker_run[docker_run.index("--volume") + 1]
    assert volume.endswith(":/tmp/agent.patch:ro")
    assert "/testbed" not in volume
    assert result.exit_code == 0
    assert result.command_started is True
    assert result.output == "1 passed\n"
    assert result.duration_seconds >= 0
    assert result.cache_hit is False
    assert result.infrastructure_error is None
    assert result.evidence_valid is True
    assert docker_timeouts == [5]


def test_sandbox_marks_missing_pytest_as_infrastructure_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing pytest runner must not be mistaken for a test failure after real execution."""

    repository = tmp_path / "repo"
    repository.mkdir()

    def fake_run(command, **kwargs):
        command = [str(item) for item in command]
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["docker", "run"]:
            shell_command = command[command.index("bash") + 2]
            marker = shell_command.split("printf '%s\\n' ", 1)[1].split(" &&", 1)[0]
            return subprocess.CompletedProcess(
                command,
                1,
                stdout=(
                    f"{marker}\n"
                    "/opt/miniconda3/envs/testbed/bin/python: "
                    "No module named pytest\n"
                ),
                stderr="",
            )
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("agent.test_sandbox.subprocess.run", fake_run)
    result = VisibleTestSandbox(
        timeout_seconds=30,
        max_output_chars=12000,
    ).run(
        repository,
        instance_id="sympy__sympy-17630",
        base_commit="0123456789abcdef0123456789abcdef01234567",
        command=("python", "-m", "pytest", "sympy/core/tests/test_basic.py"),
        apply_patch=False,
    )

    assert result.command_started is True
    assert result.infrastructure_error == (
        "test runner unavailable: pytest module is not installed"
    )
    assert result.evidence_valid is False


def test_sandbox_does_not_count_patch_apply_failure_as_test_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the container starts but patch application fails, no real test-start evidence may be produced."""

    repository = tmp_path / "repo"
    repository.mkdir()

    def fake_run(command, **kwargs):
        command = [str(item) for item in command]
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:4] == ["git", "-C", str(repository), "add"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:4] == ["git", "-C", str(repository), "diff"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="diff --git a/a.py b/a.py\n-old\n+new\n", stderr=""
            )
        if command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(
                command, 1, stdout="error: patch failed\n", stderr=""
            )
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("agent.test_sandbox.subprocess.run", fake_run)
    result = VisibleTestSandbox(
        timeout_seconds=30,
        max_output_chars=12000,
    ).run(
        repository,
        instance_id="owner__repo-7",
        base_commit="0123456789abcdef0123456789abcdef01234567",
        command=("python", "-m", "pytest"),
    )

    assert result.command_started is False
    assert result.exit_code == 1


def test_sandbox_runs_unmodified_baseline_without_collecting_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline comparison must run the image checkout directly, without reading or applying the candidate patch."""

    repository = tmp_path / "repo"
    repository.mkdir()
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        command = [str(item) for item in command]
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if "add" in command:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if "diff" in command:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["docker", "run"]:
            shell_command = command[command.index("bash") + 2]
            marker = shell_command.split("printf '%s\\n' ", 1)[1].split(" &&", 1)[0]
            return subprocess.CompletedProcess(
                command,
                1,
                stdout=f"{marker}\n1 failed\n",
                stderr="",
            )
        if command[:3] == ["docker", "rm", "--force"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("agent.test_sandbox.subprocess.run", fake_run)
    sandbox = VisibleTestSandbox(timeout_seconds=30, max_output_chars=12000)

    result = sandbox.run(
        repository,
        instance_id="owner__repo-7",
        base_commit="0123456789abcdef0123456789abcdef01234567",
        command=("python", "-m", "pytest"),
        apply_patch=False,
    )

    docker_run = next(command for command in calls if command[:2] == ["docker", "run"])
    shell_program = docker_run[docker_run.index("-lc") + 1]
    assert "[ ! -s /tmp/agent.patch ]" in shell_program
    assert docker_run[docker_run.index("--network") + 1] == "none"
    assert not any(command[:2] == ["git", "-C"] for command in calls)
    assert result.command_started is True
    assert result.exit_code == 1
    assert result.output == "1 failed\n"
    assert result.infrastructure_error is None
    assert result.evidence_valid is True
