"""验证可见测试 Docker 沙箱的镜像规则、隔离参数和输出边界。"""

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
    """双下划线必须按官方 Docker Hub 规则编码，避免拉取错误镜像。"""

    assert image_candidates("Django__Django-11951") == (
        "sweb.eval.x86_64.django__django-11951:latest",
        "swebench/sweb.eval.x86_64.django_1776_django-11951:latest",
    )


def test_truncate_output_keeps_start_and_final_failure() -> None:
    """硬截断必须同时保留初始化错误和最终测试摘要。"""

    output = "BEGIN\n" + "x" * 20000 + "\nFAILED final assertion\n"
    truncated = truncate_output(output, 12000)

    assert len(truncated) == 12000
    assert truncated.startswith("BEGIN")
    assert truncated.endswith("FAILED final assertion\n")
    assert "tool output truncated" in truncated


def test_image_digest_requires_nonempty_immutable_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """运行元数据只能记录 Docker 返回的非空 image ID，不能用可变 tag 冒充。"""

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
    """缺失镜像只能在显式允许联网的准备阶段拉取，并在拉取后重新 inspect。"""

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
    """未授权准备网络时，缺失镜像必须立即失败且不得执行 docker pull。"""

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
    """容器只能获得 Agent patch，并以无网络、限资源的一次性方式执行测试 argv。"""

    repository = tmp_path / "repo"
    repository.mkdir()
    calls: list[list[str]] = []

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


def test_sandbox_does_not_count_patch_apply_failure_as_test_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """容器已启动但补丁应用失败时，不能生成真实测试启动证据。"""

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


def test_sandbox_runs_unmodified_baseline_when_patch_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实现前即使没有补丁，也必须在官方镜像里取得真实基线测试证据。"""

    repository = tmp_path / "repo"
    repository.mkdir()

    def fake_run(command, **kwargs):
        command = [str(item) for item in command]
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if "add" in command:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if "diff" in command:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["docker", "run"]:
            shell_command = command[command.index("bash") + 2]
            marker = shell_command.split("printf '%s\\n' ", 1)[1].split(" &&", 1)[0]
            assert "[ ! -s /tmp/agent.patch ]" in shell_command
            return subprocess.CompletedProcess(
                command, 1, stdout=f"{marker}\n1 failed\n", stderr=""
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
    )

    assert result.command_started is True
    assert result.exit_code == 1
    assert result.output == "1 failed\n"
