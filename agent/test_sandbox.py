"""在 SWE-bench 官方 instance 镜像中运行仓库可见测试。

沙箱只把 Agent 当前 Git patch 应用到镜像自带的 ``/testbed``，不注入官方
``test_patch`` 或评测脚本。容器禁用网络并在每次命令后删除，因此测试依赖与宿主
调度仓库的虚拟环境完全隔离，也不会把测试产生的文件写回 Agent worktree。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile
import uuid
from typing import Sequence


class TestSandboxError(RuntimeError):
    """当镜像缺失、Git patch 无法生成或 Docker 无法启动时抛出。"""


@dataclass(frozen=True, slots=True)
class VisibleTestResult:
    """一次可见测试的启动证据、退出码、截断输出和实际镜像。"""

    exit_code: int
    output: str
    image: str
    timed_out: bool
    command_started: bool


def image_candidates(instance_id: str) -> tuple[str, str]:
    """返回本地构建名和 Docker Hub 发布名，顺序优先复用本地镜像。"""

    normalized = instance_id.lower()
    local = f"sweb.eval.x86_64.{normalized}:latest"
    remote = f"swebench/{local}".replace("__", "_1776_")
    return local, remote


def truncate_output(output: str, maximum_chars: int) -> str:
    """保留输出开头和末尾，既显示启动错误也保留最终测试摘要。"""

    if maximum_chars <= 0:
        raise ValueError("maximum_chars must be positive")
    if len(output) <= maximum_chars:
        return output
    marker = "\n... [tool output truncated by visible-test sandbox] ...\n"
    available = max(0, maximum_chars - len(marker))
    head_size = min(2000, available // 3)
    tail_size = available - head_size
    return f"{output[:head_size]}{marker}{output[-tail_size:]}"


class VisibleTestSandbox:
    """使用 Docker CLI 创建无网络、一次性的仓库测试容器。"""

    def __init__(
        self,
        *,
        timeout_seconds: int,
        max_output_chars: int,
        docker_executable: str = "docker",
    ) -> None:
        """验证资源边界；镜像只允许来自固定 SWE-bench 命名规则。"""

        if timeout_seconds <= 0 or max_output_chars <= 0:
            raise ValueError("test timeout and output limit must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.docker_executable = docker_executable

    def resolve_image(self, instance_id: str) -> str:
        """只检查本地镜像，不在正式求解期间隐式联网拉取。"""

        for image in image_candidates(instance_id):
            result = subprocess.run(
                [self.docker_executable, "image", "inspect", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                return image
        _, remote = image_candidates(instance_id)
        raise TestSandboxError(
            f"SWE-bench image is not cached for {instance_id}; "
            f"pull it during environment preparation: docker pull {remote}"
        )

    def ensure_image(self, instance_id: str, *, allow_pull: bool) -> str:
        """复用本地镜像，或仅在明确的环境准备阶段自动拉取官方镜像。

        ``allow_pull`` 必须由命令行的 ``--allow-network-preparation`` 传入；Agent
        求解过程中只调用 ``resolve_image``，因此无法借此绕过离线实验边界。
        """

        try:
            return self.resolve_image(instance_id)
        except TestSandboxError:
            if not allow_pull:
                raise

        _, remote = image_candidates(instance_id)
        completed = subprocess.run(
            [self.docker_executable, "pull", remote],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
            raise TestSandboxError(f"cannot pull SWE-bench image {remote}: {detail}")
        # pull 成功后再次 inspect，避免把 Docker 的零退出码直接当作镜像可用证据。
        return self.resolve_image(instance_id)

    def image_digest(self, image: str) -> str:
        """读取本地 immutable image ID，供运行 metadata 固定实际测试环境。"""

        result = subprocess.run(
            [
                self.docker_executable,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                image,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        digest = result.stdout.strip()
        if result.returncode != 0 or not digest:
            detail = result.stderr.strip() or "image ID is empty"
            raise TestSandboxError(f"cannot inspect cached image {image}: {detail}")
        return digest

    def run(
        self,
        repository: Path | str,
        *,
        instance_id: str,
        base_commit: str,
        command: Sequence[str],
    ) -> VisibleTestResult:
        """生成当前 patch，在隔离镜像中应用后直接执行 argv 测试命令。"""

        repository_path = Path(repository).resolve()
        if not repository_path.is_dir():
            raise TestSandboxError(f"repository does not exist: {repository_path}")
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", base_commit):
            raise TestSandboxError("base commit must be a 7-40 character Git hex id")
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise TestSandboxError("test command must contain non-empty argv items")

        image = self.resolve_image(instance_id)
        patch = self._collect_patch(repository_path, base_commit)
        container_name = self._container_name(instance_id)
        # 唯一标记只在 git apply 成功之后、exec 测试之前输出。Docker 返回结果但
        # 缺少该标记时，调度器不能把镜像或补丁准备失败误算成真实测试执行。
        started_marker = f"__LOCAL_SWE_TEST_STARTED_{uuid.uuid4().hex}__"
        with tempfile.TemporaryDirectory(prefix="local-swe-visible-test-") as directory:
            patch_path = Path(directory) / "agent.patch"
            patch_path.write_text(patch, encoding="utf-8")
            docker_command = [
                self.docker_executable,
                "run",
                "--rm",
                "--name",
                container_name,
                "--network",
                "none",
                "--user",
                "root",
                "--cpus",
                "4",
                "--memory",
                "8g",
                "--pids-limit",
                "1024",
                "--volume",
                f"{patch_path}:/tmp/agent.patch:ro",
                image,
                "bash",
                "-lc",
                # 命令参数通过 "$@" 原样传递，不把 issue 文本或 argv 拼接为 shell。
                (
                    "cd /testbed && "
                    "([ ! -s /tmp/agent.patch ] || "
                    "git apply --binary /tmp/agent.patch) && "
                    f"printf '%s\\n' {started_marker} && exec \"$@\""
                ),
                "visible-test",
                *command,
            ]
            timed_out = False
            try:
                completed = subprocess.run(
                    docker_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                )
                exit_code = completed.returncode
                output = completed.stdout
            except subprocess.TimeoutExpired as error:
                timed_out = True
                exit_code = 124
                partial = error.stdout or ""
                output = (
                    partial
                    if isinstance(partial, str)
                    else partial.decode("utf-8", errors="replace")
                )
                output += f"\nvisible test timed out after {self.timeout_seconds}s\n"
            finally:
                # timeout 时 docker 客户端可能先退出，必须按不可猜测的 UUID 名称清理
                # 精确容器，防止后台测试继续占用内存和 CPU。
                subprocess.run(
                    [self.docker_executable, "rm", "--force", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )

        command_started = started_marker in output
        output = output.replace(f"{started_marker}\n", "", 1)
        return VisibleTestResult(
            exit_code=exit_code,
            output=truncate_output(output, self.max_output_chars),
            image=image,
            timed_out=timed_out,
            command_started=command_started,
        )

    @staticmethod
    def _collect_patch(repository: Path, base_commit: str) -> str:
        """收集 committed、tracked 与 untracked 修改；基线测试允许空补丁。"""

        add = subprocess.run(
            ["git", "-C", str(repository), "add", "--intent-to-add", "--all"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if add.returncode != 0:
            raise TestSandboxError(add.stderr.strip() or "git add --intent-to-add failed")
        diff = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "diff",
                "--binary",
                "--no-ext-diff",
                base_commit,
                "--",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if diff.returncode != 0:
            raise TestSandboxError(diff.stderr.strip() or "git diff failed")
        return diff.stdout

    @staticmethod
    def _container_name(instance_id: str) -> str:
        """生成不会与正式 harness 或并发测试冲突的单次容器名。"""

        safe_id = re.sub(r"[^a-z0-9_.-]+", "-", instance_id.lower()).strip("-.")
        return f"local-swe-visible-{safe_id[:40]}-{uuid.uuid4().hex[:12]}"
