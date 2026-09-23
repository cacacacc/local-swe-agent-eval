"""为 SWE-bench 任务准备可复现、相互隔离的 Git worktree。

同一上游仓库只维护一份共享 clone，以节约磁盘；每道题从指定 base commit
创建独立 detached worktree，避免任务之间互相污染。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Sequence

from .task import SWEbenchTask


class RepositoryError(RuntimeError):
    """当仓库无法按要求安全、可复现地准备时抛出。"""


# 将外部 instance ID 转成单个安全路径组件，避免斜杠等字符改变目录层级。
_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class PreparedRepository:
    """已经验证确实位于目标 base commit 的任务 worktree。"""

    task: SWEbenchTask
    path: Path
    resolved_commit: str


class RepositoryManager:
    """管理“每仓库一个共享 clone、每任务一个 worktree”的目录结构。"""

    def __init__(
        self,
        cache_root: Path | str,
        workspace_root: Path | str,
        *,
        command_timeout_seconds: int = 600,
    ) -> None:
        """保存并规范化缓存路径，同时验证外部命令超时配置。"""

        self.cache_root = Path(cache_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        self.command_timeout_seconds = command_timeout_seconds

    def prepare(
        self,
        task: SWEbenchTask,
        *,
        allow_network: bool,
    ) -> PreparedRepository:
        """准备全新 worktree，并拒绝覆盖任何已有任务目录。

        ``allow_network`` 明确区分环境准备阶段与正式离线求解阶段：缓存或
        commit 缺失时，离线模式必须立即失败，不能偷偷访问远程仓库。
        """

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        cache_path = self._cache_path(task.repo)
        task_path = self._task_path(task.instance_id)

        if task_path.exists():
            raise RepositoryError(
                f"task workspace already exists; refusing to overwrite it: {task_path}"
            )

        # 首次准备仓库时只允许在显式开放网络的阶段执行 clone。
        if not cache_path.exists():
            if not allow_network:
                raise RepositoryError(
                    f"repository is not cached and network is disabled: {task.repo}"
                )
            self._clone(task.repo, cache_path)
        else:
            self._verify_cache(cache_path, task.repo)

        # 已有 clone 不代表包含目标历史；必要时在准备阶段 fetch 一次。
        if not self._commit_exists(cache_path, task.base_commit):
            if not allow_network:
                raise RepositoryError(
                    f"base commit {task.base_commit} is absent from the offline cache"
                )
            self._run_git(cache_path, "fetch", "--all", "--tags", "--prune")

        if not self._commit_exists(cache_path, task.base_commit):
            raise RepositoryError(
                f"base commit {task.base_commit} does not exist in {task.repo}"
            )

        # detached worktree 不绑定本地分支，保证起点严格等于数据集给定提交。
        self._run_git(
            cache_path,
            "worktree",
            "add",
            "--detach",
            str(task_path),
            task.base_commit,
        )
        # 同时解析 worktree HEAD 与缓存中的目标 commit，防止短 SHA 歧义。
        resolved_commit = self._run_git(task_path, "rev-parse", "HEAD").strip()
        expected_commit = self._run_git(
            cache_path, "rev-parse", f"{task.base_commit}^{{commit}}"
        ).strip()
        if resolved_commit != expected_commit:
            raise RepositoryError(
                f"checkout verification failed: expected {expected_commit}, "
                f"got {resolved_commit}"
            )

        return PreparedRepository(task, task_path, resolved_commit)

    def _clone(self, repo: str, destination: Path) -> None:
        """建立无工作区的共享 clone；具体任务稍后通过 worktree 检出。"""

        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{repo}.git"
        self._run(
            ["git", "clone", "--no-checkout", url, str(destination)],
            cwd=destination.parent,
        )

    def _verify_cache(self, cache_path: Path, repo: str) -> None:
        """确认缓存是 Git clone，且 origin 确实指向预期 GitHub 仓库。"""

        if not (cache_path / ".git").is_dir():
            raise RepositoryError(f"cache path is not a Git clone: {cache_path}")
        actual_url = self._run_git(
            cache_path, "remote", "get-url", "origin"
        ).strip().removesuffix("/")
        accepted_urls = {
            f"https://github.com/{repo}.git",
            f"https://github.com/{repo}",
            f"git@github.com:{repo}.git",
        }
        if actual_url not in accepted_urls:
            raise RepositoryError(
                f"cached origin mismatch for {repo}: {actual_url}"
            )

    def _commit_exists(self, cache_path: Path, commit: str) -> bool:
        """使用 ``git cat-file`` 检查标识是否能解析为 commit 对象。"""

        result = subprocess.run(
            ["git", "-C", str(cache_path), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.command_timeout_seconds,
            check=False,
        )
        return result.returncode == 0

    def _cache_path(self, repo: str) -> Path:
        """把 ``owner/name`` 映射成稳定的单层缓存目录名。"""

        return self.cache_root / repo.replace("/", "__")

    def _task_path(self, instance_id: str) -> Path:
        """为任务生成不会逃逸 workspace 根目录的路径。"""

        safe_name = _SAFE_PATH_COMPONENT.sub("_", instance_id)
        return self.workspace_root / safe_name

    def _run_git(self, repository: Path, *arguments: str) -> str:
        """在指定仓库中运行 Git，并复用统一的错误处理逻辑。"""

        return self._run(
            ["git", "-C", str(repository), *arguments],
            cwd=repository,
        )

    def _run(self, command: Sequence[str], *, cwd: Path) -> str:
        """执行外部命令，捕获输出、应用超时并将失败统一包装。"""

        try:
            result = subprocess.run(
                list(command),
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.command_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RepositoryError(
                f"command could not run: {' '.join(command)}: {error}"
            ) from error

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            raise RepositoryError(
                f"command failed ({result.returncode}): {' '.join(command)}\n{detail}"
            )
        return result.stdout
