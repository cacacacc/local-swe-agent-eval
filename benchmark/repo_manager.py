"""为 SWE-bench 任务准备不暴露后续历史的独立 Git 仓库。

同一上游仓库仍只维护一份共享 clone 以节约下载与磁盘成本，但 Agent 工作目录
不再是共享 clone 的 worktree。准备器只导出数据集指定 ``base_commit`` 的文件树，
随后在隔离目录中重新初始化一个单提交、无 remote 的 Git 仓库。这样 Agent 仍可
使用 ``git diff``，却无法通过 ``git log --all`` 或对象数据库读取题目之后的修复。
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Sequence

from .task import SWEbenchTask


class RepositoryError(RuntimeError):
    """当仓库无法按要求安全、可复现地准备时抛出。"""


# 将外部 instance ID 转成单个安全路径组件，避免斜杠等字符改变目录层级。
_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class PreparedRepository:
    """已从上游基线导出并重新初始化的单提交任务仓库。"""

    task: SWEbenchTask
    path: Path
    resolved_commit: str
    workspace_base_commit: str


class RepositoryManager:
    """管理共享只读来源缓存与不共享 Git 对象的任务仓库。"""

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
        """导出基线文件树并创建全新的单提交仓库，拒绝覆盖已有目录。

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

        # 先解析上游 commit，防止短 SHA 歧义；这个哈希只进入审计元数据，不会把
        # 上游对象数据库挂载到 Agent 工作区。
        resolved_commit = self._run_git(
            cache_path, "rev-parse", f"{task.base_commit}^{{commit}}"
        ).strip()
        workspace_base_commit = self._export_isolated_repository(
            cache_path,
            task_path,
            resolved_commit,
        )
        return PreparedRepository(
            task,
            task_path,
            resolved_commit,
            workspace_base_commit,
        )

    def _export_isolated_repository(
        self,
        cache_path: Path,
        task_path: Path,
        resolved_commit: str,
    ) -> str:
        """从指定 tree 创建单提交仓库，并返回隔离仓库的基线 commit。

        使用 ``git archive`` 而不是复制共享 clone，确保 ``.git/objects``、refs、
        hooks 和 remote 配置均不会越过准备边界。临时 tar 与目标目录位于同一受控
        workspace 根目录；任何失败都会删除尚未交给 Agent 的不完整目录。
        """

        archive_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".swe-base-",
                suffix=".tar",
                dir=self.workspace_root,
                delete=False,
            ) as archive:
                archive_path = Path(archive.name)
            self._run_git(
                cache_path,
                "archive",
                "--format=tar",
                f"--output={archive_path}",
                resolved_commit,
            )
            task_path.mkdir(parents=False, exist_ok=False)
            with tarfile.open(archive_path, mode="r:") as source:
                self._validate_archive_members(source)
                # 项目支持早期 Python 3.10 补丁版本，运行时检查 filter 参数而非
                # 假定其存在。显式验证完成后使用 fully_trusted 仅用于关闭新版默认
                # 策略警告，并保留 Git tree 中的可执行位与安全 symlink。
                if "filter" in inspect.signature(source.extractall).parameters:
                    source.extractall(task_path, filter="fully_trusted")
                else:
                    source.extractall(task_path)

            self._run_git(task_path, "init", "--initial-branch=agent-work")
            self._run_git(task_path, "config", "user.name", "Local SWE Agent")
            self._run_git(
                task_path,
                "config",
                "user.email",
                "local-swe-agent@example.invalid",
            )
            # --force 使上游已经跟踪但恰好命中 .gitignore 的文件仍进入基线提交。
            self._run_git(task_path, "add", "--all", "--force")
            self._run_git(
                task_path,
                "commit",
                "--no-gpg-sign",
                "-m",
                f"Isolated SWE-bench baseline {resolved_commit}",
            )
            workspace_base_commit = self._run_git(
                task_path, "rev-parse", "HEAD"
            ).strip()
            if self._run_git(task_path, "rev-list", "--all", "--count").strip() != "1":
                raise RepositoryError("isolated repository must contain exactly one commit")
            if self._run_git(task_path, "remote").strip():
                raise RepositoryError("isolated repository unexpectedly contains a remote")
            return workspace_base_commit
        except Exception:
            # task_path 在本方法调用前已确认不存在，因此只清理由本次准备创建的
            # 精确目录，不会触碰用户已有 workspace。
            if task_path.exists():
                shutil.rmtree(task_path)
            raise
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)

    @staticmethod
    def _validate_archive_members(source: tarfile.TarFile) -> None:
        """拒绝 archive 路径逃逸、设备文件和逃逸目标链接。

        ``git archive`` 正常只产生文件、目录与 symlink，但缓存仍属于外部输入。
        在兼容 Python 3.10 的前提下逐项验证，避免恶意仓库在 workspace 外写入。
        """

        for member in source.getmembers():
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RepositoryError(f"unsafe archive member path: {member.name}")
            if member.isdev() or member.isfifo():
                raise RepositoryError(f"unsupported archive member type: {member.name}")
            if not (member.issym() or member.islnk()):
                continue
            link_path = PurePosixPath(member.linkname)
            if link_path.is_absolute():
                raise RepositoryError(f"unsafe absolute archive link: {member.name}")
            # symlink 相对其父目录解析；hardlink 名称按 tar 约定相对 archive 根。
            parent = member_path.parent if member.issym() else PurePosixPath()
            normalized = posixpath.normpath(str(parent / link_path))
            if normalized == ".." or normalized.startswith("../"):
                raise RepositoryError(f"archive link escapes workspace: {member.name}")

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
