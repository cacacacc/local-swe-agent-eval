"""Prepare an isolated Git repository for a SWE-bench task that hides the later history.

A single shared clone is still maintained per upstream repository to save download and
disk costs, but the Agent working directory is no longer a worktree of that shared
clone. The preparer exports only the file tree at the dataset's ``base_commit`` and
then re-initializes a single-commit, remote-less Git repository in an isolated
directory. The Agent can therefore still use ``git diff``, but cannot read the fixes
that follow the task via ``git log --all`` or the object database.
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
    """Raised when a repository cannot be prepared safely and reproducibly as required."""


# Convert external instance IDs into a single safe path component so that slashes and
# similar characters cannot alter the directory hierarchy.
_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class PreparedRepository:
    """A single-commit task repository exported from the upstream baseline and re-initialized."""

    task: SWEbenchTask
    path: Path
    resolved_commit: str
    workspace_base_commit: str


class RepositoryManager:
    """Manage a shared read-only source cache and task repositories that do not share Git objects."""

    def __init__(
        self,
        cache_root: Path | str,
        workspace_root: Path | str,
        *,
        command_timeout_seconds: int = 600,
    ) -> None:
        """Store and normalize the cache paths, and validate the external command timeout."""

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
        """Export the baseline file tree and create a brand-new single-commit repository, refusing to overwrite an existing directory.

        ``allow_network`` explicitly separates the environment-preparation phase from
        the formal offline-solving phase: when the cache or commit is missing, offline
        mode must fail immediately instead of silently accessing the remote repository.
        """

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        cache_path = self._cache_path(task.repo)
        task_path = self._task_path(task.instance_id)

        if task_path.exists():
            raise RepositoryError(
                f"task workspace already exists; refusing to overwrite it: {task_path}"
            )

        # The first-time clone is only allowed during the phase that explicitly enables network access.
        if not cache_path.exists():
            if not allow_network:
                raise RepositoryError(
                    f"repository is not cached and network is disabled: {task.repo}"
                )
            self._clone(task.repo, cache_path)
        else:
            self._verify_cache(cache_path, task.repo)

        # An existing clone does not guarantee the target history is present; fetch once during preparation if needed.
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

        # Resolve the upstream commit first to avoid short-SHA ambiguity; this hash only
        # enters audit metadata and does not mount the upstream object database into the Agent workspace.
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
        """Create a single-commit repository from the given tree and return the isolated repository's baseline commit.

        ``git archive`` is used instead of copying the shared clone to ensure that
        ``.git/objects``, refs, hooks, and remote configuration never cross the
        preparation boundary. The temporary tar and the target directory live under the
        same controlled workspace root; any failure deletes the incomplete directory
        before it is handed to the Agent.
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
                # The project supports early Python 3.10 patch releases, so check for the
                # filter parameter at runtime rather than assuming it exists. After explicit
                # validation, fully_trusted is used only to silence the newer default-policy
                # warning and to preserve the executable bits and safe symlinks from the Git tree.
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
            # --force keeps files that upstream already tracks but that happen to match .gitignore in the baseline commit.
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
            # task_path was confirmed not to exist before this method was called, so only the
            # exact directory created by this preparation is removed; the user's existing workspace is never touched.
            if task_path.exists():
                shutil.rmtree(task_path)
            raise
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)

    @staticmethod
    def _validate_archive_members(source: tarfile.TarFile) -> None:
        """Reject archive path escapes, device files, and links whose targets escape.

        ``git archive`` normally produces only files, directories, and symlinks, but the
        cache is still external input. Validate each member while remaining compatible
        with Python 3.10, so a malicious repository cannot write outside the workspace.
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
            # Symlinks resolve relative to their parent directory; hardlink names are relative to the archive root per tar conventions.
            parent = member_path.parent if member.issym() else PurePosixPath()
            normalized = posixpath.normpath(str(parent / link_path))
            if normalized == ".." or normalized.startswith("../"):
                raise RepositoryError(f"archive link escapes workspace: {member.name}")

    def _clone(self, repo: str, destination: Path) -> None:
        """Create a shared clone with no working tree; individual tasks are checked out later via a worktree."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{repo}.git"
        self._run(
            ["git", "clone", "--no-checkout", url, str(destination)],
            cwd=destination.parent,
        )

    def _verify_cache(self, cache_path: Path, repo: str) -> None:
        """Confirm the cache is a Git clone and that origin points at the expected GitHub repository."""

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
        """Check whether the identifier resolves to a commit object using ``git cat-file``."""

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
        """Map ``owner/name`` to a stable single-level cache directory name."""

        return self.cache_root / repo.replace("/", "__")

    def _task_path(self, instance_id: str) -> Path:
        """Generate a task path that cannot escape the workspace root."""

        safe_name = _SAFE_PATH_COMPONENT.sub("_", instance_id)
        return self.workspace_root / safe_name

    def _run_git(self, repository: Path, *arguments: str) -> str:
        """Run Git in the given repository, reusing the unified error handling."""

        return self._run(
            ["git", "-C", str(repository), *arguments],
            cwd=repository,
        )

    def _run(self, command: Sequence[str], *, cwd: Path) -> str:
        """Run an external command, capture output, apply the timeout, and wrap failures uniformly."""

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
