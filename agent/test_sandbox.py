"""Run repository-visible tests inside the official SWE-bench instance image.

The sandbox only applies the Agent's current Git patch to the image's bundled
``/testbed``; it never injects the official ``test_patch`` or evaluation scripts. The
container has networking disabled and is removed after each command, so test
dependencies are fully isolated from the host scheduler's virtual environment, and
files produced by tests are never written back into the Agent worktree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid
from typing import Sequence

from tracking.run_manager import patch_pathspecs


class TestSandboxError(RuntimeError):
    """Raised when the image is missing, the Git patch cannot be produced, or Docker cannot start."""


@dataclass(frozen=True, slots=True)
class VisibleTestResult:
    """Startup evidence, duration, cache status, and actual image for one visible test run."""

    exit_code: int
    output: str
    image: str
    timed_out: bool
    command_started: bool
    duration_seconds: float = 0.0
    cache_hit: bool = False
    infrastructure_error: str | None = None

    @property
    def evidence_valid(self) -> bool:
        """Return ``True`` only when the test command actually started and the runner infrastructure is available."""

        return self.command_started and self.infrastructure_error is None


def image_candidates(instance_id: str) -> tuple[str, str]:
    """Return the local build name and the Docker Hub release name, in an order that prefers reusing the local image."""

    normalized = instance_id.lower()
    local = f"sweb.eval.x86_64.{normalized}:latest"
    remote = f"swebench/{local}".replace("__", "_1776_")
    return local, remote


def truncate_output(output: str, maximum_chars: int) -> str:
    """Keep the head and tail of the output, so both startup errors and the final test summary remain visible."""

    if maximum_chars <= 0:
        raise ValueError("maximum_chars must be positive")
    if len(output) <= maximum_chars:
        return output
    marker = "\n... [tool output truncated by visible-test sandbox] ...\n"
    available = max(0, maximum_chars - len(marker))
    head_size = min(2000, available // 3)
    tail_size = available - head_size
    return f"{output[:head_size]}{marker}{output[-tail_size:]}"


def _detect_infrastructure_error(
    command: Sequence[str],
    output: str,
    *,
    exit_code: int,
    command_started: bool,
    timed_out: bool,
) -> str | None:
    """Detect a missing test framework or a nonexistent entry point without swallowing ordinary test failures.

    Only narrow matches directly related to the launched command are accepted here. A
    ``ModuleNotFoundError`` for an arbitrary dependency inside the test code may be a
    real regression introduced by the candidate patch and must not be broadly labeled
    an environment fault; the current focus is the missing pytest runner that once
    produced false evidence for SymPy, plus cases where the interpreter or runner
    script simply cannot execute.
    """

    if not command_started or timed_out or exit_code == 0:
        return None
    normalized = output.lower()
    if (
        len(command) >= 3
        and command[0] in {"python", "python3"}
        and tuple(command[1:3]) == ("-m", "pytest")
        and re.search(r"no module named ['\"]?pytest(?:['\"]|\b)", normalized)
    ):
        return "test runner unavailable: pytest module is not installed"
    if re.search(
        r"(?:exec:\s*)?(?:python3?|pytest|[^\s:]+):\s*(?:command\s+)?not found",
        normalized,
    ):
        return "test executable is not available in the image"
    if (
        command[0] in {"python", "python3"}
        and len(command) >= 2
        and "can't open file" in normalized
        and ("no such file or directory" in normalized or "[errno 2]" in normalized)
    ):
        return "test runner script is not available in the image"
    return None


class VisibleTestSandbox:
    """Create an offline, one-shot repository test container using the Docker CLI."""

    def __init__(
        self,
        *,
        timeout_seconds: int,
        max_output_chars: int,
        docker_executable: str = "docker",
    ) -> None:
        """Validate resource bounds; only images from the fixed SWE-bench naming scheme are allowed."""

        if timeout_seconds <= 0 or max_output_chars <= 0:
            raise ValueError("test timeout and output limit must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.docker_executable = docker_executable

    def resolve_image(self, instance_id: str) -> str:
        """Check only local images, never implicitly pulling over the network during a formal solve."""

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
        """Reuse the local image, or auto-pull the official image only during an explicit environment-preparation phase.

        ``allow_pull`` must come from the command line's ``--allow-network-preparation``;
        the Agent solve process only calls ``resolve_image``, so it cannot use this to
        bypass the offline experiment boundary.
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
        # Inspect again after a successful pull so Docker's zero exit code is not
        # taken directly as evidence that the image is usable.
        return self.resolve_image(instance_id)

    def image_digest(self, image: str) -> str:
        """Read the local immutable image ID so run metadata pins the actual test environment."""

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
        apply_patch: bool = True,
        timeout_seconds: float | None = None,
    ) -> VisibleTestResult:
        """Execute the argv in the isolated image, letting the caller decide whether to apply the candidate patch.

        ``apply_patch=False`` tests the baseline checkout in the official instance
        image directly, for the parent process to compare against the subsequent
        patched result. The caller may tighten the per-command budget with
        ``timeout_seconds``; when omitted, the sandbox default applies, and an override
        value must never enlarge the default safety boundary.
        """

        repository_path = Path(repository).resolve()
        if not repository_path.is_dir():
            raise TestSandboxError(f"repository does not exist: {repository_path}")
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", base_commit):
            raise TestSandboxError("base commit must be a 7-40 character Git hex id")
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise TestSandboxError("test command must contain non-empty argv items")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("test timeout override must be positive")

        effective_timeout = min(
            self.timeout_seconds,
            timeout_seconds if timeout_seconds is not None else self.timeout_seconds,
        )
        started_monotonic = time.monotonic()

        image = self.resolve_image(instance_id)
        patch = (
            self._collect_patch(repository_path, base_commit)
            if apply_patch
            else ""
        )
        container_name = self._container_name(instance_id)
        # The unique marker is printed only after git apply succeeds and before the
        # test is exec'd. When Docker returns a result but this marker is absent, the
        # scheduler must not mistake an image or patch-preparation failure for a real
        # test execution.
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
                # Command arguments are passed verbatim through "$@" so the issue text
                # or argv is never concatenated into a shell.
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
                    timeout=effective_timeout,
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
                output += f"\nvisible test timed out after {effective_timeout:g}s\n"
            finally:
                # On timeout the docker client may exit first, so clean up the exact
                # container by its unguessable UUID name, preventing a background test
                # from continuing to consume memory and CPU.
                subprocess.run(
                    [self.docker_executable, "rm", "--force", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )

        command_started = started_marker in output
        output = output.replace(f"{started_marker}\n", "", 1)
        infrastructure_error = _detect_infrastructure_error(
            command,
            output,
            exit_code=exit_code,
            command_started=command_started,
            timed_out=timed_out,
        )
        return VisibleTestResult(
            exit_code=exit_code,
            output=truncate_output(output, self.max_output_chars),
            image=image,
            timed_out=timed_out,
            command_started=command_started,
            duration_seconds=round(time.monotonic() - started_monotonic, 6),
            cache_hit=False,
            infrastructure_error=infrastructure_error,
        )

    @staticmethod
    def _collect_patch(repository: Path, base_commit: str) -> str:
        """Collect committed, tracked, and untracked modifications; an empty patch is allowed for baseline tests."""

        add = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "add",
                "--intent-to-add",
                "--all",
                "--",
                *patch_pathspecs(),
            ],
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
                *patch_pathspecs(),
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
        """Generate a one-shot container name that cannot collide with the formal harness or concurrent tests."""

        safe_id = re.sub(r"[^a-z0-9_.-]+", "-", instance_id.lower()).strip("-.")
        return f"local-swe-visible-{safe_id[:40]}-{uuid.uuid4().hex[:12]}"
