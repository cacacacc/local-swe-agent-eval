"""Create run directories that cannot be silently overwritten, and persist observable experiment data.

Each run saves metadata, prompt, agent log, trajectory, Git patch, test output,
and a result summary. What is recorded here is verifiable behavior; hidden model reasoning is neither recorded nor inferred.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence

from benchmark.task import SWEbenchTask


class RunArtifactError(RuntimeError):
    """Raised when run artifacts cannot be created without losing data."""


# The instance ID comes from an external dataset and must first be converted into a safe single-level directory name.
_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")

# These directories contain only interpreter environments or caches the Agent created temporarily in the task worktree;
# writing them into the prediction would produce tens of thousands of meaningless diff lines and could even mask real source
# changes. Excluding them via Git pathspec rather than deleting files preserves the failure scene and avoids feeding run
# artifacts into the official evaluation.
_PATCH_EXCLUDE_PATHS = (
    ":(exclude,glob)**/.venv/**",
    ":(exclude,glob)**/venv/**",
    ":(exclude,glob)**/env/**",
    ":(exclude,glob)**/test_env/**",
    ":(exclude,glob)**/.tox/**",
    ":(exclude,glob)**/.nox/**",
    ":(exclude,glob)**/__pycache__/**",
    ":(exclude,glob)**/.pytest_cache/**",
    ":(exclude,glob)**/.mypy_cache/**",
    ":(exclude,glob)**/*.pyc",
)


def patch_pathspecs() -> tuple[str, ...]:
    """Return the stable Git pathspec for patch collection, shared by saving and Docker tests."""

    return (".", *_PATCH_EXCLUDE_PATHS)


def _utc_now() -> str:
    """Produce a timezone-aware UTC timestamp so different machines' local timezones cause no ambiguity."""

    return datetime.now(timezone.utc).isoformat()


class RunSession:
    """One run session for a single task; it cannot be finalized again once complete."""

    def __init__(
        self,
        path: Path,
        task: SWEbenchTask,
        metadata: dict[str, Any],
        started_monotonic: float,
        patch_base_commit: str,
    ) -> None:
        """Store the run context; the monotonic clock is used specifically to measure durations accurately."""

        self.path = path
        self.task = task
        self._metadata = metadata
        self._started_monotonic = started_monotonic
        self._patch_base_commit = patch_base_commit
        self._finished = False

    def collect_patch(self, repository: Path | str) -> str:
        """Collect committed, tracked, and untracked changes relative to the task baseline.

        ``git diff`` does not include untracked files by default, so first use ``--intent-to-add``
        to mark them as "intended to add" without actually creating a commit. The diff must explicitly take the task's
        isolated single-commit baseline as its left side; the Agent may commit on its own, and comparing only the working
        tree would misjudge an already-committed valid fix as an empty patch. The upstream original SHA is used only for
        auditing and cannot take part in the diff as an object that does not exist in the isolated repository.
        """

        repository_path = Path(repository).resolve()
        self._run_git(
            repository_path,
            "add",
            "--intent-to-add",
            "--all",
            "--",
            *patch_pathspecs(),
        )
        return self._run_git(
            repository_path,
            "diff",
            "--binary",
            "--no-ext-diff",
            self._patch_base_commit,
            "--",
            *patch_pathspecs(),
        )

    def finalize(
        self,
        *,
        exit_code: int,
        agent_log: str,
        test_output: str,
        events: Sequence[Mapping[str, Any]],
        patch: str,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically write this run's final artifacts and mark the session complete.

        ``exit_code == 0`` only means the Agent process exited normally; it does not mean the SWE-bench issue
        is resolved; the official evaluation result therefore stays ``None`` until the harness fills it in later.
        """

        if self._finished:
            raise RunArtifactError(f"run has already been finalized: {self.path}")

        end_time = _utc_now()
        runtime_seconds = round(time.monotonic() - self._started_monotonic, 6)
        # 124 follows the GNU timeout convention; classifying it separately is what allows an accurate timeout rate.
        status = "completed" if exit_code == 0 else "timeout" if exit_code == 124 else "failed"
        patch_line_count = sum(
            1
            for line in patch.splitlines()
            if (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
        )

        # Write the detailed artifacts first, then update metadata, so an exception still preserves as much evidence as possible.
        self._write_text("agent.log", agent_log)
        self._write_text("test_output.log", test_output)
        self._write_text("patch.diff", patch)
        self._write_json(
            "trajectory.json",
            {"schema_version": 1, "events": list(events)},
        )
        self._write_json(
            "result.json",
            {
                "schema_version": 1,
                "run_status": status,
                "agent_exit_code": exit_code,
                "patch_generated": bool(patch.strip()),
                "patch_line_count": patch_line_count,
                "metrics": dict(metrics) if metrics is not None else {},
                "official_evaluation": None,
            },
        )

        self._metadata.update(
            {
                "end_time": end_time,
                "runtime_seconds": runtime_seconds,
                "status": status,
                "event_count": len(events),
                "metrics": dict(metrics) if metrics is not None else {},
            }
        )
        self._write_json("metadata.json", self._metadata)
        self._finished = True

    def _write_text(self, name: str, content: str) -> None:
        """Write a temporary file in the same directory first, then replace, to avoid leaving a half-written file."""

        destination = self.path / name
        temporary = self.path / f".{name}.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)

    def _write_json(self, name: str, content: Mapping[str, Any]) -> None:
        """Save audit-friendly JSON with stable key order and UTF-8 encoding."""

        serialized = json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        self._write_text(name, f"{serialized}\n")

    @staticmethod
    def _run_git(repository: Path, *arguments: str) -> str:
        """Run the Git commands needed for patch collection and preserve failure details."""

        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            raise RunArtifactError(
                f"git command failed ({result.returncode}): "
                f"git {' '.join(arguments)}\n{detail}"
            )
        return result.stdout


class RunManager:
    """Create the run directory and write initial metadata before the Agent executes."""

    def __init__(self, runs_root: Path | str) -> None:
        """Store the resolved run root so later steps are unaffected by changes to the current working directory."""

        self.runs_root = Path(runs_root).resolve()

    def start(
        self,
        task: SWEbenchTask,
        *,
        phase: str,
        agent: str,
        model: str,
        prompt: str,
        configuration: Mapping[str, Any] | None = None,
        patch_base_commit: str | None = None,
    ) -> RunSession:
        """Start a new session and immediately persist the prompt and initial metadata.

        The directory uses ``exist_ok=False``: old results for the same instance ID are not silently
        overwritten by a new run. To repeat an experiment, the caller must supply a new run root or run ID.
        """

        safe_instance_id = _SAFE_PATH_COMPONENT.sub("_", task.instance_id)
        run_path = self.runs_root / safe_instance_id
        try:
            run_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RunArtifactError(
                f"run directory already exists; refusing to overwrite: {run_path}"
            ) from error
        except OSError as error:
            raise RunArtifactError(f"cannot create run directory {run_path}: {error}") from error

        start_time = _utc_now()
        # Persist before the Agent starts so the task and launch config are known even if the process crashes.
        metadata = {
            "schema_version": 1,
            "instance_id": task.instance_id,
            "repository": task.repo,
            "base_commit": task.base_commit,
            "experiment_phase": phase,
            "agent": agent,
            "model": model,
            "start_time": start_time,
            "end_time": None,
            "runtime_seconds": None,
            "status": "running",
            "configuration": dict(configuration) if configuration is not None else None,
        }
        # Old callers may still pass a plain checkout, in which case the original base commit remains valid; the new isolated
        # RepositoryManager explicitly passes the re-initialized workspace baseline.
        effective_patch_base = patch_base_commit or task.base_commit
        metadata["workspace_base_commit"] = effective_patch_base
        session = RunSession(
            run_path,
            task,
            metadata,
            time.monotonic(),
            effective_patch_base,
        )
        session._write_json("metadata.json", metadata)
        session._write_text("prompt.txt", prompt)
        return session
