"""Create immutable run directories and persist observable experiment data."""

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
    """Raised when run artifacts cannot be created without data loss."""


_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunSession:
    """A single task run whose directory cannot be silently reused."""

    def __init__(
        self,
        path: Path,
        task: SWEbenchTask,
        metadata: dict[str, Any],
        started_monotonic: float,
    ) -> None:
        self.path = path
        self.task = task
        self._metadata = metadata
        self._started_monotonic = started_monotonic
        self._finished = False

    def collect_patch(self, repository: Path | str) -> str:
        """Collect tracked and untracked changes as a binary-safe Git diff."""

        repository_path = Path(repository).resolve()
        self._run_git(repository_path, "add", "--intent-to-add", "--all")
        return self._run_git(
            repository_path,
            "diff",
            "--binary",
            "--no-ext-diff",
            "--",
        )

    def finalize(
        self,
        *,
        exit_code: int,
        agent_log: str,
        test_output: str,
        events: Sequence[Mapping[str, Any]],
        patch: str,
    ) -> None:
        if self._finished:
            raise RunArtifactError(f"run has already been finalized: {self.path}")

        end_time = _utc_now()
        runtime_seconds = round(time.monotonic() - self._started_monotonic, 6)
        status = "completed" if exit_code == 0 else "failed"

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
                "official_evaluation": None,
            },
        )

        self._metadata.update(
            {
                "end_time": end_time,
                "runtime_seconds": runtime_seconds,
                "status": status,
                "event_count": len(events),
            }
        )
        self._write_json("metadata.json", self._metadata)
        self._finished = True

    def _write_text(self, name: str, content: str) -> None:
        destination = self.path / name
        temporary = self.path / f".{name}.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)

    def _write_json(self, name: str, content: Mapping[str, Any]) -> None:
        serialized = json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        self._write_text(name, f"{serialized}\n")

    @staticmethod
    def _run_git(repository: Path, *arguments: str) -> str:
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
    """Start a run and establish its metadata before the agent executes."""

    def __init__(self, runs_root: Path | str) -> None:
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
    ) -> RunSession:
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
        session = RunSession(run_path, task, metadata, time.monotonic())
        session._write_json("metadata.json", metadata)
        session._write_text("prompt.txt", prompt)
        return session
